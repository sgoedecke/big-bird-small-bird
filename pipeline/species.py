"""Build data/species.json: Victorian birds with body lengths (Garnett et al. 2015)
and iNaturalist taxon ids + Victorian research-grade observation counts."""
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ABD = ROOT / "data/raw/abd.csv"
OUT = ROOT / "data/species.json"
INAT = "https://api.inaturalist.org/v1"
VIC_PLACE_ID = 7830

COL_COMMON, COL_SCI, COL_FAMILY, COL_ORDER = 2, 6, 8, 10
COL_IS_SPECIES, COL_VIC = 16, 72
COL_LEN, COL_LEN_MIN, COL_LEN_MAX = 95, 96, 97
INAT_KEYS = ("inat_taxon_id", "inat_name", "inat_common_name", "vic_observations")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def load_garnett():
    rows = list(csv.reader(open(ABD, encoding="latin-1")))[1:]
    out = {}
    for r in rows:
        if r[COL_IS_SPECIES] != "1" or r[COL_VIC] not in ("Core", "Introduced"):
            continue
        # Some species have a separate "- non-breeding population" row; merge by scientific name
        common = r[COL_COMMON].split(" - ")[0].strip()
        if r[COL_SCI] in out and out[r[COL_SCI]]["length_cm"] is not None:
            continue
        out[r[COL_SCI]] = {
            "slug": slugify(common),
            "common_name": common,
            "scientific_name": r[COL_SCI],
            "family": r[COL_FAMILY],
            "order": r[COL_ORDER],
            "introduced": r[COL_VIC] == "Introduced",
            "length_cm": num(r[COL_LEN]),
            "length_min_cm": num(r[COL_LEN_MIN]),
            "length_max_cm": num(r[COL_LEN_MAX]),
        }
    return [s for s in out.values() if s["length_cm"] is not None]


def inat_get(path, **params):
    for attempt in range(5):
        try:
            r = requests.get(f"{INAT}/{path}", params=params, timeout=30,
                         headers={"User-Agent": "bird-sizing-game/0.1"})
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(1)
        return r.json()
    raise RuntimeError("rate limited")


def match_inat(sp):
    res = inat_get("taxa", q=sp["scientific_name"], rank="species", per_page=10)["results"]
    hits = [t for t in res if t["name"].lower() == sp["scientific_name"].lower()]
    if not hits:
        res = inat_get("taxa", q=sp["common_name"], rank="species", per_page=10)["results"]
        hits = [t for t in res if t.get("iconic_taxon_name") == "Aves"][:1]
    if not hits:
        return None
    t = hits[0]
    if not t.get("is_active", True):
        # Follow synonym to the current taxon
        cur = (t.get("current_synonymous_taxon_ids") or [None])[0]
        if cur:
            t = inat_get(f"taxa/{cur}")["results"][0]
    return {"inat_taxon_id": t["id"], "inat_name": t["name"], "inat_common_name": t.get("preferred_common_name")}


def main():
    species = load_garnett()
    existing = {s["slug"]: s for s in json.loads(OUT.read_text())} if OUT.exists() else {}
    for i, sp in enumerate(species):
        prev = existing.get(sp["slug"], {})
        if "inat_taxon_id" in prev:
            sp.update({k: prev[k] for k in INAT_KEYS})
            continue
        m = match_inat(sp)
        if m:
            sp.update(m)
            sp["vic_observations"] = inat_get(
                "observations", taxon_id=m["inat_taxon_id"], place_id=VIC_PLACE_ID,
                quality_grade="research", per_page=0)["total_results"]
        else:
            print("no iNat match:", sp["common_name"], sp["scientific_name"], file=sys.stderr)
        print(f"[{i + 1}/{len(species)}] {sp['common_name']}: {sp.get('vic_observations')}", flush=True)
        if i % 20 == 0:
            OUT.write_text(json.dumps(species, indent=1))
    OUT.write_text(json.dumps(species, indent=1))


if __name__ == "__main__":
    main()
