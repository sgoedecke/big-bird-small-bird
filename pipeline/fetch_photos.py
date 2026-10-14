"""Download candidate CC-licensed photos from iNaturalist (Victorian research-grade
observations) for each species into data/photos/<slug>/."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ROOT / "data/species.json"
PHOTOS = ROOT / "data/photos"
INAT = "https://api.inaturalist.org/v1"
VIC_PLACE_ID = 7830
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"
UA = {"User-Agent": "bird-sizing-game/0.1"}
THREADS = 8


def inat_get(path, **params):
    for attempt in range(5):
        try:
            r = requests.get(f"{INAT}/{path}", params=params, timeout=30, headers=UA)
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


def candidate_photos(taxon_id, n):
    """Yield photo dicts, preferring the most-faved observations in Victoria,
    falling back to Australia-wide if Victoria is sparse."""
    seen, ids = 0, set()
    for place in (VIC_PLACE_ID, 6744):  # 6744 = Australia
        res = inat_get("observations", taxon_id=taxon_id, place_id=place, quality_grade="research",
                       photo_license=LICENSES, order_by="votes", per_page=min(2 * n, 200))["results"]
        for obs in res:
            photo = next((p for p in obs.get("photos", []) if p.get("license_code")), None)
            if not photo or photo["id"] in ids:
                continue
            ids.add(photo["id"])
            yield obs, photo
            seen += 1
            if seen >= n:
                return


def fetch_species(sp, n):
    out = PHOTOS / sp["slug"]
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
    have = {m["photo_id"] for m in meta}
    if len(meta) >= n:
        return
    todo = [(o, p) for o, p in candidate_photos(sp["inat_taxon_id"], n) if p["id"] not in have][: n - len(meta)]

    def download(item):
        obs, photo = item
        url = photo["url"].replace("/square.", "/large.")
        dest = out / f"{photo['id']}.jpg"
        try:
            r = requests.get(url, timeout=60, headers=UA)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        dest.write_bytes(r.content)
        return {
            "photo_id": photo["id"],
            "file": dest.name,
            "observation_id": obs["id"],
            "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
            "license": photo["license_code"],
            "attribution": photo.get("attribution"),
            "faves": obs.get("faves_count", 0),
            "url": url,
        }

    with ThreadPoolExecutor(THREADS) as pool:
        meta += [m for m in pool.map(download, todo) if m]
    meta_path.write_text(json.dumps(meta, indent=1))


def restore_species(slug):
    """Re-download photo files listed in an existing meta.json (e.g. on a fresh machine)."""
    out = PHOTOS / slug
    meta = json.loads((out / "meta.json").read_text())

    def get(m):
        dest = out / m["file"]
        if dest.exists():
            return
        urls = [m["url"]] if m.get("url") else [
            f"https://inaturalist-open-data.s3.amazonaws.com/photos/{m['photo_id']}/large.{ext}"
            for ext in ("jpg", "jpeg", "png", "JPG", "JPEG")]
        for url in urls:
            try:
                r = requests.get(url, timeout=60, headers=UA)
            except requests.RequestException:
                continue
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return
        print("could not restore", slug, m["photo_id"], flush=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", help="species slugs (default: all)")
    ap.add_argument("-n", type=int, default=20, help="photos per species")
    ap.add_argument("--restore", action="store_true", help="download missing files for existing meta.json")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reverse", action="store_true", help="process species in reverse order")
    ap.add_argument("--min-obs", type=int, default=20, help="skip species with fewer Victorian observations")
    args = ap.parse_args()
    global THREADS
    THREADS = args.threads
    species = [s for s in json.loads(SPECIES.read_text())
               if "inat_taxon_id" in s and (s.get("vic_observations") or 0) >= args.min_obs]
    if args.slugs:
        species = [s for s in species if s["slug"] in args.slugs]
    if args.restore:
        slugs = [p.name for p in PHOTOS.iterdir() if (p / "meta.json").exists()]
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(restore_species, slugs))
        return
    if args.reverse:
        species.reverse()
    for i, sp in enumerate(species):
        print(f"[{i + 1}/{len(species)}] {sp['common_name']}", flush=True)
        fetch_species(sp, args.n)


if __name__ == "__main__":
    main()
