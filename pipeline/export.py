"""Export reviewed silhouettes + metadata for the game into web/.

    uv run python pipeline/export.py           # reviewed species only
    uv run python pipeline/export.py --auto    # fall back to top-scoring candidate if unreviewed
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ROOT / "data/species.json"
SILS = ROOT / "data/silhouettes"
REVIEW = ROOT / "data/review.json"
WEB = ROOT / "web"
MAX_DIM = 512
MIN_OBS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--min-obs", type=int, default=MIN_OBS)
    args = ap.parse_args()
    species = json.loads(SPECIES.read_text())
    review = json.loads(REVIEW.read_text()) if REVIEW.exists() else {}
    out_dir = WEB / "sils"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    birds = []
    for sp in species:
        if (sp.get("vic_observations") or 0) < args.min_obs:
            continue
        cpath = SILS / sp["slug"] / "candidates.json"
        if not cpath.exists():
            continue
        cands = {c["photo_id"]: c for c in json.loads(cpath.read_text()) if c.get("sil_file")}
        rv = review.get(sp["slug"])
        if rv:
            if rv["status"] != "ok":
                continue
            cand = cands[rv["photo_id"]]
            length_px = rv.get("geo_length_px") or cand["geo_length_px"]
        elif args.auto and cands:
            cand = max(cands.values(), key=lambda c: c["score"])
            length_px = cand["geo_length_px"]
        else:
            continue
        img = Image.open(SILS / sp["slug"] / cand["sil_file"])
        scale = min(1.0, MAX_DIM / max(img.size))
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        img.save(out_dir / f"{sp['slug']}.png", optimize=True)
        birds.append({
            "slug": sp["slug"],
            "name": sp["common_name"],
            "sci": sp["scientific_name"],
            "length_cm": sp["length_cm"],
            "obs": sp.get("vic_observations") or 0,
            "img": f"sils/{sp['slug']}.png",
            "w": img.width,
            "h": img.height,
            "length_px": round(length_px * scale, 2),
            "credit": {
                "attribution": cand.get("attribution"),
                "license": cand.get("license"),
                "url": cand.get("observation_url"),
            },
            "reviewed": bool(rv),
        })
    (WEB / "birds.json").write_text(json.dumps(birds, separators=(",", ":")))
    print(f"exported {len(birds)} birds ({sum(b['reviewed'] for b in birds)} reviewed)")


if __name__ == "__main__":
    main()
