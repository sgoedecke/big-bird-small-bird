"""Helpers for reviewing silhouettes by looking at images (e.g. by an AI agent).

    uv run python pipeline/agent_review.py sheet <slug> [--out PATH]
        Render the top candidates: photo + silhouette with a 10% grid, the auto bill/tail
        endpoints (A = green, B = blue) and the measured body path (yellow).

    uv run python pipeline/agent_review.py pick <slug> <n> [ax ay bx by] [--out PATH]
        Choose candidate n (1-based, as numbered on the sheet). Optionally override the
        endpoints with percentages of the silhouette's width/height (0-100), e.g.
        `pick emu 2 12 8 88 55`. Renders a confirmation image of the result.

    uv run python pipeline/agent_review.py reject <slug>
        Mark the species as having no usable silhouette.

    uv run python pipeline/agent_review.py fix <slug> ax ay bx by
        Correct the endpoints of the current data/review.json pick in place (keeps the chosen
        photo, including human picks). Records "endpoints_by": "agent".

    uv run python pipeline/agent_review.py review-sheet <slug>... [--out PATH]
        Render the current review.json picks for several species on the same grid.

Decisions go to data/agent_review/<slug>.json; `merge` folds them into data/review.json
(without overwriting human decisions).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import geodesic

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ROOT / "data/species.json"
SILS = ROOT / "data/silhouettes"
PHOTOS = ROOT / "data/photos"
AGENT = ROOT / "data/agent_review"
REVIEW = ROOT / "data/review.json"
SHEETS = ROOT / "data/sheets"
TOP_K = 6
S = 300  # silhouette panel size
P = 150  # photo thumbnail size


def font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def candidates(slug):
    return [c for c in json.loads((SILS / slug / "candidates.json").read_text()) if c.get("sil_file")][:TOP_K]


def species_info(slug):
    return next(s for s in json.loads(SPECIES.read_text()) if s["slug"] == slug)


def panel(slug, c, endpoints=None, path=None, label=""):
    """Silhouette with grid, endpoints and path, plus the photo, side by side."""
    sil = Image.open(SILS / slug / c["sil_file"])
    w, h = sil.size
    scale = (S - 20) / max(w, h)
    sw, sh = round(w * scale), round(h * scale)
    ox, oy = (S - sw) // 2, (S - sh) // 2
    cell = Image.new("RGB", (S + P, S + 22), "white")
    bg = Image.new("RGB", (S, S), (245, 242, 232))
    small = sil.resize((sw, sh), Image.LANCZOS)
    bg.paste((25, 25, 25), (ox, oy), mask=small.split()[3])
    d = ImageDraw.Draw(bg)
    f = font(10)
    for i in range(11):
        x = ox + sw * i / 10
        y = oy + sh * i / 10
        col = (170, 170, 220) if i % 5 else (110, 110, 200)
        d.line([(x, oy), (x, oy + sh)], fill=col, width=1)
        d.line([(ox, y), (ox + sw, y)], fill=col, width=1)
        if i % 2 == 0:
            d.text((x - 6, max(0, oy - 11)), str(i * 10), fill=(80, 80, 160), font=f)
            d.text((max(0, ox - 18), y - 5), str(i * 10), fill=(80, 80, 160), font=f)
    ep = endpoints or c["endpoints"]
    pth = path or c.get("path") or ep
    pts = [(ox + x * scale, oy + y * scale) for x, y in pth]
    d.line(pts, fill=(240, 190, 0), width=3)
    for (x, y), colr, name in zip(ep, ((30, 180, 90), (40, 110, 230)), "AB"):
        cx, cy = ox + x * scale, oy + y * scale
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=colr, outline="white", width=2)
        d.text((cx + 7, cy - 7), name, fill=colr, font=font(13))
    cell.paste(bg, (0, 0))
    photo = Image.open(PHOTOS / slug / c["file"]).convert("RGB")
    photo.thumbnail((P - 6, S - 6))
    cell.paste(photo, (S + 3, 3))
    ImageDraw.Draw(cell).text((4, S + 3), label, fill="black", font=font(14))
    return cell


def cmd_sheet(slug, out):
    sp = species_info(slug)
    cands = candidates(slug)
    cols = 3
    rows = (len(cands) + cols - 1) // cols
    W = cols * (S + P + 10)
    img = Image.new("RGB", (W, 30 + rows * (S + 30)), "white")
    ImageDraw.Draw(img).text((6, 6), f"{sp['common_name']} ({sp['scientific_name']}) — {sp['length_cm']:g} cm",
                             fill="black", font=font(18))
    for i, c in enumerate(cands):
        cell = panel(slug, c, label=f"#{i + 1}  pose={c['profile_prob']:.2f}  score={c['score']:.2f}")
        img.paste(cell, ((i % cols) * (S + P + 10), 30 + (i // cols) * (S + 30)))
    out = Path(out or SHEETS / f"{slug}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)


def cmd_pick(slug, n, coords, out):
    c = candidates(slug)[n - 1]
    w, h = c["sil_size"]
    if coords:
        ax, ay, bx, by = coords
        endpoints = [[round(ax / 100 * w, 1), round(ay / 100 * h, 1)], [round(bx / 100 * w, 1), round(by / 100 * h, 1)]]
    else:
        endpoints = c["endpoints"]
    mask = np.asarray(Image.open(SILS / slug / c["sil_file"]))[..., 3] > 127
    length, path = geodesic(mask, *endpoints)
    entry = {"status": "ok", "photo_id": c["photo_id"], "endpoints": endpoints,
             "geo_length_px": round(length, 1), "path": path, "by": "agent"}
    AGENT.mkdir(parents=True, exist_ok=True)
    (AGENT / f"{slug}.json").write_text(json.dumps(entry))
    out = Path(out or SHEETS / f"{slug}_pick.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    panel(slug, c, endpoints, path, label=f"picked #{n}").save(out)
    print(out)


def reviewed_candidate(slug, review):
    e = review[slug]
    c = next(c for c in json.loads((SILS / slug / "candidates.json").read_text())
             if c["photo_id"] == e["photo_id"])
    return e, c


def cmd_fix(slug, coords, out):
    import fcntl
    lock = open(REVIEW.with_suffix(".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    review = json.loads(REVIEW.read_text())
    e, c = reviewed_candidate(slug, review)
    w, h = c["sil_size"]
    ax, ay, bx, by = coords
    endpoints = [[round(ax / 100 * w, 1), round(ay / 100 * h, 1)], [round(bx / 100 * w, 1), round(by / 100 * h, 1)]]
    mask = np.asarray(Image.open(SILS / slug / c["sil_file"]))[..., 3] > 127
    length, path = geodesic(mask, *endpoints)
    e.update(endpoints=endpoints, geo_length_px=round(length, 1), path=path, endpoints_by="agent")
    tmp = REVIEW.with_suffix(".tmp")
    tmp.write_text(json.dumps(review, indent=1))
    tmp.replace(REVIEW)
    fcntl.flock(lock, fcntl.LOCK_UN)
    if e.get("by") == "agent":
        (AGENT / f"{slug}.json").write_text(json.dumps(e))
    out = Path(out or SHEETS / f"{slug}_pick.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    panel(slug, c, endpoints, path, label=f"fixed {slug}").save(out)
    print(out)


def cmd_review_sheet(slugs, out):
    review = json.loads(REVIEW.read_text())
    cols = 3
    rows = (len(slugs) + cols - 1) // cols
    img = Image.new("RGB", (cols * (S + P + 10), rows * (S + 30)), "white")
    for i, slug in enumerate(slugs):
        e, c = reviewed_candidate(slug, review)
        cell = panel(slug, c, e["endpoints"], e.get("path"), label=slug)
        img.paste(cell, ((i % cols) * (S + P + 10), (i // cols) * (S + 30)))
    out = Path(out or SHEETS / "review_sheet.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out)


def cmd_reject(slug):
    AGENT.mkdir(parents=True, exist_ok=True)
    (AGENT / f"{slug}.json").write_text(json.dumps({"status": "rejected", "by": "agent"}))
    print("rejected", slug)


def cmd_merge():
    review = json.loads(REVIEW.read_text()) if REVIEW.exists() else {}
    n = 0
    for f in sorted(AGENT.glob("*.json")):
        slug = f.stem
        if slug in review and review[slug].get("by") != "agent":
            continue
        review[slug] = json.loads(f.read_text())
        n += 1
    REVIEW.write_text(json.dumps(review, indent=1))
    print(f"merged {n} agent decisions; {len(review)} total")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sheet"); a.add_argument("slug"); a.add_argument("--out")
    a = sub.add_parser("pick"); a.add_argument("slug"); a.add_argument("n", type=int)
    a.add_argument("coords", nargs="*", type=float); a.add_argument("--out")
    a = sub.add_parser("reject"); a.add_argument("slug")
    a = sub.add_parser("fix"); a.add_argument("slug"); a.add_argument("coords", nargs=4, type=float)
    a.add_argument("--out")
    a = sub.add_parser("review-sheet"); a.add_argument("slugs", nargs="+"); a.add_argument("--out")
    sub.add_parser("merge")
    args = ap.parse_args()
    if args.cmd == "sheet":
        cmd_sheet(args.slug, args.out)
    elif args.cmd == "pick":
        if args.coords and len(args.coords) != 4:
            ap.error("coords must be: ax ay bx by")
        cmd_pick(args.slug, args.n, args.coords, args.out)
    elif args.cmd == "fix":
        cmd_fix(args.slug, args.coords, args.out)
    elif args.cmd == "review-sheet":
        cmd_review_sheet(args.slugs, args.out)
    elif args.cmd == "reject":
        cmd_reject(args.slug)
    else:
        cmd_merge()


if __name__ == "__main__":
    main()
