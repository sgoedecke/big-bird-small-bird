"""Debug helper: render a contact sheet of candidates (photo + silhouette) per species."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
T = 150


def sheet(slugs, out, k=8):
    rows = []
    for slug in slugs:
        cands = json.loads((ROOT / f"data/silhouettes/{slug}/candidates.json").read_text())[:k]
        row = Image.new("RGB", (k * T, 2 * T + 14), "white")
        d = ImageDraw.Draw(row)
        for i, c in enumerate(cands):
            ph = Image.open(ROOT / f"data/photos/{slug}/{c['file']}").convert("RGB")
            ph.thumbnail((T, T)); row.paste(ph, (i * T, 0))
            if c.get("sil_file"):
                s = Image.open(ROOT / f"data/silhouettes/{slug}/{c['sil_file']}")
                bg = Image.new("RGB", s.size, (240, 236, 225)); bg.paste((20, 20, 20), mask=s.split()[3])
                dd = ImageDraw.Draw(bg); (x1, y1), (x2, y2) = c["endpoints"]
                dd.line([x1, y1, x2, y2], fill=(220, 40, 40), width=max(2, s.width // 120))
                bg.thumbnail((T - 4, T - 4)); row.paste(bg, (i * T + 2, T))
            d.text((i * T + 2, 2 * T), f"{i+1} {c['score']:.2f}", fill="black")
        d.text((2, 2), slug, fill="red")
        rows.append(row)
    img = Image.new("RGB", (k * T, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        img.paste(r, (0, y)); y += r.height
    img.save(out)


if __name__ == "__main__":
    sheet(sys.argv[2:], sys.argv[1])
