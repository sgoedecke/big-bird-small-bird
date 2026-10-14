"""Segment candidate photos into bird silhouettes, then score and rank them.

Stages (each cached, so re-running only does new work):
  1. BiRefNet salient-object mask  -> data/silhouettes/<slug>/masks/<photo_id>.png
  2. CLIP zero-shot pose check      -> data/silhouettes/<slug>/clip.json
  3. Mask analysis (always re-run)  -> <photo_id>.png silhouette + candidates.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from geometry import geodesic, geodesic_diameter

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "data/photos"
SILS = ROOT / "data/silhouettes"
PAD = 8
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
CLIP_MODEL, CLIP_REVISION = "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"

GOOD_PROMPTS = [
    "a side view photo of a single bird standing, full body visible",
    "a side view photo of a single bird perched on a branch, full body in profile",
    "a side view photo of a single bird swimming, in profile",
]
BAD_PROMPTS = [
    "a photo of a bird flying with its wings spread",
    "a photo of a bird seen from behind",
    "a photo of a bird facing the camera head-on",
    "a photo of two or more birds",
    "a close-up photo of a bird's head",
    "a photo of a bird with its body partly hidden behind leaves or branches",
    "a photo of a bird lying down or sitting on a nest",
    "a blurry distant photo of a bird",
]


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Segmenter:
    """BiRefNet salient-object segmentation."""

    def __init__(self):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        self.torch = torch
        self.device = device()
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True, revision=BIREFNET_REVISION,
        ).to(self.device, self.dtype).eval()
        self.tf = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        x = self.tf(img).unsqueeze(0).to(self.device, self.dtype)
        with self.torch.no_grad():
            pred = self.model(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
        return Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)


def _features(out):
    # transformers >= 5 returns a model output whose pooler_output is the projected embedding
    return out if hasattr(out, "norm") else out.pooler_output


class PoseScorer:
    """CLIP zero-shot probability that a photo shows a single, whole bird side-on."""

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device()
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
        with torch.no_grad():
            t = self.proc(text=GOOD_PROMPTS + BAD_PROMPTS, return_tensors="pt", padding=True).to(self.device)
            emb = _features(self.model.get_text_features(**t))
            self.text = emb / emb.norm(dim=-1, keepdim=True)

    def __call__(self, img):
        with self.torch.no_grad():
            x = self.proc(images=img, return_tensors="pt").to(self.device)
            emb = _features(self.model.get_image_features(**x))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (100 * emb @ self.text.T).softmax(-1)[0].cpu().numpy()
        return float(probs[: len(GOOD_PROMPTS)].sum())


def largest_component(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    return labels == i, float(areas.max() / areas.sum())


def fill_holes(mask):
    m = np.pad(mask.astype(np.uint8) * 255, 1)
    ff = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(m, ff, (0, 0), 255)
    return mask | (m[1:-1, 1:-1] == 0)


def farthest_pair(mask):
    pts = cv2.findNonZero(mask.astype(np.uint8))
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return hull[i], hull[j], hull


def bill_tail_endpoints(main):
    """Estimate bill tip and tail tip. Legs/feet/perches are stripped with a morphological
    opening scaled to body thickness, the geodesic diameter of what remains gives the two
    ends, and each end is then pushed outward along the body path on the full mask to
    recover the thin tip of the bill or tail."""
    dist = cv2.distanceTransform(main.astype(np.uint8), cv2.DIST_L2, 5)
    r = max(1, int(dist.max() * 0.2))
    # Cut off legs: rows at the bottom that are much narrower than the body
    counts = main.sum(1)
    rows = np.nonzero(counts)[0]
    y = rows.max()
    while y > rows.min() and counts[y] < 0.2 * counts.max():
        y -= 1
    if y < rows.max():
        main = main.copy()
        main[y + 1:] = False
    body = cv2.morphologyEx(main.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    body, _ = largest_component(body)
    if body is None:
        body = main
    a, b, ta, tb = geodesic_diameter(body)
    ys, xs = np.nonzero(main)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    out = []
    for p, t in ((a, ta), (b, tb)):
        rel = pts - p
        along = rel @ t
        perp = np.abs(rel @ np.array([-t[1], t[0]]))
        ok = (along > 0) & (perp < along * 0.4 + r * 0.5) & (along < 3 * r)
        out.append(pts[ok][np.argmax(along[ok])] if ok.any() else p)
    return out[0], out[1]


def analyse(alpha):
    """alpha: float32 HxW in [0,1]. Returns (clean_alpha, metrics, endpoints) or (None, metrics, None)."""
    h, w = alpha.shape
    binary = alpha > 0.5
    if binary.sum() < 500:
        return None, {"reject": "empty"}, None
    main, main_frac = largest_component(binary)
    main = fill_holes(main)
    near = cv2.dilate(main.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    clean = np.where(near, alpha, 0.0)
    clean[main] = 1.0

    ys, xs = np.nonzero(main)
    m = 3
    touches = int(xs.min() < m) + int(ys.min() < m) + int(xs.max() >= w - m) + int(ys.max() >= h - m)
    soft = alpha[alpha > 0.1]
    softness = float(((soft > 0.1) & (soft < 0.9)).mean())

    _, _, hull = farthest_pair(main)
    p1, p2 = bill_tail_endpoints(main)
    length_px = float(np.linalg.norm(p1 - p2))
    area = float(main.sum())
    solidity = area / max(float(cv2.contourArea(hull.astype(np.float32))), 1)

    # Mirror symmetry about the vertical axis through the centroid (front/back views are symmetric)
    cx = int(round(xs.mean()))
    half = max(cx - xs.min(), xs.max() - cx) + 1
    crop = np.zeros((h, 2 * half + 1), bool)
    lo, hi = max(cx - half, 0), min(cx + half + 1, w)
    crop[:, lo - (cx - half): lo - (cx - half) + (hi - lo)] = main[:, lo:hi]
    symmetry = float((crop & crop[:, ::-1]).sum() / max((crop | crop[:, ::-1]).sum(), 1))

    metrics = {
        "fragmentation": round(1 - main_frac, 4),
        "edges_touched": touches,
        "softness": round(softness, 4),
        "symmetry": round(symmetry, 4),
        "solidity": round(solidity, 4),
        "length_px": round(length_px, 1),
        "area_frac": round(area / (h * w), 4),
    }
    return clean, metrics, (p1, p2)


def score(m):
    if m.get("reject"):
        return 0.0
    s = 1.5 * m.get("profile_prob", 0.5)
    s += 0.3 * min(1, m["length_px"] / 700)
    s -= 0.6 * m["edges_touched"]
    s -= 2.0 * m["fragmentation"]
    s -= 1.0 * max(0, m["softness"] - 0.08)
    s -= 1.5 * max(0, m["symmetry"] - 0.6)
    s -= 1.0 * max(0, 0.55 - m["solidity"])
    return round(s, 4)


def process_species(slug, segmenter, poser):
    src = PHOTOS / slug
    meta = json.loads((src / "meta.json").read_text())
    out = SILS / slug
    (out / "masks").mkdir(parents=True, exist_ok=True)
    clip_path = out / "clip.json"
    clip = json.loads(clip_path.read_text()) if clip_path.exists() else {}
    cands = []
    for m in meta:
        if not (src / m["file"]).exists():
            continue
        mask_path = out / "masks" / f"{m['photo_id']}.png"
        img = None
        if not mask_path.exists():
            img = Image.open(src / m["file"]).convert("RGB")
            segmenter(img).save(mask_path)
        if str(m["photo_id"]) not in clip:
            img = img or Image.open(src / m["file"]).convert("RGB")
            clip[str(m["photo_id"])] = round(poser(img), 4)
        alpha = np.asarray(Image.open(mask_path), dtype=np.float32) / 255.0
        clean, metrics, ends = analyse(alpha)
        c = {**m, **metrics, "profile_prob": clip[str(m["photo_id"])]}
        if clean is not None:
            ys, xs = np.nonzero(clean > 0.02)
            y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, alpha.shape[0])
            x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, alpha.shape[1])
            rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
            rgba[..., 3] = (clean[y0:y1, x0:x1] * 255).astype(np.uint8)
            Image.fromarray(rgba).save(out / f"{m['photo_id']}.png", optimize=True)
            c.update({
                "crop": [int(x0), int(y0), int(x1), int(y1)],
                "sil_file": f"{m['photo_id']}.png",
                "sil_size": [int(x1 - x0), int(y1 - y0)],
                "endpoints": [[round(float(p[0] - x0), 1), round(float(p[1] - y0), 1)] for p in ends],
            })
            c["geo_length_px"], c["path"] = geodesic(clean[y0:y1, x0:x1] > 0.5, *c["endpoints"])
            c["geo_length_px"] = round(c["geo_length_px"], 1)
        c["score"] = score(c)
        cands.append(c)
    clip_path.write_text(json.dumps(clip))
    cands.sort(key=lambda c: -c["score"])
    (out / "candidates.json").write_text(json.dumps(cands, indent=1))
    return cands


def is_done(slug):
    cpath = SILS / slug / "candidates.json"
    if not cpath.exists():
        return False
    meta = json.loads((PHOTOS / slug / "meta.json").read_text())
    return len(json.loads(cpath.read_text())) >= len(meta)


class Lazy:
    def __init__(self, cls):
        self.cls, self.obj = cls, None

    def __call__(self, *a):
        self.obj = self.obj or self.cls()
        return self.obj(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--skip-done", action="store_true", help="skip species already analysed")
    args = ap.parse_args()
    slugs = args.slugs or sorted(p.name for p in PHOTOS.iterdir() if (p / "meta.json").exists())
    if args.skip_done:
        slugs = [s for s in slugs if not is_done(s)]
    segmenter, poser = Lazy(Segmenter), Lazy(PoseScorer)
    for i, slug in enumerate(slugs):
        cands = process_species(slug, segmenter, poser)
        print(f"[{i + 1}/{len(slugs)}] {slug}: best={cands[0]['score'] if cands else None}", flush=True)


if __name__ == "__main__":
    main()
