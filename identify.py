"""Bird Assist, run directly (no browser): same model, temperature, location filter and % formatting as the site.

Usage: python3 identify.py PHOTO [--crop LEFT TOP SIZE] [--assets DIR] [--save-crop OUT.jpg]
  --crop: square crop in original-image pixels (after EXIF rotation). Default: centred square of the short side,
          which is what the site's "Frame your bird" screen starts with.
Prints JSON: top five with Bird Assist's displayed percentages, plus how the scan was tuned.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TEMPERATURE = 0.75  # same as app/server.py and the hosted config.js


def pct(p):  # app/static/app.js: pct() / pctNum()
    if p < 0.01:
        return "<1%"
    return f"{99 if p >= 0.995 else max(1, round(p * 100))}%"


def exif_info(im):
    """Month taken and GPS position, like the site's EXIF reader (photo's own place/date win)."""
    month = lat = lon = None
    try:
        ex = im.getexif()
        dt = ex.get_ifd(0x8769).get(0x9003) or ex.get_ifd(0x8769).get(0x9004) or ex.get(0x0132)
        if dt and len(dt) >= 7:
            m = int(str(dt)[5:7])
            month = m if 1 <= m <= 12 else None
        g = ex.get_ifd(0x8825)
        if g and 2 in g and 4 in g:
            dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            lat, lon = dms(g[2]), dms(g[4])
            if g.get(1) == "S": lat = -lat
            if g.get(3) == "W": lon = -lon
    except Exception:
        pass
    return month, lat, lon


def range_adjust(probs, month, lat, lon, rng, table):
    """Copy of range.js apply(): with no GPS the season-only alpha is 0, so the scores are left alone."""
    if month is None:
        return probs, None
    row, alpha, tuned = table.shape[0] - 1, rng["alphaSeason"], "season"
    if lat is not None and lon is not None:
        st = rng["step"]
        hit = rng["cells"].get(f"{int(np.floor(lat / st) * st)},{int(np.floor(lon / st) * st)}")
        if hit is not None:
            row, alpha, tuned = hit, rng["alpha"], "place"
    if not alpha:
        return probs, None
    freq = 10.0 ** (rng["logMin"] * (1 - table[row, month - 1] / 255.0))
    probs = probs * freq ** alpha
    return probs / probs.sum(), tuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--crop", nargs=3, type=float, metavar=("LEFT", "TOP", "SIZE"))
    ap.add_argument("--assets", default=os.path.expanduser("~/bird-assist-assets"))
    ap.add_argument("--save-crop")
    a = ap.parse_args()

    raw = Image.open(a.photo)
    month, lat, lon = exif_info(raw)
    im = ImageOps.exif_transpose(raw).convert("RGB")
    w, h = im.size
    if a.crop:
        l, t, s = a.crop
    else:
        s = min(w, h); l, t = (w - s) / 2, (h - s) / 2
    s = max(16, min(s, w, h))
    l, t = min(max(0, l), w - s), min(max(0, t), h - s)
    # the site draws the square crop at 448 px, then down to 224 for the model
    sq = im.crop((round(l), round(t), round(l + s), round(t + s))).resize((448, 448), Image.BICUBIC)
    if a.save_crop:
        sq.save(a.save_crop, quality=90)
    x = (np.asarray(sq.resize((224, 224), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(a.assets, "bird_assist_fp16.onnx"), providers=["CPUExecutionProvider"])
    probs = sess.run(["probs"], {"image": x})[0][0].astype(np.float64)
    probs = probs ** (1 / TEMPERATURE)
    probs /= probs.sum()

    labels = json.load(open(os.path.join(a.assets, "labels.json"), encoding="utf-8"))
    rng = json.load(open(os.path.join(a.assets, "range.json")))
    table = np.fromfile(os.path.join(a.assets, "range.bin"), np.uint8).reshape(-1, rng["months"], rng["species"])
    probs, tuned = range_adjust(probs, month, lat, lon, rng, table)

    top = np.argsort(-probs)[:5]
    print(json.dumps({
        "crop": {"left": round(l), "top": round(t), "size": round(s), "image": [w, h]},
        "tuned": tuned, "exif_month": month, "exif_gps": lat is not None,
        "top5": [{"rank": r + 1, "common": labels[i]["common"], "scientific": labels[i]["scientific"],
                  "confidence": pct(probs[i])} for r, i in enumerate(top)],
    }, indent=1))


if __name__ == "__main__":
    sys.exit(main())
