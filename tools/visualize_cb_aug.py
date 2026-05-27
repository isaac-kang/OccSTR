"""Visualize CB precompute output (boxes + BG colors).

Shows 10 samples per dataset: original image, AABB boxes overlay, BG color swatches.

Usage:
  python tools/visualize_cb_aug.py
Output:
  /data/isaackang/STR/OccSTR/cb_aug_preview.html
"""

import lmdb, struct, cv2, numpy as np, base64, io
from pathlib import Path
from PIL import Image

DATA_ROOT = Path('/data/isaackang/data/STR/openocr/test')
AUG_ROOT  = Path('/data/isaackang/data/STR/Occ_aug/test')
CB_NAMES  = ['IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80']

CROP_H, CROP_W = 32, 128


def open_lmdb(path):
    return lmdb.open(str(path), readonly=True, lock=False, max_readers=1,
                     readahead=False, meminit=False)


def decode_img(raw):
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def decode_cb_aug(data):
    n = struct.unpack('>I', data[:4])[0]
    if n == 0:
        return np.zeros((0, 4, 2), np.float32), np.zeros((0, 3), np.uint8)
    boxes  = np.frombuffer(data[4:4+n*32],   '<f4').reshape(n, 4, 2)
    colors = np.frombuffer(data[4+n*32:],    'u1').reshape(n, 3)
    return boxes, colors


def img_to_b64(arr, max_h=128):
    h, w = arr.shape[:2]
    scale = max(1, max_h // h)
    pil = Image.fromarray(arr).resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


def draw_boxes(img_rgb, boxes_32):
    """Draw AABB boxes scaled from 32×128 back to original image size."""
    out = img_rgb.copy()
    h, w = img_rgb.shape[:2]
    sx, sy = w / CROP_W, h / CROP_H
    colors = [(255,80,80),(80,255,80),(80,80,255),(255,255,0),
              (0,255,255),(255,0,255),(255,160,0),(160,0,255)]
    for i, box in enumerate(boxes_32):
        x1 = int(box[:, 0].min() * sx); y1 = int(box[:, 1].min() * sy)
        x2 = int(box[:, 0].max() * sx); y2 = int(box[:, 1].max() * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), colors[i % len(colors)], 1)
    return out


def swatch_strip(colors, w_each=16, h=16):
    """Horizontal strip of color swatches."""
    n = len(colors)
    if n == 0:
        canvas = np.full((h, 40, 3), 200, dtype=np.uint8)
        cv2.putText(canvas, 'N/A', (2, h-4), cv2.FONT_HERSHEY_PLAIN, 0.8,
                    (100,100,100), 1)
        return canvas
    canvas = np.zeros((h, n * w_each, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        canvas[:, i*w_each:(i+1)*w_each] = c
    return canvas


def pick_samples(env_in, env_aug, n_total, n_pick=10):
    rng = np.random.default_rng(42)
    candidates = rng.choice(n_total, min(1000, n_total), replace=False)
    samples = []
    with env_in.begin() as ti, env_aug.begin() as ta:
        for idx in candidates:
            idx = int(idx) + 1
            raw_img = ti.get(f'image-{idx:09d}'.encode())
            raw_aug = ta.get(f'aug-{idx:09d}'.encode())
            if raw_img is None or raw_aug is None:
                continue
            boxes, colors = decode_cb_aug(raw_aug)
            if len(boxes) < 1:
                continue
            img = decode_img(raw_img)
            if img is None:
                continue
            samples.append((idx, img, boxes, colors))
            if len(samples) >= n_pick:
                break
    return samples


rows_html = []

for name in CB_NAMES:
    in_path  = DATA_ROOT / name
    aug_path = AUG_ROOT  / name
    if not (aug_path / 'data.mdb').exists():
        print(f'SKIP {name}: no aug data yet')
        continue
    try:
        env_in  = open_lmdb(in_path)
        env_aug = open_lmdb(aug_path)
        n_total = int(env_in.begin().get(b'num-samples'))
        samples = pick_samples(env_in, env_aug, n_total, n_pick=4)
        env_in.close(); env_aug.close()
    except Exception as e:
        print(f'SKIP {name}: {e}')
        continue

    rows_html.append(
        f'<tr><td colspan="4" class="section">{name} ({n_total:,} samples)</td></tr>')
    rows_html.append(
        '<tr><th>idx</th><th>original</th><th>boxes</th>'
        '<th>BG color (per box)</th></tr>')

    for idx, img, boxes, colors in samples:
        orig_b64  = img_to_b64(img, max_h=128)
        boxes_b64 = img_to_b64(draw_boxes(img, boxes), max_h=128)
        swatch    = img_to_b64(swatch_strip(colors, w_each=20, h=20), max_h=20)
        h_o, w_o  = img.shape[:2]
        rows_html.append(f'''
        <tr>
          <td class="idx">#{idx}<br><span class="dim">{w_o}×{h_o}</span></td>
          <td class="img"><img src="data:image/png;base64,{orig_b64}"></td>
          <td class="img"><img src="data:image/png;base64,{boxes_b64}"></td>
          <td class="sw"><img src="data:image/png;base64,{swatch}"><br>
            <span class="dim">{len(colors)} boxes</span></td>
        </tr>''')

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CB Aug Preview</title>
<style>
body {{ font-family: monospace; background:#1a1a1a; color:#ddd; padding:16px; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border:1px solid #444; padding:6px 8px; vertical-align:top; }}
th {{ background:#333; font-size:12px; }}
.section {{ background:#2a4a6a; font-weight:bold; font-size:14px; padding:10px; }}
.idx {{ color:#aaa; font-size:11px; width:60px; }}
.dim {{ color:#888; font-size:10px; }}
.img img {{ image-rendering:pixelated; display:block; max-width:400px; }}
.sw img {{ image-rendering:pixelated; display:block; }}
</style>
</head><body>
<h2>CB Aug Preview</h2>
<p style="color:#aaa;font-size:12px">
  boxes: CRAFT AABB scaled to original res |
  BG color: avg of pixels where CRAFT score&lt;0.3
</p>
<table>
{''.join(rows_html)}
</table>
</body></html>'''

out = Path('/data/isaackang/STR/OccSTR/cb_aug_preview.html')
out.write_text(html)
print(f'Written: {out}  ({out.stat().st_size // 1024}KB)')
