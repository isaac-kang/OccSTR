"""Visualize Hi-SAM (TextSeg ckpt) stroke segmentation and CRAFT character-region
heatmap on a random sample from the 6 Common Benchmark (CB) LMDB datasets.

Inference results are cached to a .pkl next to the HTML so cosmetic HTML
tweaks can rebuild the page without re-running the model. Use --force to
re-run inference.

Output: a self-contained HTML with 1 row per sample, columns:
  id | dataset | idx | gt | original | Hi-SAM mask | CRAFT heatmap

Usage:
  CUDA_VISIBLE_DEVICES=3 python tools/hisam_cb_compare.py \
      --n 100 --seed 0 \
      --out error_analysis/hisam_cb_compare.html
"""

import argparse
import base64
import io
import os
import pickle
import random
import sys
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

OCCSTR_ROOT = Path(__file__).resolve().parent.parent
HI_SAM_ROOT = OCCSTR_ROOT / 'third_party' / 'Hi-SAM'
CRAFT_ROOT = OCCSTR_ROOT / 'third_party' / 'craft'
sys.path.insert(0, str(HI_SAM_ROOT))
sys.path.insert(0, str(CRAFT_ROOT))

from hi_sam.modeling.build import model_registry  # noqa: E402
from hi_sam.modeling.predictor import SamPredictor  # noqa: E402

CB_ROOT = Path('/data/isaackang/data/STR/openocr/test')
CB_NAMES = ['IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80']

CKPT_DIR = OCCSTR_ROOT / 'weights' / 'pretrained' / 'hi_sam'
SAM_BACKBONE = CKPT_DIR / 'sam_vit_h_4b8939.pth'
TEXTSEG_CKPT = CKPT_DIR / 'sam_tss_h_textseg.pth'
CRAFT_CKPT = OCCSTR_ROOT / 'weights' / 'pretrained' / 'craft' / 'craft_ic15_20k.pth'

MASK_FILL = np.array([220, 38, 38], dtype=np.float32)   # red, RGB
MASK_BORDER = (255, 105, 180)                            # hot pink, RGB
FILL_ALPHA = 0.45
BORDER_THICKNESS = 1


def open_lmdb(path):
    env = lmdb.open(str(path), max_readers=4, readonly=True, lock=False,
                    readahead=False, meminit=False)
    txn = env.begin(write=False)
    n = int(txn.get(b'num-samples'))
    return env, txn, n


def decode_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)


def overlay_mask(image_rgb, mask, alpha=FILL_ALPHA):
    """Red fill + pink contour overlay on the original image."""
    out = image_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * MASK_FILL
    out = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(out, contours, -1, MASK_BORDER, BORDER_THICKNESS)
    return out


def to_b64_png(arr):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def build_args_namespace(model_type, ckpt):
    """SamPredictor needs a few attrs from args. Reproduce them minimally."""
    return argparse.Namespace(
        model_type=model_type,
        checkpoint=str(ckpt),
        input_size=[1024, 1024],
        attn_layers=1,
        prompt_len=12,
        hier_det=False,
    )


def load_hisam(ckpt_path, device):
    ns = build_args_namespace('vit_h', ckpt_path)
    model = model_registry['vit_h'](ns)
    model.eval().to(device)
    return SamPredictor(model)


def run_predictor(predictor, image_rgb):
    """Return binary stroke mask at original resolution."""
    predictor.set_image(image_rgb)
    _, hr_mask, _, _ = predictor.predict(multimask_output=False)
    # hr_mask shape: (1, H, W) bool
    return hr_mask[0].astype(np.uint8)


def load_craft(ckpt_path, device):
    from collections import OrderedDict
    from craft import CRAFT
    net = CRAFT()
    sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if list(sd.keys())[0].startswith('module'):
        sd = OrderedDict((k.split('.', 1)[1], v) for k, v in sd.items())
    net.load_state_dict(sd)
    net.eval().to(device)
    return net


def run_craft(net, image_rgb, device, canvas_size=2560, mag_ratio=1.5):
    """Return per-pixel character region score map matching image_rgb shape.

    Aligned correctly: the heatmap is at half-res of the *padded* network
    input, but only rows/cols 0..(active/2) correspond to actual image
    content; the rest is zero-padding. We crop to the active region before
    resizing back to the original image size, otherwise the heatmap shifts
    toward the top-left.
    """
    import imgproc
    h, w = image_rgb.shape[:2]
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image_rgb, canvas_size, interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio,
    )
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        y, _ = net(x)
    score_text = y[0, :, :, 0].cpu().numpy()  # padded H/2 x padded W/2
    # Active resized image region (pre-padding); heatmap is half-resolution.
    active_h = max(1, int(h * target_ratio) // 2)
    active_w = max(1, int(w * target_ratio) // 2)
    score_text = score_text[:active_h, :active_w]
    score_text = np.clip(score_text, 0, 1)
    score_text = cv2.resize(score_text, (w, h), interpolation=cv2.INTER_LINEAR)
    return score_text.astype(np.float32)


def overlay_craft(image_rgb, score_text, alpha=0.5):
    """Jet colormap heatmap blended onto original image."""
    heat = (np.clip(score_text, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * image_rgb.astype(np.float32) \
              + alpha * heat_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def sample_indices(rng, lmdbs, n_total):
    """Stratified-then-random sampling across CB datasets to hit n_total."""
    counts = {name: lmdbs[name][2] for name in CB_NAMES}
    n_per = n_total // len(CB_NAMES)
    leftover = n_total - n_per * len(CB_NAMES)
    pool = []
    for name in CB_NAMES:
        n = n_per + (1 if leftover > 0 else 0)
        if leftover > 0:
            leftover -= 1
        idxs = rng.sample(range(1, counts[name] + 1), min(n, counts[name]))
        for i in idxs:
            pool.append((name, i))
    rng.shuffle(pool)
    return pool


HTML_HEAD_TMPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hi-SAM CB compare</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }
h1 { color: #333; }
.summary { color: #666; margin-bottom: 16px; font-size: 14px; }
table { border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
th, td { border: 1px solid #ddd; padding: 6px 10px; vertical-align: middle; }
th { background: #f0f0f0; text-align: left; font-weight: 600; font-size: 13px; position: sticky; top: 0; }
td.id { color: #888; font-family: monospace; font-size: 12px; }
td.ds { font-family: monospace; font-size: 12px; color: #555; }
td.gt { font-family: 'Courier New', monospace; font-size: 14px; }
td.img img { display: block; max-height: 80px; image-rendering: pixelated; }
tr:nth-child(even) td { background: #fbfbfb; }
.filters { margin-bottom: 12px; font-size: 13px; color: #555; }
.filter-btn { padding: 4px 12px; margin-right: 6px; border: 1px solid #ccc; background: white; cursor: pointer; border-radius: 3px; font: inherit; }
.filter-btn.on { background: #1976d2; color: white; border-color: #1565c0; }
.filter-btn.clear { background: #f5f5f5; }
</style></head><body>
<h1>Hi-SAM stroke segmentation (TextSeg ckpt) + CRAFT character heatmap</h1>
<div class="summary">Samples: __N__ from CB (__DATASETS__) &middot; seed=__SEED__ &middot; Hi-SAM: red fill + pink contour &middot; CRAFT: jet heatmap of character region score</div>
<div class="filters">Filter by dataset:
  <button class="filter-btn clear" data-ds="">all</button>
  __FILTER_BTNS__
</div>
<table>
<thead><tr><th>#</th><th>dataset</th><th>idx</th><th>gt</th><th>original</th><th>Hi-SAM</th><th>CRAFT</th></tr></thead>
<tbody>
"""

HTML_TAIL = """</tbody></table>
<script>
(() => {
  const btns = document.querySelectorAll('.filter-btn');
  const rows = document.querySelectorAll('tbody tr');
  btns.forEach(b => b.addEventListener('click', () => {
    btns.forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    const ds = b.dataset.ds;
    rows.forEach(tr => {
      tr.style.display = (!ds || tr.dataset.ds === ds) ? '' : 'none';
    });
  }));
})();
</script>
</body></html>
"""


def run_inference(n, seed, device):
    """Sample + run Hi-SAM (TextSeg ckpt) and CRAFT. Returns list of dicts."""
    print(f'[1/4] opening CB LMDBs at {CB_ROOT}', flush=True)
    lmdbs = {}
    for name in CB_NAMES:
        env, txn, n_ds = open_lmdb(CB_ROOT / name)
        lmdbs[name] = (env, txn, n_ds)
        print(f'  {name}: num-samples={n_ds}', flush=True)

    rng = random.Random(seed)
    pool = sample_indices(rng, lmdbs, n)
    print(f'[2/4] sampled {len(pool)} indices (stratified ~{n // 6}/ds)',
          flush=True)

    print('[3/4] Hi-SAM TextSeg inference', flush=True)
    t0 = time.time()
    predictor = load_hisam(TEXTSEG_CKPT, device)
    print(f'  loaded in {time.time() - t0:.1f}s', flush=True)
    rows = []
    for ds_name, idx in tqdm(pool, desc='TextSeg'):
        _, txn, _ = lmdbs[ds_name]
        img_bytes = txn.get(f'image-{idx:09d}'.encode())
        label = txn.get(f'label-{idx:09d}'.encode())
        gt = label.decode('utf-8', errors='replace') if label else ''
        if img_bytes is None:
            continue
        img_rgb = decode_image(img_bytes)
        mask = run_predictor(predictor, img_rgb)
        rows.append({'ds': ds_name, 'idx': idx, 'gt': gt, 'orig': img_rgb,
                     'mask_ts': mask, 'craft_heat': None})
    del predictor
    torch.cuda.empty_cache()

    print('[4/4] CRAFT detection', flush=True)
    t0 = time.time()
    craft_net = load_craft(CRAFT_CKPT, device)
    print(f'  loaded in {time.time() - t0:.1f}s', flush=True)
    for row in tqdm(rows, desc='CRAFT'):
        row['craft_heat'] = run_craft(craft_net, row['orig'], device)
    del craft_net
    torch.cuda.empty_cache()

    return rows


def build_html(rows, seed, out_path):
    print(f'writing HTML to {out_path}', flush=True)
    datasets_str = ', '.join(CB_NAMES)
    filter_btns = ' '.join(
        f'<button class="filter-btn" data-ds="{n}">{n}</button>'
        for n in CB_NAMES
    )
    html = (HTML_HEAD_TMPL
            .replace('__N__', str(len(rows)))
            .replace('__DATASETS__', datasets_str)
            .replace('__SEED__', str(seed))
            .replace('__FILTER_BTNS__', filter_btns))
    import html as html_esc
    for i, r in enumerate(rows):
        orig_b64 = to_b64_png(r['orig'])
        ts_b64 = to_b64_png(overlay_mask(r['orig'], r['mask_ts']))
        craft_b64 = to_b64_png(overlay_craft(r['orig'], r['craft_heat']))
        html += (
            f'<tr data-ds="{r["ds"]}">'
            f'<td class="id">{i}</td>'
            f'<td class="ds">{r["ds"]}</td>'
            f'<td class="id">{r["idx"]}</td>'
            f'<td class="gt">{html_esc.escape(r["gt"])}</td>'
            f'<td class="img"><img src="data:image/png;base64,{orig_b64}"></td>'
            f'<td class="img"><img src="data:image/png;base64,{ts_b64}"></td>'
            f'<td class="img"><img src="data:image/png;base64,{craft_b64}"></td>'
            f'</tr>\n'
        )
    html += HTML_TAIL
    out_path.write_text(html, encoding='utf-8')
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'done: {out_path} ({size_mb:.2f} MB)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str,
                    default='error_analysis/hisam_cb_compare.html')
    ap.add_argument('--cache', type=str, default=None,
                    help='Path to inference cache pkl. Default: <out>.pkl')
    ap.add_argument('--force', action='store_true',
                    help='Ignore cache and re-run inference')
    ap.add_argument('--rerun-craft', action='store_true',
                    help='Reuse cached Hi-SAM masks but re-run CRAFT only')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache).resolve() if args.cache \
        else out_path.with_suffix('.pkl')

    rows = None
    if cache_path.exists() and not args.force:
        print(f'loading cached inference from {cache_path}', flush=True)
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        if data.get('n') == args.n and data.get('seed') == args.seed:
            rows = data['rows']
            print(f'  loaded {len(rows)} rows (cache n={data["n"]}, '
                  f'seed={data["seed"]})', flush=True)
        else:
            print(f'  cache mismatch (n={data.get("n")}, '
                  f'seed={data.get("seed")}); re-running', flush=True)

    if rows is None:
        rows = run_inference(args.n, args.seed, args.device)
    elif args.rerun_craft:
        print('re-running CRAFT on cached images', flush=True)
        t0 = time.time()
        craft_net = load_craft(CRAFT_CKPT, args.device)
        print(f'  loaded in {time.time() - t0:.1f}s', flush=True)
        for row in tqdm(rows, desc='CRAFT'):
            row['craft_heat'] = run_craft(craft_net, row['orig'], args.device)
        del craft_net
        torch.cuda.empty_cache()

    if rows is not None and (args.force or args.rerun_craft or
                             not cache_path.exists()):
        print(f'saving inference cache to {cache_path}', flush=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'n': args.n, 'seed': args.seed, 'rows': rows}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f'  cache size: {cache_path.stat().st_size / 1024 / 1024:.2f} MB',
              flush=True)

    build_html(rows, args.seed, out_path)


if __name__ == '__main__':
    main()
