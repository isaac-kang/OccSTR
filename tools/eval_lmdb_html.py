"""Run SVTRv2 (CTC-head only) inference on an LMDB dataset and dump HTML of mismatched samples.

Use the inference yml (`svtrv2_smtr_gtc_rctc_infer.yml`) which sets
`Decoder.infer_gtc: False` and `PostProcess: CTCLabelDecode` — the GTC/SMTR
head is a training auxiliary and is not forwarded at inference.

Output format matches /data/isaackang/STR/parseq/str_error_analysis/WOST_errors.html:
  - title "<TAG> error samples"
  - table cols: id, image (base64 PNG), norm pred, norm gt, raw pred, raw gt
  - mismatch judged on 36-char normalized strings (lowercase + [0-9a-z])

Usage:
  python tools/eval_lmdb_html.py \
      --config configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml \
      --ckpt   weights/pretrained/svtrv2_smtr_gtc_rctc/best.pth \
      --lmdb   /data/isaackang/data/STR/openocr/OST/weak \
      --tag    WOST \
      --out    /data/isaackang/STR/parseq/str_error_analysis/WOST_errors_svtrv2.html
"""

import argparse
import base64
import html
import io
import os
import re
import sys
import time
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import lmdb
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess.resize import RecTVResize
from tools.engine.config import Config


_NORM_RE = re.compile(r'[^a-z0-9]')


def normalize36(s: str) -> str:
    return _NORM_RE.sub('', s.lower())


def open_lmdb(path):
    env = lmdb.open(path, max_readers=4, readonly=True, lock=False,
                    readahead=False, meminit=False)
    txn = env.begin(write=False)
    n = int(txn.get(b'num-samples'))
    return env, txn, n


def load_model(config_path, ckpt_path, device):
    cfg = Config(config_path).cfg
    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = post_process.get_character_num()
    model = build_model(cfg['Architecture'])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # GTC sub-decoder weights stay in state_dict but the head isn't forwarded
    # (infer_gtc=False). strict=False silently drops unused keys.
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, post_process


HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{tag} errors</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }
h1 { color: #333; }
.summary { color: #666; margin-bottom: 16px; font-size: 14px; }
table { border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
th, td { border: 1px solid #ddd; padding: 8px 12px; vertical-align: middle; }
th { background: #f0f0f0; text-align: left; font-weight: 600; font-size: 13px; }
td.id { color: #888; font-family: monospace; font-size: 12px; }
td.img { text-align: right; }
td.img img { display: block; max-height: 60px; margin-left: auto; }
td.heat img { display: block; max-height: 60px; }

td.text { font-family: 'Courier New', monospace; font-size: 16px; white-space: pre; }
td.raw { color: #555; font-size: 14px; }
.bad { color: #e53935; font-weight: bold; }
.miss { color: #e53935; font-weight: bold; text-decoration: underline; }
tr:nth-child(even) td { background: #fbfbfb; }
.filters { margin-bottom: 12px; font-size: 13px; color: #555; }
.filter-btn { padding: 4px 12px; margin-right: 6px; border: 1px solid #ccc; background: white; cursor: pointer; border-radius: 3px; font: inherit; }
.filter-btn.on { background: #1976d2; color: white; border-color: #1565c0; }
.filter-btn.clear { background: #f5f5f5; }
.filter-count { color: #888; font-size: 12px; margin-left: 12px; }
</style></head><body>
<h1>{tag} error samples</h1>
<div class="summary">Total: {n_err} of {n_total} &middot; acc36 = {acc:.2%} &middot; mismatch (red) is computed on 36-char-normalized strings (lowercase + [0-9a-z])</div>
<div class="filters">Filter (AND of selected): <button class="filter-btn" data-kind="sub">sub</button><button class="filter-btn" data-kind="ins">ins</button><button class="filter-btn" data-kind="del">del</button><button class="filter-btn clear" id="filter-clear">clear</button><span class="filter-count" id="filter-count"></span></div>
<table>
<thead><tr><th>id</th><th>image</th><th>norm pred</th><th>norm gt</th><th>raw pred</th><th>raw gt</th></tr></thead>
<tbody>
"""

HTML_TAIL = """</tbody></table>
<script>
(() => {
  const active = new Set();
  const btns = document.querySelectorAll('.filter-btn[data-kind]');
  const countEl = document.getElementById('filter-count');
  const rows = document.querySelectorAll('tbody tr');
  const apply = () => {
    let shown = 0;
    rows.forEach(tr => {
      const ok = [...active].every(k => tr.classList.contains('has-' + k));
      tr.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });
    countEl.textContent = active.size ? `(${shown} of ${rows.length} shown)` : '';
  };
  btns.forEach(b => b.onclick = () => {
    const k = b.dataset.kind;
    if (active.has(k)) { active.delete(k); b.classList.remove('on'); }
    else { active.add(k); b.classList.add('on'); }
    apply();
  });
  document.getElementById('filter-clear').onclick = () => {
    active.clear();
    btns.forEach(b => b.classList.remove('on'));
    apply();
  };
})();
</script>
</body></html>
"""


def img_to_b64_png(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _align(pred, gt):
    """Levenshtein DP + backtrace, matching parseq/gen_error_html.py:align().

    Tie-breaking prefers substitution over insert/delete (the diagonal branch is
    checked first), so a single pred char aligned with multiple gt chars (or
    vice versa) naturally gets split via gap markers. Returns two equal-length
    lists with '-' for gaps.
    """
    n, m = len(pred), len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ap, ag = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (pred[i - 1] == gt[j - 1] or dp[i][j] == dp[i - 1][j - 1] + 1):
            ap.append(pred[i - 1]); ag.append(gt[j - 1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ap.append(pred[i - 1]); ag.append('-'); i -= 1
        else:
            ap.append('-'); ag.append(gt[j - 1]); j -= 1
    return list(reversed(ap)), list(reversed(ag))


def render_pred(norm_pred, norm_gt):
    ap, ag = _align(norm_pred, norm_gt)
    out = []
    for cp, cg in zip(ap, ag):
        if cp == '-':
            out.append('<span class="miss">_</span>')
        elif cp == cg:
            out.append(html.escape(cp))
        else:
            out.append(f'<span class="bad">{html.escape(cp)}</span>')
    return ''.join(out)


def edit_kinds(norm_pred, norm_gt):
    """Return a set of {'sub', 'ins', 'del'} describing which edit kinds are
    present in the alignment of norm_pred → norm_gt:
      - 'ins' : a gt char is missing in pred (pred dropped it) → ap has '-'
      - 'del' : a pred char is missing in gt (pred hallucinated) → ag has '-'
      - 'sub' : a pred char and gt char at the same aligned position differ
    """
    ap, ag = _align(norm_pred, norm_gt)
    kinds = set()
    for cp, cg in zip(ap, ag):
        if cp == '-':
            kinds.add('ins')
        elif cg == '-':
            kinds.add('del')
        elif cp != cg:
            kinds.add('sub')
    return kinds


def render_gt(norm_pred, norm_gt):
    ap, ag = _align(norm_pred, norm_gt)
    out = []
    for cp, cg in zip(ap, ag):
        if cg == '-':
            continue
        elif cp == cg:
            out.append(html.escape(cg))
        else:
            out.append(f'<span class="bad">{html.escape(cg)}</span>')
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True,
                    help='Inference yml (Decoder.infer_gtc=False, PostProcess=CTCLabelDecode)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--lmdb', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--image_shape', default='32,128',
                    help='H,W for fixed-resize (with right-padding to W)')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--dynamic_width', action='store_true',
                    help='Per-image resize matching SLD eval-time RatioDataSetTVResize: '
                         'ratio 1..4 → base_shape [[64,64],[96,48],[112,40],[128,32]]; '
                         'ratio 5..max → [ratio*32, 32]. Forces batch_size=1.')
    ap.add_argument('--max_ratio', type=int, default=12)
    ap.add_argument('--ins_only_out', type=str, default=None,
                    help='Optional: write a second HTML containing only rows with '
                         'at least one ins-type edit (gt has char pred dropped).')
    BASE_SHAPE = [(64, 64), (96, 48), (112, 40), (128, 32)]  # (W, H)
    args = ap.parse_args()

    H, W = [int(x) for x in args.image_shape.split(',')]
    fixed_resize = RecTVResize(image_shape=[H, W], padding=True)
    to_tensor = T.ToTensor()
    normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    bicubic = T.InterpolationMode.BICUBIC

    def resize_dyn(img):
        w, h = img.size
        r = max(1, min(args.max_ratio, round(w / max(h, 1))))
        if r <= 4:
            tw, th = BASE_SHAPE[r - 1]
        else:
            tw, th = r * 32, 32
        resized = TF.resize(img, (th, tw), interpolation=bicubic)
        return normalize(to_tensor(resized))

    model, post_process = load_model(args.config, args.ckpt, args.device)
    env, txn, n_total = open_lmdb(args.lmdb)
    print(f'[{args.tag}] LMDB at {args.lmdb} : num-samples={n_total}')

    errors = []
    n_correct = 0
    t0 = time.time()

    bs = 1 if args.dynamic_width else args.batch_size
    for bstart in range(0, n_total, bs):
        idxs = range(bstart + 1, min(bstart + bs, n_total) + 1)
        tensors, items = [], []
        for idx in idxs:
            img_bytes = txn.get(f'image-{idx:09d}'.encode())
            label = txn.get(f'label-{idx:09d}'.encode())
            if img_bytes is None or label is None:
                continue
            label = label.decode('utf-8', errors='replace')
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                print(f'  [skip] idx={idx}: decode err {e}')
                continue
            tensors.append(resize_dyn(img) if args.dynamic_width
                           else fixed_resize({'image': img})['image'])
            items.append((idx, img_bytes, label))
        if not tensors:
            continue
        x = torch.stack(tensors, dim=0).to(args.device)
        with torch.inference_mode():
            preds = model(x)
        # With infer_gtc=False + PostProcess=CTCLabelDecode, output is
        # a flat list of (text, score) tuples (one per batch item).
        decoded = post_process(preds)

        for (idx, img_bytes, gt_raw), (pred_raw, score) in zip(items, decoded):
            pred_n = normalize36(pred_raw)
            gt_n = normalize36(gt_raw)
            if pred_n == gt_n:
                n_correct += 1
                continue
            errors.append({
                'idx': idx,
                'img_bytes': img_bytes,
                'pred_raw': pred_raw,
                'gt_raw': gt_raw,
                'pred_n': pred_n,
                'gt_n': gt_n,
                'score': float(score),
            })

        if (bstart // bs) % max(1, 320 // bs) == 0:
            done = bstart + len(items)
            print(f'  {done}/{n_total}  acc36={n_correct/max(1,done):.2%}  '
                  f'errors={len(errors)}  ({time.time()-t0:.1f}s)')

    n_done = n_correct + len(errors)
    acc = n_correct / max(1, n_done)
    print(f'[{args.tag}] done: acc36 = {n_correct}/{n_done} = {acc:.2%}  '
          f'(errors={len(errors)}, {time.time()-t0:.1f}s)')

    # Pre-compute per-error kinds once so the optional second HTML reuses them.
    enriched = []
    for e in errors:
        kinds = edit_kinds(e['pred_n'], e['gt_n'])
        enriched.append((e, kinds))

    def _write_html(out_path: str, rows, title_suffix: str = ''):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            tag = args.tag + title_suffix
            f.write(HTML_HEAD.replace('{tag}', tag)
                    .replace('{n_err}', str(len(rows)))
                    .replace('{n_total}', str(n_done))
                    .replace('{acc:.2%}', f'{acc:.2%}'))
            for e, kinds in rows:
                sid = f"{args.tag}_{e['idx']:06d}"
                b64 = img_to_b64_png(e['img_bytes'])
                kind_cls = ' '.join(f'has-{k}' for k in sorted(kinds))
                f.write(
                    f'<tr class="{kind_cls}">'
                    f'<td class="id">{html.escape(sid)}</td>'
                    f'<td class="img"><img src="data:image/png;base64,{b64}" alt="{html.escape(sid)}"></td>'
                    f'<td class="text">{render_pred(e["pred_n"], e["gt_n"])}</td>'
                    f'<td class="text">{render_gt(e["pred_n"], e["gt_n"])}</td>'
                    f'<td class="text raw">{html.escape(e["pred_raw"])}</td>'
                    f'<td class="text raw">{html.escape(e["gt_raw"])}</td>'
                    f'</tr>\n'
                )
            f.write(HTML_TAIL)
        print(f'[{args.tag}] wrote {out_path}')

    _write_html(args.out, enriched)

    if args.ins_only_out:
        ins_rows = [(e, k) for e, k in enriched if 'ins' in k]
        _write_html(args.ins_only_out, ins_rows, title_suffix=' (ins only)')


if __name__ == '__main__':
    main()
