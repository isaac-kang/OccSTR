#!/usr/bin/env python3
"""Precompute CRAFT character boxes + background colors for OccSTR stroke augmentation.

Single unified pipeline (full-res CRAFT, fp32):

  1. CRAFT at full resolution, aspect-preserving (resize_aspect_ratio,
     canvas 2560, mag 1.5) — the canonical CRAFT preprocessing. Images are
     batched by zero-padding each (already /32-aligned) resized image to the
     sub-batch's common size at the top-left corner. Because conv is
     batch-independent and the extra padding is zeros (matching the network's
     own boundary zero-padding), the score maps in each image's valid region
     are bit-identical to running it one-by-one. fp32 (no AMP) keeps it
     numerically identical to the official CRAFT inference.
  2. Word instances are formed from CRAFT's affinity (link) map:
     connected components of (region > low_text) | (link > link_threshold).
     Only the word instance whose center is nearest the image center is kept;
     char boxes from all other word instances are discarded.
  3. AABB char boxes via connectedComponentsWithStats on the region score.
  4. Per-box BG color: mean of original pixels where region score < 0.3.
  5. Boxes are stored in NORMALIZED [0,1] coordinates (relative to the original
     image W,H), so the consumer can scale them to whatever size the STR model
     resizes to. No fixed 32x128 grid.

Output root: /data/isaackang/data/STR/Occ_aug/  (mirrors input relative path)
Key:   aug-XXXXXXXXX  (same index as image-XXXXXXXXX)
Value: struct.pack('>I', N) + boxes_float32 (N×4×2, normalized) + bgcolors_uint8 (N×3)
Checkpoint: __checkpoint__ → 8-byte big-endian last-written index

Decode:
  n      = struct.unpack('>I', data[:4])[0]
  boxes  = np.frombuffer(data[4:4+n*32], '<f4').reshape(n, 4, 2)   # in [0,1]
  colors = np.frombuffer(data[4+n*32:],  'u1').reshape(n, 3)
  # pixel coords: boxes * [W, H]

Usage:
  python tools/precompute_occ_aug.py --batch-size 256 --gpu-ids 0 1 2 3 4 5 6 7
"""

import argparse
import os
import struct
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE           = Path(__file__).resolve().parent
OCCSTR_ROOT     = _HERE.parent
CRAFT_ROOT      = OCCSTR_ROOT / 'third_party' / 'craft'
CRAFT_CKPT      = OCCSTR_ROOT / 'weights' / 'pretrained' / 'craft' / 'craft_ic15_20k.pth'

DATA_ROOT   = Path('/data/isaackang/data/STR/openocr')
OUTPUT_ROOT = Path('/data/isaackang/data/STR/Occ_aug')

CB_ROOT  = Path('/data/isaackang/data/STR/openocr/test')
CB_NAMES = ['IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80']

# ---------------------------------------------------------------------------
# CRAFT / detection constants
# ---------------------------------------------------------------------------
CANVAS_SIZE = 2560     # max long side after aspect-preserving resize
MAG_RATIO   = 1.5      # upscale factor (official CRAFT default for small text)

TEXT_THRESH     = 0.7
LOW_TEXT        = 0.4
LINK_THRESH     = 0.4   # affinity threshold for grouping chars into word instances
BG_SCORE_THRESH = 0.3
MIN_BOX_AREA    = 10    # min connected-component area (full-res region score)

# Batching: a sub-batch is packed up to this many "reference pixels"
# (REF_PIXELS ~= one 64x256 image) and never more than --batch-size images.
REF_PIXELS    = 64 * 256
WINDOW_FACTOR = 8       # images loaded/decoded per window = batch_size * this

MAP_SIZE       = 8 * (1 << 30)
CHECKPOINT_KEY = b'__checkpoint__'
EMPTY_BYTES    = struct.pack('>I', 0)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS: List[Tuple[str, Optional[List[str]]]] = [
    ('Union14M-L-LMDB-Filtered', [
        'filter_train_challenging', 'filter_train_easy', 'filter_train_hard',
        'filter_train_medium',      'filter_train_normal',
    ]),
    ('Union14M-U', ['book32_lmdb', 'cc_lmdb', 'openvino_lmdb']),
    ('UnionST',    ['UnionST-P', 'UnionST-R', 'UnionST-S']),
    ('RDC-U',      None),
]


# ---------------------------------------------------------------------------
# Format metadata
# ---------------------------------------------------------------------------
import json as _json

_VALUE_LAYOUT = [
    {"name": "n_boxes", "type": ">u4",  "bytes": 4,
     "desc": "number of char boxes (big-endian uint32)"},
    {"name": "boxes",   "type": "<f4",  "shape": ["n_boxes", 4, 2],
     "desc": "AABB corners NORMALIZED to [0,1] (TL,TR,BR,BL); multiply by image (W,H)"},
    {"name": "colors",  "type": "u1",   "shape": ["n_boxes", 3],
     "desc": "RGB BG color per box; avg of pixels where CRAFT region score < 0.3"},
]

FORMAT_META = {
    "description": "OccSTR precompute: char AABB boxes (normalized) + BG colors",
    "key_pattern": "aug-{idx:09d}",
    "coord_space": "normalized [0,1] relative to original image (W,H)",
    "craft": "full-res aspect-preserving (canvas 2560, mag 1.5), fp32, "
             "zero-pad batch (bit-identical to per-image)",
    "word_selection": "CRAFT affinity (link) word instances; keep only the word "
                      "whose center is nearest the image center",
    "special_keys": {
        "num-samples":    "total image count (bytes → int)",
        "__checkpoint__": "last written index (8-byte big-endian uint64)",
        "__format__":     "this JSON descriptor",
    },
    "value_layout": _VALUE_LAYOUT,
    "decode_snippet": (
        "n=struct.unpack('>I',d[:4])[0]; "
        "boxes=np.frombuffer(d[4:4+n*32],'<f4').reshape(n,4,2); "
        "colors=np.frombuffer(d[4+n*32:],'u1').reshape(n,3)  # boxes in [0,1]"
    ),
}

FORMAT_KEY = b'__format__'


def write_format_meta(env_out, meta: dict):
    """Write __format__ key if not already present."""
    with env_out.begin(write=True) as txn:
        if txn.get(FORMAT_KEY) is None:
            txn.put(FORMAT_KEY, _json.dumps(meta, ensure_ascii=False).encode())


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def encode_result(boxes: 'np.ndarray', colors: 'np.ndarray') -> bytes:
    n = len(boxes)
    if n == 0:
        return EMPTY_BYTES
    return (struct.pack('>I', n)
            + boxes.astype('<f4').tobytes()
            + colors.astype('u1').tobytes())


# ---------------------------------------------------------------------------
# LMDB / image helpers
# ---------------------------------------------------------------------------
def get_num_samples(path: Path) -> int:
    import lmdb
    env = lmdb.open(str(path), readonly=True, lock=False, max_readers=1,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        val = txn.get(b'num-samples')
    env.close()
    return int(val) if val else 0


def decode_image_fast(img_bytes: bytes) -> Optional['np.ndarray']:
    import numpy as np
    import cv2
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    try:
        import io
        from PIL import Image as PIL_Image
        return np.array(PIL_Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _log(log_path: Optional[Path], msg: str):
    ts   = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(line + '\n')


# ===========================================================================
# CRAFT inference — full-res, aspect-preserving, zero-pad batch (fp32)
# ===========================================================================
def _resized_hw(h: int, w: int, canvas: int = CANVAS_SIZE,
                mag: float = MAG_RATIO) -> Tuple[int, int, float]:
    """Replicate imgproc.resize_aspect_ratio's output size + ratio (no actual resize).

    Used for sub-batch packing (pixel budget) without paying the resize cost.
    """
    target = mag * max(h, w)
    if target > canvas:
        target = canvas
    ratio = target / max(h, w)
    th = int(h * ratio)
    tw = int(w * ratio)
    if th % 32:
        th += 32 - th % 32
    if tw % 32:
        tw += 32 - tw % 32
    return th, tw, ratio


def run_craft_batch(net, device, images_rgb: List['np.ndarray']
                    ) -> List[Tuple['np.ndarray', 'np.ndarray']]:
    """Run CRAFT on a list of RGB images, batched via top-left zero-padding.

    Each image is resized aspect-preserving exactly as the official CRAFT
    (resize_aspect_ratio), normalized individually, then placed at the top-left
    of a common zero canvas (sub-batch max size). The valid-region score maps
    are bit-identical to per-image inference.

    Returns, per image, (score_text, score_link) resized back to the ORIGINAL
    image resolution (float32, clipped to [0,1]).
    """
    import imgproc
    import torch
    import cv2
    import numpy as np

    normed, ratios, origs = [], [], []
    max_h = max_w = 0
    for img in images_rgb:
        resized, ratio, _ = imgproc.resize_aspect_ratio(
            img, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=MAG_RATIO)
        x = imgproc.normalizeMeanVariance(resized)           # HxWx3 float32, incl /32 pad
        normed.append(x)
        ratios.append(ratio)
        origs.append(img.shape[:2])
        max_h = max(max_h, x.shape[0])
        max_w = max(max_w, x.shape[1])

    B = len(images_rgb)
    batch = np.zeros((B, max_h, max_w, 3), dtype=np.float32)  # zero pad (matches conv pad)
    for i, x in enumerate(normed):
        batch[i, :x.shape[0], :x.shape[1], :] = x

    xt = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        y, _ = net(xt)                                       # [B, max_h/2, max_w/2, 2], fp32
    y = y.clamp(0, 1).float().cpu().numpy()

    out = []
    for i in range(B):
        h0, w0 = origs[i]
        ah = max(1, int(h0 * ratios[i]) // 2)
        aw = max(1, int(w0 * ratios[i]) // 2)
        st = y[i, :ah, :aw, 0]
        sl = y[i, :ah, :aw, 1]
        st = cv2.resize(st, (w0, h0), interpolation=cv2.INTER_LINEAR)
        sl = cv2.resize(sl, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out.append((st.astype(np.float32), sl.astype(np.float32)))
    return out


# ===========================================================================
# Word-instance selection + char boxes + BG colors
# ===========================================================================
def _select_center_word(chars, score_text, score_link, low_text, link_threshold,
                        w_img, h_img):
    """Keep only char boxes belonging to the word instance whose center is
    closest to the image center.

    Word instances = connected components of (region > low_text) | (link > thresh).
    Each char (AABB) is assigned to the word label that dominates its footprint;
    the word whose member-char centroid is nearest the image center is kept.
    """
    import cv2
    import numpy as np

    if len(chars) <= 1:
        return chars

    word_mask = ((score_text > low_text) | (score_link > link_threshold)).astype(np.uint8)
    n_word, wlabels = cv2.connectedComponents(word_mask, connectivity=4)
    if n_word <= 2:                       # background + at most one word → nothing to pick
        return chars

    char_word = np.zeros(len(chars), dtype=np.int32)
    centers   = np.zeros((len(chars), 2), dtype=np.float64)
    for i, (x1, y1, x2, y2) in enumerate(chars):
        patch = wlabels[y1:y2 + 1, x1:x2 + 1].ravel()
        patch = patch[patch != 0]
        char_word[i] = int(np.bincount(patch).argmax()) if patch.size else 0
        centers[i]   = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    cx_img, cy_img = (w_img - 1) / 2.0, (h_img - 1) / 2.0
    best_lbl, best_d = None, None
    for lbl in np.unique(char_word):
        if lbl == 0:
            continue
        wc = centers[char_word == lbl].mean(axis=0)
        d  = (wc[0] - cx_img) ** 2 + (wc[1] - cy_img) ** 2
        if best_d is None or d < best_d:
            best_d, best_lbl = d, lbl
    if best_lbl is None:
        return chars

    kept = [chars[i] for i in range(len(chars)) if char_word[i] == best_lbl]
    return kept if kept else chars


def detect_word_boxes(image_rgb, score_text, score_link,
                      text_threshold=TEXT_THRESH, low_text=LOW_TEXT,
                      link_threshold=LINK_THRESH, proxy_thresh=BG_SCORE_THRESH,
                      min_area=MIN_BOX_AREA):
    """Full-res char boxes (normalized [0,1]) + per-box BG color, restricted to
    the center-most word instance.

    Returns (boxes_norm (N,4,2) float32 in [0,1], colors (N,3) uint8).
    """
    import cv2
    import numpy as np

    h_img, w_img = image_rgb.shape[:2]

    # 1. character-level connected components on the region score
    bin_map = (score_text > low_text).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_map, connectivity=4)
    chars = []
    for k in range(1, n_labels):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        if score_text[labels == k].max() < text_threshold:
            continue
        x  = stats[k, cv2.CC_STAT_LEFT]
        y  = stats[k, cv2.CC_STAT_TOP]
        bw = stats[k, cv2.CC_STAT_WIDTH]
        bh = stats[k, cv2.CC_STAT_HEIGHT]
        x1 = max(0, x - 1);          y1 = max(0, y - 1)
        x2 = min(w_img - 1, x + bw); y2 = min(h_img - 1, y + bh)
        chars.append((x1, y1, x2, y2))

    if not chars:
        return np.zeros((0, 4, 2), np.float32), np.zeros((0, 3), np.uint8)

    # 2. keep only the center-most word instance (affinity-based grouping)
    chars = _select_center_word(chars, score_text, score_link,
                                low_text, link_threshold, w_img, h_img)

    # 3. sort L→R, per-box BG color, normalize coords
    chars.sort(key=lambda b: (b[0] + b[2]) / 2)
    fallback_bg = np.array([128, 128, 128], dtype=np.uint8)
    boxes, colors = [], []
    for (x1, y1, x2, y2) in chars:
        crop_orig  = image_rgb[y1:y2 + 1, x1:x2 + 1]
        crop_score = score_text[y1:y2 + 1, x1:x2 + 1]
        bg_pix = crop_orig[crop_score < proxy_thresh]
        colors.append(bg_pix.mean(axis=0).astype(np.uint8)
                      if bg_pix.size > 0 else fallback_bg)
        nx1 = x1 / w_img; ny1 = y1 / h_img
        nx2 = x2 / w_img; ny2 = y2 / h_img
        boxes.append(np.array([[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]],
                              dtype=np.float32))

    return np.array(boxes, dtype=np.float32), np.array(colors, dtype=np.uint8)


# ===========================================================================
# Per-LMDB processing (window load + size bucketing + checkpointed writes)
# ===========================================================================
def process_lmdb(net, device, in_path: Path, out_path: Path, batch_size: int,
                 log_path: Optional[Path], max_images: Optional[int] = None):
    import gc
    import lmdb
    import torch
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    env_in = lmdb.open(str(in_path), readonly=True, lock=False, max_readers=2,
                       readahead=False, meminit=False)
    with env_in.begin() as txn:
        n_total = int(txn.get(b'num-samples'))
    if max_images:
        n_total = min(n_total, max_images)

    out_path.mkdir(parents=True, exist_ok=True)
    env_out = lmdb.open(str(out_path), map_size=MAP_SIZE, sync=False, writemap=True)
    write_format_meta(env_out, FORMAT_META)
    with env_out.begin() as txn:
        cp_raw = txn.get(CHECKPOINT_KEY)
    start_idx = int.from_bytes(cp_raw, 'big') + 1 if cp_raw else 1

    tag = f'{in_path.parent.name}/{in_path.name}'
    if start_idx > n_total:
        _log(log_path, f'[{tag}] already complete ({n_total:,})')
        env_in.close(); env_out.close()
        return

    max_px = batch_size * REF_PIXELS
    win_sz = batch_size * WINDOW_FACTOR
    _log(log_path,
         f'[{tag}] {n_total - start_idx + 1:,}/{n_total:,} '
         f'(resume {start_idx}, bs<= {batch_size}, max_px {max_px:,})')
    t0 = time.time()
    n_err = 0

    txn_in = env_in.begin()

    def load_window(lo, hi):
        items, empties = [], []
        for idx in range(lo, hi):
            raw = txn_in.get(f'image-{idx:09d}'.encode())
            img = decode_image_fast(raw) if raw is not None else None
            if img is None:
                empties.append(idx)
            else:
                items.append((idx, img))
        return items, empties

    io_pool   = ThreadPoolExecutor(max_workers=1)
    win_starts = list(range(start_idx, n_total + 1, win_sz))
    fut = io_pool.submit(load_window, win_starts[0],
                         min(win_starts[0] + win_sz, n_total + 1))

    for wi, w_start in enumerate(win_starts):
        w_end = min(w_start + win_sz, n_total + 1)
        items, empties = fut.result()
        if wi + 1 < len(win_starts):
            nx = win_starts[wi + 1]
            fut = io_pool.submit(load_window, nx, min(nx + win_sz, n_total + 1))

        results = {idx: EMPTY_BYTES for idx in empties}
        n_err += len(empties)

        # Group by EXACT resized size so every sub-batch needs ZERO padding.
        # This is the only way batched CRAFT is bit-identical to per-image:
        # padding would leak through conv bias + BatchNorm into the valid
        # region (re-zeroed every layer for true single-image conv padding, but
        # accumulated as nonzero activations inside a padded batch). Sub-batch
        # size is capped by --batch-size and a pixel budget (memory guard).
        groups = defaultdict(list)
        for it in items:
            th, tw, _ = _resized_hw(it[1].shape[0], it[1].shape[1])
            groups[(th, tw)].append(it)

        for (th, tw), members in groups.items():
            per = max(1, min(batch_size, max_px // (th * tw)))
            for s in range(0, len(members), per):
                sub  = members[s:s + per]
                imgs = [it[1] for it in sub]
                try:
                    scores = run_craft_batch(net, device, imgs)
                    for (idx, img), (st, sl) in zip(sub, scores):
                        boxes, colors = detect_word_boxes(img, st, sl)
                        results[idx] = encode_result(boxes, colors)
                except Exception as e:
                    _log(log_path,
                         f'[{tag}] sub-batch err @ win {w_start} '
                         f'(size {th}x{tw}, n={len(sub)}): {e}; per-image retry')
                    for (idx, img) in sub:
                        try:
                            st, sl = run_craft_batch(net, device, [img])[0]
                            boxes, colors = detect_word_boxes(img, st, sl)
                            results[idx] = encode_result(boxes, colors)
                        except Exception as e2:
                            _log(log_path, f'[{tag}] img err idx={idx}: {e2}')
                            results[idx] = EMPTY_BYTES
                            n_err += 1

        with env_out.begin(write=True) as txn_out:
            for idx in sorted(results):
                txn_out.put(f'aug-{idx:09d}'.encode(), results[idx])
            txn_out.put(CHECKPOINT_KEY, (w_end - 1).to_bytes(8, 'big'))

        done    = w_end - start_idx
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 1
        eta     = (n_total - start_idx + 1 - done) / rate
        _log(log_path,
             f'[{tag}] {w_end - 1:,}/{n_total:,} | {rate:.0f}/s | '
             f'ETA {eta/60:.1f}m | err={n_err}')
        torch.cuda.empty_cache()
        gc.collect()

    with env_out.begin(write=True) as txn:
        txn.put(b'num-samples', str(n_total).encode())
    io_pool.shutdown()
    env_in.close(); env_out.close()
    elapsed = time.time() - t0
    n_done  = n_total - start_idx + 1
    _log(log_path,
         f'[{tag}] DONE {n_done:,} in {elapsed:.0f}s '
         f'({n_done/max(elapsed,1):.0f}/s) err={n_err}')


# ---------------------------------------------------------------------------
# Worker (one GPU)
# ---------------------------------------------------------------------------
def worker(gpu_id: int, tasks: list, batch_size: int,
           craft_root_str: str, craft_ckpt_str: str, log_dir_str: str,
           max_images: Optional[int] = None):
    # PCI_BUS_ID so --gpu-ids match `nvidia-smi` indices (torch defaults to
    # FASTEST_FIRST otherwise). Must be set before any CUDA init.
    os.environ['CUDA_DEVICE_ORDER']   = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sys.path.insert(0, craft_root_str)

    import torch
    from collections import OrderedDict
    from craft import CRAFT

    device   = 'cuda:0'
    log_dir  = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'gpu{gpu_id}.log'

    net = CRAFT()
    sd  = torch.load(craft_ckpt_str, map_location='cpu', weights_only=False)
    if next(iter(sd)).startswith('module'):
        sd = OrderedDict((k.split('.', 1)[1], v) for k, v in sd.items())
    net.load_state_dict(sd)
    net.eval().to(device)

    _log(log_path, f'[GPU {gpu_id}] ready (fp32), '
                   f'{len(tasks)} task(s): {[t[0].name for t in tasks]}')

    for in_path, out_path, _ in tasks:
        process_lmdb(net, device, in_path, out_path, batch_size, log_path,
                     max_images=max_images)

    del net
    torch.cuda.empty_cache()
    _log(log_path, f'[GPU {gpu_id}] all tasks done')


# ---------------------------------------------------------------------------
# Task collection + assignment
# ---------------------------------------------------------------------------
def collect_tasks(no_rdcu: bool, include_cb: bool,
                  out_root: Path = OUTPUT_ROOT) -> List[Tuple[Path, Path, int]]:
    tasks = []
    for ds_name, subdirs in DATASETS:
        if ds_name == 'RDC-U' and no_rdcu:
            print('Skipping RDC-U (--no-rdcu)', flush=True)
            continue
        ds_root = DATA_ROOT / ds_name
        if subdirs is None:
            n = get_num_samples(ds_root)
            if n:
                tasks.append((ds_root,
                              out_root / ds_root.relative_to(DATA_ROOT), n))
        else:
            for sub in subdirs:
                in_path = ds_root / sub
                n = get_num_samples(in_path)
                if n:
                    tasks.append((in_path,
                                  out_root / in_path.relative_to(DATA_ROOT), n))
    if include_cb:
        for name in CB_NAMES:
            in_path = CB_ROOT / name
            try:
                n = get_num_samples(in_path)
                if n:
                    tasks.append((in_path,
                                  out_root / in_path.relative_to(DATA_ROOT), n))
            except Exception as e:
                print(f'SKIP CB/{name}: {e}', flush=True)
    return tasks


def greedy_assign(tasks, n_gpus: int) -> List[List]:
    assignments = [[] for _ in range(n_gpus)]
    workloads   = [0] * n_gpus
    for t in sorted(tasks, key=lambda x: -x[2]):
        g = min(range(n_gpus), key=lambda i: workloads[i])
        assignments[g].append(t)
        workloads[g] += t[2]
    return assignments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Precompute CRAFT boxes + BG colors for OccSTR augmentation '
                    '(full-res, fp32, normalized coords)')
    ap.add_argument('--batch-size', type=int, default=256,
                    help='max images per CRAFT sub-batch (pixel-budgeted)')
    ap.add_argument('--num-gpus',   type=int, default=8,
                    help='number of GPUs to use')
    ap.add_argument('--gpu-ids', type=int, nargs='+', default=None,
                    help='physical GPU IDs to use (default: 0..num-gpus-1)')
    ap.add_argument('--no-rdcu', action='store_true', help='skip RDC-U dataset')
    ap.add_argument('--include-cb', action='store_true',
                    help='also process the CB evaluation datasets')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='only process input LMDBs whose path contains one of '
                         'these substrings (test runs)')
    ap.add_argument('--max-images', type=int, default=None,
                    help='cap images per LMDB to this (test runs)')
    ap.add_argument('--out-root', type=str, default=str(OUTPUT_ROOT),
                    help='output root (override for test runs)')
    args = ap.parse_args()

    gpu_ids  = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))
    out_root = Path(args.out_root)
    log_dir  = out_root / 'logs'

    tasks = collect_tasks(args.no_rdcu, args.include_cb, out_root)
    if args.datasets:
        tasks = [t for t in tasks
                 if any(s in str(t[0]) for s in args.datasets)]
    if args.max_images:
        tasks = [(ip, op, min(n, args.max_images)) for ip, op, n in tasks]
    if not tasks:
        print('No tasks found — check DATA_ROOT / --datasets', flush=True)
        return

    total = sum(t[2] for t in tasks)
    print(f'\n{len(tasks)} LMDBs, {total:,} images', flush=True)
    for in_p, out_p, n in sorted(tasks, key=lambda x: -x[2]):
        cp_exists = (out_p / 'data.mdb').exists()
        print(f'  {in_p.parent.name}/{in_p.name}: {n:,}'
              + (' (partial output exists)' if cp_exists else ''), flush=True)

    n_gpus      = min(len(gpu_ids), len(tasks))
    assignments = greedy_assign(tasks, n_gpus)
    print(f'\nTask assignment ({n_gpus} GPUs):', flush=True)
    for g, ts in enumerate(assignments):
        if ts:
            w = sum(t[2] for t in ts)
            print(f'  GPU {gpu_ids[g]}: {w:,} → {[t[0].name for t in ts]}', flush=True)

    processes = []
    t_start   = time.time()
    for g in range(n_gpus):
        if not assignments[g]:
            continue
        p = mp.Process(
            target=worker,
            args=(gpu_ids[g], assignments[g], args.batch_size,
                  str(CRAFT_ROOT), str(CRAFT_CKPT), str(log_dir),
                  args.max_images),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    elapsed = time.time() - t_start
    print(f'\nAll done in {elapsed/3600:.2f}h ({elapsed:.0f}s)', flush=True)
    print(f'Output: {OUTPUT_ROOT}', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
