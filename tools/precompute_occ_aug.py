#!/usr/bin/env python3
"""Precompute CRAFT character boxes + background colors for OccSTR stroke augmentation.

Two modes:

  CB mode (--mode cb)  — identical algorithm to hisam_cb_compare.py:
    1. Hi-SAM TextSeg: stroke segmentation mask at original resolution
    2. CRAFT: character region score map at original resolution
    3. AABB char boxes via connectedComponentsWithStats on score_text
    4. Per-box BG: image_rgb[crop_mask < 0.5].mean(axis=0)  (Hi-SAM mask)
    5. Scale boxes to 32×128 coordinates; write to output LMDB

  Train mode (--mode train)  — batch CRAFT on 32×128 crops (no Hi-SAM):
    1. Resize to 32×128; upscale 2× to 64×256; batched CRAFT fp16
    2. AABB char boxes from CRAFT score map (already 32×128)
    3. Per-box BG: image_32[crop_score < 0.3].mean()  (CRAFT score proxy)

Output root: /data/isaackang/data/STR/Occ_aug/
Key:   aug-XXXXXXXXX  (same index as image-XXXXXXXXX)
Value: struct.pack('>I', N) + boxes_float32 (N×4×2) + bgcolors_uint8 (N×3)
Checkpoint: __checkpoint__ → 8-byte big-endian last-written index

Decode:
  n      = struct.unpack('>I', data[:4])[0]
  boxes  = np.frombuffer(data[4:4+n*32], '<f4').reshape(n, 4, 2)
  colors = np.frombuffer(data[4+n*32:],  'u1').reshape(n, 3)

CB usage:
  python tools/precompute_occ_aug.py --mode cb --gpu-ids 4 5 6 7

Train usage:
  python tools/precompute_occ_aug.py --mode train --batch-size 256 --gpu-ids 0 1 2 3 4 5 6 7
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
HISAM_ROOT      = OCCSTR_ROOT / 'third_party' / 'Hi-SAM'
CRAFT_CKPT      = OCCSTR_ROOT / 'weights' / 'pretrained' / 'craft' / 'craft_ic15_20k.pth'
TEXTSEG_CKPT    = OCCSTR_ROOT / 'weights' / 'pretrained' / 'hi_sam' / 'sam_tss_h_textseg.pth'

DATA_ROOT   = Path('/data/isaackang/data/STR/openocr')
OUTPUT_ROOT = Path('/data/isaackang/data/STR/Occ_aug')

CB_ROOT  = Path('/data/isaackang/data/STR/openocr/test')
CB_NAMES = ['IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80']

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_H, CROP_W   = 32, 128
CRAFT_H, CRAFT_W = 64, 256
SCORE_H, SCORE_W = 32, 128

_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_STD  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

TEXT_THRESH     = 0.7
LOW_TEXT        = 0.4
BG_SCORE_THRESH = 0.3
MIN_BOX_AREA    = 4
THIN_CHAR_RATIO = 2.0   # boxes with h/w >= this are forced to LR stroke direction


def sample_stroke_direction(box_h: float, box_w: float, rng) -> str:
    """Return 'lr' or 'td'. Identical rule to hisam_cb_compare.draw_stroke_on_crop."""
    if box_w < 1e-5 or box_h / box_w >= THIN_CHAR_RATIO:
        return 'lr'
    return 'td' if rng.integers(0, 2) else 'lr'

MAP_SIZE       = 8 * (1 << 30)
CHECKPOINT_KEY = b'__checkpoint__'
EMPTY_BYTES    = struct.pack('>I', 0)

# ---------------------------------------------------------------------------
# Training datasets
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
# LMDB helpers
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
# CB MODE — identical algorithm to hisam_cb_compare.py
# ===========================================================================

def _build_hisam_args(ckpt_path):
    return argparse.Namespace(
        model_type='vit_h',
        checkpoint=str(ckpt_path),
        input_size=[1024, 1024],
        attn_layers=1,
        prompt_len=12,
        hier_det=False,
    )


def load_hisam(ckpt_path, device):
    from hi_sam.modeling.build import model_registry
    from hi_sam.modeling.predictor import SamPredictor
    ns = _build_hisam_args(ckpt_path)
    model = model_registry['vit_h'](ns)
    model.eval().to(device)
    return SamPredictor(model)


def run_hisam(predictor, image_rgb: 'np.ndarray') -> 'np.ndarray':
    """Return binary stroke mask (uint8) at original image resolution."""
    predictor.set_image(image_rgb)
    _, hr_mask, _, _ = predictor.predict(multimask_output=False)
    return hr_mask[0].astype('uint8')


def run_craft_fullres(net, image_rgb: 'np.ndarray', device,
                      canvas_size: int = 2560, mag_ratio: float = 1.5) -> 'np.ndarray':
    """CRAFT score map resized back to original image resolution.
    Identical to hisam_cb_compare.run_craft.
    """
    import imgproc
    import torch
    import cv2
    import numpy as np

    h, w = image_rgb.shape[:2]
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image_rgb, canvas_size,
        interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        y, _ = net(x)
    score_text = y[0, :, :, 0].cpu().numpy()
    active_h = max(1, int(h * target_ratio) // 2)
    active_w = max(1, int(w * target_ratio) // 2)
    score_text = score_text[:active_h, :active_w]
    score_text = np.clip(score_text, 0, 1)
    score_text = cv2.resize(score_text, (w, h), interpolation=cv2.INTER_LINEAR)
    return score_text.astype(np.float32)


def get_char_boxes_and_bg_hisam(image_rgb, mask, score_text,
                                 text_threshold=TEXT_THRESH,
                                 low_text=LOW_TEXT):
    """IDENTICAL algorithm to hisam_cb_compare.get_char_data.

    AABB boxes from CRAFT score_text via connectedComponentsWithStats;
    per-box BG color from image_rgb excluding Hi-SAM stroke pixels.

    Returns (boxes_32, colors):
      boxes_32 — (N, 4, 2) float32 in CROP_H×CROP_W (32×128) space
      colors   — (N, 3) uint8
    """
    import cv2
    import numpy as np

    h_img, w_img = image_rgb.shape[:2]
    bin_map = (score_text > low_text).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_map, connectivity=4)

    boxes = []
    for k in range(1, n_labels):
        if stats[k, cv2.CC_STAT_AREA] < 10:
            continue
        if score_text[labels == k].max() < text_threshold:
            continue
        x  = stats[k, cv2.CC_STAT_LEFT]
        y  = stats[k, cv2.CC_STAT_TOP]
        bw = stats[k, cv2.CC_STAT_WIDTH]
        bh = stats[k, cv2.CC_STAT_HEIGHT]
        x1 = max(0, x - 1);          y1 = max(0, y - 1)
        x2 = min(w_img - 1, x + bw); y2 = min(h_img - 1, y + bh)
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        return np.zeros((0, 4, 2), np.float32), np.zeros((0, 3), np.uint8)

    boxes.sort(key=lambda b: (b[0] + b[2]) / 2)

    scale_x = CROP_W / w_img
    scale_y = CROP_H / h_img
    fallback_bg = np.array([128, 128, 128], dtype=np.uint8)
    boxes_32 = []
    colors   = []
    for (x1, y1, x2, y2) in boxes:
        crop_orig = image_rgb[y1:y2 + 1, x1:x2 + 1]
        crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
        bg_pixels = crop_orig[crop_mask < 0.5]
        bg_color  = (bg_pixels.mean(axis=0).astype(np.uint8)
                     if bg_pixels.size > 0 else fallback_bg)
        x1_32 = float(np.clip(x1 * scale_x, 0, CROP_W - 1))
        y1_32 = float(np.clip(y1 * scale_y, 0, CROP_H - 1))
        x2_32 = float(np.clip(x2 * scale_x, 0, CROP_W - 1))
        y2_32 = float(np.clip(y2 * scale_y, 0, CROP_H - 1))
        box_32 = np.array([[x1_32, y1_32], [x2_32, y1_32],
                            [x2_32, y2_32], [x1_32, y2_32]], dtype=np.float32)
        boxes_32.append(box_32)
        colors.append(bg_color)

    return np.array(boxes_32, dtype=np.float32), np.array(colors, dtype=np.uint8)


def process_lmdb_cb(net, predictor, device, in_path: Path, out_path: Path,
                    log_path: Optional[Path]):
    import gc
    import lmdb
    import torch

    env_in = lmdb.open(str(in_path), readonly=True, lock=False, max_readers=1,
                       readahead=False, meminit=False)
    with env_in.begin() as txn:
        n_total = int(txn.get(b'num-samples'))

    out_path.mkdir(parents=True, exist_ok=True)
    env_out = lmdb.open(str(out_path), map_size=MAP_SIZE, sync=False, writemap=True)
    with env_out.begin() as txn:
        cp_raw = txn.get(CHECKPOINT_KEY)
    start_idx = int.from_bytes(cp_raw, 'big') + 1 if cp_raw else 1

    tag = in_path.name
    if start_idx > n_total:
        _log(log_path, f'[{tag}] already complete ({n_total:,})')
        env_in.close(); env_out.close()
        return

    _log(log_path, f'[{tag}] {n_total - start_idx + 1:,}/{n_total:,} images (Hi-SAM+CRAFT, resume {start_idx})')
    t0 = time.time()
    n_errors = 0
    COMMIT_EVERY  = 50
    CLEANUP_EVERY = 500   # flush GPU cache + Python GC

    buf = {}
    with env_in.begin() as txn_in:
        for idx in range(start_idx, n_total + 1):
            raw = txn_in.get(f'image-{idx:09d}'.encode())
            if raw is None:
                buf[idx] = EMPTY_BYTES
                n_errors += 1
            else:
                img = decode_image_fast(raw)
                if img is None:
                    buf[idx] = EMPTY_BYTES
                    n_errors += 1
                else:
                    mask = score_text = None
                    try:
                        mask       = run_hisam(predictor, img)
                        predictor.reset_image()          # free cached image features
                        score_text = run_craft_fullres(net, img, device)
                        boxes, colors = get_char_boxes_and_bg_hisam(img, mask, score_text)
                        buf[idx] = encode_result(boxes, colors)
                    except Exception as e:
                        _log(log_path, f'[{tag}] error idx={idx}: {e}')
                        buf[idx] = EMPTY_BYTES
                        n_errors += 1
                        try:
                            predictor.reset_image()
                        except Exception:
                            pass
                    finally:
                        del mask, score_text             # free large arrays immediately

            if len(buf) >= COMMIT_EVERY or idx == n_total:
                with env_out.begin(write=True) as txn_out:
                    for i, data in buf.items():
                        txn_out.put(f'aug-{i:09d}'.encode(), data)
                    txn_out.put(CHECKPOINT_KEY, idx.to_bytes(8, 'big'))
                buf.clear()

            done = idx - start_idx + 1
            if done % CLEANUP_EVERY == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if done % 100 == 0 or idx == n_total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 1
                eta     = (n_total - start_idx + 1 - done) / rate
                _log(log_path,
                     f'[{tag}] {idx:,}/{n_total:,} | {rate:.2f}/s | '
                     f'ETA {eta/60:.1f}m | err={n_errors}')

    with env_out.begin(write=True) as txn:
        txn.put(b'num-samples', str(n_total).encode())
    env_in.close(); env_out.close()
    elapsed = time.time() - t0
    n_done  = n_total - start_idx + 1
    _log(log_path,
         f'[{tag}] DONE {n_done:,} in {elapsed:.0f}s ({n_done/elapsed:.2f}/s) err={n_errors}')


def worker_cb(gpu_id: int, tasks: list,
              craft_root_str: str, craft_ckpt_str: str,
              hisam_root_str: str, textseg_ckpt_str: str,
              log_dir_str: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sys.path.insert(0, craft_root_str)
    sys.path.insert(0, hisam_root_str)

    import torch
    from collections import OrderedDict
    from craft import CRAFT

    device   = 'cuda:0'
    log_dir  = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'gpu{gpu_id}_cb.log'

    net = CRAFT()
    sd  = torch.load(craft_ckpt_str, map_location='cpu', weights_only=False)
    if next(iter(sd)).startswith('module'):
        sd = OrderedDict((k.split('.', 1)[1], v) for k, v in sd.items())
    net.load_state_dict(sd)
    net.eval().to(device)

    predictor = load_hisam(textseg_ckpt_str, device)

    _log(log_path, f'[GPU {gpu_id}] CB mode ready, {len(tasks)} task(s): '
                   f'{[t[0].name for t in tasks]}')

    for in_path, out_path in tasks:
        process_lmdb_cb(net, predictor, device, in_path, out_path, log_path)

    del net, predictor
    torch.cuda.empty_cache()
    _log(log_path, f'[GPU {gpu_id}] all CB tasks done')


def collect_cb_tasks() -> List[Tuple[Path, Path, int]]:
    tasks = []
    for name in CB_NAMES:
        in_path  = CB_ROOT / name
        out_path = OUTPUT_ROOT / name
        try:
            n = get_num_samples(in_path)
            if n:
                tasks.append((in_path, out_path, n))
        except Exception as e:
            print(f'SKIP CB/{name}: {e}', flush=True)
    return tasks


# ===========================================================================
# TRAIN MODE — batch CRAFT on 32×128 crops
# ===========================================================================

def fast_char_boxes(smap: 'np.ndarray', text_thresh: float, low_text: float,
                    min_area: int = MIN_BOX_AREA) -> List['np.ndarray']:
    """AABB char boxes (N×(4,2) float32) from CRAFT score map at 32×128."""
    import numpy as np
    import cv2

    mask = (smap > low_text).astype(np.uint8)
    if mask.sum() == 0:
        return []

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=4)

    h, w = smap.shape
    boxes = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if smap[labels == i].max() < text_thresh:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1 = max(0, x - 1);  y1 = max(0, y - 1)
        x2 = min(w - 1, x + bw); y2 = min(h - 1, y + bh)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        boxes.append(box)

    boxes.sort(key=lambda b: float(b[:, 0].mean()))
    return boxes


_PREP_POOL: Optional[object] = None
_PREP_NTHREADS = 4


def _get_prep_pool():
    global _PREP_POOL
    if _PREP_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _PREP_POOL = ThreadPoolExecutor(max_workers=_PREP_NTHREADS)
    return _PREP_POOL


def _prep_chunk(args):
    imgs, mean, std = args
    import cv2, numpy as np
    crops, normed = [], []
    for img in imgs:
        c = cv2.resize(img, (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)
        u = cv2.resize(img, (CRAFT_W, CRAFT_H), interpolation=cv2.INTER_LINEAR)
        crops.append(c)
        normed.append((u.astype(np.float32) - mean) / std)
    return crops, normed


def _post_chunk(args):
    chunk, text_thresh, low_text, bg_thresh, crop_h, crop_w, score_h = args
    import cv2, numpy as np
    fallback_bg = np.array([128, 128, 128], dtype=np.uint8)
    results = []
    for smap, img_crop in chunk:
        boxes_list = fast_char_boxes(smap, text_thresh, low_text)
        if not boxes_list:
            results.append((np.zeros((0, 4, 2), np.float32),
                            np.zeros((0, 3), np.uint8)))
            continue
        boxes_np = np.array(boxes_list)
        boxes_np[..., 0] = np.clip(boxes_np[..., 0], 0, crop_w - 1)
        boxes_np[..., 1] = np.clip(boxes_np[..., 1], 0, crop_h - 1)
        colors = []
        for box in boxes_np:
            x1 = int(box[:, 0].min()); x2 = int(box[:, 0].max())
            y1 = int(box[:, 1].min()); y2 = int(box[:, 1].max())
            crop_img   = img_crop[y1:y2 + 1, x1:x2 + 1]
            crop_score = smap[y1:y2 + 1, x1:x2 + 1]
            bg_pix = crop_img[crop_score < bg_thresh]
            colors.append(bg_pix.mean(axis=0).astype(np.uint8)
                          if bg_pix.size > 0 else fallback_bg)
        results.append((boxes_np, np.array(colors, dtype=np.uint8)))
    return results


def run_batch(net, device, images_rgb: List, use_amp: bool
              ) -> List[Tuple['np.ndarray', 'np.ndarray']]:
    import numpy as np
    import torch

    B    = len(images_rgb)
    mean = np.array(_MEAN, dtype=np.float32)
    std  = np.array(_STD,  dtype=np.float32)
    pool = _get_prep_pool()

    chunk_size = max(1, (B + _PREP_NTHREADS - 1) // _PREP_NTHREADS)
    prep_args  = [(images_rgb[i:i+chunk_size], mean, std)
                  for i in range(0, B, chunk_size)]
    chunk_res  = list(pool.map(_prep_chunk, prep_args))

    imgs_crop   = [c for crops, _ in chunk_res for c in crops]
    craft_batch = np.stack([n for _, normed in chunk_res for n in normed])

    x = torch.from_numpy(craft_batch.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda'):
                y, _ = net(x)
        else:
            y, _ = net(x)

    score_maps = y[..., 0].clamp(0, 1).float().cpu().numpy()

    post_pairs  = [(score_maps[i], imgs_crop[i]) for i in range(B)]
    post_chunks = [post_pairs[i:i+chunk_size] for i in range(0, B, chunk_size)]
    post_args   = [(c, TEXT_THRESH, LOW_TEXT, BG_SCORE_THRESH,
                    CROP_H, CROP_W, SCORE_H) for c in post_chunks]
    post_res    = list(pool.map(_post_chunk, post_args))

    return [r for chunk in post_res for r in chunk]


def _run_single(net, device, img_rgb, use_amp):
    return run_batch(net, device, [img_rgb], use_amp)[0]


def process_lmdb(net, device, use_amp,
                 in_path: Path, out_path: Path, batch_size: int,
                 log_path: Optional[Path]):
    import lmdb

    env_in  = lmdb.open(str(in_path), readonly=True, lock=False, max_readers=1,
                        readahead=False, meminit=False)
    txn_in  = env_in.begin()
    n_total = int(txn_in.get(b'num-samples'))

    out_path.mkdir(parents=True, exist_ok=True)
    env_out = lmdb.open(str(out_path), map_size=MAP_SIZE, sync=False, writemap=True)

    with env_out.begin() as txn:
        cp_raw = txn.get(CHECKPOINT_KEY)
    start_idx = int.from_bytes(cp_raw, 'big') + 1 if cp_raw else 1

    tag = in_path.name
    if start_idx > n_total:
        _log(log_path, f'[{tag}] already complete ({n_total:,})')
        env_in.close(); env_out.close()
        return

    _log(log_path,
         f'[{tag}] {n_total - start_idx + 1:,}/{n_total:,} (resume from {start_idx})')
    t0 = time.time()
    n_errors    = 0
    report_every = 50 * batch_size

    pool = _get_prep_pool()

    def _load_batch(indices):
        imgs, valid, empty = [], [], []
        for idx in indices:
            raw = txn_in.get(f'image-{idx:09d}'.encode())
            if raw is None:
                empty.append(idx)
                continue
            img = decode_image_fast(raw)
            if img is None:
                empty.append(idx)
            else:
                imgs.append(img)
                valid.append(idx)
        return imgs, valid, empty

    batch_ranges = list(range(start_idx, n_total + 1, batch_size))
    first_end    = min(batch_ranges[0] + batch_size, n_total + 1)
    prefetch_fut = pool.submit(_load_batch, range(batch_ranges[0], first_end))

    for b_num, batch_start in enumerate(batch_ranges):
        last_idx = min(batch_start + batch_size, n_total + 1) - 1
        buf_imgs, buf_idxs, empty_idxs = prefetch_fut.result()

        if b_num + 1 < len(batch_ranges):
            nx_start     = batch_ranges[b_num + 1]
            nx_end       = min(nx_start + batch_size, n_total + 1)
            prefetch_fut = pool.submit(_load_batch, range(nx_start, nx_end))

        txn_out = env_out.begin(write=True)
        for idx in empty_idxs:
            txn_out.put(f'aug-{idx:09d}'.encode(), EMPTY_BYTES)
        n_errors += len(empty_idxs)

        if buf_imgs:
            try:
                batch_res = run_batch(net, device, buf_imgs, use_amp)
                for b_idx, (boxes, colors) in zip(buf_idxs, batch_res):
                    txn_out.put(f'aug-{b_idx:09d}'.encode(),
                                encode_result(boxes, colors))
            except Exception as e:
                _log(log_path,
                     f'[{tag}] batch error @ {batch_start}: {e}; retrying one-by-one')
                for b_idx, img in zip(buf_idxs, buf_imgs):
                    try:
                        boxes, colors = _run_single(net, device, img, use_amp)
                        txn_out.put(f'aug-{b_idx:09d}'.encode(),
                                    encode_result(boxes, colors))
                    except Exception as e2:
                        _log(log_path, f'[{tag}] single error idx={b_idx}: {e2}')
                        txn_out.put(f'aug-{b_idx:09d}'.encode(), EMPTY_BYTES)
                        n_errors += 1

        txn_out.put(CHECKPOINT_KEY, last_idx.to_bytes(8, 'big'))
        txn_out.commit()

        done = last_idx - start_idx + 1
        if done > 0 and (batch_start - start_idx) % report_every < batch_size:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 1
            eta     = (n_total - start_idx + 1 - done) / rate
            _log(log_path,
                 f'[{tag}] {last_idx:,}/{n_total:,} | '
                 f'{rate:.0f}/s | ETA {eta/60:.1f}m | err={n_errors}')

    with env_out.begin(write=True) as txn:
        txn.put(b'num-samples', str(n_total).encode())
    env_in.close(); env_out.close()
    elapsed = time.time() - t0
    n_done  = n_total - start_idx + 1
    _log(log_path,
         f'[{tag}] DONE {n_done:,} in {elapsed:.0f}s ({n_done/elapsed:.0f}/s) err={n_errors}')


def worker(gpu_id: int, tasks: list, batch_size: int,
           craft_root_str: str, craft_ckpt_str: str, log_dir_str: str):
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
        from collections import OrderedDict as OD
        sd = OD((k.split('.', 1)[1], v) for k, v in sd.items())
    net.load_state_dict(sd)
    net.eval().to(device)

    use_amp = torch.cuda.is_available()
    _log(log_path, f'[GPU {gpu_id}] ready (amp={use_amp}), '
                   f'{len(tasks)} task(s): {[t[0].name for t in tasks]}')

    for in_path, out_path, _ in tasks:
        process_lmdb(net, device, use_amp, in_path, out_path, batch_size, log_path)

    del net
    torch.cuda.empty_cache()
    _log(log_path, f'[GPU {gpu_id}] all tasks done')


# ---------------------------------------------------------------------------
# Task collection + assignment
# ---------------------------------------------------------------------------
def collect_train_tasks(no_rdcu: bool) -> List[Tuple[Path, Path, int]]:
    tasks = []
    for ds_name, subdirs in DATASETS:
        if ds_name == 'RDC-U' and no_rdcu:
            print(f'Skipping RDC-U (--no-rdcu)', flush=True)
            continue
        ds_root = DATA_ROOT / ds_name
        if subdirs is None:
            n = get_num_samples(ds_root)
            if n:
                tasks.append((ds_root, OUTPUT_ROOT / ds_name, n))
        else:
            for sub in subdirs:
                n = get_num_samples(ds_root / sub)
                if n:
                    tasks.append((ds_root / sub,
                                  OUTPUT_ROOT / ds_name / sub, n))
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
        description='Precompute CRAFT boxes + BG colors for OccSTR augmentation')
    ap.add_argument('--mode', choices=['train', 'cb'], default='train',
                    help='cb: Hi-SAM+CRAFT on CB datasets; train: batch CRAFT on train sets')
    ap.add_argument('--batch-size', type=int, default=256,
                    help='(train mode) images per CRAFT batch per GPU')
    ap.add_argument('--num-gpus',   type=int, default=8,
                    help='number of GPUs to use')
    ap.add_argument('--gpu-ids', type=int, nargs='+', default=None,
                    help='physical GPU IDs to use (default: 0..num-gpus-1)')
    ap.add_argument('--no-rdcu', action='store_true',
                    help='(train mode) skip RDC-U dataset')
    args = ap.parse_args()

    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))
    log_dir = OUTPUT_ROOT / 'logs'

    if args.mode == 'cb':
        tasks = collect_cb_tasks()
        if not tasks:
            print('No CB tasks found — check CB_ROOT', flush=True)
            return
        total = sum(t[2] for t in tasks)
        print(f'\nCB mode: {len(tasks)} LMDBs, {total:,} images total', flush=True)
        for in_p, out_p, n in tasks:
            cp_exists = (out_p / 'data.mdb').exists()
            print(f'  {in_p.name}: {n:,}'
                  + (' (partial output exists)' if cp_exists else ''), flush=True)

        n_gpus      = min(len(gpu_ids), len(tasks))
        assignments = greedy_assign(tasks, n_gpus)
        print(f'\nTask assignment ({n_gpus} GPUs):', flush=True)
        for g, ts in enumerate(assignments):
            if ts:
                print(f'  GPU {gpu_ids[g]}: {sum(t[2] for t in ts):,} → '
                      f'{[t[0].name for t in ts]}', flush=True)

        processes = []
        t_start   = time.time()
        for g in range(n_gpus):
            if not assignments[g]:
                continue
            actual_gpu = gpu_ids[g]
            task_pairs = [(t[0], t[1]) for t in assignments[g]]
            p = mp.Process(
                target=worker_cb,
                args=(actual_gpu, task_pairs,
                      str(CRAFT_ROOT), str(CRAFT_CKPT),
                      str(HISAM_ROOT), str(TEXTSEG_CKPT),
                      str(log_dir)),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        elapsed = time.time() - t_start
        print(f'\nCB done in {elapsed/60:.1f}m ({elapsed:.0f}s)', flush=True)
        print(f'Output: {OUTPUT_ROOT}', flush=True)

    else:  # train mode
        tasks = collect_train_tasks(args.no_rdcu)
        if not tasks:
            print('No tasks found — check DATA_ROOT', flush=True)
            return
        total = sum(t[2] for t in tasks)
        print(f'\nTrain mode: {len(tasks)} LMDBs, {total:,} images', flush=True)
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
                print(f'  GPU {gpu_ids[g]}: {w:,} → {[t[0].name for t in ts]}',
                      flush=True)

        processes = []
        t_start   = time.time()
        for g in range(n_gpus):
            if not assignments[g]:
                continue
            actual_gpu = gpu_ids[g]
            p = mp.Process(
                target=worker,
                args=(actual_gpu, assignments[g], args.batch_size,
                      str(CRAFT_ROOT), str(CRAFT_CKPT), str(log_dir)),
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
