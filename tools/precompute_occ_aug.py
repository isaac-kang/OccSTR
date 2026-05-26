#!/usr/bin/env python3
"""Precompute CRAFT character boxes + background colors for OccSTR stroke augmentation.

For every image in each training LMDB:
  1. Resize to 32×128 (canonical STR crop size)
  2. Upscale 4× to 128×512 and run batched CRAFT inference (fp16 autocast)
  3. Detect character boxes with fast cv2 connected-components (axis-aligned)
     and scale to 32×128 coordinates
  4. Compute a single per-image background color from CRAFT score map
     (pixels with score < 0.3 are background); replicated for every box
  5. Write to parallel output LMDB

Output: /data/isaackang/data/STR/openocr/occ_aug/<dataset>/<sub>/
Key format: aug-XXXXXXXXX (same index as image-XXXXXXXXX)
Value format: struct.pack('>I', N) + boxes_float32 (N×4×2) + bgcolors_uint8 (N×3)
Checkpoint key: __checkpoint__ → 8-byte big-endian last-written index (for resume)

Decode example:
  n      = struct.unpack('>I', data[:4])[0]
  boxes  = np.frombuffer(data[4:4+n*32], '<f4').reshape(n, 4, 2)
  colors = np.frombuffer(data[4+n*32:],  'u1').reshape(n, 3)

Datasets processed:
  Union14M-L-LMDB-Filtered (5 sub-LMDBs, ~3.2M)
  Union14M-U               (3 sub-LMDBs, ~10.7M)
  UnionST                  (3 sub-LMDBs, ~15M)
  RDC-U                    (flat LMDB,   ~10M)

Usage:
  python tools/precompute_occ_aug.py [--batch-size 256] [--num-gpus 8] [--no-rdcu]
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
# Module-level paths  (re-evaluated in each spawned worker — safe)
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
OCCSTR_ROOT = _HERE.parent
CRAFT_ROOT  = OCCSTR_ROOT / 'third_party' / 'craft'
CRAFT_CKPT  = OCCSTR_ROOT / 'weights' / 'pretrained' / 'craft' / 'craft_ic15_20k.pth'
DATA_ROOT   = Path('/data/isaackang/data/STR/openocr')
OUTPUT_ROOT = DATA_ROOT / 'occ_aug'

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
CROP_H, CROP_W    = 32, 128        # canonical output coordinate space
CRAFT_H, CRAFT_W  = 64, 256        # 2× upscale — score map output == CROP size
SCORE_H, SCORE_W  = 32, 128        # CRAFT output = CRAFT_input / 2 = CROP size

_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_STD  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

TEXT_THRESH      = 0.7
LOW_TEXT         = 0.4
BG_SCORE_THRESH  = 0.3    # CRAFT score below this → background pixel
MIN_BOX_AREA     = 4      # skip blobs < this area (in score_map pixels)

MAP_SIZE        = 8 * (1 << 30)   # 8 GB virtual LMDB address space
CHECKPOINT_KEY  = b'__checkpoint__'
EMPTY_BYTES     = struct.pack('>I', 0)

# ---------------------------------------------------------------------------
# Dataset definitions
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
    """cv2-first image decode (releases GIL — good for parallel use)."""
    import numpy as np
    import cv2
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    # PIL fallback
    try:
        import io
        from PIL import Image as PIL_Image
        return np.array(PIL_Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Fast character box detection using cv2 (replaces skimage-based getDetBoxes)
# ---------------------------------------------------------------------------
def fast_char_boxes(smap: 'np.ndarray', text_thresh: float, low_text: float,
                    min_area: int = MIN_BOX_AREA) -> List['np.ndarray']:
    """Return axis-aligned char boxes (N×(4,2) float32) from CRAFT score map.

    Boxes are in score_map coordinate space (SCORE_H × SCORE_W = 32×128 = CROP size).
    Uses cv2.connectedComponentsWithStats — much faster than skimage.measure.
    """
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
        # Check max score in this component
        comp_smap = smap[labels == i]
        if comp_smap.max() < text_thresh:
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

    # Sort left-to-right
    boxes.sort(key=lambda b: float(b[:, 0].mean()))
    return boxes


# ---------------------------------------------------------------------------
# Process-level thread pool for parallel preprocessing (cv2 releases GIL)
# ---------------------------------------------------------------------------
_PREP_POOL: Optional[object] = None
_PREP_NTHREADS = 4   # chunk-based: 4 tasks total regardless of batch size

def _get_prep_pool():
    global _PREP_POOL
    if _PREP_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _PREP_POOL = ThreadPoolExecutor(max_workers=_PREP_NTHREADS)
    return _PREP_POOL


def _prep_chunk(args):
    """Preprocess a chunk of images; returns (crops_list, normalized_list)."""
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
    """Post-process a chunk of (score_map, img_crop) pairs."""
    chunk, text_thresh, low_text, bg_thresh, crop_h, crop_w, score_h = args
    import cv2, numpy as np
    results = []
    for smap, img_crop in chunk:
        # smap is already CROP_H×CROP_W (32×128) when CRAFT input is 64×256
        bg_pix  = img_crop[smap < bg_thresh]
        bg_color = (bg_pix.mean(axis=0).astype(np.uint8)
                    if bg_pix.size > 0
                    else img_crop.mean(axis=(0, 1)).astype(np.uint8))
        boxes_list = fast_char_boxes(smap, text_thresh, low_text)
        if not boxes_list:
            results.append((np.zeros((0, 4, 2), np.float32),
                            np.zeros((0, 3), np.uint8)))
            continue
        boxes_np   = np.array(boxes_list)   # already in CROP coords (scale=1)
        boxes_np[..., 0] = np.clip(boxes_np[..., 0], 0, crop_w - 1)
        boxes_np[..., 1] = np.clip(boxes_np[..., 1], 0, crop_h - 1)
        N          = len(boxes_np)
        colors_arr = np.tile(bg_color, (N, 1)).astype(np.uint8)
        results.append((boxes_np, colors_arr))
    return results


# ---------------------------------------------------------------------------
# CRAFT batch inference (fp16 autocast + chunk-parallel pre/post-processing)
# ---------------------------------------------------------------------------
def run_batch(net, device, images_rgb: List, use_amp: bool
              ) -> List[Tuple['np.ndarray', 'np.ndarray']]:
    """
    CRAFT batch inference with fp16 autocast.
    Preprocessing and postprocessing are parallelised across 4 threads
    using chunks so that thread-submission overhead is minimal.

    Returns per-image (boxes, bg_colors):
      boxes:     (N, 4, 2) float32  in CROP_H×CROP_W coordinates
      bg_colors: (N, 3)    uint8    replicated per-image background RGB
    """
    import numpy as np
    import torch

    B    = len(images_rgb)
    mean = np.array(_MEAN, dtype=np.float32)
    std  = np.array(_STD,  dtype=np.float32)
    pool = _get_prep_pool()

    # --- Parallel preprocessing (chunk-based: 4 chunks, low task overhead) ---
    chunk_size = max(1, (B + _PREP_NTHREADS - 1) // _PREP_NTHREADS)
    prep_args  = [(images_rgb[i:i+chunk_size], mean, std)
                  for i in range(0, B, chunk_size)]
    chunk_res  = list(pool.map(_prep_chunk, prep_args))

    imgs_crop   = [c for crops, _ in chunk_res for c in crops]
    craft_batch = np.stack([n for _, normed in chunk_res for n in normed])  # (B,H,W,3)

    x = torch.from_numpy(craft_batch.transpose(0, 3, 1, 2)).to(device)

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda'):
                y, _ = net(x)   # (B, SCORE_H, SCORE_W, 2)
        else:
            y, _ = net(x)

    score_maps = y[..., 0].clamp(0, 1).float().cpu().numpy()  # (B, 32, 128)

    # --- Parallel postprocessing (chunk-based) ---
    post_pairs   = [(score_maps[i], imgs_crop[i]) for i in range(B)]
    post_chunks  = [post_pairs[i:i+chunk_size] for i in range(0, B, chunk_size)]
    post_args    = [(c, TEXT_THRESH, LOW_TEXT, BG_SCORE_THRESH,
                     CROP_H, CROP_W, SCORE_H) for c in post_chunks]
    post_res     = list(pool.map(_post_chunk, post_args))

    results = [r for chunk in post_res for r in chunk]
    return results


def _run_single(net, device, img_rgb, use_amp):
    return run_batch(net, device, [img_rgb], use_amp)[0]


# ---------------------------------------------------------------------------
# Single LMDB processing
# ---------------------------------------------------------------------------
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
    n_errors = 0
    report_every = 50 * batch_size  # log every ~50 batches

    pool = _get_prep_pool()

    def _load_batch(indices):
        """Read + decode a batch from LMDB; returns (imgs, valid_idxs, empty_idxs)."""
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

    # Pre-fetch first batch while setup completes
    batch_ranges = list(range(start_idx, n_total + 1, batch_size))
    first_end    = min(batch_ranges[0] + batch_size, n_total + 1)
    prefetch_fut = pool.submit(_load_batch, range(batch_ranges[0], first_end))

    for b_num, batch_start in enumerate(batch_ranges):
        last_idx = min(batch_start + batch_size, n_total + 1) - 1

        # Get current batch (already pre-fetched)
        buf_imgs, buf_idxs, empty_idxs = prefetch_fut.result()

        # Submit next batch fetch while GPU processes current batch
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


# ---------------------------------------------------------------------------
# Worker (one per GPU)
# ---------------------------------------------------------------------------
def worker(gpu_id: int, tasks: list, batch_size: int,
           craft_root_str: str, craft_ckpt_str: str, log_dir_str: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sys.path.insert(0, craft_root_str)

    import torch
    from collections import OrderedDict
    from craft import CRAFT

    device  = 'cuda:0'
    log_dir = Path(log_dir_str)
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
        _log(log_path, f'[GPU {gpu_id}] starting {in_path.name}')
        process_lmdb(net, device, use_amp, in_path, out_path, batch_size, log_path)

    del net
    torch.cuda.empty_cache()
    _log(log_path, f'[GPU {gpu_id}] all tasks done')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_tasks(no_rdcu: bool) -> List[Tuple[Path, Path, int]]:
    tasks: List[Tuple[Path, Path, int]] = []
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


def main():
    ap = argparse.ArgumentParser(
        description='Precompute CRAFT boxes + BG colors for OccSTR augmentation')
    ap.add_argument('--batch-size', type=int, default=256,
                    help='Images per CRAFT batch per GPU (default 256)')
    ap.add_argument('--num-gpus',   type=int, default=8)
    ap.add_argument('--no-rdcu',    action='store_true', help='Skip RDC-U')
    args = ap.parse_args()

    log_dir = OUTPUT_ROOT / 'logs'
    tasks   = collect_tasks(args.no_rdcu)
    if not tasks:
        print('No tasks found — check DATA_ROOT', flush=True)
        return

    total = sum(t[2] for t in tasks)
    print(f'\nTotal: {len(tasks)} LMDBs, {total:,} images', flush=True)
    for in_p, out_p, n in sorted(tasks, key=lambda x: -x[2]):
        cp_exists = (out_p / 'data.mdb').exists()
        print(f'  {in_p.parent.name}/{in_p.name}: {n:,}'
              + (' (partial output exists)' if cp_exists else ''), flush=True)

    n_gpus      = min(args.num_gpus, len(tasks))
    assignments = greedy_assign(tasks, n_gpus)

    print(f'\nTask assignment ({n_gpus} GPUs):', flush=True)
    for g, ts in enumerate(assignments):
        if ts:
            w = sum(t[2] for t in ts)
            print(f'  GPU {g}: {w:,} → {[t[0].name for t in ts]}', flush=True)

    processes = []
    t_start   = time.time()
    for g in range(n_gpus):
        if not assignments[g]:
            continue
        p = mp.Process(
            target=worker,
            args=(g, assignments[g], args.batch_size,
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
