import lmdb, struct, cv2, numpy as np, base64, io
from pathlib import Path
from PIL import Image

DATA_ROOT   = Path('/data/isaackang/data/STR/openocr')
OCC_ROOT    = DATA_ROOT / 'occ_aug'
CROP_H, CROP_W = 32, 128

DATASETS = [
    ('Union14M-L-LMDB-Filtered', ['filter_train_challenging','filter_train_easy','filter_train_hard','filter_train_medium','filter_train_normal']),
    ('Union14M-U', ['book32_lmdb','cc_lmdb','openvino_lmdb']),
    ('UnionST', ['UnionST-P','UnionST-R','UnionST-S']),
    ('RDC-U', None),
]

def decode_img(raw):
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

def decode_aug(data):
    n = struct.unpack('>I', data[:4])[0]
    if n == 0:
        return np.zeros((0,4,2), np.float32), np.zeros((0,3), np.uint8)
    boxes  = np.frombuffer(data[4:4+n*32], '<f4').reshape(n,4,2)
    colors = np.frombuffer(data[4+n*32:], 'u1').reshape(n,3)
    return boxes, colors

def img_to_b64(arr, max_h=128):
    h, w = arr.shape[:2]
    scale = max(1, max_h // h)
    pil = Image.fromarray(arr).resize((w*scale, h*scale), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()

def open_lmdb(path):
    return lmdb.open(str(path), readonly=True, lock=False, max_readers=1,
                     readahead=False, meminit=False)

def pick_samples(env_in, env_aug, n_total, n_pick=4):
    samples = []
    rng = np.random.default_rng(42)
    candidates = rng.choice(n_total, min(500, n_total), replace=False)
    with env_in.begin() as ti, env_aug.begin() as ta:
        for idx in candidates:
            idx = int(idx) + 1
            raw_img = ti.get(f'image-{idx:09d}'.encode())
            raw_aug = ta.get(f'aug-{idx:09d}'.encode())
            if raw_img is None or raw_aug is None:
                continue
            boxes, colors = decode_aug(raw_aug)
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

for ds_name, subdirs in DATASETS:
    sub_list = subdirs if subdirs else [None]
    for sub in sub_list:
        in_path  = DATA_ROOT / ds_name / (sub if sub else '')
        aug_path = OCC_ROOT  / ds_name / (sub if sub else '')
        label    = f'{ds_name}/{sub}' if sub else ds_name

        try:
            env_in  = open_lmdb(in_path)
            env_aug = open_lmdb(aug_path)
            n_total = int(env_in.begin().get(b'num-samples'))
            samples = pick_samples(env_in, env_aug, n_total, n_pick=4)
            env_in.close(); env_aug.close()
        except Exception as e:
            print(f'SKIP {label}: {e}')
            continue

        rows_html.append(f'<tr><td colspan="3" class="section">{label} ({n_total:,} samples)</td></tr>')
        rows_html.append('<tr><th>idx</th><th>원본 이미지 (native res)</th><th>character crops (32×128 기준) → bg colors</th></tr>')

        for idx, img, boxes, colors in samples:
            orig_h, orig_w = img.shape[:2]

            # 1열: 원본 이미지 그대로 (최대 height 128px로만 표시 스케일)
            orig_b64 = img_to_b64(img, max_h=128)

            # character crop용: 32×128 resize
            crop32 = cv2.resize(img, (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)

            crops_html = '<div class="crops">'
            for box, color in zip(boxes, colors):
                x1 = int(np.clip(box[:,0].min(), 0, CROP_W-1))
                y1 = int(np.clip(box[:,1].min(), 0, CROP_H-1))
                x2 = int(np.clip(box[:,0].max(), 0, CROP_W-1))
                y2 = int(np.clip(box[:,1].max(), 0, CROP_H-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                char_patch = crop32[y1:y2+1, x1:x2+1]
                h_p, w_p   = char_patch.shape[:2]
                char_b64   = img_to_b64(char_patch, max_h=48)
                rgb_str    = f'rgb({color[0]},{color[1]},{color[2]})'
                crops_html += f'''
                <div class="pair">
                  <div class="label">{w_p}×{h_p}</div>
                  <img src="data:image/png;base64,{char_b64}" title="char crop">
                  <div class="swatch" style="background:{rgb_str}" title="{rgb_str}"></div>
                </div>'''
            crops_html += '</div>'

            rows_html.append(f'''
            <tr>
              <td class="idx">#{idx}<br><span class="dim">{orig_w}×{orig_h}</span></td>
              <td class="orig"><img src="data:image/png;base64,{orig_b64}"></td>
              <td class="crops-cell">{crops_html}</td>
            </tr>''')

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>OccSTR Aug Preview</title>
<style>
body {{ font-family: monospace; background:#1a1a1a; color:#ddd; padding:16px; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border:1px solid #444; padding:8px; vertical-align:top; }}
th {{ background:#333; font-size:12px; }}
.section {{ background:#2a4a6a; font-weight:bold; font-size:14px; padding:10px; }}
.idx {{ color:#aaa; font-size:11px; width:70px; }}
.dim {{ color:#888; }}
.orig img {{ image-rendering:pixelated; display:block; max-width:400px; }}
.crops {{ display:flex; flex-wrap:wrap; gap:10px; }}
.pair {{ display:flex; flex-direction:column; align-items:center; gap:2px; }}
.pair img {{ image-rendering:pixelated; border:1px solid #555; display:block; }}
.label {{ font-size:9px; color:#888; }}
.swatch {{ width:24px; height:6px; border:1px solid #666; margin-top:1px; }}
</style>
</head><body>
<h2>OccSTR Aug Visualization — 4 samples × 12 LMDBs</h2>
<p style="color:#aaa;font-size:12px">1열: 원본 이미지 (native 해상도) | 3열: 32×128 기준 character crops (위) → 동일 크기 bg 패치 (아래)</p>
<table>
{''.join(rows_html)}
</table>
</body></html>'''

out = Path('/data/isaackang/STR/OccSTR/occ_aug_preview.html')
out.write_text(html)
print(f'Written: {out}  ({out.stat().st_size//1024}KB)')
