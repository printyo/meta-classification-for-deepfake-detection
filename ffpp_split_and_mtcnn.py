# ffpp_split_and_mtcnn.py
# End-to-end FF++ pipeline:
#  - Index folders (original + manipulations)
#  - Confidence-aware MTCNN face extraction (facenet-pytorch)
#  - Identity/source-disjoint splits (GroupShuffleSplit + GroupKFold)
#  - Outputs: faces_ffpp/*.npz, train_ffpp.csv, test_ffpp.csv, train_folds_ffpp.csv, failed_videos.tsv, ffpp_paths.tsv

import os
import re
import csv
import glob
import sys
import argparse
import pathlib
import random
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from facenet_pytorch import MTCNN

# -----------------------
# Defaults (override via CLI)
# -----------------------
DEFAULT_ROOT = r"C:\Users\admin\Documents\te\datasets\ffpp"  # contains: original, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, (optional) csv
DEFAULT_OUT  = r"C:\Users\admin\Documents\te\datasets"       # where CSVs + faces_ffpp will be written

FACES_SUBDIR = "faces_ffpp"

K_FRAMES     = 8
TARGET_SIZE  = 224
MIN_FACE_PX  = 60
MARGIN       = 0.20
MIN_PROB     = 0.80      # relaxed default to keep more data
MIN_OK       = 6         # require at least 6/8 confident frames

TEST_SIZE    = 0.20
N_SPLITS     = 5
SEED         = 42

NUM_WORKERS  = 2
QUEUE_MULT   = 4

EXTS = ("*.mp4","*.avi","*.mov","*.mkv")

# -----------------------
# Helpers
# -----------------------
def list_videos(folder: str) -> List[str]:
    vids = []
    for ext in EXTS:
        vids += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    return vids

def normalize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+','_', s).lower()

def expand_square(box, w, h, margin):
    x1, y1, x2, y2 = box
    bw, bh = (x2-x1), (y2-y1)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(bw, bh) * (1.0 + margin)
    x1 = int(max(0, cx - side/2)); y1 = int(max(0, cy - side/2))
    x2 = int(min(w-1, cx + side/2)); y2 = int(min(h-1, cy + side/2))
    return [x1, y1, x2, y2]

def resize_with_padding(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    pad = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(127,127,127))
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)

def sample_indices(n: int, k: int) -> List[int]:
    if n <= 0: n = k
    idx = np.linspace(0, max(0, n-1), num=k)
    return [int(i) for i in idx]

# -----------------------
# Index FF++ (unique IDs + group by original)
# -----------------------
def index_ffpp(root: str) -> List[Dict]:
    real_dir = os.path.join(root, "original")
    fake_dirs = [d for d in ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures"]
                 if os.path.isdir(os.path.join(root, d))]
    rows = []

    # originals
    orig_map = {}  # normalized stem -> canonical original stem
    for p in list_videos(real_dir):
        stem = pathlib.Path(p).stem
        vid  = f"original__{stem}"  # unique video id
        rows.append({"video_id": vid, "abs_path": p, "label": 0, "group": stem})
        orig_map[normalize(stem)] = stem

    # helper to map fake to original
    def map_to_original(stem: str) -> str:
        ns = normalize(stem)
        for t in ["deepfakes","face2face","faceswap","faceshifter","neuraltextures","df","f2f","fs","nt","fake","manipulated"]:
            ns = ns.replace("_"+t+"_","_").replace("_"+t,"").replace(t+"_","")
        if ns in orig_map:
            return orig_map[ns]
        # fallback: longest common prefix
        best,score=None,0
        for n,o in orig_map.items():
            c = len(os.path.commonprefix([ns, n]))
            if c > score: best, score = o, c
        return best if best is not None else stem

    # fakes
    for folder in fake_dirs:
        froot = os.path.join(root, folder)
        for p in list_videos(froot):
            stem = pathlib.Path(p).stem
            vid  = f"{folder.lower()}__{stem}"  # unique id per file
            orig = map_to_original(stem)
            rows.append({"video_id": vid, "abs_path": p, "label": 1, "group": orig})

    print(f"[LOG] Indexed originals: {sum(1 for r in rows if r['label']==0)}")
    for d in fake_dirs:
        cnt = len(list_videos(os.path.join(root, d)))
        print(f"[LOG] Indexed fakes in {d}: {cnt}")

    if not rows:
        raise RuntimeError("No videos found. Check --root path and folder names.")

    return rows

# -----------------------
# Worker: extract K faces with confidence; accept if >= min_ok frames pass
# -----------------------
def worker(args) -> Tuple[str, bool, str, List[float], int]:
    """
    return: (video_id, accepted, error_msg, probs_per_frame, saved_count)
    accepted=True iff >=min_ok frames pass (prob >= MIN_PROB & size >= MIN_FACE_PX)
    """
    (video_id, vpath, out_dir, k, target, min_px, margin, min_prob, min_ok) = args
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(
            keep_all=True,
            device=device,
            select_largest=True,
            post_process=False
        )

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return (video_id, False, "cannot_open", [], 0)

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_indices(n, k)

        frames: List[Optional[np.ndarray]] = []
        for f in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            frames.append(fr if ok and fr is not None else None)
        cap.release()

        # Build detection batch (only valid frames)
        pil = [Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
               for fr in frames if fr is not None]

        if len(pil) == 0:
            return (video_id, False, "no_frames_read", [0.0]*k, 0)

        # Run detector
        boxes_list, probs_list = mtcnn.detect(pil)

        # --- CRUCIAL NORMALIZATION (fixes your error) ---
        # Make sure we always have Python lists with length == len(pil)
        if boxes_list is None:
            boxes_seq = [None] * len(pil)
        else:
            # facenet-pytorch may return a numpy array of objects; cast to list
            boxes_seq = list(boxes_list)

        if probs_list is None:
            probs_seq = [None] * len(pil)
        else:
            probs_seq = list(probs_list)
        # -------------------------------------------------

        # Map detections back to the original frame indices (0..k-1)
        mapping = [i for i, fr in enumerate(frames) if fr is not None]

        ok_crops: List[Tuple[int, np.ndarray]] = []
        per_frame_probs: List[float] = [0.0] * k  # default 0 for missing/failed frames

        for mp_i, boxes, probs in zip(mapping, boxes_seq, probs_seq):
            fr = frames[mp_i]
            if fr is None or boxes is None or len(boxes) == 0:
                per_frame_probs[mp_i] = 0.0
                continue

            # Choose the largest face and its prob
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            j = int(np.argmax(areas))

            b = boxes[j].astype(float)

            # probs might be None, a scalar, or a 1D array — normalize to float
            if probs is None:
                p = 0.0
            else:
                # Ensure probs is 1D array-like
                try:
                    p = float(probs[j])
                except Exception:
                    # fallback if probs is scalar
                    try:
                        p = float(probs)
                    except Exception:
                        p = 0.0

            h, w = fr.shape[:2]
            x1, y1, x2, y2 = expand_square(b, w, h, margin)

            if (x2 - x1) < min_px or (y2 - y1) < min_px or p < min_prob:
                per_frame_probs[mp_i] = p
                continue

            face = fr[y1:y2, x1:x2]
            crop = resize_with_padding(face, target)
            ok_crops.append((mp_i, crop))
            per_frame_probs[mp_i] = p

        # Acceptance rule: need >= min_ok confident crops
        passed = len(ok_crops) >= min_ok
        if not passed:
            return (video_id, False, "insufficient_confident_frames", per_frame_probs, len(ok_crops))

        # Ensure exactly k crops (sorted by original frame index; pad by repeating last)
        ok_crops.sort(key=lambda t: t[0])
        crops = [c for _, c in ok_crops]
        while len(crops) < k:
            crops.append(crops[-1] if crops else np.zeros((target, target, 3), np.uint8))

        arr = np.stack(crops[:k], axis=0).astype(np.uint8)  # (k,224,224,3)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"{video_id}.npz"), faces=arr)
        return (video_id, True, "", per_frame_probs, len(ok_crops))

    except Exception as e:
        return (video_id, False, f"exception:{e}", [], 0)


# -----------------------
# Write CSVs (only accepted videos) + grouped splits
# -----------------------
def write_splits(rows: List[Dict], accepted_ids: set, out_dir: str):
    rows_ok = [r for r in rows if r["video_id"] in accepted_ids]

    X = [r["video_id"] for r in rows_ok]
    y = [r["label"] for r in rows_ok]
    g = [r["group"] for r in rows_ok]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, g))
    train = [rows_ok[i] for i in tr_idx]
    test  = [rows_ok[i] for i in te_idx]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_of = {}
    Xtr = [r["video_id"] for r in train]
    ytr = [r["label"] for r in train]
    gtr = [r["group"] for r in train]
    for fold, (_, val_idx) in enumerate(gkf.split(Xtr, ytr, gtr)):
        for i in val_idx:
            fold_of[Xtr[i]] = fold

    def wb(items, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video_id","frames_path","label"])
            for r in items:
                w.writerow([r["video_id"], f"{FACES_SUBDIR}/{r['video_id']}.npz", r["label"]])

    def wf(items, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video_id","frames_path","label","fold"])
            for r in items:
                w.writerow([r["video_id"], f"{FACES_SUBDIR}/{r['video_id']}.npz", r["label"], fold_of.get(r["video_id"],0)])

    os.makedirs(out_dir, exist_ok=True)
    wb(train, os.path.join(out_dir, "train_ffpp.csv"))
    wb(test,  os.path.join(out_dir, "test_ffpp.csv"))
    wf(train, os.path.join(out_dir, "train_folds_ffpp.csv"))

    # manifest for convenience
    with open(os.path.join(out_dir, "ffpp_paths.tsv"), "w", encoding="utf-8") as f:
        for r in train + test:
            f.write(f"{r['video_id']}\t{r['abs_path']}\n")

    print(f"[OK] Wrote CSVs to {out_dir}")
    print(f"     train: {len(train)} | test: {len(test)} | total kept: {len(rows_ok)}")

# -----------------------
# Main
# -----------------------
def main():
    global TEST_SIZE, N_SPLITS  # <-- FIX 1: declare globals at the very top of main()

    ap = argparse.ArgumentParser("FF++ split + MTCNN (confidence-aware, grouped)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="FF++ root with original/ and manipulation folders")
    ap.add_argument("--out",  default=DEFAULT_OUT,  help="Output directory for CSVs and faces_ffpp/")
    ap.add_argument("--k", type=int, default=K_FRAMES)
    ap.add_argument("--size", type=int, default=TARGET_SIZE)
    ap.add_argument("--min-face", type=int, default=MIN_FACE_PX)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--min-prob", type=float, default=MIN_PROB)
    ap.add_argument("--min-ok", type=int, default=MIN_OK, help="Minimum #frames (of k) that must pass")
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--queue-mult", type=int, default=QUEUE_MULT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # update globals (Fix 1 continues)
    TEST_SIZE = args.test_size
    N_SPLITS = args.folds

    # 1) Index
    rows = index_ffpp(args.root)
    faces_dir = os.path.join(args.out, FACES_SUBDIR)
    os.makedirs(faces_dir, exist_ok=True)

    # 2) Extract faces in parallel (confidence-aware)
    tasks = [(r["video_id"], r["abs_path"], faces_dir,
              args.k, args.size, args.min_face, args.margin, args.min_prob, args.min_ok)
             for r in rows]

    accepted = set()
    fail_log = []
    queue_cap = max(1, args.workers * args.queue_mult)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        it = iter(tasks); pending=set()
        for _ in range(min(queue_cap, len(tasks))):
            try: pending.add(ex.submit(worker, next(it)))
            except StopIteration: break

        pbar = tqdm(total=len(tasks), desc=f"Extracting faces (k={args.k}, min_prob={args.min_prob}, min_ok={args.min_ok})")
        while pending:
            for fut in as_completed(pending, timeout=None):
                pending.remove(fut)
                vid, ok, err, probs, saved = fut.result()
                if ok:
                    accepted.add(vid)
                else:
                    fail_log.append((vid, err, saved, ",".join([f"{p:.3f}" for p in probs])))
                pbar.update(1)
                try: pending.add(ex.submit(worker, next(it)))
                except StopIteration: pass
        pbar.close()

    # 3) Log failures
    if fail_log:
        with open(os.path.join(args.out, "failed_videos.tsv"), "w", encoding="utf-8") as f:
            f.write("video_id\treason\tsaved_frames\tframe_probs\n")
            for vid, err, saved, probs in fail_log:
                f.write(f"{vid}\t{err}\t{saved}\t{probs}\n")
        print(f"[WARN] Excluded videos: {len(fail_log)} → {os.path.join(args.out, 'failed_videos.tsv')}")

    print(f"[LOG] Indexed total: {len(rows)} | Kept: {len(accepted)} | Excluded: {len(fail_log)}")

    # 4) Write grouped splits using only accepted videos
    write_splits(rows, accepted, args.out)

    print("Done. Faces:", faces_dir)

if __name__ == "__main__":
    # Windows-safe entry for multiprocessing
    main()
