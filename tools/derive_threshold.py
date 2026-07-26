#!/usr/bin/env python3
"""Derive a same/different-person threshold for facecomp's recognition backends.

Runs the same pipeline facecomp uses (YuNet detect -> 5-point similarity-transform
align -> embed -> cosine similarity) over a labelled pair set, then picks the
threshold by 10-fold cross-validation: for each fold the threshold is chosen on
the other nine and scored on the held-out one. That both derives the value and
reports how stable it is across independent splits, so a single lucky split
can't masquerade as a result.

Usage
-----
  # LFW, standard protocol (pairs.txt + lfw/<Name>/<Name>_0001.jpg layout)
  python3 derive_threshold.py \
      --images-dir /path/to/lfw \
      --pairs      /path/to/pairs.txt \
      --detector   face_detection_yunet_2023mar.onnx \
      --recognizer arcfaceresnet100-11-int8.onnx \
      --backend    arcface

  # Any second dataset, as a simple TSV: <imgA>\t<imgB>\t<1|0>
  python3 derive_threshold.py \
      --images-dir / --pairs mypairs.tsv --pairs-format tsv \
      --detector ... --recognizer ... --backend arcface

Requires: opencv-python (>=4.10) and numpy.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ArcFace's canonical destination landmarks for a 112x112 aligned crop, from
# onnx/models face_preprocess.py (base points, +8.0 on x for a 112-wide crop).
# The row order here matches YuNet's own landmark order; swapping the eyes and
# mouth corners does NOT error, it silently destroys accuracy, so don't reorder.
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def parse_lfw_pairs(path):
    """Parse LFW's standard pairs.txt. Returns (pairs, fold_ids).

    pairs is a list of (nameA, idxA, nameB, idxB, same); fold_ids the 0-based
    fold each pair belongs to, preserving LFW's official 10-fold split.
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    header = lines[0].split()
    n_folds, per_class = int(header[0]), int(header[1])
    pairs, folds, cursor = [], [], 1
    for fold in range(n_folds):
        for _ in range(per_class):                      # same-person block
            name, a, b = lines[cursor].split("\t"); cursor += 1
            pairs.append((name, int(a), name, int(b), 1)); folds.append(fold)
        for _ in range(per_class):                      # different-person block
            n1, a, n2, b = lines[cursor].split("\t"); cursor += 1
            pairs.append((n1, int(a), n2, int(b), 0)); folds.append(fold)
    return pairs, np.array(folds)


def parse_tsv_pairs(path):
    """Parse a plain <pathA>\t<pathB>\t<1|0> pair list, or a deepface-style
    <fileA>,<fileB>,<Yes|No> CSV. Folds are assigned stratified (see below)."""
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t") if "\t" in ln else ln.split(",")
            if len(parts) < 3:
                continue
            a, b, lab = parts[0].strip(), parts[1].strip(), parts[2].strip().lower()
            if lab in ("decision", "same", "label"):     # header row
                continue
            same = 1 if lab in ("1", "yes", "true") else 0
            rows.append((a, None, b, None, same))
    return rows, stratified_folds(np.array([r[4] for r in rows]))


def stratified_folds(labels, n_folds=10):
    """Assign folds so each holds a proportional share of same/different pairs.

    With only a few dozen positive pairs, naive sequential folds can end up with
    zero same-person pairs, making a fold's score meaningless. Round-robin
    within each class keeps every fold representative.
    """
    folds = np.zeros(len(labels), dtype=int)
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def lfw_path(root, name, idx):
    return os.path.join(root, name, "%s_%04d.jpg" % (name, idx))


class Embedder:
    def __init__(self, detector, recognizer, backend, det_score=0.9):
        self.backend = backend
        self.detector = cv2.FaceDetectorYN.create(detector, "", (320, 320), det_score, 0.3, 5000)
        if backend == "sface":
            self.sface = cv2.FaceRecognizerSF.create(recognizer, "")
        else:
            self.net = cv2.dnn.readNetFromONNX(recognizer)
        self.cache = {}

    def _detect_largest(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        # Largest bounding box, matching facecomp's own "most prominent face".
        return faces[np.argmax(faces[:, 2] * faces[:, 3])]

    def embed(self, path):
        if path in self.cache:
            return self.cache[path]
        emb = None
        img = cv2.imread(path)
        if img is not None:
            face = self._detect_largest(img)
            if face is not None:
                if self.backend == "sface":
                    aligned = self.sface.alignCrop(img, face)
                    emb = self.sface.feature(aligned).flatten()
                    emb = emb / np.linalg.norm(emb)
                else:
                    src = face[4:14].reshape(5, 2).astype(np.float32)
                    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST,
                                                       method=cv2.RANSAC,
                                                       ransacReprojThreshold=1000.0)
                    aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
                    blob = cv2.dnn.blobFromImage(aligned, 1.0, (112, 112),
                                                 (0, 0, 0), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    emb = self.net.forward().flatten()
                    emb = emb / np.linalg.norm(emb)
        self.cache[path] = emb
        return emb


def score_at(sims, labels, t, metric="balanced"):
    """Score a threshold. 'balanced' = mean of TPR and TNR.

    Plain accuracy is misleading on imbalanced pair sets: if only 13% of pairs
    are same-person, "always different" already scores 87%. Balanced accuracy
    weights both classes equally, so it can't be gamed that way.
    """
    pred = (sims >= t).astype(int)
    pos, neg = labels == 1, labels == 0
    if metric == "accuracy" or not pos.any() or not neg.any():
        return float((pred == labels).mean())
    tpr = float((pred[pos] == 1).mean())
    tnr = float((pred[neg] == 0).mean())
    return 0.5 * (tpr + tnr)


def best_threshold(sims, labels, metric="balanced"):
    """Threshold maximising `metric`, scanned over candidate cut points."""
    s = np.unique(sims)
    cands = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2.0, [s[-1] + 1e-6]]) \
        if len(s) > 1 else np.array([s[0] - 1e-6, s[0] + 1e-6])
    best_sc, best_t = -1.0, 0.0
    for t in cands:
        sc = score_at(sims, labels, t, metric)
        if sc > best_sc:
            best_sc, best_t = sc, float(t)
    return best_t, best_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pairs-format", choices=["lfw", "tsv"], default="lfw")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--backend", choices=["arcface", "sface"], default="arcface")
    ap.add_argument("--metric", choices=["balanced", "accuracy"], default="balanced")
    ap.add_argument("--det-score", type=float, default=0.9,
                    help="YuNet detection confidence. Use 0.5 when deriving, so hard images "
                         "are scored rather than silently skipped - a pair set that quietly "
                         "drops its difficult half derives a threshold for the easy half. "
                         "(facecomp's own runtime default is 0.7; this stays at OpenCV's 0.9 "
                         "so the value is always a deliberate choice.)")
    args = ap.parse_args()

    parse = parse_lfw_pairs if args.pairs_format == "lfw" else parse_tsv_pairs
    pairs, folds = parse(args.pairs)
    print("loaded %d pairs across %d folds" % (len(pairs), len(set(folds.tolist()))))

    emb = Embedder(args.detector, args.recognizer, args.backend, args.det_score)
    sims, labels, keep, skipped = [], [], [], 0
    for i, (na, ia, nb, ib, same) in enumerate(pairs):
        pa = lfw_path(args.images_dir, na, ia) if ia is not None else os.path.join(args.images_dir, na)
        pb = lfw_path(args.images_dir, nb, ib) if ib is not None else os.path.join(args.images_dir, nb)
        ea, eb = emb.embed(pa), emb.embed(pb)
        if ea is None or eb is None:
            skipped += 1
            continue
        sims.append(float(np.dot(ea, eb)))
        labels.append(same)
        keep.append(i)
        if (i + 1) % 500 == 0:
            print("  %d/%d pairs..." % (i + 1, len(pairs)), flush=True)

    sims = np.array(sims)
    labels = np.array(labels)
    folds = folds[np.array(keep, dtype=int)]
    print("\nusable pairs: %d  (skipped %d - no face detected)" % (len(sims), skipped))
    print("same-person    mean sim: %.4f  (n=%d)" % (sims[labels == 1].mean(), (labels == 1).sum()))
    print("different      mean sim: %.4f  (n=%d)" % (sims[labels == 0].mean(), (labels == 0).sum()))

    # 10-fold cross-validation: choose the threshold on nine folds, score on the tenth.
    ts, accs = [], []
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        if not labels[te].any() or labels[te].all():
            continue                      # fold lacks one class; score undefined
        t, _ = best_threshold(sims[tr], labels[tr], args.metric)
        a = score_at(sims[te], labels[te], t, args.metric)
        ts.append(t); accs.append(a)
        print("  fold %2d: threshold %.4f -> held-out %s %.4f" % (f, t, args.metric, a))

    ts, accs = np.array(ts), np.array(accs)
    print("\n=== RESULT (%s, metric=%s) ===" % (args.backend, args.metric))
    print("threshold : %.4f  +/- %.4f   (min %.4f, max %.4f)" % (ts.mean(), ts.std(), ts.min(), ts.max()))
    print("%-10s: %.4f  +/- %.4f" % (args.metric, accs.mean(), accs.std()))
    print("\nSuggested DEFAULT_THRESHOLD: %.3f" % ts.mean())
    if ts.std() > 0.05:
        print("WARNING: threshold varies a lot across folds - treat it as unstable.")


if __name__ == "__main__":
    sys.exit(main())
