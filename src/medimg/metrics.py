"""Evaluation metrics (numpy only): sensitivity, specificity, AUROC (+bootstrap CI), confusion matrix, Dice, IoU, latency."""
import time
import numpy as np

def confusion(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return dict(tp=int(((y == 1) & (pred == 1)).sum()), fn=int(((y == 1) & (pred == 0)).sum()),
                fp=int(((y == 0) & (pred == 1)).sum()), tn=int(((y == 0) & (pred == 0)).sum()))

def sens_spec(y, pred):
    c = confusion(y, pred)
    return c["tp"] / max(1, c["tp"] + c["fn"]), c["tn"] / max(1, c["tn"] + c["fp"])

def auroc(y, score):
    """Mann-Whitney U with average ranks for ties."""
    y, s = np.asarray(y).astype(int), np.asarray(score, float)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                      # average ties
        m = s == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def bootstrap_ci(fn, y, score, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v = fn(y[i], score[i])
        if not np.isnan(v): vals.append(v)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))

def threshold_at_sensitivity(y, score, target=0.95, target_sens=None):
    """Operating point for triage: highest threshold whose sensitivity >= target."""
    if target_sens is not None:
        target = target_sens
    y, s = np.asarray(y), np.asarray(score)
    for t in np.unique(s)[::-1]:
        if sens_spec(y, s >= t)[0] >= target: return float(t)
    return float(s.min())

def dice(pred, gt, eps=1e-7):
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    return float((2 * (pred & gt).sum() + eps) / (pred.sum() + gt.sum() + eps))

def iou(pred, gt, eps=1e-7):
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    return float(((pred & gt).sum() + eps) / ((pred | gt).sum() + eps))

def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def ap_iou(preds, gts, iou_thresh=0.5):
    """Compute AP for given IoU threshold."""
    recs = [(s, i, b) for i, (bx, sc) in enumerate(preds) for b, s in zip(bx, sc)]; recs.sort(key=lambda t: -t[0])
    used = [np.zeros(len(g), bool) for g in gts]; npos = sum(len(g) for g in gts); tp = []
    for s, i, b in recs:
        ious = [box_iou(b, g) for g in gts[i]]; k = int(np.argmax(ious)) if ious else -1
        ok = k >= 0 and ious[k] >= iou_thresh and not used[i][k]
        if ok: used[i][k] = True
        tp.append(ok)
    tp = np.array(tp, float); ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / max(1, npos); prec = ctp / np.maximum(1, ctp + cfp)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

def ap50(preds, gts):
    """All-point interpolated AP at IoU 0.5."""
    return ap_iou(preds, gts, 0.5)

def ap_50_95(preds, gts):
    """Mean AP across IoU thresholds 0.50 to 0.95 (step 0.05)."""
    thresholds = np.arange(0.50, 1.00, 0.05)
    return float(np.mean([ap_iou(preds, gts, t) for t in thresholds]))

def time_fn(fn, warmup=3, runs=20):
    """Latency in ms: returns (median, p95)."""
    for _ in range(warmup): fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t)), float(np.percentile(t, 95))
