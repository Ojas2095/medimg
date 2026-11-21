"""Dedup + leakage-safe split.
1. exact duplicates (SHA-256) and near duplicates (8x8 average hash, Hamming <= --hamming)
2. group = connected component over {same patient_id} U {duplicate pairs}  (union-find)
3. greedy stratified *group* split into train/val/test; audit proves no group crosses splits.
Writes splits.csv, split_report.json."""
import argparse, hashlib, json, os
import numpy as np, pandas as pd
from PIL import Image

class UF:
    def __init__(s): s.p = {}
    def find(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b): s.p[s.find(a)] = s.find(b)

def thumb(path, n=32):
    return np.asarray(Image.open(path).convert("L").resize((n, n), Image.BILINEAR), np.float32) / 255.0

def ahash(t):
    """16x16 average hash computed from the 32x32 thumbnail (256 bits)."""
    a = t.reshape(16, 2, 16, 2).mean((1, 3))
    return (a > a.mean()).flatten()

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def build_groups(df, root, hamming, mad=0.02):
    uf = UF()
    group_col = "lesion_id" if "lesion_id" in df.columns else "patient_id"
    for _, r in df.iterrows():
        val = r.get(group_col, r.get("patient_id", r.image_id))
        uf.union("img:" + str(r.image_id), f"grp:{val}")
    thumbs = [thumb(os.path.join(root, p)) for p in df.image_path]
    hashes = [ahash(t) for t in thumbs]
    shas = [sha(os.path.join(root, p)) for p in df.image_path]
    ids = df.image_id.tolist()
    n_exact = n_near = 0
    seen = {}
    for i, s in enumerate(shas):
        if s in seen: uf.union("img:" + ids[i], "img:" + ids[seen[s]]); n_exact += 1
        else: seen[s] = i
    H = np.stack(hashes)
    for i in range(len(ids)):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.where(d <= hamming)[0]:       # hash = cheap candidate filter ...
            if np.abs(thumbs[i] - thumbs[i + 1 + j]).mean() < mad:   # ... pixel check = confirmation
                uf.union("img:" + ids[i], "img:" + ids[i + 1 + j]); n_near += 1
    return [uf.find("img:" + i) for i in ids], n_exact, n_near

def greedy_split(df, ratios, seed):
    rng = np.random.default_rng(seed)
    g = df.groupby("group").agg(n=("label", "size"), pos=("label", "sum")).reset_index()
    g = g.sample(frac=1, random_state=seed).sort_values("n", ascending=False, kind="stable")
    names = list(ratios)
    tot_n, tot_p = len(df), max(1, int(df.label.sum()))
    cur = {k: [0, 0] for k in names}
    assign = {}
    for _, r in g.iterrows():
        def deficit(k):
            return (ratios[k] * tot_n - cur[k][0]) / (ratios[k] * tot_n) + \
                   (ratios[k] * tot_p - cur[k][1]) / (ratios[k] * tot_p) * (r.pos > 0)
        k = max(names, key=deficit)
        assign[r.group] = k; cur[k][0] += r.n; cur[k][1] += r.pos
    return df.group.map(assign)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--hamming", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    meta = a.meta or os.path.join(a.root, "metadata.csv")
    df = pd.read_csv(meta)
    df["group"], n_exact, n_near = build_groups(df, a.root, a.hamming)
    r = [float(x) for x in a.ratios.split(",")]
    df["split"] = greedy_split(df, dict(zip(["train", "val", "test"], r)), a.seed)
    biggest = df.group.value_counts().iloc[0] / len(df)
    if biggest > 0.5:
        raise SystemExit(f"Degenerate grouping: largest group holds {biggest:.0%} of data - dedup thresholds are too loose")
    # --- leakage audit (hard failure) ---
    assert df.groupby("group").split.nunique().max() == 1, "LEAK: a group spans several splits"
    assert df.groupby("patient_id").split.nunique().max() == 1, "LEAK: a patient spans several splits"
    rep = {"n": len(df), "exact_dup_links": n_exact, "near_dup_links": n_near, "n_groups": int(df.group.nunique()),
           "per_split": {k: {"n": int(v.shape[0]), "pos": int(v.label.sum()), "pos_rate": round(float(v.label.mean()), 3)}
                         for k, v in df.groupby("split")}, "seed": a.seed, "ratios": r, "leakage_audit": "passed"}
    df.drop(columns=["group"]).to_csv(os.path.join(a.root, "splits.csv"), index=False)
    json.dump(rep, open(os.path.join(a.root, "split_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
