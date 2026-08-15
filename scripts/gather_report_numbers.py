#!/usr/bin/env python
"""gather_report_numbers.py — collect every ablation stat the figures need,
into one printout. Run on the login node, paste the output back."""
import glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

A = "/dl_scratch2/ege/ender/TRACER-Reproduced/ablations"
CELLS = ["qwen32B-int4_retail_50runs", "qwen32B-int4_telecom_50runs", "qwen32B-int4_airline_50runs"]
rng = np.random.default_rng(0)


def auroc(g):
    y, s = g.label.values, g.score.values
    if len(np.unique(y)) < 2: return None
    pt = roc_auc_score(y, s)
    b = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if len(np.unique(y[i])) > 1]
    return pt, np.percentile(b, 2.5), np.percentile(b, 97.5)


def traj(df, col, agg="mean"):
    d = df[df[col].notna() & df.label_fail.notna()]
    g = d.groupby("traj_id").agg(score=(col, agg), label=("label_fail", "first")).reset_index()
    return g if g.label.nunique() > 1 else None


print("### G1 QUANTIZATION ###")
i4 = pd.read_csv(f"{A}/surprisal_qwen14B-int4_retail_50runs_int4.csv")
bf = pd.read_csv(f"{A}/surprisal_qwen14B-int4_retail_50runs_bf16.csv")
m = i4.merge(bf, on=["traj_id", "step_idx"], suffixes=("_i4", "_bf"))
for c in ["U_all", "U_stopfilter", "U_content"]:
    r = spearmanr(m[f"{c}_i4"], m[f"{c}_bf"])[0]
    print(f"  Spearman {c}: {r:.4f}  | mean int4={i4[c].mean():.4f} bf16={bf[c].mean():.4f}  (n={len(m)})")

print("\n### G2/G3/G4 per cell ###")
for cell in CELLS:
    print(f"\n-- {cell} --")
    base = pd.read_csv(f"{A}/embed_{cell}_bgem3_baseline.csv")
    for emb in ["bgem3", "qwen3emb"]:
        f = f"{A}/embed_{cell}_{emb}_baseline.csv"
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        for col in ["d_a_sem", "d_a_lex", "d_oA", "d_oU"]:
            g = traj(d, col)
            if g is not None:
                a = auroc(g); print(f"  {emb:9} {col:8} AUROC={a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]")
    # D_oA moments raw/verbalized/calibrated
    verb = pd.read_csv(f"{A}/embed_{cell}_bgem3_verbalized.csv")
    for tag, df, col in [("raw", base, "d_oA"), ("verbalized", verb, "d_oA"), ("calibrated", base, "d_oA_calibrated")]:
        v = df[col].dropna()
        print(f"  D_oA {tag:11} mean={v.mean():.4f} std={v.std():.4f} p05={v.quantile(.05):.4f} p95={v.quantile(.95):.4f}")
    # telecom D_oU split
    if base.user_is_tool_turn.sum() > 0:
        for tag, sub in [("all", base[base.d_oU.notna()]),
                         ("msg-only", base[(base.user_is_tool_turn == 0) & base.d_oU.notna()])]:
            g = sub.groupby("traj_id").agg(score=("d_oU", "mean"), label=("label_fail", "first")).reset_index()
            g = g[g.label.notna()]
            a = auroc(g.rename(columns={"label": "label"}))
            print(f"  D_oU {tag:8} mean={sub.d_oU.mean():.4f} n={len(sub)} AUROC={a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]")


if __name__ == "__main__":
    pass
