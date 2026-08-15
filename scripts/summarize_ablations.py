#!/usr/bin/env python
"""
summarize_ablations.py — turn the ablation CSVs into the three verdict tables.

Prints, per cell:
  G1  Int4 vs bf16 surprisal: mean U, solo AUROC, and the paired per-step
      correlation. The question is whether AUROC crosses 0.5, not whether the
      means differ — a uniform shift in U changes nothing about ranking.
  G2  bge-m3 vs Qwen3-Embedding: solo AUROC of each coherence signal.
  G3/G4  raw vs verbalized vs calibrated D_o^A.

AUROC is reported with a 1000-sample bootstrap CI over TRAJECTORIES (not steps),
because steps within a trajectory are not independent and a step-level CI would
be far too narrow.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def traj_agg(df, col, how="mean"):
    """Step-level signal -> one value per trajectory, with its label."""
    d = df[df[col].notna() & df["label_fail"].notna()]
    if d.empty:
        return None
    g = d.groupby("traj_id").agg(score=(col, how), label=("label_fail", "first")).reset_index()
    return g if g["label"].nunique() > 1 else None


def auroc_ci(g, n_boot=1000, seed=0):
    y, s = g["label"].values, g["score"].values
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            boots.append(roc_auc_score(y[idx], s[idx]))
    if not boots:
        return point, np.nan, np.nan
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def line(name, g, n_pos_note=True):
    if g is None:
        return f"  {name:<28} n/a (no variance in label, or signal all-NaN)"
    a, lo, hi = auroc_ci(g)
    flag = "INVERTED" if hi < 0.5 else ("above chance" if lo > 0.5 else "spans 0.5")
    extra = f"  [n={len(g)}, fails={int(g.label.sum())}]" if n_pos_note else ""
    return f"  {name:<28} AUROC={a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {flag}{extra}"


def read(pattern):
    fs = glob.glob(pattern)
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    args = ap.parse_args()
    D = args.dir

    for cell in args.cells:
        print("\n" + "=" * 78)
        print(f"CELL: {cell}")
        print("=" * 78)

        # ---------------- G1 ----------------
        i4 = read(os.path.join(D, f"surprisal_{cell}_int4.csv"))
        bf = read(os.path.join(D, f"surprisal_{cell}_bf16.csv"))
        if i4 is not None or bf is not None:
            print("\n[G1] QUANTIZATION — content-aware surprisal, identical code path")
            for tag, df in (("int4", i4), ("bf16", bf)):
                if df is None:
                    print(f"  {tag}: MISSING (job did not complete)")
                    continue
                print(f"  --- {tag}: mean U_content={df.U_content.mean():.4f} "
                      f"(U_all={df.U_all.mean():.4f}, "
                      f"content tokens/step={df.n_content_tokens.mean():.1f})")
                for col in ("U_all", "U_stopfilter", "U_content"):
                    print(line(f"{tag} {col} (traj mean)", traj_agg(df, col, "mean")))
                    print(line(f"{tag} {col} (traj max)", traj_agg(df, col, "max"), False))
            if i4 is not None and bf is not None:
                m = i4.merge(bf, on=["traj_id", "step_idx"], suffixes=("_i4", "_bf"))
                if len(m) > 2:
                    r = m["U_content_i4"].corr(m["U_content_bf"], method="spearman")
                    print(f"\n  paired step-level Spearman(int4, bf16) U_content = {r:.3f}  "
                          f"[n={len(m)} steps]")
                    print("  VERDICT: inversion is behavioural if bf16 AUROC also sits below 0.5;")
                    print("           it is a quantization artifact only if bf16 crosses above it.")
        else:
            print("\n[G1] no surprisal CSVs found — check Phase 2/3 in the slurm log.")

        # ---------------- G2 ----------------
        base = read(os.path.join(D, f"embed_{cell}_bgem3_baseline.csv"))
        alt = read(os.path.join(D, f"embed_{cell}_qwen3emb_baseline.csv"))
        if base is not None:
            print("\n[G2] EMBEDDER — same formulas, different geometry")
            for tag, df in (("bge-m3", base), ("qwen3-emb", alt)):
                if df is None:
                    print(f"  {tag}: MISSING")
                    continue
                for col in ("d_a_sem", "d_a_lex", "d_a_gated", "d_oA", "d_oU"):
                    print(line(f"{tag} {col}", traj_agg(df, col, "mean"), False))
                print()
            print("  VERDICT: findings that flip sign between embedders are geometry,")
            print("           not behaviour, and cannot be reported as TRACER properties.")

            # telecom-specific check
            if base["user_is_tool_turn"].sum() > 0:
                msg_only = base[base.user_is_tool_turn == 0]
                print(f"\n  [D_o^U split] {int(base.user_is_tool_turn.sum())} user turns carry "
                      f"tool calls (definition of D_o^U does not hold for these)")
                print(line("    d_oU all user turns", traj_agg(base, "d_oU", "mean"), False))
                print(line("    d_oU message-turns only", traj_agg(msg_only, "d_oU", "mean"), False))

        # ---------------- G3 / G4 ----------------
        verb = read(os.path.join(D, f"embed_{cell}_bgem3_verbalized.csv"))
        cal = base   # G4 rides along in the baseline file (same embeddings)
        print("\n[G3/G4] AGENT-COHERENCE GAP — is D_o^A semantics or format?")
        for tag, df, col in (("raw", base, "d_oA"),
                             ("verbalized obs", verb, "d_oA"),
                             ("calibrated", cal, "d_oA_calibrated")):
            if df is None or col not in df:
                print(f"  {tag}: MISSING")
                continue
            v = df[col].dropna()
            if len(v):
                print(f"  {tag:<16} mean={v.mean():.4f} std={v.std():.4f} "
                      f"range=[{v.quantile(.05):.4f}, {v.quantile(.95):.4f}]")
            print(line(f"  {tag} AUROC", traj_agg(df, col, "mean"), False))
        print("\n  VERDICT: if raw D_o^A is pinned near its ceiling with tiny std while the")
        print("           calibrated version has real spread, the raw signal was measuring")
        print("           NL-vs-JSON distance, and that is the report's finding.")

    print("\n" + "=" * 78)
    print("Reminder: these are confound checks, not the headline result. They say")
    print("whether each signal's behaviour is real; they do not say why it is real.")
    print("The loop-hypothesis test (U_t vs D_a correlation) is a separate CPU job.")
    print("=" * 78)


if __name__ == "__main__":
    main()
