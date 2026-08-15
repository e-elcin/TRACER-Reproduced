#!/usr/bin/env python
"""
loop_hypothesis.py — the headline analysis. CPU-only, reads the grid's OWN
inline signals from the simulation JSON (no recompute, no GPU, no join).

Claim: surprisal and repetition are one phenomenon with opposite sign. A looping
agent conditions on its own repeated turns, so token probability rises, U falls,
and low U ends up predicting failure — the inversion. If true, one mechanism
explains three results: inverted U, working D_a, collapsed max-composite.

Reads per assistant turn:
    U_t  = messages[i]['uncertainty']['normalized_entropy']   (grid's own U)
    D_a  = messages[i]['da_score']                             (grid's own D_a)
Trajectory label = failure (1) / success (0), auto-detected from reward fields.

Two tests per cell:
    T1  step-level Spearman(U, D_a). Prediction: strongly NEGATIVE.
    T2  trajectory-U solo AUROC on ALL steps vs LOW-repetition steps only
        (D_a below per-trajectory median). If the inversion weakens on
        low-repetition steps, repetition WAS driving it.

Runs on any number of cells; pass the simulation JSON paths.
    python loop_hypothesis.py data/simulations/qwen14B-int4_retail_50runs.json ...
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------- loaders
def load_sims(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        for k in ("simulations", "results", "runs", "trajectories"):
            if isinstance(d.get(k), list):
                return d[k]
    return d if isinstance(d, list) else []


def get_label(sim):
    """1 = FAILURE (positive class, matching TRACER). Auto-detect the field."""
    ri = sim.get("reward_info") or {}
    for src in (ri, sim):
        for key in ("reward", "success", "passed", "task_success"):
            if key in src and src[key] is not None:
                v = src[key]
                if isinstance(v, bool):
                    return 0 if v else 1
                try:
                    return 0 if float(v) >= 0.999 else 1
                except (TypeError, ValueError):
                    continue
    return np.nan


def get_U(msg):
    """Grid's own per-turn surprisal proxy: normalized entropy (mean per-token
    surprisal). Fall back to total/token_count, then to 1-mean_probability."""
    u = msg.get("uncertainty")
    if not isinstance(u, dict):
        return np.nan
    if u.get("normalized_entropy") is not None:
        return float(u["normalized_entropy"])
    te, tc = u.get("total_entropy"), u.get("token_count")
    if te is not None and tc:
        return float(te) / float(tc)
    if u.get("mean_probability") is not None:
        return 1.0 - float(u["mean_probability"])
    return np.nan


def get_messages(sim):
    for k in ("messages", "trajectory", "conversation", "history"):
        m = sim.get(k)
        if isinstance(m, list) and m:
            return m
    return []


# ----------------------------------------------------------------- build table
def build(path):
    sims = load_sims(path)
    cell = os.path.basename(path).replace(".json", "")
    rows = []
    for ti, sim in enumerate(sims):
        label = get_label(sim)
        tid = sim.get("id", sim.get("task_id", ti))
        for si, m in enumerate(get_messages(sim)):
            if m.get("role") != "assistant":
                continue
            U = get_U(m)
            da = m.get("da_score")
            rows.append(dict(
                cell=cell, traj_id=tid, step_idx=si,
                U=U, da=(float(da) if da is not None else np.nan),
                label_fail=label,
            ))
    return pd.DataFrame(rows), cell


# ----------------------------------------------------------------- stats
def auroc_ci(g, n_boot=1000, seed=0):
    d = g[g.score.notna() & g.label.notna()]
    if d.label.nunique() < 2:
        return None
    y, s = d.label.values, d.score.values
    pt = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    b = [roc_auc_score(y[i], s[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
         if len(np.unique(y[i])) > 1]
    lo, hi = (np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan))
    return pt, lo, hi


def traj_U(df):
    d = df[df.U.notna() & df.label_fail.notna()]
    g = d.groupby("traj_id").agg(score=("U", "mean"),
                                 label=("label_fail", "first")).reset_index()
    return g


def analyze(df, cell):
    print("\n" + "=" * 70)
    print(f"CELL: {cell}")
    print("=" * 70)
    both = df[df.U.notna() & df.da.notna()]
    print(f"assistant steps with both U and D_a: {len(both)}  "
          f"(of {len(df)} assistant turns)")
    if len(both) < 20:
        print("  too few steps — skipping.")
        return None

    # ---- T1: step-level correlation ----
    rho, p = spearmanr(both.U, both.da)
    verdict = "CONFIRMED (negative)" if rho < -0.15 else \
              ("weak/none" if abs(rho) <= 0.15 else "POSITIVE (unexpected)")
    print(f"\nT1  Spearman(U, D_a) = {rho:+.3f}  (p={p:.1e}, n={len(both)})  -> {verdict}")
    per = []
    for _, g in both.groupby("traj_id"):
        if len(g) >= 4 and g.U.std() > 0 and g.da.std() > 0:
            per.append(spearmanr(g.U, g.da)[0])
    if per:
        per = np.array(per)
        print(f"    within-trajectory median rho = {np.nanmedian(per):+.3f}, "
              f"{(per < 0).mean()*100:.0f}% of {len(per)} trajectories negative")

    # ---- T2: inversion on low-repetition steps ----
    print("\nT2  trajectory-U solo AUROC (positive class = failure)")
    df2 = df.copy()
    df2["da_med"] = df2.groupby("traj_id")["da"].transform("median")
    low = df2[df2.da.notna() & (df2.da <= df2.da_med)]
    out = {}
    for tag, sub in (("all steps", df2), ("low-repetition only", low)):
        g = traj_U(sub)
        r = auroc_ci(g)
        if r is None:
            print(f"    {tag:<22} n/a (no label variance)")
            out[tag] = None
        else:
            a, lo, hi = r
            flag = "INVERTED" if hi < 0.5 else ("above 0.5" if lo > 0.5 else "spans 0.5")
            print(f"    {tag:<22} AUROC={a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {flag}  "
                  f"[n={len(g)} trajectories]")
            out[tag] = (a, lo, hi)

    if out.get("all steps") and out.get("low-repetition only"):
        a_all = out["all steps"][0]
        a_low = out["low-repetition only"][0]
        moved = a_low - a_all
        toward = a_low > a_all  # toward 0.5 from below == inversion weakening
        print(f"\n    Δ(low − all) = {moved:+.3f}  "
              f"-> inversion {'WEAKENS on low-repetition steps (supports loop hypothesis)' if toward else 'persists (mechanism may be elsewhere)'}")
    return dict(cell=cell, rho=rho, n=len(both),
                auroc_all=out.get("all steps"), auroc_low=out.get("low-repetition only"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_files", nargs="+", help="simulation JSON paths")
    ap.add_argument("--csv", default=None, help="optional: dump per-step table here")
    args = ap.parse_args()

    all_rows, results = [], []
    for path in args.sim_files:
        if not os.path.exists(path):
            print(f"skip (not found): {path}", file=sys.stderr)
            continue
        df, cell = build(path)
        all_rows.append(df)
        r = analyze(df, cell)
        if r:
            results.append(r)

    if args.csv and all_rows:
        pd.concat(all_rows).to_csv(args.csv, index=False)
        print(f"\nper-step table -> {args.csv}")

    # ---- cross-cell summary ----
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS CELLS")
        print("=" * 70)
        print(f"{'cell':<34}{'rho(U,Da)':>11}{'AUROC all':>11}{'AUROC low':>11}")
        for r in results:
            aa = f"{r['auroc_all'][0]:.3f}" if r['auroc_all'] else "n/a"
            al = f"{r['auroc_low'][0]:.3f}" if r['auroc_low'] else "n/a"
            print(f"{r['cell']:<34}{r['rho']:>+11.3f}{aa:>11}{al:>11}")
        print("\nInterpretation: consistently negative rho + AUROC rising toward 0.5 on")
        print("low-repetition steps == surprisal and repetition are one phenomenon with")
        print("opposite sign. That single mechanism explains the inverted U, the working")
        print("D_a, and the collapsed max-composite together.")


if __name__ == "__main__":
    main()
