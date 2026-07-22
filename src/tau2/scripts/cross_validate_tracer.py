#!/usr/bin/env python3
"""
Cross-validated TRACER evaluation.

WHY THIS EXISTS
---------------
diagnose_and_optimize_tracer.py grid-searches alpha/beta/gamma/k/w and reports the
best AUROC measured on the SAME trajectories it searched over. With ~35-50
trajectories per cell and ~20k configs, that best-of-N is fit to the noise of those
specific trajectories -> inflated, non-reproducible ("tuned-on-test").

This wrapper fixes the PROTOCOL, not the scoring. For each cross-validation fold it:
  1. tunes the parameters on the TRAINING folds only,
  2. freezes them, and
  3. scores the HELD-OUT fold once.
Held-out scores are pooled across folds and AUROC is computed on them. That number
is honest: it measures the method, not the size of the search.

It also reports an untuned REPETITION-ONLY baseline (Da alone, no parameters), because
the real question is: does the tuned multi-signal composite beat the single best
signal OUT OF SAMPLE?

Reuses the paper's own extraction + scoring from diagnose_and_optimize_tracer.py so the
per-trajectory scores are identical to the original method.

USAGE
-----
  python cross_validate_tracer.py FILE.json [FILE2.json ...]
  python cross_validate_tracer.py data/simulations/qwen*_50runs.json --folds 5 --variant additive
"""

import argparse, json, sys, itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --- reuse the paper's own code so scoring is identical to the original method ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from diagnose_and_optimize_tracer import (
        _sim_ground_truth_pass, _sim_uncertainty_scores,
        calculate_variant_score, TRACERConfig,
    )
    _HAVE_REPO = True
except Exception as e:
    _HAVE_REPO = False
    _IMPORT_ERR = e


# ----------------------------------------------------------------------
# Fallbacks (only used if the repo import fails, e.g. running standalone).
# Kept byte-compatible with the repo's logic.
# ----------------------------------------------------------------------
if not _HAVE_REPO:
    from dataclasses import dataclass
    @dataclass
    class TRACERConfig:
        alpha: float = 1.0; beta: float = 1.0; gamma: float = 1.0
        top_k_percentile: float = 0.2; ensemble_weight_max: float = 0.0

    def _sim_ground_truth_pass(sim):
        if sim.get('ground_truth_pass') is not None:
            return bool(sim['ground_truth_pass'])
        ri = sim.get('reward_info') or {}
        r = ri.get('reward')
        return None if r is None else r >= 0.5

    def _sim_uncertainty_scores(sim):
        if sim.get('uncertainty_scores') is not None:
            return sim['uncertainty_scores']
        out=[]
        for m in sim.get('messages', []):
            role=m.get('role')
            if role=='assistant': actor='agent'
            elif role=='user': actor='user'
            else: continue
            unc=m.get('uncertainty') or {}
            out.append({'actor':actor,
                        'normentropy_filtered_score':unc.get('normalized_entropy',0.0),
                        'da_score':m.get('da_score'),
                        'do_score':m.get('do_score'),
                        'do_type':m.get('do_type')})
        return out

    def calculate_variant_score(step_data, variant, config):
        if not step_data: return 0.0
        risks=[]
        for s in step_data:
            ui=s.get('ui',0.0) or 0.0
            da=s.get('da',0.0) or 0.0
            doa=s.get('do_agent',0.0) or 0.0
            dou=s.get('do_user',0.0) or 0.0
            pen=config.alpha*da+config.beta*doa+config.gamma*dou
            if variant=='additive': risks.append(ui+pen)
            elif variant=='multiplicative': risks.append(ui*(1.0+pen))
            elif variant=='max': risks.append(max(ui,config.alpha*da,config.beta*doa,config.gamma*dou))
            else: risks.append(ui+pen)
        N=len(risks)
        if config.top_k_percentile>=1.0: topk=risks
        else:
            k=max(1,int(config.top_k_percentile*N)); topk=sorted(risks,reverse=True)[:k]
        mtk=float(np.mean(topk))
        if config.ensemble_weight_max>0.0:
            return (1-config.ensemble_weight_max)*mtk+config.ensemble_weight_max*float(np.max(risks))
        return mtk


# ----------------------------------------------------------------------
def build_step_data(sim):
    """One dict per step with the four raw signals — matches the repo's evaluate path."""
    steps=[]
    for sc in _sim_uncertainty_scores(sim):
        steps.append({
            'ui': sc.get('normentropy_filtered_score', sc.get('ui_score',0.0)) or 0.0,
            'da': sc['da_score'] if sc.get('da_score') is not None else 0.0,
            'do_agent': sc['do_score'] if sc.get('do_type')=='agent_coherence' and sc.get('do_score') is not None else 0.0,
            'do_user': sc['do_score'] if sc.get('do_type')=='user_coherence' and sc.get('do_score') is not None else 0.0,
        })
    return steps


def load_cell(path):
    """Return (list_of_step_data, labels) where label 1 = failure."""
    with open(path) as f:
        data=json.load(f)
    sims=data.get('results') or data.get('simulations') or []
    X, y = [], []
    for sim in sims:
        gp=_sim_ground_truth_pass(sim)
        if gp is None: continue
        steps=build_step_data(sim)
        if not steps: continue
        X.append(steps)
        y.append(0 if gp else 1)   # failure = positive
    return X, np.array(y)


# --- parameter grid for tuning (kept SMALL on purpose: fewer configs = less overfit,
#     and we only ever tune on the training portion anyway) ---
def param_grid(variant):
    alphas=[0.0,0.5,1.0,2.0]
    betas =[0.0,0.5,1.0,2.0]
    gammas=[0.0,0.5,1.0,2.0]
    ks    =[0.2,0.5,1.0]
    ws    =[0.0,0.2]
    for a,b,g,k,w in itertools.product(alphas,betas,gammas,ks,ws):
        yield TRACERConfig(alpha=a,beta=b,gamma=g,top_k_percentile=k,ensemble_weight_max=w)


def score_set(X, cfg, variant):
    return np.array([calculate_variant_score(steps, variant, cfg) for steps in X])


def safe_auroc(y, s):
    if len(np.unique(y))<2: return np.nan
    try: return roc_auc_score(y, s)
    except Exception: return np.nan


def repetition_only_score(X):
    """Untuned baseline: mean Da per trajectory. No parameters, nothing to overfit."""
    out=[]
    for steps in X:
        das=[st['da'] for st in steps]
        out.append(float(np.mean(das)) if das else 0.0)
    return np.array(out)


def cross_validate(path, variant='additive', folds=5, seed=0):
    X, y = load_cell(path)
    n=len(y); npos=int(y.sum()); nneg=n-npos
    result={'file':Path(path).name,'n':n,'failures':npos,'successes':nneg}

    if nneg<2 or npos<2:
        result['status']=f'SKIP (need >=2 of each class; have {npos} fail / {nneg} succ)'
        return result

    k=min(folds, npos, nneg)   # can't have more folds than minority-class members
    if k<folds:
        result['note']=f'folds reduced {folds}->{k} (minority class = {min(npos,nneg)})'
    skf=StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    oof_tracer=np.full(n, np.nan)   # out-of-fold tuned-TRACER scores
    Xarr=np.array(X, dtype=object)

    for tr, te in skf.split(np.zeros(n), y):
        ytr=y[tr]
        if len(np.unique(ytr))<2:
            # degenerate training fold: fall back to a neutral config
            best=TRACERConfig(alpha=1.0,beta=0.0,gamma=0.0,top_k_percentile=1.0)
        else:
            best,best_auc=None,-1
            for cfg in param_grid(variant):
                s=score_set(Xarr[tr], cfg, variant)
                a=safe_auroc(ytr, s)
                if not np.isnan(a) and a>best_auc:
                    best_auc, best = a, cfg
            if best is None:
                best=TRACERConfig(alpha=1.0,beta=0.0,gamma=0.0,top_k_percentile=1.0)
        # score held-out fold with FROZEN params
        oof_tracer[te]=score_set(Xarr[te], best, variant)

    # pooled out-of-fold AUROC — the honest number
    cv_tracer = safe_auroc(y, oof_tracer)

    # untuned repetition-only baseline, same pooled evaluation (no folds needed: no params)
    rep = repetition_only_score(X)
    cv_rep = safe_auroc(y, rep)

    # for reference: the in-sample tuned number (what the original script reports) — to show the gap
    best_in,best_in_auc=None,-1
    for cfg in param_grid(variant):
        a=safe_auroc(y, score_set(X, cfg, variant))
        if not np.isnan(a) and a>best_in_auc: best_in_auc,best_in=a,cfg

    result.update({
        'status':'ok',
        'variant':variant,
        'folds_used':k,
        'cv_tracer_auroc':None if np.isnan(cv_tracer) else round(float(cv_tracer),4),
        'cv_repetition_baseline':None if np.isnan(cv_rep) else round(float(cv_rep),4),
        'insample_tuned_auroc':round(float(best_in_auc),4),   # the inflated number, for contrast
    })
    if result['cv_tracer_auroc'] is not None and result['cv_repetition_baseline'] is not None:
        result['tracer_minus_baseline']=round(result['cv_tracer_auroc']-result['cv_repetition_baseline'],4)
    return result


def main():
    ap=argparse.ArgumentParser(description="Cross-validated TRACER vs untuned repetition baseline")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--variant", default="additive", choices=["additive","multiplicative","max"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None, help="also write results as JSON")
    args=ap.parse_args()

    if not _HAVE_REPO:
        print(f"[warn] could not import repo scoring ({_IMPORT_ERR}); using built-in fallback.\n", file=sys.stderr)

    rows=[]
    print(f"{'cell':<34} {'n':>4} {'fail':>4} {'succ':>4}  "
          f"{'CV-TRACER':>9} {'CV-Da(base)':>11} {'Δ':>7}  {'in-sample':>9}")
    print("-"*92)
    for path in args.files:
        r=cross_validate(path, args.variant, args.folds, args.seed)
        rows.append(r)
        name=r['file'].replace('.json','')
        if r.get('status')!='ok':
            print(f"{name:<34} {r['n']:>4} {r.get('failures','?'):>4} {r.get('successes','?'):>4}  {r.get('status')}")
            continue
        cvt=r['cv_tracer_auroc']; base=r['cv_repetition_baseline']
        d=r.get('tracer_minus_baseline'); ins=r['insample_tuned_auroc']
        print(f"{name:<34} {r['n']:>4} {r['failures']:>4} {r['successes']:>4}  "
              f"{cvt:>9} {base:>11} {d:>+7} {ins:>9}")

    print("-"*92)
    print("CV-TRACER   = cross-validated tuned composite (HONEST: tuned on train folds, scored on held-out)")
    print("CV-Da(base) = untuned repetition-only baseline (mean Da per trajectory)")
    print("Δ           = CV-TRACER - baseline  (>0 means combining signals helps out-of-sample)")
    print("in-sample   = best-of-grid on ALL data (the inflated number the original script reports)")

    if args.json_out:
        with open(args.json_out,"w") as f: json.dump(rows,f,indent=2)
        print(f"\nwrote {args.json_out}")

if __name__=="__main__":
    main()
