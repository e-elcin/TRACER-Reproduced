#!/usr/bin/env python3
"""Evaluate TRACER-U unsafe-run prediction without stored-round pseudoreplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPRESENTATIVE_ROUNDS = {
    "single_generation": 1,
    "generation_plus_selector": 7,
    "three_generations_plus_selector": 3,
}

SIGNALS = {
    "selected_generation": "selected_generation_U_content",
    "generation_max": "generation_max_U_content",
    "selector": "selector_U_content",
    "step_mean": "step_mean_U_content",
    "tracer_u_max": "tracer_u_score",
}


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_value(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    positives = target == 1
    n_positive = int(positives.sum())
    n_negative = len(target) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    rank_sum = float(average_ranks(score)[positives].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    n_positive = int(target.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ordered_target = target[order]
    ordered_score = score[order]
    cumulative_true = np.cumsum(ordered_target)
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_score) != 0),
        len(ordered_score) - 1,
    ]
    true_positive = cumulative_true[threshold_ends]
    predicted_positive = threshold_ends + 1
    recall = true_positive / n_positive
    precision = true_positive / predicted_positive
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-runs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    return args


def finite_metric_values(target: np.ndarray, score: np.ndarray) -> tuple:
    mask = np.isfinite(score)
    y = target[mask].astype(int)
    s = score[mask].astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    rho = spearman_value(s, y)
    return (
        binary_auroc(y, s),
        binary_average_precision(y, s),
        float(rho),
        int(len(y)),
        float(y.mean()),
    )


def percentile_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def evaluate_scope(
    frame: pd.DataFrame,
    scope: str,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    frame = frame.reset_index(drop=True)
    prompt_ids = sorted(frame["prompt_id"].unique())
    group_positions = {
        prompt_id: np.flatnonzero(frame["prompt_id"].to_numpy() == prompt_id)
        for prompt_id in prompt_ids
    }
    target = frame["label_unsafe"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    bootstrap_positions = []
    for _ in range(bootstrap):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        bootstrap_positions.append(
            np.concatenate([group_positions[prompt_id] for prompt_id in sampled])
        )

    rows = []
    for signal_name, column in SIGNALS.items():
        score = frame[column].to_numpy(dtype=float)
        point = finite_metric_values(target, score)
        if point is None:
            continue
        auc_values, ap_values, rho_values = [], [], []
        for positions in bootstrap_positions:
            estimate = finite_metric_values(target[positions], score[positions])
            if estimate is None:
                continue
            auc_values.append(estimate[0])
            ap_values.append(estimate[1])
            rho_values.append(estimate[2])
        if len(auc_values) < int(bootstrap * 0.9):
            raise ValueError(
                f"too few valid bootstrap samples for {scope}/{signal_name}: "
                f"{len(auc_values)}"
            )
        auc_low, auc_high = percentile_interval(auc_values)
        ap_low, ap_high = percentile_interval(ap_values)
        rho_low, rho_high = percentile_interval(rho_values)
        rows.append(
            {
                "scope": scope,
                "signal": signal_name,
                "rows": point[3],
                "prompts": len(prompt_ids),
                "unsafe_prevalence": point[4],
                "unsafe_auroc": point[0],
                "unsafe_auroc_ci_low": auc_low,
                "unsafe_auroc_ci_high": auc_high,
                "inverted_auroc": 1.0 - point[0],
                "unsafe_average_precision": point[1],
                "unsafe_ap_ci_low": ap_low,
                "unsafe_ap_ci_high": ap_high,
                "spearman_with_unsafe": point[2],
                "spearman_ci_low": rho_low,
                "spearman_ci_high": rho_high,
                "bootstrap_valid": len(auc_values),
            }
        )
    return rows


def paired_pathway_changes(
    distinct: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    prompt = (
        distinct.groupby(["pathway", "prompt_id"], as_index=False)
        .agg(
            safe_rate=("label_safe", "mean"),
            **{
                signal_name: (column, "mean")
                for signal_name, column in SIGNALS.items()
                if signal_name != "selector"
            },
        )
    )
    baseline = prompt[prompt["pathway"] == "single_generation"]
    rows = []
    rng = np.random.default_rng(seed + 1000)
    for comparison in (
        "generation_plus_selector",
        "three_generations_plus_selector",
    ):
        other = prompt[prompt["pathway"] == comparison]
        paired = baseline.merge(
            other,
            on="prompt_id",
            suffixes=("_baseline", "_comparison"),
            validate="one_to_one",
        )
        if len(paired) != 32:
            raise ValueError(f"expected 32 paired prompts for {comparison}")
        delta_safe = (
            paired["safe_rate_comparison"] - paired["safe_rate_baseline"]
        ).to_numpy()
        sampled_positions = rng.integers(0, len(paired), size=(bootstrap, len(paired)))
        for signal_name in SIGNALS:
            if signal_name == "selector":
                continue
            delta_score = (
                paired[f"{signal_name}_comparison"]
                - paired[f"{signal_name}_baseline"]
            ).to_numpy()
            rho = spearman_value(delta_score, delta_safe)
            boot_safe, boot_score, boot_rho = [], [], []
            for positions in sampled_positions:
                ds = delta_safe[positions]
                du = delta_score[positions]
                boot_safe.append(float(np.mean(ds)))
                boot_score.append(float(np.mean(du)))
                estimate = spearman_value(du, ds)
                if np.isfinite(estimate):
                    boot_rho.append(float(estimate))
            safe_low, safe_high = percentile_interval(boot_safe)
            score_low, score_high = percentile_interval(boot_score)
            rho_low, rho_high = percentile_interval(boot_rho)
            rows.append(
                {
                    "comparison": f"single_generation_to_{comparison}",
                    "signal": signal_name,
                    "prompts": len(paired),
                    "mean_delta_safe_rate": float(np.mean(delta_safe)),
                    "delta_safe_ci_low": safe_low,
                    "delta_safe_ci_high": safe_high,
                    "mean_delta_score": float(np.mean(delta_score)),
                    "delta_score_ci_low": score_low,
                    "delta_score_ci_high": score_high,
                    "spearman_delta_safety_vs_score": rho,
                    "spearman_ci_low": rho_low,
                    "spearman_ci_high": rho_high,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    runs = pd.read_csv(args.labeled_runs)
    required = {
        "round",
        "pathway",
        "prompt_id",
        "run_index",
        "label_safe",
        "label_unsafe",
        *SIGNALS.values(),
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"labeled run table is missing columns: {missing}")
    if len(runs) != 3200 or runs.duplicated(
        ["round", "prompt_id", "run_index"]
    ).any():
        raise ValueError("expected 3,200 unique stored-round runs")

    representative_parts = []
    for pathway, round_number in REPRESENTATIVE_ROUNDS.items():
        part = runs[(runs["pathway"] == pathway) & (runs["round"] == round_number)]
        if len(part) != 320:
            raise ValueError(f"expected 320 rows for {pathway}/round {round_number}")
        representative_parts.append(part)
    distinct = pd.concat(representative_parts, ignore_index=True)
    if len(distinct) != 960:
        raise AssertionError("effective-pathway table must contain 960 runs")

    metric_rows = []
    for pathway in REPRESENTATIVE_ROUNDS:
        metric_rows.extend(
            evaluate_scope(
                distinct[distinct["pathway"] == pathway],
                pathway,
                args.bootstrap,
                args.seed,
            )
        )
    metric_rows.extend(
        evaluate_scope(
            distinct,
            "pooled_three_effective_pathways",
            args.bootstrap,
            args.seed + 1,
        )
    )
    metrics = pd.DataFrame(metric_rows)
    deltas = paired_pathway_changes(distinct, args.bootstrap, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "campaign_run_prediction_metrics.csv"
    deltas_path = args.out_dir / "campaign_pathway_delta_metrics.csv"
    distinct_path = args.out_dir / "campaign_distinct_pathway_runs.csv"
    summary_path = args.out_dir / "campaign_run_prediction_summary.json"
    metrics.to_csv(metrics_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    distinct.to_csv(distinct_path, index=False)
    summary = {
        "stored_round_rows": len(runs),
        "distinct_pathway_rows": len(distinct),
        "representative_rounds": REPRESENTATIVE_ROUNDS,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_unit": "prompt_id",
        "primary_target": "label_unsafe",
        "score_direction": "higher_TRACER_U_predicts_unsafe",
        "metrics_path": str(metrics_path),
        "deltas_path": str(deltas_path),
        "distinct_runs_path": str(distinct_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    display = metrics[
        [
            "scope",
            "signal",
            "rows",
            "unsafe_prevalence",
            "unsafe_auroc",
            "unsafe_auroc_ci_low",
            "unsafe_auroc_ci_high",
            "unsafe_average_precision",
            "unsafe_ap_ci_low",
            "unsafe_ap_ci_high",
        ]
    ]
    print("Run-level unsafe prediction (high U = unsafe):")
    print(display.round(4).to_string(index=False))
    print("\nPaired pathway changes:")
    print(deltas.round(4).to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
