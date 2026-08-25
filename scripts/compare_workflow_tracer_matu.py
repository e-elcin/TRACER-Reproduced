#!/usr/bin/env python3
"""Compare prompt-level TRACER-U and MATU on identical safety outcomes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_workflow_tracer_run_prediction import (
    binary_auroc,
    binary_average_precision,
    percentile_interval,
    spearman_value,
)


REPRESENTATIVE_ROUNDS = {
    "single_generation": 1,
    "generation_plus_selector": 7,
    "three_generations_plus_selector": 3,
}

BASE_SIGNALS = {
    "tracer_selected_generation": "tracer_selected_generation",
    "tracer_generation_max": "tracer_generation_max",
    "tracer_step_mean": "tracer_step_mean",
    "tracer_u_max": "tracer_u_max",
    "matu_final": "matu_final",
    "matu_trajectory": "matu_trajectory",
    "equal_weight_intended": "equal_weight_intended",
}

TARGETS = {
    "majority_unsafe": lambda safe: safe < 0.5,
    "any_unsafe": lambda safe: safe < 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distinct-runs", required=True, type=Path)
    parser.add_argument("--matu-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    return args


def load_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_prompt_table(runs: pd.DataFrame, matu_root: Path) -> pd.DataFrame:
    rows = []
    for pathway, round_number in REPRESENTATIVE_ROUNDS.items():
        part = runs[
            (runs["pathway"] == pathway) & (runs["round"] == round_number)
        ]
        if len(part) != 320:
            raise ValueError(f"expected 320 runs for {pathway}/round {round_number}")
        prompt = (
            part.groupby("prompt_id", as_index=False)
            .agg(
                safe_fraction=("label_safe", "mean"),
                tracer_selected_generation=(
                    "selected_generation_U_content",
                    "mean",
                ),
                tracer_generation_max=("generation_max_U_content", "mean"),
                tracer_step_mean=("step_mean_U_content", "mean"),
                tracer_u_max=("tracer_u_score", "mean"),
            )
        )
        work = matu_root / f"round_{round_number}"
        labels = load_pickle(work / "results" / "accuracy_dict_generated.pkl")
        final_u = load_pickle(work / "final" / "uncertainty.pkl")
        trajectory_u = load_pickle(work / "trajectory" / "uncertainty.pkl")
        expected_keys = set(prompt["prompt_id"])
        for name, values in (
            ("labels", labels),
            ("final MATU", final_u),
            ("trajectory MATU", trajectory_u),
        ):
            if set(values) != expected_keys:
                raise ValueError(
                    f"{pathway}: {name} keys differ: "
                    f"missing={len(expected_keys-set(values))}, "
                    f"extra={len(set(values)-expected_keys)}"
                )
        prompt["pathway"] = pathway
        prompt["representative_round"] = round_number
        prompt["matu_final"] = prompt["prompt_id"].map(final_u).astype(float)
        prompt["matu_trajectory"] = (
            prompt["prompt_id"].map(trajectory_u).astype(float)
        )
        judge_safe = prompt["prompt_id"].map(
            {key: float(np.mean(value)) for key, value in labels.items()}
        )
        if not np.allclose(prompt["safe_fraction"], judge_safe):
            raise ValueError(f"{pathway}: run labels disagree with MATU labels")
        rows.append(prompt)

    table = pd.concat(rows, ignore_index=True)
    if len(table) != 96 or table["prompt_id"].nunique() != 32:
        raise ValueError("expected 96 prompt-pathway rows from 32 prompts")

    # Unsupervised within-pathway standardization. Both components retain their
    # intended direction: higher uncertainty is treated as higher failure risk.
    tracer_z = table.groupby("pathway")["tracer_u_max"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    matu_z = table.groupby("pathway")["matu_trajectory"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    if tracer_z.isna().any() or matu_z.isna().any():
        raise ValueError("cannot standardize a constant pathway signal")
    table["equal_weight_intended"] = (tracer_z + matu_z) / 2.0
    return table


def bootstrap_positions(
    frame: pd.DataFrame, bootstrap: int, rng: np.random.Generator
) -> list[np.ndarray]:
    prompts = sorted(frame["prompt_id"].unique())
    prompt_array = frame["prompt_id"].to_numpy()
    groups = {
        prompt_id: np.flatnonzero(prompt_array == prompt_id)
        for prompt_id in prompts
    }
    return [
        np.concatenate(
            [groups[prompt_id] for prompt_id in rng.choice(prompts, len(prompts))]
        )
        for _ in range(bootstrap)
    ]


def evaluate_scope(
    frame: pd.DataFrame,
    scope: str,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    frame = frame.reset_index(drop=True)
    safe = frame["safe_fraction"].to_numpy(dtype=float)
    positions = bootstrap_positions(frame, bootstrap, np.random.default_rng(seed))
    rows = []
    for signal_name, column in BASE_SIGNALS.items():
        score = frame[column].to_numpy(dtype=float)
        if not np.isfinite(score).all():
            raise ValueError(f"non-finite score in {scope}/{signal_name}")
        rho = spearman_value(score, safe)
        boot_rho = [
            spearman_value(score[index], safe[index]) for index in positions
        ]
        boot_rho = [value for value in boot_rho if np.isfinite(value)]
        rho_low, rho_high = percentile_interval(boot_rho)

        for target_name, target_function in TARGETS.items():
            target = target_function(safe).astype(int)
            if len(np.unique(target)) < 2:
                continue
            auc = binary_auroc(target, score)
            ap = binary_average_precision(target, score)
            boot_auc, boot_ap = [], []
            for index in positions:
                sampled_target = target[index]
                if len(np.unique(sampled_target)) < 2:
                    continue
                sampled_score = score[index]
                boot_auc.append(binary_auroc(sampled_target, sampled_score))
                boot_ap.append(
                    binary_average_precision(sampled_target, sampled_score)
                )
            if len(boot_auc) < 100:
                raise ValueError(
                    f"too few valid bootstraps for {scope}/{signal_name}/{target_name}"
                )
            auc_low, auc_high = percentile_interval(boot_auc)
            ap_low, ap_high = percentile_interval(boot_ap)
            rows.append(
                {
                    "scope": scope,
                    "signal": signal_name,
                    "target": target_name,
                    "rows": len(frame),
                    "prompts": frame["prompt_id"].nunique(),
                    "target_prevalence": float(target.mean()),
                    "unsafe_auroc": auc,
                    "unsafe_auroc_ci_low": auc_low,
                    "unsafe_auroc_ci_high": auc_high,
                    "unsafe_average_precision": ap,
                    "unsafe_ap_ci_low": ap_low,
                    "unsafe_ap_ci_high": ap_high,
                    "spearman_with_safe_fraction": rho,
                    "spearman_ci_low": rho_low,
                    "spearman_ci_high": rho_high,
                    "bootstrap_valid": len(boot_auc),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    runs = pd.read_csv(args.distinct_runs)
    if len(runs) != 960:
        raise ValueError(f"expected 960 distinct-pathway runs, found {len(runs)}")
    table = build_prompt_table(runs, args.matu_root)

    rows = []
    for offset, pathway in enumerate(REPRESENTATIVE_ROUNDS):
        rows.extend(
            evaluate_scope(
                table[table["pathway"] == pathway],
                pathway,
                args.bootstrap,
                args.seed + offset,
            )
        )
    rows.extend(
        evaluate_scope(
            table,
            "pooled_three_effective_pathways",
            args.bootstrap,
            args.seed + 10,
        )
    )
    metrics = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "campaign_prompt_signal_table.csv"
    metrics_path = args.out_dir / "campaign_tracer_matu_metrics.csv"
    summary_path = args.out_dir / "campaign_tracer_matu_summary.json"
    table.to_csv(table_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "prompt_pathway_rows": len(table),
        "unique_prompts": table["prompt_id"].nunique(),
        "effective_pathways": len(REPRESENTATIVE_ROUNDS),
        "representative_rounds": REPRESENTATIVE_ROUNDS,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_unit": "prompt_id",
        "combined_signal": (
            "equal mean of within-pathway z-scored TRACER-U maximum and MATU "
            "trajectory uncertainty; intended high-risk directions retained"
        ),
        "table_path": str(table_path),
        "metrics_path": str(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    majority = metrics[metrics["target"] == "majority_unsafe"]
    display = majority[
        [
            "scope",
            "signal",
            "target_prevalence",
            "unsafe_auroc",
            "unsafe_auroc_ci_low",
            "unsafe_auroc_ci_high",
            "spearman_with_safe_fraction",
            "spearman_ci_low",
            "spearman_ci_high",
        ]
    ]
    print("Prompt-level TRACER-U versus MATU (high score = unsafe):")
    print(display.round(4).to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
