#!/usr/bin/env python3
"""Map deduplicated TRACER-U scores back to workflow steps and runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = [
    "U_all",
    "U_stopfilter",
    "U_content",
    "n_tokens",
    "n_content_tokens",
]
METRICS = ["U_all", "U_stopfilter", "U_content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cached-scores", required=True, type=Path)
    parser.add_argument("--new-scores", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-steps", type=int, default=6400)
    parser.add_argument("--expected-runs", type=int, default=3200)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_score_table(cached_path: Path, new_path: Path) -> pd.DataFrame:
    cached = pd.read_csv(cached_path)
    new = pd.read_csv(new_path)
    required = ["traj_id", *SCORE_COLUMNS]
    require_columns(cached, required, "cached scores")
    require_columns(new, required, "new scores")

    cached = cached[required].copy()
    cached["score_source"] = "round3_cache"
    new = new[required].copy()
    new["score_source"] = "campaign_gpu"
    scores = pd.concat([cached, new], ignore_index=True)

    if scores["traj_id"].duplicated().any():
        duplicates = scores.loc[scores["traj_id"].duplicated(), "traj_id"].head()
        raise ValueError(f"duplicate score keys: {duplicates.tolist()}")
    for column in SCORE_COLUMNS:
        values = pd.to_numeric(scores[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"non-finite values in score column {column}")
        scores[column] = values
    if (scores[["n_tokens", "n_content_tokens"]] <= 0).any().any():
        raise ValueError("token counts must be positive")
    return scores


def pathway_name(n_generation: int, n_selector: int) -> str:
    mapping = {
        (1, 0): "single_generation",
        (1, 1): "generation_plus_selector",
        (3, 1): "three_generations_plus_selector",
    }
    try:
        return mapping[(n_generation, n_selector)]
    except KeyError as error:
        raise ValueError(
            f"unsupported pathway: generations={n_generation}, selectors={n_selector}"
        ) from error


def aggregate_run(group: pd.DataFrame) -> dict:
    group = group.sort_values("step_index")
    generations = group[group["step_type"] == "generation"]
    selectors = group[group["step_type"] == "selector"]
    selected = generations[generations["selected_generation"] == True]  # noqa: E712

    if len(selected) != 1:
        raise ValueError(
            f"expected one selected generation for "
            f"{group.iloc[0]['round']}/{group.iloc[0]['prompt_id']}/"
            f"{group.iloc[0]['run_index']}; found {len(selected)}"
        )
    if len(selectors) not in (0, 1):
        raise ValueError("expected zero or one selector")

    critical_index = group["U_content"].idxmax()
    critical = group.loc[critical_index]
    row = {
        "round": int(group.iloc[0]["round"]),
        "prompt_id": group.iloc[0]["prompt_id"],
        "run_index": int(group.iloc[0]["run_index"]),
        "pathway": pathway_name(len(generations), len(selectors)),
        "n_steps": len(group),
        "n_generation_steps": len(generations),
        "n_selector_steps": len(selectors),
        "critical_step_index": int(critical["step_index"]),
        "critical_step_type": critical["step_type"],
        "critical_candidate_index": critical["candidate_index"],
        "all_raw_verified": bool(group["raw_verified"].all()),
        "all_generation_raw_verified": bool(generations["raw_verified"].all()),
        "total_response_tokens": int(group["n_tokens"].sum()),
        "total_content_tokens": int(group["n_content_tokens"].sum()),
    }
    for metric in METRICS:
        row[f"selected_generation_{metric}"] = float(selected.iloc[0][metric])
        row[f"generation_mean_{metric}"] = float(generations[metric].mean())
        row[f"generation_max_{metric}"] = float(generations[metric].max())
        row[f"selector_{metric}"] = (
            float(selectors.iloc[0][metric]) if len(selectors) else float("nan")
        )
        row[f"step_mean_{metric}"] = float(group[metric].mean())
        row[f"step_max_{metric}"] = float(group[metric].max())
    row["tracer_u_score"] = row["step_max_U_content"]
    return row


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    if len(manifest_rows) != args.expected_steps:
        raise ValueError(
            f"expected {args.expected_steps} manifest steps, found {len(manifest_rows)}"
        )
    manifest = pd.DataFrame(manifest_rows)
    require_columns(
        manifest,
        [
            "record_id",
            "score_key",
            "round",
            "prompt_id",
            "run_index",
            "step_index",
            "step_type",
            "candidate_index",
            "selected_generation",
            "raw_verified",
        ],
        "manifest",
    )
    if manifest["record_id"].duplicated().any():
        raise ValueError("manifest record_id values must be unique")

    scores = load_score_table(args.cached_scores, args.new_scores)
    manifest_keys = set(manifest["score_key"])
    score_keys = set(scores["traj_id"])
    if manifest_keys != score_keys:
        raise ValueError(
            f"score coverage mismatch: missing={len(manifest_keys-score_keys)}, "
            f"extra={len(score_keys-manifest_keys)}"
        )

    steps = manifest.merge(
        scores,
        left_on="score_key",
        right_on="traj_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["traj_id"])
    if steps[SCORE_COLUMNS].isna().any().any():
        raise ValueError("score merge produced missing values")

    run_rows = [
        aggregate_run(group)
        for _, group in steps.groupby(
            ["round", "prompt_id", "run_index"], sort=True, dropna=False
        )
    ]
    runs = pd.DataFrame(run_rows).sort_values(
        ["round", "prompt_id", "run_index"]
    )
    if len(runs) != args.expected_runs:
        raise ValueError(f"expected {args.expected_runs} runs, found {len(runs)}")
    if runs.duplicated(["round", "prompt_id", "run_index"]).any():
        raise ValueError("run identities must be unique")
    if not (runs["tracer_u_score"] == runs["step_max_U_content"]).all():
        raise AssertionError("TRACER-U aggregation does not equal maximum step U")

    round_rows = []
    for round_number, group in runs.groupby("round", sort=True):
        pathways = sorted(group["pathway"].unique())
        if len(pathways) != 1:
            raise ValueError(f"round {round_number} has multiple pathways: {pathways}")
        round_rows.append(
            {
                "round": int(round_number),
                "pathway": pathways[0],
                "runs": len(group),
                "mean_tracer_u": float(group["tracer_u_score"].mean()),
                "std_tracer_u": float(group["tracer_u_score"].std()),
                "median_tracer_u": float(group["tracer_u_score"].median()),
                "min_tracer_u": float(group["tracer_u_score"].min()),
                "max_tracer_u": float(group["tracer_u_score"].max()),
                "mean_selected_generation_u": float(
                    group["selected_generation_U_content"].mean()
                ),
                "mean_generation_max_u": float(
                    group["generation_max_U_content"].mean()
                ),
                "mean_selector_u": float(group["selector_U_content"].mean()),
                "selector_critical_runs": int(
                    (group["critical_step_type"] == "selector").sum()
                ),
                "fully_raw_verified_runs": int(group["all_raw_verified"].sum()),
            }
        )
    rounds = pd.DataFrame(round_rows)
    if len(rounds) != 10 or not (rounds["runs"] == 320).all():
        raise ValueError("expected ten rounds with 320 runs each")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "campaign_unique_scores.csv"
    step_path = args.out_dir / "campaign_step_scores.csv"
    run_path = args.out_dir / "campaign_run_scores.csv"
    round_path = args.out_dir / "campaign_round_summary.csv"
    summary_path = args.out_dir / "campaign_finalize_summary.json"
    scores.sort_values("traj_id").to_csv(score_path, index=False)
    steps.sort_values(["round", "prompt_id", "run_index", "step_index"]).to_csv(
        step_path, index=False
    )
    runs.to_csv(run_path, index=False)
    rounds.to_csv(round_path, index=False)

    summary = {
        "unique_scores": len(scores),
        "cached_unique_scores": int((scores["score_source"] == "round3_cache").sum()),
        "gpu_unique_scores": int((scores["score_source"] == "campaign_gpu").sum()),
        "step_rows": len(steps),
        "run_rows": len(runs),
        "round_rows": len(rounds),
        "paths": {
            "unique_scores": str(score_path),
            "steps": str(step_path),
            "runs": str(run_path),
            "rounds": str(round_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nPer-round TRACER-U summary:")
    print(rounds.to_string(index=False))


if __name__ == "__main__":
    main()
