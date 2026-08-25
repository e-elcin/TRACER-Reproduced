#!/usr/bin/env python3
"""Build a deduplicated TRACER-U scoring campaign from workflow traces.

Every original LLM call receives a manifest row. Exact user-prompt/raw-response
pairs share one score key, so deterministic duplicates across stored rounds are
teacher-forced only once. Existing Round-3 scores can seed the score cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from extract_workflow_generations import parse_collection_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--round3-cache",
        type=Path,
        help="Optional Round-3 tracer_u_steps.csv to reuse",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_key(prompt: str, response: str) -> str:
    payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def load_workflow_helpers(workflow_root: Path):
    sys.path.insert(0, str((workflow_root / "src").resolve()))
    from scripts.formatter import CodeFormatter, XmlFormatter
    from workspace_matu_noframe_trajectory.HumanEval.workflows.template.op_prompt import (
        SC_ENSEMBLE_PROMPT,
    )
    from workspace_matu_noframe_trajectory.HumanEval.workflows.template.operator_an import (
        ScEnsembleOp,
    )

    return CodeFormatter, XmlFormatter, SC_ENSEMBLE_PROMPT, ScEnsembleOp


def generation_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in trace.get("events", [])
        if event.get("event_type") == "generation"
        and event.get("operator") == "CustomCodeGenerate"
    ]


def llm_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in trace.get("events", [])
        if event.get("event_type") == "llm_call"
    ]


def ensemble_event(trace: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in trace.get("events", [])
            if event.get("event_type") == "ensemble"
            and event.get("operator") == "ScEnsemble"
            and "candidates" in event
        ),
        None,
    )


def generation_is_selected(
    candidate_index: int,
    generation_count: int,
    ensemble: dict[str, Any] | None,
) -> bool:
    if ensemble is None:
        return True
    candidates = ensemble.get("candidates") or []
    selected_index = ensemble.get("selected_index")
    if generation_count == len(candidates):
        return candidate_index == selected_index
    if generation_count == 1 and candidates and len(set(candidates)) == 1:
        return True
    return False


def load_round3_cache(path: Path | None) -> dict[tuple[Any, ...], dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Round-3 cache not found: {path}")

    cache: dict[tuple[Any, ...], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            candidate = row.get("candidate_index", "")
            candidate_index = None if candidate in ("", None) else int(float(candidate))
            key = (
                int(row["round"]),
                row["prompt_id"],
                int(row["run_index"]),
                row["step_type"],
                candidate_index,
            )
            if key in cache:
                raise ValueError(f"duplicate Round-3 cache key: {key}")
            cache[key] = row
    return cache


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    workflow_root = args.workflow_root.resolve()
    spec = json.loads(args.spec.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    CodeFormatter, XmlFormatter, selector_template, ScEnsembleOp = (
        load_workflow_helpers(workflow_root)
    )
    selector_formatter = XmlFormatter.from_model(ScEnsembleOp)
    round3_cache = load_round3_cache(args.round3_cache)

    expected_prompts = int(spec["expected_prompts"])
    expected_runs = int(spec["expected_runs_per_prompt"])
    manifest: list[dict[str, Any]] = []
    unique_sims: dict[str, dict[str, Any]] = {}
    round_counts: dict[int, dict[str, int]] = {}
    verification = Counter()

    for round_spec in spec["rounds"]:
        round_number = int(round_spec["round"])
        expected_calls = int(round_spec["expected_calls_per_run"])
        traces_dir = (
            workflow_root
            / "results_matu"
            / "trajectory_pilot_n32"
            / f"round_{round_number}"
            / "trajectories"
        )
        collection_log = workflow_root / round_spec["collection_log"]
        trace_paths = sorted(traces_dir.rglob("run_*.json"))
        expected_trace_count = expected_prompts * expected_runs
        if len(trace_paths) != expected_trace_count:
            raise ValueError(
                f"round {round_number}: expected {expected_trace_count} traces, "
                f"found {len(trace_paths)}"
            )
        raw_runs = parse_collection_log(collection_log)
        if len(raw_runs) != expected_trace_count:
            raise ValueError(
                f"round {round_number}: expected {expected_trace_count} raw runs, "
                f"found {len(raw_runs)}"
            )

        step_count = 0
        run_keys: set[tuple[str, int]] = set()
        for trace_path in trace_paths:
            trace = json.loads(trace_path.read_text())
            prompt_id = trace.get("prompt_id")
            run_index = trace.get("run")
            request = trace.get("request")
            entry_point = trace.get("entry_point")
            if not isinstance(prompt_id, str) or not isinstance(run_index, int):
                raise ValueError(f"missing prompt/run identity: {trace_path}")
            if not isinstance(request, str) or not isinstance(entry_point, str):
                raise ValueError(f"missing request/entry point: {trace_path}")
            run_key = (prompt_id, run_index)
            run_keys.add(run_key)

            calls = llm_events(trace)
            raws = raw_runs.get(run_key, ())
            generations = generation_events(trace)
            ensemble = ensemble_event(trace)
            if len(calls) != expected_calls or len(raws) != expected_calls:
                raise ValueError(
                    f"round {round_number} {run_key}: calls={len(calls)}, "
                    f"raws={len(raws)}, expected={expected_calls}"
                )

            generation_index = 0
            for call_index, (call, raw) in enumerate(zip(calls, raws)):
                operator = call.get("operator")
                candidate_index: int | None = None
                selected_generation: bool | None = None
                selected_index = ensemble.get("selected_index") if ensemble else None

                if operator == "CustomCodeGenerate":
                    if generation_index >= len(generations):
                        raise ValueError(f"unmatched generation call: {trace_path}")
                    generation = generations[generation_index]
                    candidate_index = generation_index
                    instruction = generation.get("instruction", "")
                    stored_response = generation.get("response")
                    if not isinstance(instruction, str) or not isinstance(stored_response, str):
                        raise ValueError(f"invalid generation event: {trace_path}")
                    formatter = CodeFormatter(function_name=entry_point)
                    prompt = formatter.prepare_prompt(instruction + request)
                    if raw == stored_response:
                        raw_verified = True
                        verification_kind = "identity"
                    else:
                        with redirect_stdout(StringIO()):
                            valid, parsed = formatter.validate_response(raw)
                        normalized = parsed.get("response") if valid and isinstance(parsed, dict) else None
                        raw_verified = normalized == stored_response
                        verification_kind = (
                            "workflow_formatter" if raw_verified else "formatter_mismatch"
                        )
                    selected_generation = generation_is_selected(
                        generation_index, len(generations), ensemble
                    )
                    step_type = "generation"
                    generation_index += 1

                elif operator == "ScEnsemble":
                    if ensemble is None:
                        raise ValueError(f"selector call without ensemble event: {trace_path}")
                    candidates = ensemble.get("candidates")
                    if not isinstance(candidates, list) or not all(
                        isinstance(value, str) for value in candidates
                    ):
                        raise ValueError(f"invalid ensemble candidates: {trace_path}")
                    solution_text = "".join(
                        f"{chr(65 + index)}: \n{candidate}\n\n\n"
                        for index, candidate in enumerate(candidates)
                    )
                    base_prompt = selector_template.format(
                        problem=request, solutions=solution_text
                    )
                    prompt = selector_formatter.prepare_prompt(base_prompt)
                    valid, parsed = selector_formatter.validate_response(raw)
                    raw_verified = (
                        valid
                        and isinstance(parsed, dict)
                        and parsed == ensemble.get("selector_response")
                    )
                    verification_kind = (
                        "workflow_xml_formatter" if raw_verified else "formatter_mismatch"
                    )
                    step_type = "selector"

                else:
                    raise ValueError(
                        f"unsupported LLM operator {operator!r} in {trace_path}"
                    )

                key = score_key(prompt, raw)
                sim = {
                    "id": key,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw},
                    ],
                }
                previous = unique_sims.setdefault(key, sim)
                if previous != sim:
                    raise AssertionError(f"score-key collision: {key}")

                manifest.append(
                    {
                        "record_id": (
                            f"round_{round_number}::{prompt_id}::run_{run_index}"
                            f"::step_{call_index}"
                        ),
                        "score_key": key,
                        "round": round_number,
                        "prompt_id": prompt_id,
                        "run_index": run_index,
                        "step_index": call_index,
                        "step_type": step_type,
                        "operator": operator,
                        "candidate_index": candidate_index,
                        "selected_generation": selected_generation,
                        "selected_index": selected_index,
                        "raw_verified": raw_verified,
                        "raw_verification": verification_kind,
                        "prompt_sha256": sha256_text(prompt),
                        "response_sha256": sha256_text(raw),
                        "trace_path": str(trace_path),
                    }
                )
                verification[(step_type, verification_kind)] += 1
                step_count += 1

            if generation_index != len(generations):
                raise ValueError(f"unmatched generation events: {trace_path}")

        if len(run_keys) != expected_trace_count:
            raise ValueError(f"round {round_number}: duplicate run identities")
        round_counts[round_number] = {
            "runs": len(run_keys),
            "steps": step_count,
            "expected_calls_per_run": expected_calls,
        }

    expected_steps = sum(
        expected_prompts
        * expected_runs
        * int(round_spec["expected_calls_per_run"])
        for round_spec in spec["rounds"]
    )
    if len(manifest) != expected_steps:
        raise ValueError(f"expected {expected_steps} steps, found {len(manifest)}")

    cached_by_score_key: dict[str, dict[str, Any]] = {}
    if round3_cache:
        for row in manifest:
            if row["round"] != 3:
                continue
            lookup_key = (
                3,
                row["prompt_id"],
                row["run_index"],
                row["step_type"],
                row["candidate_index"],
            )
            cached = round3_cache.get(lookup_key)
            if cached is None:
                raise ValueError(f"missing Round-3 cached score: {lookup_key}")
            score = {
                "traj_id": row["score_key"],
                "U_all": float(cached["U_all"]),
                "U_stopfilter": float(cached["U_stopfilter"]),
                "U_content": float(cached["U_content"]),
                "n_tokens": int(float(cached["n_tokens"])),
                "n_content_tokens": int(float(cached["n_content_tokens"])),
                "score_source": "round3_cache",
            }
            previous = cached_by_score_key.setdefault(row["score_key"], score)
            comparable = {key: value for key, value in score.items() if key != "score_source"}
            previous_comparable = {
                key: value for key, value in previous.items() if key != "score_source"
            }
            if previous_comparable != comparable:
                raise ValueError(f"conflicting cached score: {row['score_key']}")

    uncached = [
        sim
        for key, sim in sorted(unique_sims.items())
        if key not in cached_by_score_key
    ]

    write_jsonl(args.out_dir / "campaign_manifest.jsonl", manifest)
    (args.out_dir / "uncached_sims.json").write_text(
        json.dumps(uncached, ensure_ascii=False) + "\n"
    )
    cached_path = args.out_dir / "cached_scores.csv"
    with cached_path.open("w", newline="") as handle:
        fieldnames = [
            "traj_id",
            "U_all",
            "U_stopfilter",
            "U_content",
            "n_tokens",
            "n_content_tokens",
            "score_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(cached_by_score_key.values(), key=lambda row: row["traj_id"]))

    summary = {
        "campaign": spec["campaign"],
        "model": spec["model"],
        "rounds": round_counts,
        "workflow_runs": expected_prompts * expected_runs * len(spec["rounds"]),
        "manifest_steps": len(manifest),
        "unique_prompt_response_pairs": len(unique_sims),
        "cached_unique_pairs": len(cached_by_score_key),
        "uncached_unique_pairs": len(uncached),
        "deduplicated_step_savings": len(manifest) - len(unique_sims),
        "verification": {
            f"{step_type}:{kind}": count
            for (step_type, kind), count in sorted(verification.items())
        },
    }
    (args.out_dir / "campaign_build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
