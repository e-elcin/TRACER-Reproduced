#!/usr/bin/env python
"""Join Workflow-Misevolution traces with raw collection-log responses.

This writes a new JSONL sidecar and never modifies the source traces or logs.
Raw text is marked verified only when Workflow's own formatter reproduces the
candidate stored in the corresponding trajectory JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROGRESS_RE = re.compile(
    r"^\[(\d+)/(\d+)\] (\S+) run (\d+)/(\d+) "
    r"(OK|FAILED) events=(\d+)$"
)


def is_boundary(line: str) -> bool:
    return (
        line.startswith("Cost: $")
        or line.startswith("output:")
        or PROGRESS_RE.match(line) is not None
    )


def parse_collection_log(path: Path) -> dict[tuple[str, int], tuple[str, ...]]:
    lines = path.read_text(errors="replace").splitlines()
    pending: list[str] = []
    runs: dict[tuple[str, int], tuple[str, ...]] = {}

    for index, line in enumerate(lines):
        if line.startswith("Token usage:"):
            start = index - 1
            while start >= 0 and not is_boundary(lines[start]):
                start -= 1
            pending.append("\n".join(lines[start + 1 : index]))
            continue

        progress = PROGRESS_RE.match(line)
        if progress is None:
            continue

        key = (progress.group(3), int(progress.group(4)) - 1)
        if key in runs:
            raise ValueError(f"duplicate run in collection log: {key}")
        runs[key] = tuple(pending)
        pending = []

    if pending:
        raise ValueError(
            f"collection log ended with {len(pending)} unassigned responses"
        )
    return runs


def load_normalizer(workflow_root: Path):
    sys.path.insert(0, str((workflow_root / "src").resolve()))
    from scripts.formatter import CodeFormatter

    def normalize(raw: str, entry_point: str):
        with redirect_stdout(StringIO()):
            valid, parsed = CodeFormatter(
                function_name=entry_point
            ).validate_response(raw)
        if not valid or not isinstance(parsed, dict):
            return None
        return parsed.get("response")

    return normalize


def generation_events(trace: dict) -> list[dict]:
    return [
        event
        for event in trace.get("events", [])
        if event.get("event_type") == "generation"
        and event.get("operator") == "CustomCodeGenerate"
    ]


def ensemble_event(trace: dict):
    return next(
        (
            event
            for event in trace.get("events", [])
            if event.get("event_type") == "ensemble"
            and "candidates" in event
        ),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", required=True, type=Path)
    parser.add_argument("--collection-log", required=True, type=Path)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    trace_paths = sorted(args.traces_dir.rglob("run_*.json"))
    if not trace_paths:
        raise SystemExit(f"no run_*.json files found under {args.traces_dir}")

    raw_runs = parse_collection_log(args.collection_log)
    normalize = load_normalizer(args.workflow_root)
    records = []

    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text())
        prompt_id = trace.get("prompt_id")
        run_index = trace.get("run")
        entry_point = trace.get("entry_point")
        request = trace.get("request")
        round_number = trace.get("round")

        if not isinstance(prompt_id, str) or not isinstance(run_index, int):
            raise ValueError(f"missing prompt_id/run in {trace_path}")
        if not isinstance(entry_point, str) or not isinstance(request, str):
            raise ValueError(f"missing entry_point/request in {trace_path}")

        generations = generation_events(trace)
        if not generations:
            raise ValueError(f"no generation events in {trace_path}")

        ensemble = ensemble_event(trace)
        if ensemble is None:
            selected_index = 0 if len(generations) == 1 else None
        else:
            selected_index = ensemble.get("selected_index")
            stored_candidates = [event.get("response") for event in generations]
            if ensemble.get("candidates") != stored_candidates:
                raise ValueError(
                    f"ensemble/generation candidate mismatch in {trace_path}"
                )

        raw_responses = raw_runs.get((prompt_id, run_index), ())
        if len(raw_responses) < len(generations):
            raise ValueError(
                f"too few raw calls for {prompt_id} run {run_index}: "
                f"{len(raw_responses)} < {len(generations)}"
            )

        for candidate_index, event in enumerate(generations):
            stored = event.get("response")
            if not isinstance(stored, str):
                raise ValueError(
                    f"non-text candidate {candidate_index} in {trace_path}"
                )

            raw = raw_responses[candidate_index]
            if raw == stored:
                raw_verified = True
                verification = "identity"
            else:
                try:
                    raw_verified = normalize(raw, entry_point) == stored
                    verification = (
                        "workflow_formatter"
                        if raw_verified
                        else "formatter_mismatch"
                    )
                except Exception as error:
                    raw_verified = False
                    verification = f"normalizer_error:{type(error).__name__}"

            records.append(
                {
                    "trace_path": str(trace_path),
                    "round": round_number,
                    "prompt_id": prompt_id,
                    "run_index": run_index,
                    "candidate_index": candidate_index,
                    "request": request,
                    "entry_point": entry_point,
                    "stored_response": stored,
                    "selected_index": selected_index,
                    "selected": candidate_index == selected_index,
                    "raw_response": raw,
                    "raw_verified": raw_verified,
                    "raw_verification": verification,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    outcomes = Counter(record["raw_verification"] for record in records)
    verified = sum(record["raw_verified"] for record in records)
    print("traces:", len(trace_paths))
    print("generation records:", len(records))
    print("raw verified:", verified)
    print("raw coverage:", f"{verified / len(records):.2%}")
    print("verification outcomes:", dict(sorted(outcomes.items())))
    print("output:", args.out)


if __name__ == "__main__":
    main()
