#!/usr/bin/env python
"""
rescore_surprisal.py — teacher-forced content-aware surprisal re-scoring.

Purpose (G1): decide whether the inverted surprisal in the 3x3 grid is a
property of agent behaviour or an artifact of GPTQ-Int4 logprobs.

Key design point: this script is run TWICE, once against an Int4 server and once
against a bf16 server, with everything else held identical. Do not compare its
bf16 output against the U values already in the grid — those were collected
inline during generation, so that comparison would confound precision with code
path. The Int4 run here is the control.

Method: for each agent turn, rebuild the exact prompt with the chat template,
re-send prompt+response to /v1/completions with echo=True and max_tokens=0, and
read back per-token logprobs for the response span. No generation happens, so
the trajectory is untouched and the comparison is exact.

Output: one row per agent step with U under three content filters, so the filter
itself can be ablated later without another GPU run.
"""

import argparse
import json
import os
import re
import string
import sys

# Small built-in stopword set — avoids an nltk download under HF_HUB_OFFLINE.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())

PUNCT = set(string.punctuation)


def is_numeric(tok: str) -> bool:
    s = tok.strip().strip("".join(PUNCT))
    return len(s) > 0 and re.fullmatch(r"[\d.,:/\-]+", s) is not None


def is_content_token(tok: str, prob: float, pi0: float) -> bool:
    """TRACER's content-bearing predicate: not a stopword, not numeric, not
    structural punctuation/whitespace, and not trivially predictable."""
    s = tok.strip()
    if not s:
        return False
    if all(c in PUNCT for c in s):
        return False
    if s.lower() in STOPWORDS:
        return False
    if is_numeric(s):
        return False
    if prob > pi0:
        return False
    return True


def load_sims(path):
    """Tolerant loader — tau2 has shipped a few shapes for this file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("simulations", "results", "runs", "trajectories"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"No trajectory list found in {path}; top-level keys: {list(data)[:10]}")
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON root type in {path}: {type(data)}")


def get_messages(sim):
    for key in ("messages", "trajectory", "conversation", "history"):
        msgs = sim.get(key)
        if isinstance(msgs, list) and msgs:
            return msgs
    return []


def get_label(sim):
    """1 = FAILURE (TRACER predicts failure, so failure is the positive class)."""
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
    return float("nan")


def msg_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # multipart content blocks
        return " ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
    if c is None and m.get("tool_calls"):
        return json.dumps(m["tool_calls"], ensure_ascii=False)
    return "" if c is None else str(c)


def score_span(api_base, served_name, prompt_text, full_text, n_prefix_tokens, timeout=180):
    """Return (tokens, logprobs) for the response span via echo re-scoring."""
    import requests

    r = requests.post(
        f"{api_base}/completions",
        json={
            "model": served_name,
            "prompt": full_text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]
    toks = lp["tokens"][n_prefix_tokens:]
    vals = lp["token_logprobs"][n_prefix_tokens:]
    keep = [(t, v) for t, v in zip(toks, vals) if v is not None]
    if not keep:
        return [], []
    return [t for t, _ in keep], [v for _, v in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-file", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--served-name", default="scorer")
    ap.add_argument("--tokenizer", required=True, help="HF repo id used for the chat template")
    ap.add_argument("--precision", required=True, choices=["int4", "bf16"])
    ap.add_argument("--pi0", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-trajs", type=int, default=0, help="0 = all; use a small number for a smoke test")
    args = ap.parse_args()

    # Keep heavyweight imports after argument parsing so --help remains fast on
    # network filesystems with large conda environments.
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sims = load_sims(args.sim_file)
    if args.max_trajs:
        sims = sims[: args.max_trajs]
    print(f"[{args.precision}] {args.cell}: {len(sims)} trajectories", flush=True)

    rows, failures = [], 0
    for ti, sim in enumerate(sims):
        msgs = get_messages(sim)
        label = get_label(sim)
        traj_id = sim.get("id", sim.get("task_id", ti))
        n_steps = len(msgs)

        for si, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            resp = msg_text(m)
            if not resp.strip():
                continue

            # rebuild the exact generation-time prompt
            try:
                prefix = tok.apply_chat_template(
                    msgs[:si], tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                failures += 1
                if failures <= 3:
                    print(f"  chat-template failed traj={traj_id} step={si}: {e}", flush=True)
                continue

            n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
            try:
                toks, logps = score_span(
                    args.api_base, args.served_name, prefix, prefix + resp, n_prefix
                )
            except Exception as e:
                failures += 1
                if failures <= 3:
                    print(f"  rescore failed traj={traj_id} step={si}: {e}", flush=True)
                continue
            if not toks:
                continue

            probs = np.exp(np.array(logps))
            surp = -np.array(logps)

            # three filter levels, so the filter can be ablated offline later
            m_all = np.ones(len(toks), dtype=bool)
            m_stop = np.array(
                [t.strip().lower() not in STOPWORDS and t.strip() != "" for t in toks]
            )
            m_content = np.array(
                [is_content_token(t, p, args.pi0) for t, p in zip(toks, probs)]
            )

            def mean_or_nan(mask):
                return float(surp[mask].mean()) if mask.any() else np.nan

            rows.append(
                dict(
                    cell=args.cell,
                    precision=args.precision,
                    traj_id=traj_id,
                    step_idx=si,
                    n_steps=n_steps,
                    actor="agent",
                    n_tokens=len(toks),
                    n_content_tokens=int(m_content.sum()),
                    U_all=mean_or_nan(m_all),
                    U_stopfilter=mean_or_nan(m_stop),
                    U_content=mean_or_nan(m_content),
                    mean_prob=float(probs.mean()),
                    label_fail=label,
                )
            )

        if (ti + 1) % 10 == 0:
            print(f"  ...{ti + 1}/{len(sims)} trajectories, {len(rows)} steps", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[{args.precision}] wrote {len(df)} rows -> {args.out}  (skipped {failures} steps)")

    if len(df):
        print(df[["U_all", "U_stopfilter", "U_content", "n_content_tokens"]].describe().round(4))
    else:
        print("WARNING: zero rows written — check the assistant-role key and message schema.")
        sys.exit(2)


if __name__ == "__main__":
    main()
