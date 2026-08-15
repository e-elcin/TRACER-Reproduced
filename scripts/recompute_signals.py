#!/usr/bin/env python
"""
recompute_signals.py — recompute D_a, D_o^A, D_o^U under embedding-side ablations.

Variants:
  baseline    signals exactly as defined in TRACER, with the chosen embedder.
              Run with BOTH bge-m3 and Qwen3-Embedding-0.6B -> that pair is G2.
  verbalized  observations rendered to natural language before embedding, so
              D_o^A stops partly measuring NL-vs-JSON format distance (G3).
Both variants also emit the CALIBRATED gap (G4) as column d_oA_calibrated:
      D~ = d(x_t, o_t) - mean_{o' in traj} d(x_t, o')
  It is a post-hoc transform of the same embeddings, so it needs no pass of its
  own. Negative values are meaningful: the action is CLOSER to its own
  observation than to a random one, i.e. coherent.

Also emits d_oA_raw_mean / d_oA_raw_std per row's trajectory so the "is D_o^A
pinned near ceiling with no variance?" diagnostic can be read straight off.

!! ALIGN-ME: the D_a / D_o formulas below are a faithful reading of the paper,
   not a copy of the in-pipeline implementation. Before reporting G2 as an
   embedder effect, confirm the baseline+bge-m3 output here reproduces the grid
   numbers. If it does not, the difference is this file, not the embedder.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rescore_surprisal import load_sims, get_messages, get_label, msg_text

WINDOW = 5           # look-back window for repetition
MAX_CONTRAST = 20    # observations sampled for the calibration baseline


# ---------------------------------------------------------------- verbalizer
def verbalize(obs_text: str) -> str:
    """JSON/structured tool output -> one flat NL sentence. Deterministic on
    purpose: an LLM paraphrase would add its own variance to a controlled
    ablation, and the point is only to remove the format gap."""
    t = (obs_text or "").strip()
    if not t:
        return ""
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return t

    def flatten(o, prefix=""):
        out = []
        if isinstance(o, dict):
            for k, v in o.items():
                key = re.sub(r"[_\-]+", " ", str(k))
                out += flatten(v, f"{prefix} {key}".strip())
        elif isinstance(o, list):
            for i, v in enumerate(o[:10]):
                out += flatten(v, f"{prefix} item {i + 1}".strip())
        else:
            out.append(f"{prefix} is {o}" if prefix else str(o))
        return out

    parts = flatten(obj)
    return "The tool returned: " + "; ".join(parts[:60]) + "." if parts else t


# ---------------------------------------------------------------- signals
def lexical_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def entity_set(s: str) -> frozenset:
    """Capitalised words, IDs and numbers — the things that change during a
    legitimate enumeration but not during a degenerate loop."""
    ents = set(re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", s))
    ents |= set(re.findall(r"\b[A-Za-z]*\d{3,}\b", s))
    ents |= set(re.findall(r"\b\d+(?:\.\d+)?\b", s))
    return frozenset(ents)


def cos(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if d == 0 else float(np.dot(u, v) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-file", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--variant", required=True, choices=["baseline", "verbalized"],
                    help="observation representation; the calibrated gap (G4) is "
                         "emitted in both, as column d_oA_calibrated")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-trajs", type=int, default=0)
    args = ap.parse_args()

    print(f"[{args.variant}|{os.path.basename(args.embed_model)}] {args.cell}", flush=True)
    model = SentenceTransformer(args.embed_model, trust_remote_code=True)
    # Cap sequence length. bge-m3 self-caps at 8192, but Qwen3-Embedding has a
    # very long context window and does NOT truncate by default — a single long
    # airline observation (30k+ tokens) then builds an O(L^2) attention matrix
    # and OOMs (observed: 50GB alloc for a 0.6B model). 8192 tokens is ample for
    # a coherence embedding and makes both embedders behave identically.
    try:
        if model.max_seq_length is None or model.max_seq_length > 8192:
            model.max_seq_length = 8192
    except Exception:
        pass

    sims = load_sims(args.sim_file)
    if args.max_trajs:
        sims = sims[: args.max_trajs]

    rows = []
    for ti, sim in enumerate(sims):
        msgs = get_messages(sim)
        if not msgs:
            continue
        label = get_label(sim)
        traj_id = sim.get("id", sim.get("task_id", ti))

        # ---- collect step texts -------------------------------------------
        acts, obss, roles, is_tool_turn = [], [], [], []
        for m in msgs:
            role = m.get("role", "")
            text = msg_text(m)
            obs = ""
            if role == "assistant":
                actor = "agent"
            elif role in ("user", "human"):
                actor = "user"
            elif role in ("tool", "function", "observation"):
                actor = "tool"
                obs = text
            else:
                actor = role or "other"
            roles.append(actor)
            acts.append(text)
            obss.append(obs)
            # A turn is a "tool turn" if the MESSAGE OBJECT carries a structured
            # tool_calls / function_call field. Reading acts[t] text misses these
            # entirely (tau2 stores the call as structured data, not in content),
            # which is why the regex version flagged zero in telecom despite 506
            # user tool-calls. This is the field that matters for the D_o^U split.
            has_tc = bool(m.get("tool_calls") or m.get("function_call")
                          or (isinstance(m.get("content"), list)
                              and any(isinstance(b, dict)
                                      and b.get("type") in ("tool_use", "tool_call")
                                      for b in m["content"])))
            is_tool_turn.append(has_tc)

        # attach each tool observation to the agent turn that caused it
        obs_for_step = [""] * len(acts)
        for i, r in enumerate(roles):
            if r == "tool":
                for j in range(i - 1, -1, -1):
                    if roles[j] == "agent":
                        obs_for_step[j] = obss[i]
                        break

        if args.variant == "verbalized":
            obs_for_step = [verbalize(o) for o in obs_for_step]

        # ---- embed once per trajectory ------------------------------------
        texts = acts + obs_for_step
        nonempty = [t if t.strip() else " " for t in texts]
        emb = model.encode(nonempty, batch_size=args.batch_size,
                           show_progress_bar=False, normalize_embeddings=True)
        E_act, E_obs = emb[: len(acts)], emb[len(acts):]

        obs_idx = [i for i, o in enumerate(obs_for_step) if o.strip()]
        n_steps = len(acts)

        for t in range(n_steps):
            d_a_sem = d_a_lex = d_a_gated = np.nan
            d_oA = d_oA_cal = d_oU = np.nan

            # --- repetition (agent turns only) ---
            if roles[t] == "agent":
                prev = [j for j in range(max(0, t - WINDOW), t) if roles[j] == "agent"]
                if prev:
                    sims_sem = [cos(E_act[t], E_act[j]) for j in prev]
                    sims_lex = [lexical_overlap(acts[t], acts[j]) for j in prev]
                    k = int(np.argmax(sims_sem))
                    d_a_sem = float(max(sims_sem))
                    d_a_lex = float(max(sims_lex))
                    # entity-gated: fires only when nothing new was introduced
                    same_entities = entity_set(acts[t]) == entity_set(acts[prev[k]])
                    d_a_gated = d_a_lex if same_entities else 0.0

                # --- agent coherence gap ---
                if obs_for_step[t].strip():
                    d_oA = 1.0 - cos(E_act[t], E_obs[t])
                    # G4 is a post-hoc transform of the SAME embeddings as the raw
                    # gap, so it is computed here rather than in a separate pass.
                    others = [i for i in obs_idx if i != t][:MAX_CONTRAST]
                    if others:
                        floor = np.mean([1.0 - cos(E_act[t], E_obs[i]) for i in others])
                        d_oA_cal = float(d_oA - floor)

            # --- user coordination gap: agent turn followed by a user turn ---
            if t > 0 and roles[t] == "user" and roles[t - 1] == "agent":
                d_oU = 1.0 - cos(E_act[t], E_act[t - 1])

            rows.append(dict(
                cell=args.cell, variant=args.variant,
                embedder=os.path.basename(args.embed_model),
                traj_id=traj_id, step_idx=t, n_steps=n_steps, actor=roles[t],
                d_a_sem=d_a_sem, d_a_lex=d_a_lex, d_a_gated=d_a_gated,
                d_oA=d_oA, d_oA_calibrated=d_oA_cal, d_oU=d_oU,
                # telecom split: tool-emitting user turns break D_o^U's definition
                user_is_tool_turn=int(roles[t] == "user" and is_tool_turn[t]),
                label_fail=label,
            ))

        if (ti + 1) % 20 == 0:
            print(f"  ...{ti + 1}/{len(sims)} trajectories", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")

    cols = ["d_a_sem", "d_a_lex", "d_a_gated", "d_oA", "d_oA_calibrated", "d_oU"]
    print(df[cols].describe().round(4))
    if df["d_oA"].notna().any():
        print(f"D_oA dynamic range: mean={df.d_oA.mean():.4f} std={df.d_oA.std():.4f} "
              f"p05={df.d_oA.quantile(.05):.4f} p95={df.d_oA.quantile(.95):.4f}")
        print("  (mean near ceiling + tiny std == format-mismatch diagnosis confirmed)")


if __name__ == "__main__":
    main()
