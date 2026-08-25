# changes_by_ender.md

Fork of `sinatayebati/agent-tracer`, branched at `dba3de2`.
`main` is 9 commits ahead; `ablation-test` adds 2 more.

# Code Changes

**src/tau2/config.py:**

* replaced the four hardcoded OpenAI models with env vars so the benchmark runs on a local vLLM server
* `TAU2_LLM` sets agent/user/env-interface; `TAU2_JUDGE_LLM` sets the judge separately, so the judge can stay fixed at 72B while agent size varies

**src/tau2/metrics/uncertainty.py:**

* replaced Vertex AI embeddings with local `sentence-transformers`, since compute nodes run offline
* embedder made selectable via `TRACER_EMBED_MODEL`; runs use `bge-m3`

**src/tau2/scripts/diagnose_and_optimize_tracer.py:**

* added adapters to read the current tau2-bench `simulations` JSON format instead of the old `results` format
* fixed the actor-routing bug: the `user_coherence` check sat in the agent-only branch, so `Do_user` was always 0.0 and always gave AUROC 0.5000
* scoring logic itself is untouched — only the file reading changed
* the diff looks large but is mostly whitespace cleanup

**src/tau2/scripts/cross_validate_tracer.py:**

* added because the diagnostic tunes ~20k configs and reports the best AUROC on the same data — inflated and not reproducible
* tunes on training folds, freezes, scores held-out folds, and pools them
* also reports an untuned `D_a`-only baseline, to check whether the tuned composite beats the single best signal out of sample

**src/tau2/scripts/cv_results.json:**

* cross-validated AUROCs for the 9 grid cells

**.gitignore:**

* added the new `data/tau2/` output paths and unignored `results/`

**scripts/slurms/run_smoke.slurm:**

* small single-GPU job to validate the pipeline before spending grid GPU hours

**scripts/slurms/run_grid.slurm:**

* runs the 3×3 grid (14B/32B/72B × retail/airline/telecom, 50 tasks)
* loads judge and agent servers sequentially, since loading both saturates NFS and the 72B crawls
* all sizes served as Int4, so precision does not confound the size comparison
* preflight checks the env in seconds instead of failing after a long model load — a missing sentence-transformers would make `D_a`/`D_o` silently zero
* post-run check warns if the judge server got zero requests
* on `ablation-test` only, paths updated from `elcin` to `ender`

**scripts/rescore_surprisal.py:** (ablation-test)

* checks whether the inverted surprisal is real behaviour or an Int4 logprob artifact
* re-scores existing turns teacher-forced with `echo=True, max_tokens=0`, so nothing is regenerated
* run twice, Int4 and bf16, everything else identical

**scripts/recompute_signals.py:** (ablation-test)

* created for G2/G3/G4
* recomputes `D_a` and `D_o` with a second embedder (G2) and with observations verbalized to natural language (G3), plus the calibrated gap (G4)
* caps sequence length at 8192 — Qwen3-Embedding does not truncate by default and OOMed on long airline observations
* tool-turn detection reads the structured `tool_calls` field; the earlier text-regex version missed all 506 telecom user tool-calls
* still carries an `ALIGN-ME` note: must first reproduce the grid's numbers before G2 can be called an embedder effect

**scripts/loop_hypothesis.py:** (ablation-test)

* main analysis: tests whether looping raises token probability, lowers `U`, and produces the inversion
* reads the grid's own inline `U` and `D_a`, so it cannot drift from the reported numbers
* T1 = Spearman(`U`, `D_a`); T2 = `U` AUROC on all steps vs low-repetition steps only

**scripts/summarize_ablations.py:** (ablation-test)

* builds the G1/G2/G3-G4 verdict tables
* bootstraps CIs over trajectories, not steps, since steps within a trajectory are not independent

**scripts/gather_report_numbers.py:** (ablation-test)

* prints all ablation stats for the report in one pass

**scripts/slurms/run_ablations_test.slurm:** (ablation-test)

* runs the ablation stages; re-scores cached trajectories only, no new rollouts
* needs one GPU instead of two, since there is no judge to host
* refuses at preflight if the bf16 checkpoint is missing (only 14B has one)

# Directories

**diagnose_results/**

* it includes the diagnostic tables for the 3×3 grid

**variant_results/**

* it includes the same grid under the variant sweep

**results/**

* it includes small curated outputs; raw trajectories stay on scratch and are gitignored

**cluster/**

* it includes the requirements lock file

**ablations/** (ablation-test)

* it includes the recomputed signal CSVs, the G1 surprisal CSVs, the loop-hypothesis table, and the summary files

**src/tau2/scripts/data/ablation/**

* it includes one retail trajectory used while developing the diagnostic
