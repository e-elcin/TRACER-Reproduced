#!/bin/bash
# ===========================================================================
# run_ablations_test.slurm — TRACER signal confound checks, parametrized by
#                            NODE, stage, agent size, domain
#
# Usage:  sbatch -w <NODE> run_ablations_test.slurm <NODE> <STAGE> <SIZE> <DOMAIN...>
#   NODE   : adep2 | romer3
#   STAGE  : signals   — G2 embedder / G3 verbalization / G4 calibrated gap.
#                        Embedders only, no LLM. Minutes per domain.
#            precision — G1 quantization. Teacher-forced re-scoring of the SAME
#                        trajectories with Int4 then bf16. Needs the bf16
#                        checkpoint (14B only, as of now — see CACHE below).
#   SIZE   : 14B | 32B | 72B   (selects which grid row's trajectories to re-score)
#   DOMAIN : retail | airline | telecom   (one or more)
#
# Example: sbatch -w adep2 run_ablations_test.slurm adep2 signals 32B retail telecom airline
#          sbatch -w adep2 run_ablations_test.slurm adep2 precision 14B retail
#
# NOTHING IS RE-GENERATED. Both stages read the cached trajectories under
# data/simulations/ and re-score them. No tau2 rollouts, no agent sampling.
#
# ONE GPU IS ENOUGH. Unlike run_grid.slurm there is no judge to co-host: the
# signals stage runs embedders only, and the precision stage loads its two
# checkpoints sequentially (same NFS reasoning as run_grid — one load at a time
# gets full bandwidth). Requesting gpu:1 also schedules faster.
#
# CACHE: 14B has BOTH Int4 and bf16 under $HF_HOME. 32B/72B are Int4-only, so
#        `precision 32B` will refuse at preflight until the bf16 weights exist.
#        14B is the right cell for G1 anyway — see the note at the bottom.
#
# Path check with no allocation (runs on the login node, ~seconds):
#   PREFLIGHT_ONLY=1 bash run_ablations_test.slurm adep2 signals 32B retail
# ===========================================================================
#SBATCH --job-name=tracer-ablate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/dl_scratch2/ege/ender/TRACER-Reproduced/logs/ablate-%j.out
#SBATCH --error=/dl_scratch2/ege/ender/TRACER-Reproduced/logs/ablate-%j.err
# NOTE: these two paths are ABSOLUTE and slurm does NOT create the directory.
#       If logs/ is missing or the path is stale, the job dies at submit with
#       NO log file and no error — indistinguishable from "nothing happened".
#       `mkdir -p /dl_scratch2/ege/ender/TRACER-Reproduced/logs` first, and
#       update BOTH lines if the project directory is ever renamed again.

set -uo pipefail

USAGE="Usage: sbatch -w <node> run_ablations_test.slurm <adep2|romer3> <signals|precision> <14B|32B|72B> <retail|airline|telecom ...>"

# ---------------- arguments ----------------
NODE="${1:?$USAGE}"
STAGE="${2:?$USAGE}"
SIZE="${3:?$USAGE}"
shift 3
DOMAINS=("$@")
[ "${#DOMAINS[@]}" -eq 0 ] && { echo "$USAGE"; exit 1; }

BASE=/dl_scratch2/ege/ender/TRACER-Reproduced
SIM_DIR="${SIM_DIR:-$BASE/data/simulations}"
OUT="${OUT:-$BASE/ablations}"
SCRIPTS="${SCRIPTS:-$BASE/scripts}"
SIM_TAG="${SIM_TAG:-50runs}"          # file stem suffix; set 20runs1 etc. if needed
# which embedder passes to run in the signals stage:
#   all (default) = bge-m3 baseline + qwen3 + verbalized
#   bgem3         = bge-m3 baseline + verbalized only
#   qwen3         = qwen3 baseline only (use to backfill a single failed pass)
SIGNALS_ONLY="${SIGNALS_ONLY:-all}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

# ---------------- node-specific setup ----------------
case "$NODE" in
  adep2)
    GPU_MEM_GB=96
    export CUDA_HOME=/dl_scratch2/ege/cuda-13.0
    CONDA_ENV=/dl_scratch2/ege/envs/misevolution_agent_safety
    ;;
  romer3)
    GPU_MEM_GB=48
    export CUDA_HOME=/usr/local/cuda
    CONDA_ENV=tracer
    ;;
  *) echo "Bad NODE: $NODE (use adep2|romer3)"; exit 1 ;;
esac
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# guard: submitted node must match the NODE argument (env/CUDA depend on it)
if [ "$PREFLIGHT_ONLY" != "1" ]; then
  ACTUAL_NODE="$(hostname -s)"
  if [ "$ACTUAL_NODE" != "$NODE" ]; then
    echo "REFUSING: running on '$ACTUAL_NODE' but NODE arg says '$NODE'."
    echo "  -> Submit with: sbatch -w $NODE run_ablations_test.slurm $NODE $STAGE $SIZE ${DOMAINS[*]}"
    echo "  -> Checking paths from the login node? PREFLIGHT_ONLY=1 bash \$0 ..."
    exit 1
  fi
fi

case "$STAGE" in
  signals|precision) ;;
  *) echo "Bad STAGE: $STAGE (use signals|precision)"; exit 1 ;;
esac

# ---------------- map size -> checkpoints ----------------
# Int4 is the checkpoint the grid actually ran; bf16 is the G1 comparison.
# *_WAIT are per-model load budgets in 10s ticks, same convention as run_grid.
case "$SIZE" in
  14B) INT4_MODEL="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"; INT4_GB=10; INT4_WAIT=180
       BF16_MODEL="Qwen/Qwen2.5-14B-Instruct";           BF16_GB=30; BF16_WAIT=300 ;;
  32B) INT4_MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"; INT4_GB=20; INT4_WAIT=270
       BF16_MODEL="Qwen/Qwen2.5-32B-Instruct";           BF16_GB=64; BF16_WAIT=600 ;;
  72B) INT4_MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"; INT4_GB=42; INT4_WAIT=420
       BF16_MODEL="Qwen/Qwen2.5-72B-Instruct";           BF16_GB=145; BF16_WAIT=900 ;;
  *)   echo "Bad SIZE: $SIZE (use 14B|32B|72B)"; exit 1 ;;
esac

for D in "${DOMAINS[@]}"; do
  case "$D" in
    retail|airline|telecom) ;;
    *) echo "Bad DOMAIN: $D (use retail|airline|telecom)"; exit 1 ;;
  esac
done

# embedders: base = the one the grid used, alt = the G2 comparison
EMB_BASE="${TRACER_EMBED_MODEL:-BAAI/bge-m3}"
EMB_ALT="${TRACER_EMBED_MODEL_ALT:-Qwen/Qwen3-Embedding-0.6B}"

PORT=8000

# ---------------- fit guards ----------------
if [ "$STAGE" = "precision" ]; then
  if [ "$INT4_GB" -gt "$((GPU_MEM_GB - 10))" ]; then
    echo "REFUSING: ${SIZE} Int4 (~${INT4_GB}GB) won't fit one ${GPU_MEM_GB}GB card on ${NODE}."
    exit 1
  fi
  if [ "$BF16_GB" -gt "$((GPU_MEM_GB - 12))" ]; then
    echo "REFUSING: ${SIZE} bf16 (~${BF16_GB}GB) won't fit one ${GPU_MEM_GB}GB card on ${NODE}."
    echo "  ${SIZE}=72B bf16 needs tensor-parallel across both cards — out of scope here."
    echo "  -> G1 on 14B answers the same question: the statistic is the paired"
    echo "     per-step Int4-vs-bf16 correlation, which does not need a"
    echo "     label-reliable cell.  sbatch -w $NODE \$0 $NODE precision 14B ${DOMAINS[*]}"
    exit 1
  fi
fi

echo "=== ABLATE: node=${NODE} stage=${STAGE} size=${SIZE} domains=${DOMAINS[*]} signals_only=${SIGNALS_ONLY} ==="

# ---------------- conda env ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# fail in seconds, not after a 30-min model load
python -c "import transformers, sentence_transformers, sklearn, pandas, numpy, requests" 2>/dev/null || {
  echo "FATAL: missing a dependency in $CONDA_ENV"
  echo "       need: transformers sentence-transformers scikit-learn pandas numpy requests"
  echo "       (without sentence-transformers, Da/Do come out SILENTLY ZERO — not an error)"
  exit 1
}
echo "env check OK: transformers + sentence-transformers + sklearn + pandas"

cd "$BASE" || { echo "FATAL: cannot cd $BASE"; exit 1; }
mkdir -p logs "$OUT"

# ---------------- preflight: scripts, models, trajectories ----------------
echo "--- preflight ---"

for f in rescore_surprisal.py recompute_signals.py summarize_ablations.py; do
  [ -f "$SCRIPTS/$f" ] || { echo "FATAL: $SCRIPTS/$f not found"; exit 1; }
done
echo "  scripts OK ($SCRIPTS)"

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITELLM_LOG=CRITICAL
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ease embedder fragmentation

# model cache check — HF_HUB_OFFLINE means a missing repo fails at load, not now
check_cached() {
  local repo="$1" label="$2"
  local dir="$HF_HOME/hub/models--${repo//\//--}"
  if [ -d "$dir" ]; then
    echo "  ok       $label  $repo"
    return 0
  fi
  echo "  MISSING  $label  $repo   (no $dir)"
  return 1
}

CACHE_MISS=0
if [ "$STAGE" = "precision" ]; then
  check_cached "$INT4_MODEL" "INT4    " || CACHE_MISS=1
  check_cached "$BF16_MODEL" "BF16    " || CACHE_MISS=1
else
  check_cached "$EMB_BASE" "EMB_BASE" || CACHE_MISS=1
  check_cached "$EMB_ALT"  "EMB_ALT " || CACHE_MISS=1
fi
if [ "$CACHE_MISS" -eq 1 ]; then
  echo "FATAL: fetch the missing repo(s) first (scratch is writable again):"
  echo "  HF_HOME=$HF_HOME huggingface-cli download <repo>"
  exit 1
fi

# trajectory files: cell stem follows the grid's own naming
CELLS=()
for D in "${DOMAINS[@]}"; do
  CELL="qwen${SIZE}-int4_${D}_${SIM_TAG}"
  SIM="$SIM_DIR/${CELL}.json"
  if [ ! -f "$SIM" ]; then
    echo "FATAL: no trajectory file at $SIM"
    echo "       Available in $SIM_DIR:"
    ls -1 "$SIM_DIR" 2>/dev/null | head -30
    echo "       -> wrong SIZE/DOMAIN, or set SIM_TAG=... (current: $SIM_TAG)"
    exit 1
  fi
  CELLS+=("$CELL")
  python - "$SIM" "$CELL" <<'PY'
import json, sys
p, cell = sys.argv[1], sys.argv[2]
d = json.load(open(p))
s = d if isinstance(d, list) else []
if isinstance(d, dict):
    for k in ("simulations", "results", "runs", "trajectories"):
        if isinstance(d.get(k), list):
            s = d[k]; break
print(f"  cell {cell}: {len(s)} trajectories")
PY
done
echo "preflight OK"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "PREFLIGHT_ONLY=1 -> stopping here. Paths resolve; safe to sbatch."
  exit 0
fi

echo "=== GPUs allocated ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# ---------------- helper: wait for one server ----------------
wait_for_server() {
  local label="$1" port="$2" pid="$3" ticks="$4" logfile="$5"
  local mins=$(( ticks * 10 / 60 ))
  echo "waiting for ${label} on port ${port} (budget ~${mins} min)..."
  local i
  for i in $(seq 1 "$ticks"); do
    if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
      echo "${label} ready on port ${port} after ~$((i*10))s."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${label} process died — see ${logfile}"
      tail -30 "$logfile"
      return 1
    fi
    sleep 10
  done
  echo "TIMED OUT waiting for ${label} after ~${mins} min — see ${logfile}"
  tail -30 "$logfile"
  return 1
}

RC=0

# ======================= STAGE: signals (G2 / G3 / G4) =======================
if [ "$STAGE" = "signals" ]; then
  for CELL in "${CELLS[@]}"; do
    SIM="$SIM_DIR/${CELL}.json"

    # G2 control: the grid's own embedder through THIS code path. Not a rerun —
    # it is what makes the EMB_ALT difference attributable to the embedder
    # rather than to a formula mismatch between this script and the pipeline.
    # Also emits d_oA_calibrated (G4), which is a post-hoc transform of these
    # same embeddings and therefore costs no extra pass.
    if [ "$SIGNALS_ONLY" = "all" ] || [ "$SIGNALS_ONLY" = "bgem3" ]; then
      echo "[G2/G4] ${CELL}: ${EMB_BASE} (baseline + calibrated gap)"
      python "$SCRIPTS/recompute_signals.py" \
        --sim-file "$SIM" --cell "$CELL" --embed-model "$EMB_BASE" \
        --variant baseline --out "$OUT/embed_${CELL}_bgem3_baseline.csv" || RC=1
    fi

    if [ "$SIGNALS_ONLY" = "all" ] || [ "$SIGNALS_ONLY" = "qwen3" ]; then
      echo "[G2]    ${CELL}: ${EMB_ALT}"
      python "$SCRIPTS/recompute_signals.py" \
        --sim-file "$SIM" --cell "$CELL" --embed-model "$EMB_ALT" \
        --variant baseline --out "$OUT/embed_${CELL}_qwen3emb_baseline.csv" || RC=1
    fi

    if [ "$SIGNALS_ONLY" = "all" ] || [ "$SIGNALS_ONLY" = "bgem3" ]; then
      echo "[G3]    ${CELL}: verbalized observations (${EMB_BASE})"
      python "$SCRIPTS/recompute_signals.py" \
        --sim-file "$SIM" --cell "$CELL" --embed-model "$EMB_BASE" \
        --variant verbalized --out "$OUT/embed_${CELL}_bgem3_verbalized.csv" || RC=1
    fi
  done
fi

# ======================= STAGE: precision (G1) =======================
# Int4 is re-scored too, and that is not redundant: the U already in the grid
# was collected inline during generation, so comparing it to a bf16 re-score
# would confound precision with code path. Same script, same trajectories, only
# precision moves. Both checkpoints load sequentially — one at a time gets full
# NFS bandwidth — and every requested domain is scored inside one load.
if [ "$STAGE" = "precision" ]; then

  # ---- PHASE 1: Int4 (the control) ----
  INT4_LOG="logs/ablate-vllm-int4-${SLURM_JOB_ID}.log"
  echo "Starting vLLM SCORER ($INT4_MODEL) on GPU 0, port $PORT ..."
  vllm serve "$INT4_MODEL" \
    --served-model-name scorer --port "$PORT" \
    --quantization gptq \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    > "$INT4_LOG" 2>&1 &
  INT4_PID=$!

  if wait_for_server "SCORER(${SIZE}-int4)" "$PORT" "$INT4_PID" "$INT4_WAIT" "$INT4_LOG"; then
    for CELL in "${CELLS[@]}"; do
      echo "[G1a] ${CELL}: int4 re-score"
      python "$SCRIPTS/rescore_surprisal.py" \
        --sim-file "$SIM_DIR/${CELL}.json" --cell "$CELL" \
        --api-base "http://localhost:${PORT}/v1" --served-name scorer \
        --tokenizer "$INT4_MODEL" --precision int4 \
        --out "$OUT/surprisal_${CELL}_int4.csv" || RC=1
    done
  else
    echo "WARNING: Int4 scorer failed to start — G1 has no control arm."
    RC=1
  fi
  kill "$INT4_PID" 2>/dev/null; wait "$INT4_PID" 2>/dev/null
  sleep 30   # let the card actually free before the next load

  # ---- PHASE 2: bf16 ----
  BF16_LOG="logs/ablate-vllm-bf16-${SLURM_JOB_ID}.log"
  echo "Starting vLLM SCORER ($BF16_MODEL) on GPU 0, port $PORT ..."
  # prompt-logprob re-scoring materializes a [prompt_len x vocab] fp32 logits
  # tensor; with vocab~152k a single long prompt spikes several GB. 0.92 util
  # left the 14B run with <500MB free and the engine OOM'd after 80 rows. Lower
  # the util for headroom, and cap the prefill batch so one scheduler step's
  # logits tensor is bounded regardless of prompt length.
  vllm serve "$BF16_MODEL" \
    --served-model-name scorer --port "$PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 --max-model-len 32768 \
    --max-num-batched-tokens 8192 --max-num-seqs 8 \
    > "$BF16_LOG" 2>&1 &
  BF16_PID=$!

  if wait_for_server "SCORER(${SIZE}-bf16)" "$PORT" "$BF16_PID" "$BF16_WAIT" "$BF16_LOG"; then
    for CELL in "${CELLS[@]}"; do
      echo "[G1b] ${CELL}: bf16 re-score"
      python "$SCRIPTS/rescore_surprisal.py" \
        --sim-file "$SIM_DIR/${CELL}.json" --cell "$CELL" \
        --api-base "http://localhost:${PORT}/v1" --served-name scorer \
        --tokenizer "$BF16_MODEL" --precision bf16 \
        --out "$OUT/surprisal_${CELL}_bf16.csv" || RC=1
    done
  else
    echo "WARNING: bf16 scorer failed to start — G1 is Int4-only."
    RC=1
  fi
  kill "$BF16_PID" 2>/dev/null; wait "$BF16_PID" 2>/dev/null
fi

# ======================= collect =======================
echo "=== SUMMARY TABLES ==="
python "$SCRIPTS/summarize_ablations.py" --dir "$OUT" --cells "${CELLS[@]}" \
  | tee "$OUT/summary_${STAGE}_${SIZE}_${SLURM_JOB_ID}.txt"

echo "=== done: node=${NODE} stage=${STAGE} size=${SIZE} domains=${DOMAINS[*]} rc=${RC} ==="
echo "Results in $OUT"
ls -lh "$OUT" | tail -20

# ---------------------------------------------------------------------------
# WHY G1 RUNS ON 14B
# The scorer has to be the model that produced the trajectories, so a bf16
# scorer requires a cell of that size — and 14B is the only size with a cached
# bf16 checkpoint. That is not a compromise: the statistic answering G1 is the
# paired per-step Spearman between Int4 and bf16, computed over thousands of
# steps, which is unaffected by the 6-12-success label problem in the 14B cells.
# Read that line in the summary; ignore the AUROC columns for 14B. A correlation
# of 0.95+ means precision is not rewriting surprisal, and the inversion is
# behavioural. Re-running on 32B needs a ~62GB download and would answer the
# same question.
# ---------------------------------------------------------------------------

