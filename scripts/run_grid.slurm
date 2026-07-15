#!/bin/bash
# ===========================================================================
# run_grid.slurm — TRACER grid, parametrized by NODE, agent size, domain
#
# Usage:  sbatch -w <NODE> run_grid.slurm <NODE> <AGENT_SIZE> <DOMAIN>
#   NODE       : adep2 | romer3
#   AGENT_SIZE : 14B | 32B | 72B   (ALL GPTQ-Int4 — same precision for a fair
#                                   size comparison; logprobs/surprisal are
#                                   quantization-sensitive, so precision is
#                                   held constant across the gradient)
#   DOMAIN     : retail | airline | telecom
# Example: sbatch -w adep2 run_grid.slurm adep2 14B telecom
#
# Judge is ALWAYS Qwen2.5-72B-Instruct-GPTQ-Int4, GPU 1, port 8001.
# Agent (model under test) on GPU 0, port 8000.
#
# Models load SEQUENTIALLY (judge first, then agent). Loading both at once
# saturates the NFS checkpoint storage and each load crawls (observed: 72B
# projecting ~2h when contended). Sequential = each gets full NFS bandwidth.
# ===========================================================================
#SBATCH --job-name=tracer-grid
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/TRACER-Reproduced/logs/tracer-%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/TRACER-Reproduced/logs/tracer-%j.err

set -uo pipefail

# ---------------- arguments ----------------
NODE="${1:?Usage: sbatch -w <node> run_grid.slurm <adep2|romer3> <14B|32B|72B> <retail|airline|telecom>}"
AGENT_SIZE="${2:?Usage: sbatch -w <node> run_grid.slurm <adep2|romer3> <14B|32B|72B> <retail|airline|telecom>}"
DOMAIN="${3:?Usage: sbatch -w <node> run_grid.slurm <adep2|romer3> <14B|32B|72B> <retail|airline|telecom>}"
NUM_TASKS=50

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
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# guard: submitted node must match the NODE argument (env/CUDA depend on it)
ACTUAL_NODE="$(hostname -s)"
if [ "$ACTUAL_NODE" != "$NODE" ]; then
  echo "REFUSING: running on '$ACTUAL_NODE' but NODE arg says '$NODE'."
  echo "  -> Submit with: sbatch -w $NODE run_grid.slurm $NODE $AGENT_SIZE $DOMAIN"
  exit 1
fi

# ---------------- map size -> checkpoint (ALL Int4) ----------------
# AGENT_WAIT / JUDGE_WAIT are per-model load budgets, in 10s ticks.
case "$AGENT_SIZE" in
  14B) AGENT_MODEL="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"; AGENT_GB=10; AGENT_WAIT=180 ;;  # ~30 min
  32B) AGENT_MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"; AGENT_GB=20; AGENT_WAIT=270 ;;  # ~45 min
  72B) AGENT_MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"; AGENT_GB=42; AGENT_WAIT=420 ;;  # ~70 min
  *)   echo "Bad AGENT_SIZE: $AGENT_SIZE (use 14B|32B|72B)"; exit 1 ;;
esac
AGENT_QUANT="--quantization gptq"
PREC="int4"

case "$DOMAIN" in
  retail|airline|telecom) ;;
  *) echo "Bad DOMAIN: $DOMAIN (use retail|airline|telecom)"; exit 1 ;;
esac

JUDGE_MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
JUDGE_QUANT="--quantization gptq"
JUDGE_GB=42
JUDGE_WAIT=420   # ~70 min — 72B over NFS is the slowest load

AGENT_PORT=8000
JUDGE_PORT=8001

# ---------------- fit guards ----------------
if [ "$AGENT_GB" -gt "$((GPU_MEM_GB - 10))" ]; then
  echo "REFUSING: agent ${AGENT_SIZE}(${PREC}) ~${AGENT_GB}GB won't fit one ${GPU_MEM_GB}GB card on ${NODE}."
  exit 1
fi
if [ "$JUDGE_GB" -gt "$((GPU_MEM_GB - 6))" ]; then
  echo "WARNING: 72B judge (~${JUDGE_GB}GB) is tight on ${GPU_MEM_GB}GB — watch for OOM."
fi

echo "=== RUN: node=${NODE} agent=${AGENT_SIZE}(${PREC}) domain=${DOMAIN} judge=72B(int4) tasks=${NUM_TASKS} ==="

# ---------------- conda env ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# fail in seconds, not after a 20-min model load
command -v tau2 >/dev/null 2>&1 || { echo "FATAL: tau2 not found in $CONDA_ENV"; exit 1; }
python -c "import litellm, sentence_transformers" 2>/dev/null || {
  echo "FATAL: litellm or sentence-transformers missing in $CONDA_ENV"
  echo "       (without sentence-transformers, Da/Do are SILENTLY ZERO — not an error)"
  exit 1
}
echo "env check OK: tau2 + litellm + sentence-transformers"

cd /dl_scratch2/ege/elcin/TRACER-Reproduced
mkdir -p logs

echo "=== GPUs allocated ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITELLM_LOG=CRITICAL

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

# ================= PHASE 1: JUDGE (loads first, alone) =================
JUDGE_LOG="logs/vllm-judge-${SLURM_JOB_ID}.log"
echo "Starting vLLM JUDGE ($JUDGE_MODEL) on GPU 1, port $JUDGE_PORT ..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$JUDGE_MODEL" \
  --served-model-name judge --port "$JUDGE_PORT" \
  $JUDGE_QUANT \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!

if ! wait_for_server "JUDGE(72B)" "$JUDGE_PORT" "$JUDGE_PID" "$JUDGE_WAIT" "$JUDGE_LOG"; then
  kill "$JUDGE_PID" 2>/dev/null
  exit 1
fi

# ================= PHASE 2: AGENT (loads only after judge is up) =================
AGENT_LOG="logs/vllm-agent-${SLURM_JOB_ID}.log"
echo "Starting vLLM AGENT ($AGENT_MODEL) on GPU 0, port $AGENT_PORT ..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$AGENT_MODEL" \
  --served-model-name qwen --port "$AGENT_PORT" \
  $AGENT_QUANT \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  > "$AGENT_LOG" 2>&1 &
AGENT_PID=$!

if ! wait_for_server "AGENT(${AGENT_SIZE})" "$AGENT_PORT" "$AGENT_PID" "$AGENT_WAIT" "$AGENT_LOG"; then
  kill "$AGENT_PID" "$JUDGE_PID" 2>/dev/null
  exit 1
fi

echo "both vLLM servers ready."

# ---------------- routing ----------------
export OPENAI_API_BASE=http://localhost:${AGENT_PORT}/v1
export OPENAI_API_KEY=dummy
export TAU2_LLM=openai/qwen
export TAU2_JUDGE_LLM=openai/judge
export TAU2_JUDGE_API_BASE=http://localhost:${JUDGE_PORT}/v1   # only effective if tau2 reads it — verified post-run below
export TRACER_EMBED_MODEL=BAAI/bge-m3

# ---------------- run ----------------
tau2 run --domain "$DOMAIN" --num-tasks "$NUM_TASKS" --num-trials 1 --calculate-uncertainty
RUN_RC=$?

# ---------------- judge-routing verification ----------------
JUDGE_HITS=$(grep -c "chat/completions" "$JUDGE_LOG" 2>/dev/null || echo 0)
echo "=== judge server received ${JUDGE_HITS} request(s) ==="
if [ "$JUDGE_HITS" -eq 0 ]; then
  echo "WARNING: judge got ZERO requests — judge calls likely went to the agent port."
  echo "         Labels for this run may NOT reflect the fixed 72B judge."
fi

# ---------------- shutdown ----------------
kill "$AGENT_PID" "$JUDGE_PID" 2>/dev/null
echo "=== done: node=${NODE} agent=${AGENT_SIZE}(${PREC}) domain=${DOMAIN} rc=${RUN_RC} ==="
