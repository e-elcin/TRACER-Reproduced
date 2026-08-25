#!/bin/bash
#SBATCH --job-name=tracer_r3_smoke
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/TRACER-Workflow/logs/tracer_smoke_%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/TRACER-Workflow/logs/tracer_smoke_%j.err

set -Eeuo pipefail

source /home/ege/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/envs/misevolution_agent_safety

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/dl_scratch2/ege/elcin/TRACER-Workflow
MODEL=Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4
SIM="$REPO/data/uncertainty/workflow_misevolution/round_3/tracer_smoke_sim.json"
OUT="$REPO/data/uncertainty/workflow_misevolution/round_3/tracer_smoke_scores.csv"
PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_LOG="$REPO/logs/tracer_smoke_vllm_${SLURM_JOB_ID}.log"
VLLM_PID=""

mkdir -p "$REPO/logs" "$(dirname "$OUT")"
cd "$REPO"

cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! -s "$SIM" ]]; then
    echo "Missing smoke input: $SIM" >&2
    exit 1
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --quantization gptq \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on port $PORT"
READY=0
for attempt in $(seq 1 600); do
    if curl --fail --silent \
        "http://127.0.0.1:${PORT}/v1/models" >/dev/null
    then
        READY=1
        echo "vLLM ready after $attempt checks"
        break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited; see $VLLM_LOG" >&2
        exit 1
    fi
    sleep 10
done

if [[ "$READY" -ne 1 ]]; then
    echo "Timed out waiting for vLLM; see $VLLM_LOG" >&2
    exit 1
fi

python scripts/rescore_surprisal.py \
    --sim-file "$SIM" \
    --cell workflow_round_3_smoke \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --served-name "$MODEL" \
    --tokenizer "$MODEL" \
    --precision int4 \
    --pi0 0.9 \
    --out "$OUT" \
    --max-trajs 1

python - "$OUT" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 1:
    raise SystemExit(f"expected one score row, found {len(rows)}")

row = rows[0]
for field in ("U_all", "U_stopfilter", "U_content"):
    value = float(row[field])
    if not math.isfinite(value):
        raise SystemExit(f"{field} is not finite: {value}")

if int(row["n_tokens"]) <= 0 or int(row["n_content_tokens"]) <= 0:
    raise SystemExit("token counts must be positive")

print("TRACER smoke validation: PASS")
print("scores:", {
    key: row[key]
    for key in (
        "U_all",
        "U_stopfilter",
        "U_content",
        "n_tokens",
        "n_content_tokens",
    )
})
print("output:", path)
PY
