#!/bin/bash
#SBATCH --job-name=tracer-smoke
#SBATCH --nodelist=romer3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/dl_scratch2/ege/elcin/TRACER-Reproduced/logs/tracer-%j.out
#SBATCH --error=/dl_scratch2/ege/elcin/TRACER-Reproduced/logs/tracer-%j.err

echo "=== GPU allocated ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

source $HOME/miniconda3/bin/activate
conda activate tracer
cd /dl_scratch2/ege/elcin/TRACER-Reproduced

export HF_HOME=/dl_scratch2/ege/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export LITELLM_LOG=CRITICAL

vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --served-model-name qwen --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.80 --max-model-len 32768 \
  > logs/vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!

echo "waiting for vLLM to load..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM failed to start — check logs/vllm-$SLURM_JOB_ID.log"; exit 1
  fi
  sleep 10
done
echo "vLLM ready."

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export TAU2_LLM=openai/qwen
export TRACER_EMBED_MODEL=BAAI/bge-m3

tau2 run --domain retail --num-tasks 20 --num-trials 1 --calculate-uncertainty

kill $VLLM_PID
echo "=== done ==="
