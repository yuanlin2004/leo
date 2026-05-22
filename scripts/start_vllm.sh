docker run -d --name vllm --privileged --gpus all --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node \
  bash -c "vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 262144 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.5 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder"
