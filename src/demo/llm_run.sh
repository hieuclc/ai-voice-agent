python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 7999 \
  --download-dir ./models \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4