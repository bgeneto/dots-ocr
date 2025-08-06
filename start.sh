#!/bin/bash
set -e

echo 'Modifying vllm entrypoint...'
sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)

echo 'Starting vLLM server...'
vllm serve /workspace/weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code &

# Wait for vLLM to become available
until curl -f http://localhost:8000/v1/models > /dev/null 2>&1; do
  echo "Waiting for vLLM server..."
  sleep 5
done

echo "vLLM ready! Starting Streamlit..."
#exec streamlit run streamlit_app.py --browser.gatherUsageStats=false
exec python3 gradio_app.py

