FROM vllm/vllm-openai:v0.14.1

RUN mkdir -p /usr/lib64 && \
	ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib64/libcuda.so

RUN uv pip install --system --no-cache-dir "runpod>=1.8,<2.0" huggingface-hub hf-transfer

ARG MODEL_NAME="Qwen/Qwen3-Embedding-8B"
ARG BASE_PATH="/runpod-volume"
ARG MODEL_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
	MODEL_REVISION=$MODEL_REVISION \
	BASE_PATH=$BASE_PATH

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PYTHONUNBUFFERED=1

# Chunked processing configuration for long texts
ENV ENABLE_CHUNKED_PROCESSING="true"
ENV MAX_EMBED_LEN="3072000"
ENV POOLING_TYPE="LAST"

# HuggingFace cache configuration
# Models will be automatically downloaded to /runpod-volume
ENV HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
	HF_HOME="${BASE_PATH}/.cache/huggingface" \
	HF_HUB_ENABLE_HF_TRANSFER=0

ENV PYTHONPATH="/:/vllm-workspace"

COPY src .

RUN --mount=type=secret,id=HF_TOKEN,required=false \
	if [ -f /run/secrets/HF_TOKEN ]; then \
	export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
	fi && \
	if [ -n "$MODEL_NAME" ]; then \
	python3 ./download_model.py; \
	fi

ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]