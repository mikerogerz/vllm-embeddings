FROM vllm/vllm-openai:v0.25.1-cu129-ubuntu2404

RUN uv pip install --system --no-cache-dir "runpod>=1.8,<2.0" huggingface-hub hf-transfer

# MODEL_NAME / MODEL_REVISION are RUNTIME settings — set them on the endpoint
# (Manage -> Edit Endpoint -> Environment Variables). MODEL_NAME must match the
# HF id you put in the endpoint's *Model* field so RunPod caches the right repo.
ARG BASE_PATH="/runpod-volume"
ENV BASE_PATH=$BASE_PATH

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PYTHONUNBUFFERED=1

# Chunked processing configuration for long texts
ENV ENABLE_CHUNKED_PROCESSING="true"
ENV MAX_EMBED_LEN="3072000"
ENV POOLING_TYPE="LAST"

# --- HuggingFace cache: RunPod's MANAGED model-cache location ---------------
# This is where RunPod's model caching feature (endpoint "Model" field) stores
# models: /runpod-volume/huggingface-cache/hub. HF_HUB_CACHE derives from
# HF_HOME as ${HF_HOME}/hub, so pointing HF_HOME one level up makes HF and vLLM
# read the SAME managed directory. Do NOT set a second, divergent cache path.
ENV HF_HOME="${BASE_PATH}/huggingface-cache"

# Load ONLY from the managed cache; never touch the network. This is what kills
# the sporadic "Cannot find any model weights": with offline forced, no worker
# ever falls through to a runtime download, so no cold-starting worker can glob
# a half-written snapshot. If the model isn't cached, the worker fails fast with
# a clear offline error instead of silently downloading.
ENV HF_HUB_OFFLINE=1 \
	TRANSFORMERS_OFFLINE=1 \
	HF_HUB_ENABLE_HF_TRANSFER=1

ENV PYTHONPATH="/:/vllm-workspace"

COPY src .

# NOTE: no build-time model download. RunPod populates the managed cache for
# you when the endpoint's "Model" field is set; a build-time download to a
# different path (the old HF_HOME=/runpod-volume/.cache/huggingface) was both
# wrong-path and shadowed by the volume mount at runtime. src/download_model.py
# is now unused and can be deleted.

ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]