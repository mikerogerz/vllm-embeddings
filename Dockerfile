# Model baked INTO the image. No RunPod managed model cache, no network volume.
#
# Why: "image ready, initializing model files" is RunPod populating the host
# model cache before your container starts. Baking the weights into an image
# layer removes that phase entirely — when the image is ready, the model is
# already on disk. It also removes the partial-snapshot failure mode, since
# image layers are pulled atomically rather than assembled file-by-file.
#
# Cost: ~16 GB larger image and a slower FIRST pull per host. After that, hosts
# cache image layers, so subsequent cold starts are faster than cache population.
#
# IMPORTANT: detach the network volume AND clear the endpoint's "Model" field
# when using this image, or RunPod will still run the caching phase you're
# trying to avoid.

FROM vllm/vllm-openai:v0.27.1-cu129-ubuntu2404

RUN uv pip install --system --no-cache-dir "runpod>=1.8,<2.0" huggingface-hub

# Baked into the image at BUILD time, so these are ARGs, not runtime settings.
# Pin the revision so rebuilds are reproducible and match what you tested.
ARG MODEL_NAME="Qwen/Qwen3-Embedding-8B"
ARG MODEL_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

# Cache lives in the IMAGE, deliberately NOT under /runpod-volume — anything at
# that path is shadowed the moment a network volume mounts over it.
ENV HF_HOME="/opt/hf"
ENV HF_CACHE_ROOT="/opt/hf/hub"

# Fetch weights at build time. Only this step needs network; the running worker
# never does. Excluding non-safetensors formats avoids pulling duplicate
# variants of the same weights into the image.
RUN --mount=type=secret,id=hf_token,required=false \
	HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
	python3 -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download( \
	repo_id='${MODEL_NAME}', \
	revision='${MODEL_REVISION}', \
	cache_dir='/opt/hf/hub', \
	ignore_patterns=['*.pth','*.bin','*.msgpack','*.h5','*.onnx','*.gguf'], \
	max_workers=8, \
); \
print('baked snapshot:', p)"

# Fail the BUILD if the weights didn't land, rather than shipping a broken image
# that only fails at request time on a live worker.
RUN python3 -c "\
import glob, sys; \
s = glob.glob('/opt/hf/hub/models--*/snapshots/*/*.safetensors'); \
print(f'verified {len(s)} shard(s)'); \
sys.exit(0 if s else 1)"

ENV MODEL_NAME=$MODEL_NAME
ENV MODEL_REVISION=$MODEL_REVISION

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PYTHONUNBUFFERED=1

# Chunked processing configuration for long texts
ENV ENABLE_CHUNKED_PROCESSING="true"
ENV MAX_EMBED_LEN="3072000"
ENV POOLING_TYPE="LAST"

# Offline: the weights are local and complete, so nothing should ever be fetched.
ENV HF_HUB_OFFLINE=1 \
	TRANSFORMERS_OFFLINE=1

# Weights are on local disk in the image, so a stalled cache can't happen —
# keep the wait short and let a genuinely broken worker exit fast.
ENV CACHE_WAIT_SECONDS=30

ENV PYTHONPATH="/:/vllm-workspace"

COPY src .

ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]