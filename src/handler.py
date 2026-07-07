import math
import os
import re
import threading
import time
import asyncio
import uuid
import base64
import struct
import runpod

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import PoolerConfig
from vllm.inputs import TextPrompt
from vllm.pooling_params import PoolingParams
from vllm.v1.engine.async_llm import AsyncLLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", None)
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "False").lower() == "true"
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 5))
ENABLE_CHUNKED_PROCESSING = os.environ.get("ENABLE_CHUNKED_PROCESSING", "true").lower() == "true"
MAX_EMBED_LEN = int(os.environ.get("MAX_EMBED_LEN", "3072000"))
POOLING_TYPE = os.environ.get("POOLING_TYPE", "LAST")
# Print the raw embedding tensor shape for the first request, to confirm the
# extraction assumption against the actual model/vLLM build in production.
EMBED_DEBUG = os.environ.get("EMBED_DEBUG", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Lazy async engine singleton
# ---------------------------------------------------------------------------

_detected_concurrency: int | None = None # set from vLLM's KV-cache log line
_engine: AsyncLLM | None = None
_max_model_len: int | None = None
_engine_lock = asyncio.Lock()
_debug_logged = False  # ensures the shape debug line prints only once

async def get_engine() -> tuple[AsyncLLM, int]:
	"""Initialise AsyncLLM once; subsequent calls return the cached instance."""
	global _engine, _max_model_len

	# Fast path – engine already ready
	if _engine is not None:
		return _engine, _max_model_len

	async with _engine_lock:
		# Second check inside the lock guards against simultaneous initialisers.
		# NOTE: this lock only protects one-time engine construction. It does
		# NOT wrap encode() — embedding requests run fully concurrently.
		if _engine is not None:
			return _engine, _max_model_len

		pooler_config = PoolerConfig(
			pooling_type=POOLING_TYPE,
			use_activation=True,
			enable_chunked_processing=ENABLE_CHUNKED_PROCESSING,
			max_embed_len=MAX_EMBED_LEN,
		)

		engine_args = AsyncEngineArgs(
			model=MODEL_NAME,
			runner="pooling",
			convert="embed",
			trust_remote_code=TRUST_REMOTE_CODE,
			max_model_len=None, # auto-detect from model config
			enforce_eager=True,
			enable_prefix_caching=True,
			gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
			download_dir=DOWNLOAD_DIR,
			pooler_config=pooler_config,
		)

		_engine = AsyncLLM.from_engine_args(engine_args)
		_max_model_len = _engine.model_config.max_model_len

		print(f"[init] Engine ready — model={MODEL_NAME}  max_model_len={_max_model_len}")

	return _engine, _max_model_len

# ---------------------------------------------------------------------------
# Embedding extraction + per-text helper
# ---------------------------------------------------------------------------

def _extract_embedding(output, request_id: str) -> list[float]:
	"""
	Pull the embedding vector out of a single request's PoolingRequestOutput.

	This output belongs to exactly one request_id — vLLM routes each request's
	results to its own encode() generator, which is what makes concurrent
	encode() calls safe. So whatever shape .data has, it is *this request's*
	data only; there is no cross-request mixing to defend against here.

	Shape handling (intra-request only):
	  • 1-D [hidden]        → the pooled embedding, use directly (normal case).
	  • 2-D [1, hidden]     → a leading singleton dim, take row 0.
	  • 2-D [seq, hidden]   → per-token rows; for LAST pooling the pooled vector
	                          is the final row.
	"""
	global _debug_logged

	data = output.outputs.data

	if EMBED_DEBUG and not _debug_logged:
		shape = getattr(data, "shape", f"(no shape, type={type(data).__name__})")
		print(f"[debug] request {request_id}: outputs.data shape = {shape}")
		_debug_logged = True

	if hasattr(data, "ndim"):
		if data.ndim == 1:
			vec = data
		elif data.shape[0] == 1:
			vec = data[0]
		else:
			# per-token rows; LAST pooling → final row
			vec = data[-1]
		return vec.tolist()

	# Fallbacks for list / other sequence types
	if data and isinstance(data[0], (list, tuple)):
		return list(data[-1])
	return list(data)

async def embed_text(
	engine: AsyncLLM,
	text: str,
	request_id: str,
	truncate_len: int,
) -> list[float]:
	"""
	Embed a single text string via its own encode() request.

	encode() is an async generator keyed to request_id; for pooling models it
	yields a single PoolingRequestOutput. Draining to the final output and
	extracting its data gives this request's embedding, independent of any
	other concurrent request.
	"""
	pooling_params = PoolingParams()
	final_output = None

	async for output in engine.encode(
		TextPrompt(prompt=text),
		pooling_params=pooling_params,
		request_id=request_id,
		tokenization_kwargs=dict(truncate_prompt_tokens=truncate_len),
	):
		final_output = output

	if final_output is None:
		raise RuntimeError(f"Engine returned no output for request '{request_id}'")

	return _extract_embedding(final_output, request_id)

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def to_base64(embedding: list[float]) -> str:
	"""Pack a float32 embedding as a base64 string (OpenAI-compatible format)."""
	return base64.b64encode(
		struct.pack(f"{len(embedding)}f", *embedding)
	).decode("utf-8")

# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

async def handler(job):
	"""
	Plain async handler — RunPod returns 'output' as a single object.

	Concurrency model:
	  • Multiple RunPod jobs run concurrently (concurrency_modifier > 1).
	  • Within a job, asyncio.gather fans out one encode() per text.
	  • All those requests — across every concurrent job — are visible to the
	    engine at once, and vLLM's continuous batching loop packs them into
	    shared GPU forward passes. THIS is what the KV-cache-derived
	    concurrency value is meant to exploit.
	  • Each request carries a globally-unique request_id (uuid4), so the
	    engine routes every result back to the correct caller with no chance
	    of id collision between jobs.
	"""
	job_input = job["input"]
	prompt    = job_input.get("prompt")

	# ---- Input validation ----

	if isinstance(prompt, str):
		texts = [prompt]
	elif isinstance(prompt, list):
		texts = prompt
	else:
		return {"error": "'prompt' must be a string or list of strings"}

	if not texts:
		return {"error": "Empty input"}

	if not all(isinstance(t, str) for t in texts):
		return {"error": "All items in 'prompt' must be strings"}

	encoding_format = job_input.get("encoding_format", "float")
	if encoding_format not in ("float", "base64"):
		return {"error": "encoding_format must be 'float' or 'base64'"}

	# ---- Engine init (idempotent) ----

	try:
		engine, max_model_len = await get_engine()
	except Exception as exc:
		return {"error": f"Engine initialisation failed: {exc}"}

	truncate_len = max_model_len - 1
	job_id       = job.get("id", uuid.uuid4().hex)

	text_lengths = [len(t) for t in texts]
	print(
		f"[{job_id}] Embedding {len(texts)} text(s) — "
		f"chars: min={min(text_lengths)}, max={max(text_lengths)}, "
		f"avg={sum(text_lengths) // len(text_lengths)}"
	)

	# ---- Concurrent embedding ----
	#
	# Each text gets a globally-unique request_id so results can never be
	# cross-routed between concurrent requests or concurrent jobs. gather
	# submits them all at once; the engine batches them on the GPU.

	start = time.time()

	try:
		embeddings: list[list[float]] = await asyncio.gather(*[
			embed_text(engine, text, f"{job_id}-{i}-{uuid.uuid4().hex}", truncate_len)
			for i, text in enumerate(texts)
		])
	except Exception as exc:
		return {"error": f"Embedding failed: {exc}"}

	elapsed = time.time() - start

	# ---- Sanity check: every embedding must have identical dimension ----
	# A ragged result would signal the extraction picked up the wrong shape.
	dims = {len(e) for e in embeddings}
	if len(dims) > 1:
		return {
			"error": f"Inconsistent embedding dimensions {sorted(dims)} — "
			         f"check outputs.data shape (set EMBED_DEBUG=true)"
		}

	# ---- Build response ----

	data = [
		{
			"object": "embedding",
			"index": idx,
			"embedding": to_base64(emb) if encoding_format == "base64" else emb,
		}
		for idx, emb in enumerate(embeddings)
	]

	estimated_tokens = sum(len(t) for t in texts) // 4  # ~4 chars per token

	print(
		f"[{job_id}] Done — {len(embeddings)} embedding(s) in {elapsed:.2f}s "
		f"({elapsed / len(embeddings):.3f}s avg, dim={dims.pop()})"
	)

	return {
		"object": "list",
		"data": data,
		"model": MODEL_NAME,
		"usage": {
			"prompt_tokens": estimated_tokens,
			"total_tokens": estimated_tokens,
		},
	}

# ---------------------------------------------------------------------------
# Dynamic concurrency — driven by vLLM's KV-cache analysis log line
# ---------------------------------------------------------------------------

_CONCURRENCY_PATTERN = re.compile(
	r'Maximum concurrency for [\d,]+ tokens per request:\s+([\d.]+)x'
)

class _FdCapture:
	"""
	Redirects stdout (fd 1) through a pipe and relays it in a daemon thread,
	scanning each line for the vLLM KV-cache concurrency value.

	The EngineCore subprocess writes directly to the inherited fd 1 — it does
	not forward records through the parent's Python logging system. This rules
	out logging.Handler (any logger), sys.stdout wrapping (StreamHandler caches
	the reference at dictConfig time), and logging.Handler on "vllm" (records
	never arrive there from the subprocess). Redirecting the raw fd before the
	fork is the only interception point that reliably captures the output.
	"""

	def __init__(self, target_fd: int) -> None:
		read_fd, write_fd = os.pipe()
		self._out = open(os.dup(target_fd), 'wb', buffering=0)
		os.dup2(write_fd, target_fd)
		os.close(write_fd)
		threading.Thread(
			target=self._relay,
			args=(open(read_fd, 'r', encoding='utf-8', errors='replace'),),
			daemon=True,
			name=f'fd{target_fd}-capture',
		).start()

	def _relay(self, pipe) -> None:
		for line in pipe:
			self._out.write(line.encode())
			m = _CONCURRENCY_PATTERN.search(line)
			if m:
				self._on_match(float(m.group(1)))

	def _on_match(self, raw: float) -> None:
		global _detected_concurrency
		_detected_concurrency = max(1, math.floor(raw))
		self._out.write(
			f'[concurrency] vLLM KV-cache reports {raw}x '
			f'→ RunPod concurrency = {_detected_concurrency}\n'
			.encode()
		)

def _install_concurrency_capture() -> None:
	"""
	Redirect stdout (fd 1) through _FdCapture before runpod.serverless.start().
	The EngineCore subprocess is forked during the first get_engine() call, so
	installing here ensures it inherits the redirected fd.
	"""
	_FdCapture(1)

def concurrency_modifier(current_concurrency: int) -> int:
	"""
	RunPod calls this periodically to ask how many concurrent jobs this worker
	should accept. Returns the GPU-derived value once vLLM's KV-cache analysis
	has run, otherwise falls back to the MAX_CONCURRENCY environment variable.

	Because encode() now runs concurrently (no lock, no serialising worker),
	accepting up to this many jobs lets vLLM batch their requests together on
	the GPU — the whole point of deriving the value from the KV-cache report.
	"""
	return _detected_concurrency if _detected_concurrency is not None else MAX_CONCURRENCY

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	# Start intercepting vLLM logs before the engine is created so we
	# don't miss the KV-cache concurrency line emitted at startup.
	_install_concurrency_capture()

	runpod.serverless.start({
		"handler": handler,
		"concurrency_modifier": concurrency_modifier,
	})