import logging
import math
import os
import re
import sys
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

# ---------------------------------------------------------------------------
# Lazy async engine singleton
# ---------------------------------------------------------------------------

_detected_concurrency: int | None = None # set from vLLM's KV-cache log line
_engine: AsyncLLM | None = None
_max_model_len: int | None = None
_engine_lock = asyncio.Lock()

async def get_engine() -> tuple[AsyncLLM, int]:
	"""Initialise AsyncLLMEngine once; subsequent calls return the cached instance."""
	global _engine, _max_model_len

	# Fast path – engine already ready
	if _engine is not None:
		return _engine, _max_model_len

	async with _engine_lock:
		# Second check inside the lock to guard against simultaneous initialisers
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
# Per-text embedding helper
# ---------------------------------------------------------------------------

async def embed_text(
	engine: AsyncLLM,
	text: str,
	request_id: str,
	truncate_len: int,
) -> list[float]:
	"""
	Embed a single piece of text.

	vLLM's AsyncLLMEngine.encode() is itself an async generator that yields
	incremental EmbeddingRequestOutput objects as the request progresses through
	the engine's continuous batching loop.  We drain the generator and keep only
	the final output, which contains the complete embedding vector.
	"""
	pooling_params = PoolingParams()
	final_output = None

	async for output in engine.encode(
		TextPrompt(prompt=text),
		pooling_params=pooling_params,
		request_id=request_id,
		tokenization_kwargs=dict(truncate_prompt_tokens=truncate_len),
	):
		# Each iteration may be a partial/intermediate result; overwrite until done
		final_output = output

	if final_output is None:
		raise RuntimeError(f"Engine returned no output for request '{request_id}'")

	return final_output.outputs.data.tolist()

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def to_base64(embedding: list[float]) -> str:
	"""Pack a float32 embedding as a base64 string (OpenAI-compatible format)."""
	return base64.b64encode(
		struct.pack(f"{len(embedding)}f", *embedding)
	).decode("utf-8")

# ---------------------------------------------------------------------------
# RunPod handler  (async generator → enables streaming + concurrent jobs)
# ---------------------------------------------------------------------------

async def handler(job):
	"""
	RunPod serverless handler.

	Declaring this as an async generator (uses `yield`) lets RunPod:
	  • run multiple jobs concurrently within the same worker process, and
	  • stream partial results back to callers when appropriate.

	For embedding workloads we yield once with the complete response; the
	concurrent-request benefit comes from AsyncLLMEngine batching all
	per-text encode() calls that arrive during the same scheduling window.
	"""
	job_input = job["input"]
	prompt    = job_input.get("prompt")

	# ---- Input validation ----

	if isinstance(prompt, str):
		texts = [prompt]
	elif isinstance(prompt, list):
		texts = prompt
	else:
		yield {"error": "'prompt' must be a string or list of strings"}
		return

	if not texts:
		yield {"error": "Empty input"}
		return

	if not all(isinstance(t, str) for t in texts):
		yield {"error": "All items in 'prompt' must be strings"}
		return

	encoding_format = job_input.get("encoding_format", "float")
	if encoding_format not in ("float", "base64"):
		yield {"error": "encoding_format must be 'float' or 'base64'"}
		return

	# ---- Engine init (idempotent) ----

	try:
		engine, max_model_len = await get_engine()
	except Exception as exc:
		yield {"error": f"Engine initialisation failed: {exc}"}
		return

	truncate_len = max_model_len - 1
	job_id = job.get("id", str(uuid.uuid4()))

	text_lengths = [len(t) for t in texts]
	print(
		f"[{job_id}] Embedding {len(texts)} text(s) — "
		f"chars: min={min(text_lengths)}, max={max(text_lengths)}, "
		f"avg={sum(text_lengths) // len(text_lengths)}"
	)

	# ---- Concurrent embedding ----
	#
	# asyncio.gather fans out one engine.encode() coroutine per text.
	# Because AsyncLLMEngine runs a background scheduling loop, all of these
	# requests are visible to the engine at once and get batched together,
	# giving true GPU-level concurrency rather than sequential inference.

	start = time.time()

	try:
		embeddings: list[list[float]] = await asyncio.gather(*[
			embed_text(engine, text, f"{job_id}-{i}", truncate_len)
			for i, text in enumerate(texts)
		])
	except Exception as exc:
		yield {"error": f"Embedding failed: {exc}"}
		return

	elapsed = time.time() - start

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
		f"({elapsed / len(embeddings):.3f}s avg)"
	)

	yield {
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

# Compiled once at module level — used by the relay thread
_CONCURRENCY_PATTERN = re.compile(
	r'Maximum concurrency for [\d,]+ tokens per request:\s+([\d.]+)x'
)

class _FdCapture:
	"""
	Intercepts a raw file descriptor (stdout=1, stderr=2) at the OS level.

	Why fd-level and not sys.stdout / logging.Handler?
	- vLLM v1 runs its EngineCore in a forked subprocess (pid differs from
	  the main process).
	- On fork the child inherits the parent's raw file descriptors, so it
	  writes directly to fd 1 — without touching the parent's Python
	  sys.stdout object or the logging module at all.
	- The only way to see that output in the parent is to redirect the fd
	  itself to a pipe *before* the fork, then relay the bytes in a thread.

	How it works:
	  1. os.dup(target_fd)        → save a copy of the original fd
	  2. os.pipe()                → create (read_fd, write_fd)
	  3. os.dup2(write_fd, fd)    → fd now points at the write end
	  4. daemon thread reads from read_fd, forwards to saved_fd, scans lines
	"""

	def __init__(self, target_fd: int) -> None:
		self._saved_fd = os.dup(target_fd)   # preserve the real destination
		read_fd, write_fd = os.pipe()
		os.dup2(write_fd, target_fd)         # redirect fd → pipe write-end
		os.close(write_fd)                   # parent no longer needs this end
		self._read_fd = read_fd
		threading.Thread(
			target=self._relay,
			daemon=True,
			name=f'fd{target_fd}-capture',
		).start()

	def _relay(self) -> None:
		"""Read from pipe, forward verbatim, scan each line for the pattern."""
		buf = b''
		while True:
			try:
				chunk = os.read(self._read_fd, 4096)
			except OSError:
				break
			if not chunk:
				break
			os.write(self._saved_fd, chunk)   # pass through to real destination
			buf += chunk
			while b'\n' in buf:
				line, buf = buf.split(b'\n', 1)
				self._check(line.decode('utf-8', errors='replace'))

	def _check(self, line: str) -> None:
		global _detected_concurrency
		m = _CONCURRENCY_PATTERN.search(line)
		if m:
			raw = float(m.group(1))
			_detected_concurrency = max(1, math.floor(raw))
			os.write(
				self._saved_fd,
				(f'[concurrency] vLLM KV-cache reports {raw}x max '
					f'→ RunPod concurrency set to {_detected_concurrency}\n').encode(),
			)

def _install_concurrency_capture() -> None:
	"""
	Redirect stdout (fd 1) and stderr (fd 2) through _FdCapture pipes.
	Must be called before the vLLM engine is created (i.e. before the
	EngineCore subprocess is forked) so it inherits the redirected fds.
	"""
	_FdCapture(1)  # stdout
	_FdCapture(2)  # stderr

def concurrency_modifier(current_concurrency: int) -> int:
	"""
	RunPod calls this function periodically to ask how many concurrent
	jobs this worker should accept.

	Returns the GPU-derived value once vLLM's KV-cache analysis has run
	(emitted during engine initialisation), otherwise falls back to the
	MAX_CONCURRENCY environment variable.
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
		# Aggregate all yielded chunks so callers receive a single JSON response
		# rather than a raw newline-delimited stream.
		"return_aggregate_stream": True,
	})