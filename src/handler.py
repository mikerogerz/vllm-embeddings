import os
import time
import runpod

from dataclasses import asdict

from vllm import LLM, EngineArgs

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", None)
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
TRUST_REMOTE_CODE = os.environ.get('TRUST_REMOTE_CODE', 'False').lower() == 'true'

# Chunked processing configuration for handling long texts
ENABLE_CHUNKED_PROCESSING = os.environ.get("ENABLE_CHUNKED_PROCESSING", 'true').lower() == 'true'
MAX_EMBED_LEN = int(os.environ.get("MAX_EMBED_LEN", "3072000")) # Max tokens for embedding input
POOLING_TYPE = os.environ.get("POOLING_TYPE", "LAST")

llm = None

def initialize_model():
	global llm
	if llm is None:
		pooler_config = {
			"pooling_type": POOLING_TYPE,
			"use_activation": True,
			"enable_chunked_processing": ENABLE_CHUNKED_PROCESSING,
			"max_embed_len": MAX_EMBED_LEN
		}
		
		engine_args = EngineArgs(
			model=MODEL_NAME,
			runner="pooling",
			trust_remote_code=TRUST_REMOTE_CODE,
			max_model_len=-1,
			enforce_eager=True,
			gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
			download_dir=DOWNLOAD_DIR,
			pooler_config=pooler_config
		)
		
		try:
			llm = LLM(**asdict(engine_args))
		except Exception as e:
			print(f"Error loading model: {str(e)}")
			raise
	return llm

#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#
#    Returns:
#       Any: The result to be returned to the client
def handler(event):
	input = event['input']
	
	prompt = input.get('prompt')
	seconds = input.get('seconds', 0)
	
	# Convert input to list format
	if isinstance(prompt, str):
		texts = [prompt]
	elif isinstance(prompt, list):
		texts = prompt
	else:
		return {
			"error": "'input' must be a string or list of strings"
		}
	
	if len(texts) == 0:
		return {
			"error": "Empty input"
		}
	
	# Validate all inputs are strings
	if not all(isinstance(text, str) for text in texts):
		return {
			"error": "All inputs must be strings"
		}
	
	# Get encoding format (default: float)
	encoding_format = input.get("encoding_format", "float")
	if encoding_format not in ["float", "base64"]:
		return {
			"error": "encoding_format must be 'float' or 'base64'"
		}
	
	model = initialize_model()
	
	# Log information about input lengths
	text_lengths = [len(text) for text in texts]
	print(f"Generating embeddings for {len(texts)} text(s)")
	print(f"Text lengths (chars): min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths)//len(text_lengths)}")
	
	# Check for potentially long texts
	long_text_threshold = model.llm_engine.model_config.max_model_len * 3  # Rough char estimate
	long_texts = [i for i, length in enumerate(text_lengths) if length > long_text_threshold]
	if long_texts and ENABLE_CHUNKED_PROCESSING:
		print(f"Detected {len(long_texts)} potentially long text(s) - chunked processing will handle automatically")
	
	start_time = time.time()
	outputs = model.embed(texts, use_tqdm=False)
	inference_time = time.time() - start_time
	
	embeddings = [output.outputs.embedding for output in outputs]
	
	data = []
	for idx, embedding in enumerate(embeddings):
		data.append({
			"object": "embedding",
			"embedding": embedding,
			"index": idx
		})
	
	# Estimate token count (rough approximation)
	total_chars = sum(len(text) for text in texts)
	estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
	
	response = {
		"object": "list",
		"data": data,
		"model": MODEL_NAME,
		"usage": {
			"prompt_tokens": estimated_tokens,
			"total_tokens": estimated_tokens
		}
	}
	
	print(f"Generated {len(embeddings)} embeddings in {inference_time:.2f}s")
	print(f"Avg time per embedding: {inference_time/len(embeddings):.3f}s")
	
	return response

# Start the Serverless function when the script is run
if __name__ == '__main__':
	runpod.serverless.start({
		'handler': handler
	})