import os
import json
import logging
import glob

from huggingface_hub import snapshot_download
from utils import timer_decorator

BASE_DIR = "/"
TOKENIZER_PATTERNS = [["*.json", "tokenizer*"]]
MODEL_PATTERNS = [["*.safetensors"], ["*.bin"], ["*.pt"]]

@timer_decorator
def download(name, revision, cache_dir):
	pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
	try:
		for pattern_set in pattern_sets:
			path = snapshot_download(name, revision=revision, cache_dir=cache_dir, allow_patterns=pattern_set)
			for pattern in pattern_set:
				if glob.glob(os.path.join(path, pattern)):
					logging.info(f"Successfully downloaded {pattern} model files.")
					return path
	except ValueError:
		raise ValueError(f"No patterns matching {pattern_sets} found for download.")

if __name__ == "__main__":
	cache_dir = os.getenv("HF_HOME")
	model_name, model_revision = os.getenv("MODEL_NAME"), os.getenv("MODEL_REVISION") or None
	
	model_path = download(model_name, model_revision, cache_dir)
	
	metadata = {
		"MODEL_NAME": model_path,
		"MODEL_REVISION": os.getenv("MODEL_REVISION")
	}
	
	with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
		json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)