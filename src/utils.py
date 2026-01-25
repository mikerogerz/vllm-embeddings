import os
import logging

from http import HTTPStatus
from functools import wraps
from time import time
try:
	from vllm.utils import random_uuid
	from vllm.entrypoints.openai.protocol import ErrorResponse
	from vllm import SamplingParams
except ImportError:
	logging.warning("Error importing vllm, skipping related imports. This is ONLY expected when baking model into docker image from a machine without GPUs")
	pass

logging.basicConfig(level=logging.INFO)

def create_error_response(message: str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
	return ErrorResponse(message=message,
							type=err_type,
							code=status_code.value)

def timer_decorator(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time()
		result = func(*args, **kwargs)
		end = time()
		logging.info(f"{func.__name__} completed in {end - start:.2f} seconds")
		return result
	return wrapper