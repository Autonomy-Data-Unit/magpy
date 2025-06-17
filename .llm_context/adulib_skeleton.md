# git.py

__all__ = ['find_root_repo_path']
from pathlib import Path
import os

def find_root_repo_path(path=None):
    ...



# asynchronous.py

__all__ = ['is_in_event_loop', 'batch_executor']
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Callable, Tuple, Any, Dict, Iterable, Optional

def is_in_event_loop():
    ...

async def batch_executor(func: Callable, constant_kwargs: Dict[str, Any]={}, batch_args: Optional[Iterable[Tuple[Any, ...]]]=None, batch_kwargs: Optional[Iterable[Dict[str, Any]]]=None, concurrency_limit: Optional[int]=None, verbose: bool=True, progress_bar_desc: str='Processing'):
    """Executes a batch of asynchronous tasks.

Parameters:
- func (Callable): The asynchronous function to execute for each batch.
- constant_kwargs (Dict[str, Any], optional): Constant keyword arguments to pass to each function call.
- batch_args (Optional[Iterable[Tuple[Any, ...]]], optional): Iterable of argument tuples for each function call.
- batch_kwargs (Optional[Iterable[Dict[str, Any]]], optional): Iterable of keyword argument dictionaries for each function call.
- concurrency_limit (Optional[int], optional): Maximum number of concurrent tasks. If None, no limit is applied.
- verbose (bool, optional): If True, displays a progress bar. Default is True.
- progress_bar_desc (str, optional): Description for the progress bar. Default is "Processing".

Returns:
- List of results from the executed tasks.

Raises:
- ValueError: If both 'batch_args' and 'batch_kwargs' are empty or if their lengths do not match."""
    ...



# reflection.py

__all__ = ['is_valid_python_name', 'find_module_root', 'get_module_path_hierarchy', 'get_function_from_py_file', 'method_from_py_file', 'mod_property', 'cached_mod_property']
from pathlib import Path
import os
import sys
import importlib
import inspect
import types
import functools
import keyword
import re

def is_valid_python_name(name: str) -> bool:
    ...

def find_module_root(path):
    ...

def __get_module_path_hierarchy(path, hierarchy):
    ...

def get_module_path_hierarchy(path):
    ...

def get_function_from_py_file(file_path, func_name=None, args=[], is_async=False, return_func_key=''):
    ...

def method_from_py_file(file_path: str):
    ...

def update_module_class(mod):
    ...

def mod_property(func, cached=False):
    """Used to create module-level properties.

Example:
```python
@mod_property
def my_prop():
    print('my_prop called')
    return 42
```"""
    ...

def cached_mod_property(func):
    ...



# rest.py

__all__ = ['async_get', 'async_put', 'async_post', 'async_delete', 'get', 'post', 'put', 'delete', 'AsyncAPIHandler']
import requests
from urllib.parse import urljoin
import diskcache
import aiohttp
import tempfile
from asynciolimiter import Limiter

async def async_get(endpoint, params=None, headers=None):
    """Fetch data from a given RESTful API endpoint using an HTTP GET request.

:param endpoint: The API endpoint URL (string).
:param params: A dictionary of query parameters (default is None).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

async def async_put(endpoint, data=None, headers=None):
    """Update data at a given RESTful API endpoint using an HTTP PUT request.

:param endpoint: The API endpoint URL (string).
:param data: A dictionary of data to send in the body of the request (default is None).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

async def async_post(endpoint, data=None, headers=None):
    """Send data to a given RESTful API endpoint using an HTTP POST request.

:param endpoint: The API endpoint URL (string).
:param data: A dictionary of data to send in the body of the request (default is None).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

async def async_delete(endpoint, headers=None):
    """Delete a resource at a given RESTful API endpoint using an HTTP DELETE request.

:param endpoint: The API endpoint URL (string).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

def get(endpoint, params=None, headers=None):
    """Fetch data from a given RESTful API endpoint using an HTTP GET request.

:param endpoint: The API endpoint URL (string).
:param params: A dictionary of query parameters (default is None).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

def post(endpoint, data=None, headers=None):
    """Send data to a given RESTful API endpoint using an HTTP POST request.

:param endpoint: The API endpoint URL (string).
:param data: A dictionary of data to send in the body of the request (default is None).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

def put(endpoint, data=None, headers=None):
    """Update data at a given RESTful API endpoint using an HTTP PUT request.

:param endpoint: The API endpoint URL (string).
:param data: A dictionary of data to send in the body of the request (default is None).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

def delete(endpoint, headers=None):
    """Delete a resource at a given RESTful API endpoint using an HTTP DELETE request.

:param endpoint: The API endpoint URL (string).
:param headers: A dictionary of HTTP headers (default is None).
:return: The JSON response as a dictionary, or an error message."""
    ...

class AsyncAPIHandler:
    GET = 'get'
    PUT = 'put'
    POST = 'post'
    DELETE = 'delete'

    def __init__(self, base_url=None, default_params=None, default_headers=None, rate_limit=None, use_cache=True, cache_dir=None, call_quota=None):
        """A handler for making asynchronous API calls with support for caching, rate limiting, and default parameters.

:param base_url: The base URL of the API. This will be prepended to all endpoint calls.
:param default_params: A dictionary of default query parameters to be included in every request.
:param default_headers: A dictionary of default headers to be included in every request.
:param rate_limit: The rate limit for API calls, specified as the number of calls per second.
:param use_cache: A boolean indicating whether to enable caching of API responses.
:param cache_dir: The directory where cached responses will be stored. If None, a temporary directory will be created.
:param call_quota: An optional limit on the number of API calls that can be made. If None, there is no limit.

This class provides methods for making GET, POST, PUT, and DELETE requests asynchronously, while managing
caching and rate limiting. It also allows checking and clearing the cache for specific API calls."""
        ...

    @property
    def remaining_call_quota(self):
        ...

    def reset_quota(self):
        ...

    def __get_defaults(self, method, endpoint, params, headers):
        ...

    async def __load_cache_or_make_call(self, func, args, only_use_cache, cache_key):
        ...

    async def call(self, method, endpoint=None, params=None, data=None, headers=None, only_use_cache=False, **param_kwargs):
        """Make a request to the API.

:param method: The HTTP method to use (e.g., "get", "put", "post", "delete").
:param endpoint: The API endpoint to request.
:param params: A dictionary of query parameters for the request."""
        ...

    async def get(self, endpoint=None, params=None, headers=None, only_use_cache=False, **param_kwargs):
        ...

    async def put(self, endpoint=None, data=None, only_use_cache=False, headers=None):
        ...

    async def post(self, endpoint=None, data=None, only_use_cache=False, headers=None):
        ...

    async def delete(self, endpoint=None, only_use_cache=False, headers=None):
        ...

    def check_cache(self, method, endpoint=None, params=None, headers=None, **param_kwargs):
        ...

    def clear_cache_key(self, method, endpoint=None, params=None, headers=None, **param_kwargs):
        ...



# __init__.py

from .asynchronous import *
from .caching import *
from .cli import *
from .git import *
from .reflection import *
from .rest import *
from .utils import *



# caching.py

__all__ = ['set_default_cache_path', 'get_default_cache_path', 'get_default_cache', 'get_cache', 'clear_cache_key', 'is_in_cache', 'memoize']
import diskcache
from pathlib import Path
from diskcache.core import ENOVAL, args_to_key, full_name
import functools as ft
import asyncio
from typing import Union
from .utils import check_mutual_exclusivity
_caches = {}
_default_cache = None
_default_cache_path = None

def set_default_cache_path(cache_path: Path):
    """Set the path for the temporary cache."""
    ...

def get_default_cache_path() -> Path | None:
    """Set the path for the temporary cache."""
    ...

def _create_cache(cache_path: Union[Path, None]=None, temp: bool=False):
    """Creates a new cache with the right policies. This ensures that no data is lost as the cache grows."""
    ...

def get_default_cache():
    """Retrieve the default cache."""
    ...

def get_cache(cache_path: Union[Path, None]=None):
    """Retrieve a cache instance for the given path. If no path is provided, 
the default cache is used. If the cache does not exist, it is created 
using the specified cache path or the default cache path."""
    ...

def clear_cache_key(cache_key, cache: Union[Path, diskcache.Cache, None]=None):
    ...

def is_in_cache(key: tuple, cache: Union[Path, diskcache.Cache, None]=None):
    ...
__memoized_function_names = set()

def memoize(cache: Union[Path, diskcache.Cache, None]=None, temp=False, typed=True, expire=None, tag=None, return_cache_key=False):
    """Decorator for memoizing function results to improve performance.

This decorator stores the results of function calls, allowing subsequent
calls with the same arguments to retrieve the result from the cache instead
of recomputing it. You can specify a cache object or use a temporary cache
if none is provided.

Parameters:
- cache (Union[Path, diskcache.Cache, None], optional): A cache object or a
  path to the cache directory. Defaults to a temporary cache if None.
- temp (bool, optional): If True, use a temporary cache. Cannot be True if
  a cache is provided. Defaults to False.
- typed (bool, optional): If True, cache function arguments of different
  types separately. Defaults to True.
- expire (int, optional): Cache expiration time in seconds. If None, cache
  entries do not expire.
- tag (str, optional): A tag to associate with cache entries.
- return_cache_key (bool, optional): If True, return the cache key along
  with the result, in the order `(cache_key, result)`. Defaults to False.

Returns:
- function: A decorator that applies memoization to the target function."""
    ...



# llm/text_completions.py

"""See the [`litellm` documention](https://docs.litellm.ai/docs/text_completion)."""
__all__ = ['text_completion', 'async_text_completion']
try:
    import litellm
    import functools
    from adulib.llm._utils import _llm_func_factory, _llm_async_func_factory
    from adulib.llm.tokens import token_counter
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e
text_completion = _llm_func_factory(func=litellm.text_completion, func_name='text_completion', func_cache_name='text_completion', retrieve_log_data=lambda model, func_kwargs, response, cache_args: {'method': 'text_completion', 'input_tokens': token_counter(model=model, text=func_kwargs['prompt'], **cache_args), 'output_tokens': sum([token_counter(model=model, text=c.text, **cache_args) for c in response.choices]), 'cost': response._hidden_params['response_cost']})
text_completion.__doc__ = '\nThis function is a wrapper around a corresponding function in the `litellm` library, see [this](https://docs.litellm.ai/docs/text_completion) for a full list of the available arguments.\n'.strip()
async_text_completion = _llm_async_func_factory(func=functools.wraps(litellm.text_completion)(litellm.atext_completion), func_name='async_text_completion', func_cache_name='text_completion', retrieve_log_data=lambda model, func_kwargs, response, cache_args: {'method': 'text_completion', 'input_tokens': token_counter(model=model, text=func_kwargs['prompt'], **cache_args), 'output_tokens': sum([token_counter(model=model, text=c.text, **cache_args) for c in response.choices]), 'cost': response._hidden_params['response_cost']})
text_completion.__doc__ = '\nThis function is a wrapper around a corresponding function in the `litellm` library, see [this](https://docs.litellm.ai/docs/text_completion) for a full list of the available arguments.\n'.strip()



# llm/__init__.py

from .base import *
from .rate_limits import *
from .call_logging import *
from .caching import *
from ._utils import *
from .tokens import *
from .completions import *
from .text_completions import *
from .embeddings import *



# llm/tokens.py

__all__ = ['token_counter']
try:
    import litellm
    from adulib.llm._utils import _llm_func_factory
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e
token_counter = _llm_func_factory(func=litellm.token_counter, func_name='token_counter', func_cache_name='token_counter')



# llm/completions.py

"""Otherwise known as *chat completions*. See the [`litellm` documention](https://docs.litellm.ai/docs/completion)."""
__all__ = ['completion', 'async_completion', 'sig', 'single', 'async_single']
try:
    import litellm
    import inspect
    import functools
    from typing import List, Dict
    from adulib.llm._utils import _llm_func_factory, _llm_async_func_factory
    from adulib.llm.tokens import token_counter
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e
completion = _llm_func_factory(func=litellm.completion, func_name='completion', func_cache_name='completion', retrieve_log_data=lambda model, func_kwargs, response, cache_args: {'method': 'completion', 'input_tokens': token_counter(model=model, messages=func_kwargs['messages'], **cache_args), 'output_tokens': sum([token_counter(model=model, messages=[{'role': c.message.role, 'content': c.message.content}], **cache_args) for c in response.choices]), 'cost': response._hidden_params['response_cost']})
completion.__doc__ = '\nThis function is a wrapper around a corresponding function in the `litellm` library, see [this](https://docs.litellm.ai/docs/completion/input) for a full list of the available arguments.\n'.strip()
async_completion = _llm_async_func_factory(func=litellm.acompletion, func_name='async_completion', func_cache_name='completion', retrieve_log_data=lambda model, func_kwargs, response, cache_args: {'method': 'completion', 'input_tokens': token_counter(model=model, messages=func_kwargs['messages'], **cache_args), 'output_tokens': sum([token_counter(model=model, messages=[{'role': c.message.role, 'content': c.message.content}], **cache_args) for c in response.choices]), 'cost': response._hidden_params['response_cost']})
completion.__doc__ = '\nThis function is a wrapper around a corresponding function in the `litellm` library, see [this](https://docs.litellm.ai/docs/completion/input) for a full list of the available arguments.\n'.strip()

def _get_msgs(orig_msgs, response):
    ...

@functools.wraps(completion)
def single(prompt: str, model: str | None=None, system: str | None=None, *args, multi: bool | Dict | None=None, return_full_response: bool=False, **kwargs):
    ...
sig = inspect.signature(completion)
single.__signature__ = sig.replace(parameters=[p for p in sig.parameters.values() if p.name != 'messages'])
single.__name__ = 'prompt'
single.__doc__ = '\nSimplified chat completions designed for single-turn tasks like classification, summarization, or extraction. For a full list of the available arguments see the [documentation](https://docs.litellm.ai/docs/completion/input) for the `completion` function in `litellm`.\n'.strip()

@functools.wraps(completion)
async def async_single(prompt: str, model: str | None=None, system: str | None=None, *args, multi: bool | Dict | None=None, return_full_response: bool=False, **kwargs):
    ...
sig = inspect.signature(completion)
async_single.__signature__ = sig.replace(parameters=[p for p in sig.parameters.values() if p.name != 'messages'])
async_single.__name__ = 'async_prompt'
async_single.__doc__ = '\nSimplified chat completions designed for single-turn tasks like classification, summarization, or extraction. For a full list of the available arguments see the [documentation](https://docs.litellm.ai/docs/completion/input) for the `completion` function in `litellm`.\n'.strip()



# llm/embeddings.py

"""See the [`litellm` documention](https://docs.litellm.ai/docs/embedding/supported_embedding)."""
__all__ = ['embedding', 'async_embedding']
try:
    import litellm
    import functools
    from adulib.llm._utils import _llm_func_factory, _llm_async_func_factory
    from adulib.llm.tokens import token_counter
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e
embedding = _llm_func_factory(func=litellm.embedding, func_name='embedding', func_cache_name='embedding', retrieve_log_data=lambda model, func_kwargs, response, cache_args: {'method': 'embedding', 'input_tokens': sum([token_counter(model=model, text=inp, **cache_args) for inp in func_kwargs['input']]), 'output_tokens': None, 'cost': response._hidden_params['response_cost']})
embedding.__doc__ = '\nThis function is a wrapper around a corresponding function in the `litellm` library, see [this](https://docs.litellm.ai/docs/embedding/supported_embedding) for a full list of the available arguments.\n'.strip()
async_embedding = _llm_async_func_factory(func=functools.wraps(litellm.embedding)(litellm.aembedding), func_name='async_embedding', func_cache_name='embedding', retrieve_log_data=lambda model, func_kwargs, response, cache_args: {'method': 'embedding', 'input_tokens': sum([token_counter(model=model, text=inp, **cache_args) for inp in func_kwargs['input']]), 'output_tokens': None, 'cost': response._hidden_params['response_cost']})
embedding.__doc__ = '\nThis function is a wrapper around a corresponding function in the `litellm` library, see [this](https://docs.litellm.ai/docs/embedding/supported_embedding) for a full list of the available arguments.\n'.strip()



# llm/caching.py

__all__ = ['get_cache_key']
try:
    from pathlib import Path
    from typing import Dict, Union, Callable, Coroutine
    from adulib.caching import get_cache, clear_cache_key, is_in_cache
    from diskcache import ENOVAL
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e

def get_cache_key(model: str, func_name, content: any, key_prefix: Union[str, None]=None, include_model_in_cache_key: bool=True) -> tuple:
    ...

def _cache_execute(cache_key: tuple, execute_func: Callable, cache_enabled: bool=True, cache_path: Union[str, Path, None]=None):
    ...

async def _async_cache_execute(cache_key: tuple, execute_func: Callable, cache_enabled: bool=True, cache_path: Union[str, Path, None]=None):
    ...



# llm/rate_limits.py

__all__ = ['default_rpm', 'default_retry_on_exception', 'default_max_retries', 'default_retry_delay', 'default_timeout', 'set_default_request_rate_limit', 'set_request_rate_limit']
try:
    import litellm
    from asynciolimiter import Limiter
    import asyncio
    from typing import Dict, Literal, Union
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e
default_rpm = 1000
default_retry_on_exception = [litellm.RateLimitError, asyncio.TimeoutError]
default_max_retries = 5
default_retry_delay = 10
default_timeout = None
_request_rate_limiters: Dict[str, Limiter] = {}

def _convert_to_per_minute(rate: float, unit: Literal['per-second', 'per-minute', 'per-hour']='per-minute') -> float:
    ...

def set_default_request_rate_limit(request_rate: float, request_rate_unit: Literal['per-second', 'per-minute', 'per-hour']='per-minute'):
    ...

def _get_limiter(model: str, api_key: Union[str, None]=None) -> Limiter:
    ...

def set_request_rate_limit(model: str, api_key: str | None, request_rate: float, request_rate_unit: Literal['per-second', 'per-minute', 'per-hour']='per-minute'):
    ...



# llm/base.py

__all__ = ['available_models', 'search_models']
try:
    import litellm
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e
available_models = list(litellm.model_cost.keys())
available_models.remove('sample_spec')

def search_models(query: str):
    ...



# llm/_utils.py

__all__ = []
try:
    import litellm
    import inspect
    import time
    import asyncio
    from typing import Callable, Optional, Union
    from pathlib import Path
    from adulib.llm.caching import _cache_execute, _async_cache_execute, get_cache_key, is_in_cache
    from adulib.llm.call_logging import _log_call
    from adulib.llm.rate_limits import _get_limiter, default_retry_on_exception, default_max_retries, default_retry_delay, default_timeout
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e

class MaximumRetriesException(Exception):

    def __init__(self, retry_exceptions: list[Exception]):
        ...

def _llm_func_factory(func: Callable, func_name: str, func_cache_name: str, retrieve_log_data: Optional[Callable]=None):
    ...

def _llm_async_func_factory(func: Callable, func_name: str, func_cache_name: str, retrieve_log_data: Optional[Callable]=None):
    ...



# llm/call_logging.py

__all__ = ['CallLog', 'set_call_log_save_path', 'get_call_logs', 'get_total_costs', 'get_total_input_tokens', 'get_total_output_tokens', 'get_total_tokens', 'save_call_log', 'load_call_log_file']
try:
    from datetime import datetime, timezone
    from pydantic import BaseModel, Field
    from typing import List, Optional
    from pathlib import Path
    import json
    from adulib.llm.base import available_models
    import uuid
except ImportError as e:
    raise ImportError(f'Install adulib[llm] to use this API.') from e

class CallLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    model: str
    cost: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
_call_logs: List[CallLog] = []
_call_log_save_path: Optional[Path] = None

def set_call_log_save_path(path: Path):
    ...

def _log_call(**log_kwargs):
    ...

def get_call_logs(model: Optional[str]=None) -> List[CallLog]:
    ...

def get_total_costs(model: Optional[str]=None) -> float:
    ...

def get_total_input_tokens(model: Optional[str]=None) -> int:
    ...

def get_total_output_tokens(model: Optional[str]=None) -> int:
    ...

def get_total_tokens(model: Optional[str]=None) -> int:
    ...

def save_call_log(path: Path, combine_with_existing: bool=True):
    ...

def load_call_log_file(path: Optional[Path]=None) -> List[CallLog]:
    ...



# utils/daemons.py

__all__ = ['create_interval_daemon', 'create_watchdog_daemon']
import time
from pathlib import Path
from typing import Callable, Union, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread, Lock, Timer
from pathlib import Path
import datetime
import os

def create_interval_daemon(lock_file: str, callback: Callable[[], None], interval: float=1.0, verbose: bool=False, error_callback: Callable[[BaseException], None]=None) -> Callable[[], None]:
    """Creates a daemon that calls the callback function at fixed intervals.

Args:
    callback: The function to call at each interval.
    interval: Number of seconds between callbacks.
    verbose: Whether to print status messages.

Returns:
    A (start, stop) function pair for the daemon."""
    ...

def create_watchdog_daemon(folder_paths: Union[str, List[str]], lock_file: str, callback: Callable[[object], None], recursive: bool=True, verbose: bool=False, rate_limit: float=1) -> Callable[[], None]:
    """Starts a background daemon that watches `folder_paths` for changes.
Calls `callback(event)` whenever a file changes.

Args:
    folder_paths: A path or list of paths to watch.
    callback: The function to call when a file changes. Receives the event as argument.
    recursive: Whether to watch folders recursively.
    lock_file: Optional path to a lock file to ensure only one daemon is running.
    rate_limit: Minimum number of seconds between callbacks.

Returns:
    A (start, stop) function pair for the daemon."""
    ...



# utils/__init__.py

from .base import *
from .daemons import *



# utils/base.py

__all__ = ['as_dict', 'check_mutual_exclusivity', 'run_script']
import subprocess, os
from pathlib import Path
import tempfile

def as_dict(**kwargs):
    """Convert keyword arguments to a dictionary."""
    ...

def check_mutual_exclusivity(*args, check_falsy=True):
    """Check if only one of the arguments is falsy (or truthy, if check_falsy is False)."""
    ...

def run_script(script_path: Path, cwd: Path=None, env: dict=None, interactive: bool=False, raise_on_error: bool=True):
    """Execute a Python or Bash script with specified parameters and environment variables.

Args:
    script_path (Path): Path to the script file to execute (.py or .sh)
    cwd (Path, optional): Working directory for script execution. Defaults to None.
    env (dict, optional): Additional environment variables to pass to the script. Defaults to None.
    interactive (bool, optional): Whether to run the script in interactive mode. Defaults to False.
    raise_on_error (bool, optional): Whether to raise an exception on non-zero exit code. Defaults to True.

Returns:
    tuple: A tuple containing:
        - int: Return code from the script execution
        - str or None: Standard output (None if interactive mode)
        - bytes: Contents of the temporary output file

Raises:
    FileNotFoundError: If the specified script does not exist
    ValueError: If the script type is not supported (.py or .sh)
    Exception: If the script fails and raise_on_error is True

Notes:
    - The script's output can be captured in two ways:
      1. Through stdout/stderr capture when not in interactive mode
      2. Through a temporary file accessible via the OUT_PATH environment variable
    - In interactive mode, the script uses the parent process's stdin/stdout/stderr"""
    ...



# cli/__init__.py

from .base import *
from .data_questionnaire import data_questionnaire



# cli/data_questionnaire.py

__all__ = ['data_questionnaire']
from typing import Any, Dict, List, Union, Optional, Type, get_args, get_origin
import inspect
import enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin
from pydantic import BaseModel
import questionary
import inspect
import enum

def is_optional(field_type: Any) -> bool:
    ...

def get_inner_type(field_type: Any) -> Any:
    ...

def prompt_for_value(field_name: str, field_type: Any, default: Optional[Any]=None) -> Any:
    ...

def prompt_for_list(field_name: str, item_type: Any, initial: Optional[list]=None) -> list:
    ...

def prompt_for_dict(field_name: str, value_type: Any, initial: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    ...

def data_questionnaire(model_cls: Type[BaseModel], initial_data: Optional[Dict[str, Any]]=None, print_final: bool=True) -> BaseModel:
    ...



# cli/base.py

__all__ = ['run_fzf']
from typing import List, Optional

def run_fzf(terms: List[str], disp_terms: Optional[List[str]]=None):
    """Launches the fzf command-line fuzzy finder with a list of terms and returns
the selected term.

Parameters:
terms (List[str]): A list of strings to be presented to fzf for selection.

Returns:
str or None: The selected string from fzf, or None if no selection was made
or if fzf encountered an error.

Raises:
RuntimeError: If fzf is not installed or not found in the system PATH."""
    ...