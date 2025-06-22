# %% [markdown]
# # config

# %%
#|default_exp core.config

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
from pathlib import Path
from typing import Optional, Union
import adulib.llm
from adulib.caching import set_default_cache_path

# %%
#|export
_current_config = {}

def set_magpy_config(
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    cache_path: Union[str, Path, None] = None,
    caching: bool = True,
    **kwargs
):
    """Configure LLM settings for magpy.
    
    Args:
        api_key: API key for the LLM provider
        model_name: Name of the model to use (default: gpt-4o)
        temperature: Temperature for generation (default: 0.1)
        cache_path: Path to cache directory (default: None for default cache)
        caching: Whether to enable caching (default: True)
        **kwargs: Additional arguments passed to adulib.llm functions
    """
    global _current_config
    
    config = {
        'model': model_name,
        'temperature': temperature,
        'cache_enabled': caching,
        **kwargs
    }
    
    if api_key:
        config['api_key'] = api_key
    
    if caching:
        if cache_path:
            set_default_cache_path(Path(cache_path))
        else:
            raise ValueError("Cache path must be provided if caching is enabled.")
    
    _current_config = config

def get_llm_config():
    """Get current LLM configuration."""
    return _current_config.copy()

def get_model_name():
    """Get the currently configured model name."""
    return _current_config.get('model', 'gpt-4o')

def get_temperature():
    """Get the currently configured temperature."""
    return _current_config.get('temperature', 0.1)

def is_caching_enabled():
    """Check if caching is enabled."""
    return _current_config.get('cache_enabled', True)
