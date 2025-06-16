# %% [markdown]
# # config

# %%
#|default_exp config

# %%
#|hide
import nblite; from nblite import show_doc; nblite.nbl_export()
import magpy.config as this_module

# %%
#|export
from importlib import resources
import toml
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, model_validator, Field
from pathlib import Path
import magpy as proj

# %%
#|export
root_path = resources.files(f'{proj.__name__}').joinpath('').resolve()


# %%
#|export
class PathConfig(BaseModel):
    parent: Optional[str] = None
    path: Path
    
    @model_validator(mode='before')
    @classmethod
    def process_path(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = {'path': data}
        data['path'] = Path(data['path'])
        if not data.get('parent') and not Path(data['path']).is_absolute(): 
            raise ValueError(f"The path '{data}' must be an absolute path.")
        return data


# %%
#|export
class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    pkg_root_path: PathConfig = PathConfig(path=resources.files(f'{proj.__name__}').joinpath('').resolve())
    store_path: PathConfig
    caches_path: PathConfig
    data_path: PathConfig
    misc_path: PathConfig
    pre_output: PathConfig
    output_path: PathConfig
    
    def get_path(self, field: str) -> Path:
        field = getattr(self, field)
        if field.parent: return self.get_path(f'{field.parent}_path') / field.path
        else: return field.path


# %%
#|export
_config = None
def load_config_from_path(config_path: str):
    global _config
    toml_config = toml.load(config_path)
    _config = Config.model_validate(toml_config)
config_path = resources.files(f'{proj.__name__}').joinpath('assets/config.toml').resolve()
load_config_from_path(config_path)


# %%
#|export
def get_config(): return _config


# %%
#|export
def set_config(field_key: str, field_value: str):
    """
    Update the configuration by setting a specified field to a new value.

    This function allows updating nested fields within the configuration
    by using dot notation in the field_key. For example, to update a field
    within a sub-configuration, use 'sub_config.field_name'.

    Call `validate_config` after using to ensure that the model data is correct.

    Args:
        field_key (str): The key of the field to update. Can be a nested key
                         using dot notation.
        field_value (str): The new value to set for the specified field.

    Raises:
        AttributeError: If the specified field_key does not exist in the configuration.
    """
    def _set_config_helper(model, field_key: str, field_value: str):
        if '.' in field_key:
            field_parent_key, field_child_key = field_key.split('.', 1)
            _set_config_helper(getattr(model, field_parent_key), field_child_key, field_value)
            return
        setattr(model, field_key, field_value)
    _set_config_helper(_config, field_key, field_value)


# %%
#|export
def update_config_from_pairs(key_value_pairs: list[str]):
    for key_value_pair in key_value_pairs:
        key, value = key_value_pair.split('=', 1)
        set_config(key, value)


# %%
#|export
def validate_config():
    global _config
    _config = Config.model_validate(_config.model_dump())
