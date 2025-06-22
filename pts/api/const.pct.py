# %% [markdown]
# # const

# %%
#|default_exp const

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
from pathlib import Path
import toml
from typing import Optional
from pydantic import BaseModel, ConfigDict, model_validator
from pathlib import Path
from importlib import resources
import adulib, adulib.llm
from dotenv import load_dotenv
import magpy as proj

# %% [markdown]
# Set up paths

# %%
#|export
scripts_path = resources.files(f'{proj.__name__}').joinpath('scripts').resolve()
assets_path = resources.files(f'{proj.__name__}').joinpath('assets').resolve()
