# %% [markdown]
# # magpy
#
# > A powerful Python library for extracting structured data from unstructured text using Large Language Models (LLMs). magpy provides a simple, flexible interface for converting free-form text into structured, machine-readable data formats.

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|hide
import magpy as proj

# %% [markdown]
# ## Installation
#
# ```bash
# pip install git+https://github.com/Autonomy-Data-Unit/magpy
# ```
#
# Execute the following commands in your terminal to quickly set up a testing environment where you can try out magpy's features:
#
# ```bash
# mkdir magpy-test # Create folder to test MagPy in
# cd magpy-test
# # Download example notebooks
# curl -O https://raw.githubusercontent.com/Autonomy-Data-Unit/magpy/refs/heads/main/nbs/examples/extract/00_data_extraction_from_text.ipynb
# curl -O https://raw.githubusercontent.com/Autonomy-Data-Unit/magpy/refs/heads/main/nbs/examples/extract/01_pdfs_to_csv.ipynb
# # Create a virtual environment and install dependencies
# python -m venv venv
# source ./venv/bin/activate
# pip install git+https://github.com/Autonomy-Data-Unit/magpy
# pip install jupyterlab
# # Create a .env file in the directory
# echo "OPENAI_API_KEY=" > .env
# # Run jupyterlab
# jupyter lab
# ```
#
# **Note:** To run the notebooks, you must first register for an API key from an LLM provider. In the example scripts we are using OpenAI's models, so you'll need to get an API key from https://openai.com/api/, and then paste it into the `.env` file created in one of the commands above. As Jupyter does not by default display hidden files you cannot edit `.env` from within Jupyter Lab. On Mac you can open the file by executing `open .env` in the terminal.

# %% [markdown]
# ## Quick Start
#
# For detailed examples see the [examples](./nbs/examples/) folder.
#
# ### Basic Usage
#
# ```python
# from magpy import extract_structured, set_magpy_config
# import os
#
# # Configure the LLM
# set_magpy_config(
#     api_key=os.getenv("OPENAI_API_KEY"), 
#     model_name="gpt-4o",    
#     temperature=0.1,
#     cache_path='.cache'
# )
#
# # Define your extraction schema
# schema = {
#     "name": str,
#     "amount": int,
#     "date": datetime,
#     "category": str
# }
#
# # Extract structured data from text
# text = "John Doe donated $500 to the charity on 2024-01-15 for education programs."
# result = extract_structured(text=text, schema=schema)
# print(result)
# # Output: {'name': 'John Doe', 'amount': 500, 'date': datetime(2024, 1, 15), 'category': 'education'}
# ```

# %% [markdown]
# ## Contributing
#
# To contribute to the development of the package, follow the below instructions.
#
# ### Prerequisites
#
# - Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
# - Install [direnv](https://direnv.net/) to automatically load the project virtual environment when entering it.
#     - Mac: `brew install direnv`
#     - Linux: `curl -sfL https://direnv.net/install.sh | bash`
#
# ### Setting up the environment
#
# Run the following:
#
# ```bash
# # In the root of the repo folder
# uv sync --all-extras # Installs the virtual environment at './.venv'
# direnv allow # Allows the automatic running of the script './.envrc'
# nbl install-hooks # Installs a git hooks that ensures that notebooks are added properly
# ```
#
# You are now set up to develop the codebase.
#
# Further instructions:
#
# - To export notebooks run `nbl export`.
# - To clean notebooks run `nbl clean`.
# - To see other available commands run just `nbl`.
# - To add a new dependency run `uv add package-name`. See the the [uv documentation](https://docs.astral.sh/uv/) for more details.
# - You need to `git add` all 'twinned' notebooks for the commit to be validated by the git-hook. For example, if you add `nbs/my-nb.ipynb`, you must also add `pts/my-nb.pct.py`.
# - To render the documentation, run `nbl render-docs`. To preview it run `nbl preview-docs`
# - To upgrade all dependencies run `uv sync --upgrade --all-extras`
# <!-- #endregion -->
#
# ## Support
#
# For questions, issues, or contributions, please open an issue on the GitHub repository.
