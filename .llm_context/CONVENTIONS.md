
# CONVENTIONS

This document contains various conventions pertaining how you contribute to the repository, coding-related or otherwise.

## Writing code

The repository uses a very specific structure. All the package code and the test code reside in `REPO_FOLDER/magpy` and `REPO_FOLDER/tests` as is common in many Python codebases. However, this project uses `nblite` to do the development, which means that all the actual code-writing is done *in notebook form*. See (`.llm_context/nblite_README.md` for details on how `nblite` works).

All source code should be written in a notebook format, which will then be exported into the package code in `magpy` by running `nbl export`. There are three *code locations* for the package code in the repository:

- `pts`: Plaintext notebooks
- `nbs`: Juptyer notebook
- `magpy`: The actual package code, that is being exported from the notebooks.

The reason we have both plaintext notebooks and Jupyter notebooks is that the former is easier to manipulate for an LLM, and is also generally easier to deal with using `git`. The reason we have the latter is that it makes it easy for human developers to test the code interactively. Furthermore, running `nbl prepare` will execute all cells in the notebooks, which serves as a way to test the code (in addition to `pytest`).

The test code has a similar structure: `test_pts`, `test_nbs` and `magpy`.

What all the above means is that all development should be done in the `pts` and `test_pts` folders, for package and test development respectively. You should then run `nbl export` to export your changes to the `magpy` (and `nbs`) folder. To then test that the code is working you can run `nbl prepare` as well as `pytest`.

### VERY IMPORTANT: Run `nbl export --export-pipeline 'pts->nbs'` to export your code

Note that `nbl export` will export the code in `nbs` to `pts`. That means that if you write code in `pts`, you **MUST** run the reverse export `nbl export --export-pipeline 'pts->nbs'` for it to properly export *back* into `nbs`. If you run `nbl export` by itself, your code will be overwritten by the code in `nbs`. Furthermore, since most notebooks start with a `nblite.nbl_export()`, if you run a notebook without first having run the reverse export, your contribution will be overwritten.

### Testing

After implementing a new feature, it is good practice to write a test and then run the whole test suite by executing `pytest` in the repo root folder. To add a new test, add code to files in the `test_pts` folder.

### Difference between tests created in `test_pts` and unexported cells in `pts`

Running `pytest` tests all tests exported into `test`, and running `nbl prepare` will run all cells in all notebooks in `nbs`. In principle these are just two ways of defining tests. In practice however, you should see the unexported cells as rather opportunity to do *literate programming*. That is, you should not so much write tests, but rather write *example code* that shows how you use the functions you define, produce visualisations or other useful explanations.

## Submodule template

Any new submodule that you write should start of with the following general template:

```python
# %% [markdown]
# # app

# %%
#|default_exp MODULE_PATH

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
# Import statements here

# %%
#|export
# Code to be exported

```

The `MODULE_PATH` is the module path relative to the module root `magpy`. So if you are writing a new submodule `magpy.extract.schema` then you should have `extract.schema` as the module path.

## `_scratch` code

If you want to prototype or ideate code for a new feature before implementing it in full, you can create 'scratch' code in `pts/_scratch`. The basic form of a scratch file is:


```python
# %% [markdown]
# # TITLE

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
import magpy as proj
from magpy import const

# %%
# Code cells here and below
```

You should prepend scratch filenames with a number like `00_...pct.py`, `01_...pct.py` and so on. The title should reflect what you are doing in the file.


## Adding logs to `PROJECT_LOG.md`

After making a new contribution to the codebase, please write a summary of what you've done to `.llm_context/PROJECT_LOG.md`. Use the following format:

```markdown
## YYYY-MM-DD HH:MM:SS

Commit hash: {COMMIT_HASH}

Files edited:
- ...

Files created:
- ...

{Detailed summary of changes you've made.}
```