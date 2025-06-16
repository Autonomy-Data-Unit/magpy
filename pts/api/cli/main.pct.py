# %% [markdown]
# # main

# %%
#|default_exp cli.main

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
import typer
from typer import Argument, Option
from typing_extensions import Annotated
from types import FunctionType
from typing import Callable, Union, List
from pathlib import Path
import importlib

import magpy as proj
from magpy import const
from magpy.cli.app import app
from magpy.config import get_config, update_config_from_pairs, validate_config, load_config_from_path


# %% [markdown]
# ## Main command

# %%
#|export
@app.callback()
def entrypoint(
    ctx: typer.Context,
    config_path: str = typer.Option(None, "--config-path", help="Path to config file."),
    config_mods: List[str] = typer.Option(None, "--config", "-c", help="Set config values. Use key=value pairs. Run `list-config` to see all available config variables.")
):
    if config_path:
        load_config_from_path(config_path)
    if config_mods:
        update_config_from_pairs(config_mods)
    if config_path or config_mods:
        validate_config()
        importlib.reload(const)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# %%
# !magpy

# %% [markdown]
# ## `run-core`

# %%
#|export
@app.command(name='run-core')
def cli_run_core(
):
    """
    Run all scripts in the `core` submodule in alphanumerical order
    """
    proj.core.run_all()


# %% [markdown]
# ## `list-config`

# %%
#|export
@app.command(name='list-config')
def cli_list_config(
):
    """
    List all config variables.
    """
    from rich.table import Table
    from rich.console import Console
    from rich.markup import escape
    from pydantic import BaseModel

    table = Table(title=f"{proj.__name__} configuration schema")

    table.add_column("Config key", justify="left", style="cyan", no_wrap=True)
    table.add_column("Type", justify="left", style="magenta")

    def add_fields(field_key_prefix, model_class):
        for field_key, field_type in model_class.__annotations__.items():
            if issubclass(field_type, BaseModel):
                add_fields(field_key, field_type)
                continue
            _field_key = f'{field_key_prefix}.{field_key}' if field_key_prefix else field_key
            table.add_row(escape(_field_key), escape(str(field_type)))

    add_fields("", get_config().__class__)

    console = Console()
    console.print(table)


# %%
#|export
@app.command(name='config-path')
def cli_config_path(
):
    """
    Get the path to the configuration file.
    """
    from repos_to_obsidian_notes.config import config_path
    print(config_path)


# %%
cli_list_config()

# %% [markdown]
# ## `a-sub-command`
#
# Uncomment the following cell to add a subcommand to the CLI.

# %%
#|export
# @app.command(name='a-sub-command')
# def cli_a_sub_command(
#     postional_arg: Annotated[Union[List[str], None], Argument(help="This will be displayed in the help message.")] = None,
#     optional_arg: Annotated[Union[str,None], Option("-r", "--root", help="This is an optional argument.")] = None,
# ):
#     """
#     This is a subcommand. This text will be displayed in the help message.
#     """
#     print("Hello from a subcommand.")
