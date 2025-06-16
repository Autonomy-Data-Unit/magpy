# %% [markdown]
# # utils

# %%
#|default_exp utils

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
import magpy as proj
from magpy import const

# %%
import magpy.utils as this_module

# %%
show_doc(this_module.foo)


# %%
#|export
def foo():
    print("Hello")
