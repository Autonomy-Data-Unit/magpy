import os, pkgutil, importlib, sys

def run_all():
    # Run all scripts in this folder in alphanumerical order
    scripts = list(f"{__package__}.{module_name}" for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))
    scripts = sorted(scripts)
    for script_full_import_path in scripts:
        script_name = script_full_import_path.split(".")[-1]
        if script_name.startswith("_"): continue
        print(f"{'#' * (len(script_full_import_path)+4)}")
        print(f"# {script_full_import_path} #")
        print(f"{'#' * (len(script_full_import_path)+4)}")
        if script_full_import_path in sys.modules:
            module = importlib.import_module(script_full_import_path)
            importlib.reload(module)
        else:
            importlib.import_module(script_full_import_path)