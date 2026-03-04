from typing import List, Union


def _build_module_setup(modules: Union[List[str], str, None]) -> str:
    """Normalize module setup to a shell command sequence.

    If the input looks like a list of raw module names (e.g. "gacode"), convert it to
    a module load command with common initialization for non-interactive shells.
    """
    if not modules:
        return ""
    if not isinstance(modules, list):
        modules = [modules]
    strip_module_list = [module.strip() for module in modules]
    if not strip_module_list:
        return ""

    return f"module load {' '.join(strip_module_list)}"
