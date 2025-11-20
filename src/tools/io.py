import numpy as np

def isfloat(x):
    return isinstance(x, (np.floating, float))


def isint(x):
    return isinstance(x, (np.integer, int))


def isnum(x):
    return isinstance(x, (np.floating, float)) or isinstance(x, (np.integer, int))


def islistarray(x):
    return type(x) in [list, np.ndarray]


def isAnyNan(x):
    try:
        aux = len(x)
    except:
        aux = None

    if aux is None:
        isnan = np.isnan(x)
    else:
        isnan = False
        for j in range(aux):
            isnan = isnan or np.isnan(x[j]).any()

    return isnan

def clipstr(txt, chars=40):
    if not isinstance(txt, str):
        txt = f"{txt}"
    return f"{'...' if len(txt) > chars else ''}{txt[-chars:]}" if txt is not None else None


def get_root():
    """Get the root directory of the SMARTS solver standalone package."""
    import os
    from pathlib import Path

    return Path(os.path.dirname(os.path.abspath(__file__))).parents[3]