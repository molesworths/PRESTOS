"""Transport models package."""

from .TransportBase import TransportBase
from .Fingerprints import Fingerprints
from .Tglf import Tglf
from .Cgyro import Cgyro
from .Qlgyro import Qlgyro
from .Fixed import Fixed
from .Analytic import Analytic
from .CH_fingerprints import CH_fingerprints
from .utils import (
    _enforce_cgyro_thread_multiple,
    _format_namelist_value,
    _get_controls_path,
    _get_template_path,
    _load_controls,
    _load_template,
    _merge_controls_and_locpargen,
    _parse_namelist_lines,
    _pick_locpargen_file,
    _read_numeric_file,
    _resolve_gacode_executable,
    _resolve_locpargen_executable,
    _run_locpargen,
    _update_namelist_lines,
    _with_python_in_path,
)

TRANSPORT_MODELS = {
    "fingerprints": Fingerprints,
    "CH_fingerprints": CH_fingerprints,
    "tglf": Tglf,
    "cgyro": Cgyro,
    "qlgyro": Qlgyro,
    "fixed": Fixed,
    "analytic": Analytic,
}


def create_transport_model(config):
    """Factory to create a transport model instance using a config dict."""
    if isinstance(config, str):
        model_type = config
        kwargs = {}
    else:
        model_type = (config or {}).get("type", "fingerprints")
        kwargs = (config or {}).get("kwargs", {})

    cls = TRANSPORT_MODELS.get(model_type.lower())
    if cls is None:
        raise ValueError(f"Unknown transport model: {model_type}")
    return cls(**kwargs)


for _cls in (TransportBase, Fingerprints, Tglf, Cgyro, Qlgyro, Fixed, Analytic):
    _cls.__module__ = __name__

__all__ = [
    "TransportBase",
    "Fingerprints",
    "Tglf",
    "Cgyro",
    "Qlgyro",
    "Fixed",
    "Analytic",
    "TRANSPORT_MODELS",
    "create_transport_model",
    "_enforce_cgyro_thread_multiple",
    "_format_namelist_value",
    "_get_controls_path",
    "_get_template_path",
    "_load_controls",
    "_load_template",
    "_merge_controls_and_locpargen",
    "_parse_namelist_lines",
    "_pick_locpargen_file",
    "_read_numeric_file",
    "_resolve_gacode_executable",
    "_resolve_locpargen_executable",
    "_run_locpargen",
    "_update_namelist_lines",
    "_with_python_in_path",
]
