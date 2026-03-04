from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def _merge_platform_dicts(
    base: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge platform dictionaries with nested slurm support."""
    merged = dict(base or {})
    for key, value in (overrides or {}).items():
        if key == "slurm" and isinstance(value, dict) and isinstance(merged.get("slurm"), dict):
            slurm_cfg = dict(merged.get("slurm") or {})
            slurm_cfg.update(value)
            merged["slurm"] = slurm_cfg
        else:
            merged[key] = value
    return merged


def load_platform_definitions(
    platforms_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load platform definitions from a YAML file."""
    if platforms_file is None:
        platforms_file = Path(__file__).resolve().parent.parent / "platforms.yaml"
    else:
        platforms_file = Path(platforms_file)

    if not platforms_file.exists():
        return {}

    with open(platforms_file, "r") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        return {}

    return data


def resolve_platform_config(
    platform_cfg: Optional[Union[str, Dict[str, Any], "PlatformSpec"]],
    platforms_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Resolve a platform config against shared platform definitions."""
    if platform_cfg is None:
        return {}

    try:
        from .platformspec import PlatformSpec
    except Exception:
        PlatformSpec = None

    if PlatformSpec is not None and isinstance(platform_cfg, PlatformSpec):
        return asdict(platform_cfg)

    name = None
    overrides: Dict[str, Any] = {}

    if isinstance(platform_cfg, str):
        name = platform_cfg
    elif isinstance(platform_cfg, dict):
        name = platform_cfg.get("name")
        overrides = dict(platform_cfg.get("args", {}) or {})
        for key, value in platform_cfg.items():
            if key in ("name", "args"):
                continue
            overrides[key] = value
    else:
        return {}

    if name:
        platforms = load_platform_definitions(platforms_file)
        base = platforms.get(name, {}) if isinstance(platforms, dict) else {}
        merged = _merge_platform_dicts(base, overrides)
        merged.setdefault("name", name)
        return merged

    return overrides
