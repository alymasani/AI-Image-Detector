"""Shared helpers: configuration loading, seeding, device selection, hashing, environment reporting."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- config


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config.

    A config may name a parent with ``inherits: other.yaml`` (resolved relative to the
    file). ``overrides`` are ``dotted.key=value`` strings parsed as YAML scalars, e.g.
    ``training.batch_size=16`` or ``data.jpeg_quality=null``.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path: absolute paths pass through, relative ones are relative to the repo."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else (base / p)


def portable_path(path: str | Path) -> str:
    """Path relative to the repository root when inside it (keeps exported results machine-independent)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_data_root(cfg: Dict[str, Any], cli_value: Optional[str] = None) -> Path:
    """Dataset root priority: CLI flag > GENIMAGE_ROOT env var > config ``data.root``."""
    value = cli_value or os.environ.get("GENIMAGE_ROOT") or cfg.get("data", {}).get("root", "data/genimage")
    return resolve_path(value)


# --------------------------------------------------------------------------- reproducibility


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA, MPS)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive NumPy/random seeds from the worker's torch seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(requested: str = "auto"):
    """Return a torch.device. ``auto`` prefers CUDA, then Apple MPS, then CPU."""
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return torch.device(requested)


# --------------------------------------------------------------------------- hashing / IO


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def git_commit() -> Optional[str]:
    """Current git commit hash, or None outside a repository."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # git missing is not an error for training
        return None


# --------------------------------------------------------------------------- environment


def environment_report() -> Dict[str, Any]:
    """Collect the actual software/hardware capabilities of this machine."""
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        import torchvision

        report.update(
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
            mps_available=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report.update(gpu_name=props.name, gpu_memory_gb=round(props.total_memory / 1024**3, 2),
                          gpu_count=torch.cuda.device_count())
    except ImportError:
        report["torch"] = None
    try:
        if platform.system() == "Darwin":
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
            report["system_memory_gb"] = round(int(mem) / 1024**3, 1)
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            report["cpu_name"] = chip or None
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            report["system_memory_gb"] = round(kb / 1024**2, 1)
    except Exception:
        pass
    return report


if __name__ == "__main__":
    print(json.dumps(environment_report(), indent=2))
