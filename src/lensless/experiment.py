"""Create traceable run folders: experiments/<name>/<YYMMDD>_<technique>_v<N>/."""
import json
import platform
import random
import re
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml


def create_run_dir(root: str | Path, name: str, technique: str, today: date | None = None) -> Path:
    """Create the next free version, so an earlier run is never overwritten."""
    stamp = (today or date.today()).strftime("%y%m%d")
    parent = Path(root) / name
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{stamp}_{re.escape(technique)}_v(\d+)$")
    versions = [int(m.group(1)) for p in parent.iterdir() if (m := pattern.match(p.name))]
    run_dir = parent / f"{stamp}_{technique}_v{max(versions, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def git_state() -> dict[str, object]:
    """Record the code version, since uncommitted changes would make a result untraceable."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "uncommitted_changes": None}


def write_run_metadata(run_dir: Path, config: dict, device: str) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "platform": platform.platform(),
    }
    (run_dir / "run_info.json").write_text(json.dumps({"git": git_state(), "environment": environment}, indent=2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
