import csv
import importlib.util
from pathlib import Path

import torch


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "train_le_admm.py"
    spec = importlib.util.spec_from_file_location("train_le_admm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_config(root: Path, denoiser=None) -> dict:
    return {
        "experiment": {"root": str(root), "name": "smoke", "technique": "le_admm"},
        "data": {"source": "synthetic", "synthetic_examples": 4, "val_size": 2, "split_seed": 1, "max_train_examples": None,
                 "max_val_examples": None, "val_from_train": False, "num_workers": 0},
        "psf": {"source": "synthetic", "seed": 1},
        "model": {"iterations": 2, "tau": 1e-3, "mu": 0.03, "denoiser": denoiser},
        "train": {"seed": 1, "epochs": 2, "batch_size": 2, "learning_rate": 1e-2, "lpips_weight": 0.0, "eval_every": 1},
    }


def test_full_run_writes_traceable_outputs_and_learns_penalties(tmp_path):
    train_le_admm = load_script()
    run_dir = train_le_admm.run(synthetic_config(tmp_path))
    for name in ("config.yaml", "run_info.json", "metrics.csv", "checkpoints/best.pt", "previews/epoch_0002.png"):
        assert (run_dir / name).exists(), name
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    saved = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert set(saved) == {"log_tau", "log_mu", "log_gain"}
    assert not torch.allclose(saved["log_mu"], torch.full((2, 3), float(torch.log(torch.tensor(0.03)))))


def test_denoiser_variant_runs(tmp_path):
    train_le_admm = load_script()
    run_dir = train_le_admm.run(synthetic_config(tmp_path, denoiser={"base_channels": 4, "depth": 2}))
    saved = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert any(name.startswith("denoiser.") for name in saved)
