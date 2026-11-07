import csv
import importlib.util
from pathlib import Path

import pytest
import torch


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_admm.py"
    spec = importlib.util.spec_from_file_location("run_admm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_run_writes_traceable_outputs(tmp_path):
    run_admm = load_script()
    config = {
        "experiment": {"root": str(tmp_path), "name": "smoke", "technique": "admm"},
        "seed": 1,
        "data": {"source": "synthetic", "synthetic_examples": 6, "val_size": 3, "split_seed": 1, "max_val_examples": 2, "num_workers": 0, "batch_size": 2},
        "psf": {"source": "synthetic", "seed": 1},
        "admm": {"tau": 1e-3, "mu1": 0.1, "mu2": 0.1, "mu3": 0.1, "iterations": 3, "eval_at": [1], "output_scaling": "max"},
    }
    run_dir = run_admm.run(config)
    for name in ("config.yaml", "run_info.json", "metrics.csv", "previews/iteration_0003.png"):
        assert (run_dir / name).exists(), name
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [row["iteration"] for row in rows] == ["1", "3"]
    assert float(rows[-1]["val_screen_psnr"]) > 0 and float(rows[-1]["val_seconds_per_image"]) > 0


def test_scale_output_uses_each_images_own_maximum():
    run_admm = load_script()
    recon = torch.stack([torch.full((3, 4, 4), 2.0), torch.full((3, 4, 4), 0.5)])
    scaled = run_admm.scale_output(recon, "max")
    assert torch.allclose(scaled, torch.ones_like(recon))
    assert torch.equal(run_admm.scale_output(recon, "none"), recon)
    with pytest.raises(ValueError):
        run_admm.scale_output(recon, "mean")
