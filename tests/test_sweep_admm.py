import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_sweep_config(root: Path) -> dict:
    return {
        "experiment": {"root": str(root), "name": "sweep_smoke", "technique": "admm"},
        "seed": 1,
        "data": {"source": "synthetic", "synthetic_examples": 6, "val_size": 3, "split_seed": 1, "max_val_examples": 2, "num_workers": 0, "batch_size": 2},
        "psf": {"source": "synthetic", "seed": 1},
        "admm": {"tau": None, "mu1": None, "mu2": None, "mu3": None, "iterations": 2, "eval_at": [1], "output_scaling": "max"},
        "sweep": {"tau": [1e-3, 2e-3], "mu": [0.1]},
    }


def test_overrides_replace_nested_values_without_touching_the_original():
    run_admm = load_script("run_admm")
    config = {"admm": {"tau": 1e-3, "mu1": 0.1}, "data": {"max_val_examples": 8}}
    key, value = run_admm.parse_override("admm.tau=3e-4")
    updated = run_admm.apply_overrides(config, {key: value, "data.max_val_examples": None})
    assert updated["admm"]["tau"] == pytest.approx(3e-4) and updated["data"]["max_val_examples"] is None
    assert config["admm"]["tau"] == 1e-3
    with pytest.raises(KeyError):
        run_admm.apply_overrides(config, {"admm.mu9": 1.0})
    with pytest.raises(ValueError):
        run_admm.parse_override("admm.tau")


def test_sweep_makes_one_run_per_point_and_skips_finished_points(tmp_path):
    sweep_admm = load_script("sweep_admm")
    config = synthetic_sweep_config(tmp_path)
    summary = sweep_admm.sweep(config)
    experiment_dir = tmp_path / "sweep_smoke"
    run_dirs = sorted(p for p in experiment_dir.iterdir() if p.is_dir())
    assert len(run_dirs) == 2
    rows = list(csv.DictReader(summary.open()))
    assert len(rows) == 4  # 2 settings x 2 scored iterations
    assert sorted({float(row["tau"]) for row in rows}) == [1e-3, 2e-3]
    assert all(float(row["mu1"]) == float(row["mu3"]) == 0.1 for row in rows)

    sweep_admm.sweep(config)
    assert sorted(p for p in experiment_dir.iterdir() if p.is_dir()) == run_dirs
