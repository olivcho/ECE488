import csv
import importlib.util
import json
from pathlib import Path

import torch


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "train_flow.py"
    spec = importlib.util.spec_from_file_location("train_flow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_config(root: Path) -> dict:
    return {
        "experiment": {"root": str(root), "name": "smoke", "technique": "flow"},
        "data": {"source": "synthetic", "synthetic_examples": 4, "val_size": 2, "split_seed": 1, "max_train_examples": None,
                 "max_val_examples": None, "val_from_train": False, "num_workers": 0},
        "model": {"base_channels": 4, "depth": 2, "time_dim": 16, "attention_heads": 2, "sample_steps": 2},
        "train": {"seed": 1, "epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "lpips_weight": 0.0, "eval_every": 1},
    }


def test_full_run_writes_traceable_outputs(tmp_path):
    train_flow = load_script()
    run_dir = train_flow.run(synthetic_config(tmp_path))
    assert run_dir.name.endswith("_flow_v1")
    for name in ("config.yaml", "run_info.json", "metrics.csv", "checkpoints/best.pt", "previews/epoch_0002.png"):
        assert (run_dir / name).exists(), name
    assert "commit" in json.loads((run_dir / "run_info.json").read_text())["git"]
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert float(rows[-1]["val_screen_psnr"]) > 0
    saved = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert saved["config"]["experiment"]["technique"] == "flow"
    assert any(name.startswith("velocity.attention.") for name in saved["model"])


def test_resume_continues_from_saved_epoch(tmp_path):
    train_flow = load_script()
    config = synthetic_config(tmp_path)
    config["train"]["epochs"] = 1
    run_dir = train_flow.run(config)
    config["train"]["epochs"] = 2
    assert train_flow.run(config, resume_run=str(run_dir)) == run_dir
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
