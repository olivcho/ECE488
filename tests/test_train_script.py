import csv
import importlib.util
from pathlib import Path



def load_script():
    path = Path(__file__).parents[1] / "scripts" / "train_unet.py"
    spec = importlib.util.spec_from_file_location("train_unet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_run_writes_traceable_outputs(tmp_path):
    train_unet = load_script()
    config = {
        "experiment": {"root": str(tmp_path), "name": "smoke", "technique": "unet"},
        "data": {"source": "synthetic", "synthetic_examples": 4, "val_size": 2, "split_seed": 1, "max_train_examples": None,
                 "max_val_examples": None, "val_from_train": False, "num_workers": 0},
        "model": {"base_channels": 4, "depth": 4},
        "train": {"seed": 1, "epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "lpips_weight": 0.5, "eval_every": 1},
    }
    run_dir = train_unet.run(config)
    for name in ("config.yaml", "run_info.json", "metrics.csv", "checkpoints/best.pt", "previews/epoch_0002.png"):
        assert (run_dir / name).exists(), name
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert float(rows[-1]["val_screen_psnr"]) > 0


def test_overfit_mode_scores_training_images():
    train_unet = load_script()
    train, val = train_unet.build_datasets({"source": "synthetic", "synthetic_examples": 6, "val_size": 2, "split_seed": 1,
                                            "max_train_examples": 2, "max_val_examples": None, "val_from_train": True})
    assert train.indices == val.indices and len(train) == 2


def test_validation_stays_separate_by_default():
    train_unet = load_script()
    train, val = train_unet.build_datasets({"source": "synthetic", "synthetic_examples": 6, "val_size": 2, "split_seed": 1,
                                            "max_train_examples": None, "max_val_examples": None, "val_from_train": False})
    assert not set(train.indices) & set(val.indices)


def test_resume_continues_from_saved_epoch(tmp_path):
    train_unet = load_script()
    config = {
        "experiment": {"root": "unused", "name": "resume", "technique": "unet"},
        "data": {"source": "synthetic", "synthetic_examples": 4, "val_size": 2, "split_seed": 1, "max_train_examples": None,
                 "max_val_examples": None, "val_from_train": False, "num_workers": 0},
        "model": {"base_channels": 4, "depth": 4},
        "train": {"seed": 1, "epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "lpips_weight": 0.0, "eval_every": 1},
    }
    run_dir = train_unet.run(config, experiments_root=str(tmp_path))
    assert run_dir.parent.parent == tmp_path
    config["train"]["epochs"] = 3
    assert train_unet.run(config, resume_run=str(run_dir)) == run_dir
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    assert float(rows[0]["train_seconds"]) >= 0
    assert not list((run_dir / "checkpoints").glob("*.tmp"))
