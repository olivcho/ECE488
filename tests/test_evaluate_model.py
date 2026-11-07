import csv
import importlib.util
from pathlib import Path

import torch

from lensless.admm import ForwardModel, simulate_measurement
from lensless.data import select_split
from lensless.psf import transform_psf

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_split_gives_the_same_rows_to_every_model():
    examples = list(range(20))
    train = select_split(examples, "train", 5, 1, max_examples=3)
    val = select_split(examples, "val", 5, 1)
    test = select_split(examples, "test", 5, 1, test_examples=list(range(7)))
    assert len(train) == 3 and len(val) == 5 and len(test) == 7
    assert not set(train.indices) & set(val.indices)
    assert train.indices == select_split(examples, "train", 5, 1, max_examples=3).indices
    for bad in ({"split": "validation"}, {"split": "test"}):
        try:
            select_split(examples, bad["split"], 5, 1)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_transform_psf_rearranges_without_changing_mass():
    psf = torch.rand(3, 4, 6)
    for name in ("rotate180", "flip_horizontal", "flip_vertical"):
        moved = transform_psf(psf, name)
        assert moved.shape == psf.shape and torch.allclose(moved.sum(), psf.sum()) and not torch.equal(moved, psf)
    assert torch.equal(transform_psf(psf, "rotate180"), transform_psf(transform_psf(psf, "flip_horizontal"), "flip_vertical"))


def test_simulated_measurement_is_full_range_nonnegative_and_repeatable():
    psf = torch.rand(3, 8, 10) ** 4
    model = ForwardModel(psf / psf.sum(dim=(-2, -1), keepdim=True))
    scene = torch.rand(2, 3, 8, 10)
    first = simulate_measurement(model, scene, 0.05, torch.Generator().manual_seed(3))
    second = simulate_measurement(model, scene, 0.05, torch.Generator().manual_seed(3))
    assert first.shape == scene.shape and (first >= 0).all()
    assert torch.allclose(first.flatten(1).amax(dim=1), torch.ones(2))
    assert torch.equal(first, second)
    assert not torch.equal(first, simulate_measurement(model, scene, 0.0))


def evaluation_config(root: Path, checkpoint: Path, split: str = "val") -> dict:
    return {
        "experiment": {"root": str(root), "name": "evaluate", "technique": "evaluate"},
        "seed": 1,
        "checkpoint": str(checkpoint),
        "data": {"source": "synthetic", "synthetic_examples": 6, "split": split, "val_size": 3, "split_seed": 1, "max_examples": 2, "num_workers": 0, "batch_size": 2},
        "psf": {"source": "synthetic", "seed": 1},
        "psf_swap": {"new_psf": {"transform": "rotate180"}, "noise_std": 0.01},
    }


def test_le_admm_checkpoint_gets_all_four_conditions_on_the_test_split(tmp_path):
    train_le_admm, evaluate_model = load_script("train_le_admm"), load_script("evaluate_model")
    from test_train_le_admm import synthetic_config

    train_dir = train_le_admm.run(synthetic_config(tmp_path))
    run_dir = evaluate_model.run(evaluation_config(tmp_path, train_dir / "checkpoints" / "best.pt", split="test"))
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [row["condition"] for row in rows] == ["real", "simulated_same_psf", "new_psf_no_swap", "new_psf_swapped"]
    assert all(float(row["test_screen_psnr"]) > 0 for row in rows)
    assert (run_dir / "previews" / "new_psf_swapped.png").exists()
    assert rows[2]["test_screen_psnr"] != rows[3]["test_screen_psnr"]


def test_unet_checkpoint_gets_three_conditions(tmp_path):
    train_unet, evaluate_model = load_script("train_unet"), load_script("evaluate_model")
    config = {
        "experiment": {"root": str(tmp_path), "name": "unet", "technique": "unet"},
        "data": {"source": "synthetic", "synthetic_examples": 6, "val_size": 3, "split_seed": 1, "max_train_examples": None,
                 "max_val_examples": None, "val_from_train": False, "num_workers": 0},
        "model": {"base_channels": 4, "depth": 2},
        "train": {"seed": 1, "epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "lpips_weight": 0.0, "eval_every": 1},
    }
    train_dir = train_unet.run(config)
    run_dir = evaluate_model.run(evaluation_config(tmp_path, train_dir / "checkpoints" / "best.pt"))
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [row["condition"] for row in rows] == ["real", "simulated_same_psf", "new_psf_no_swap"]
    assert "val_screen_psnr" in rows[0]


def test_prepared_psf_is_recovered_exactly_from_the_forward_model():
    evaluate_model = load_script("evaluate_model")
    psf = torch.rand(3, 6, 8)
    recovered = evaluate_model.prepare_psf_from_model(ForwardModel(psf))
    assert torch.allclose(recovered, psf, atol=1e-6)


def test_flow_checkpoint_samples_under_three_conditions(tmp_path):
    train_flow, evaluate_model = load_script("train_flow"), load_script("evaluate_model")
    from test_train_flow import synthetic_config

    train_dir = train_flow.run(synthetic_config(tmp_path))
    run_dir = evaluate_model.run(evaluation_config(tmp_path, train_dir / "checkpoints" / "best.pt"))
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert [row["condition"] for row in rows] == ["real", "simulated_same_psf", "new_psf_no_swap"]
    assert all(float(row["val_screen_psnr"]) > 0 for row in rows)
