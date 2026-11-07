import json
from datetime import date

import yaml

from lensless.experiment import create_run_dir, write_run_metadata


def test_run_folders_follow_naming_and_never_collide(tmp_path):
    day = date(2026, 11, 6)
    assert create_run_dir(tmp_path, "baseline", "unet", day).name == "261106_unet_v1"
    assert create_run_dir(tmp_path, "baseline", "unet", day).name == "261106_unet_v2"
    assert create_run_dir(tmp_path, "baseline", "unet_small", day).name == "261106_unet_small_v1"
    assert (tmp_path / "baseline" / "261106_unet_v2").is_dir()


def test_metadata_records_config_and_code_version(tmp_path):
    write_run_metadata(tmp_path, {"train": {"seed": 1}}, "cpu")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["train"]["seed"] == 1
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert {"commit", "uncommitted_changes"} <= set(info["git"]) and info["environment"]["device"] == "cpu"
