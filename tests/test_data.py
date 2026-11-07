import numpy as np
import pytest
import torch

from lensless.data import LensedPairDataset, image_to_tensor, split_train_val, train_shard_paths


def test_split_is_disjoint_complete_and_repeatable():
    train, val = split_train_val(100, 10, seed=1)
    assert len(val) == 10 and not set(train) & set(val) and set(train) | set(val) == set(range(100))
    assert (train, val) == split_train_val(100, 10, seed=1)
    assert val != split_train_val(100, 10, seed=2)[1]


@pytest.mark.parametrize("val_size", [0, 100])
def test_split_rejects_empty_sides(val_size):
    with pytest.raises(ValueError):
        split_train_val(100, val_size, seed=1)


def test_image_to_tensor_gives_channels_first_unit_range():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 1, 2] = 255
    tensor = image_to_tensor(image)
    assert tensor.shape == (3, 2, 3) and tensor.dtype == torch.float32
    assert tensor[2, 0, 1] == 1.0 and tensor.sum() == 1.0


def test_image_to_tensor_rejects_wrong_inputs():
    with pytest.raises(TypeError):
        image_to_tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        image_to_tensor(np.zeros((2, 2), dtype=np.uint8))


def test_dataset_returns_selected_rows_in_order():
    examples = [{"lensless": np.full((2, 2, 3), i, np.uint8), "lensed": np.full((2, 2, 3), i + 10, np.uint8)} for i in range(5)]
    dataset = LensedPairDataset(examples, [4, 1])
    assert len(dataset) == 2
    assert dataset[0][0][0, 0, 0] == pytest.approx(4 / 255)
    assert dataset[1][1][0, 0, 0] == pytest.approx(11 / 255)


def test_shard_paths():
    assert train_shard_paths(1) == ["data/train-00000-of-00015.parquet"]
    with pytest.raises(ValueError):
        train_shard_paths(16)
