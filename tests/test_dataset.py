import numpy as np
import pytest

from deepanedet.training.dataset import NO_AUGMENTATION, Dataset, get_dataloaders

from conftest import TINY_TRAINING, compose_config


# an aneurysm straddling the border of the 8 mm patch centered at (20, 20, 20)
STRADDLING = {
    "status": False,
    "point": np.array([20.0, 20.0, 20.0]),
    "data": np.zeros((40, 40, 40), dtype=np.float32),
    "vox2met": np.eye(4),
    "aneurysms": np.array([[23.0, 20.0, 20.0], [26.0, 20.0, 20.0]]),
}


def test_dataloaders_include_validation(patient_db, tmp_path):
    cfg = compose_config("train", [*TINY_TRAINING, f"data_dir={tmp_path}"])

    loaders, _ = get_dataloaders(train_db=patient_db[:1], valid_db=patient_db[1:], cfg=cfg)

    batch = next(iter(loaders["valid"]))

    assert batch["volume"].shape[1:] == (1, 16, 16, 16)


def dataset(negative):
    no_flip = {"probability": 0, "axes": [0]}
    return Dataset([dict(STRADDLING)], [8, 8, 8], [16, 16, 16], negative, NO_AUGMENTATION, no_flip, scales=[2])


def test_patch_without_augmentation_is_kept_with_its_straddling_aneurysm_left_out():
    sample = dataset(NO_AUGMENTATION)[0]

    assert sample["volume"].shape == (1, 16, 16, 16) and sample["truth"][0][0].sum() == 0


def test_augmentation_that_cannot_avoid_a_straddling_aneurysm_fails_loudly():
    tiny_shift = {**NO_AUGMENTATION, "shift": 1e-6}

    with pytest.raises(RuntimeError, match="attempts"):
        dataset(tiny_shift)[0]
