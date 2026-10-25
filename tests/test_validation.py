"""Invalid settings and inputs fail with an explicit error, also when Python runs with -O."""

import numpy as np
import pytest
import torch

from deepanedet.data import io as dio
from deepanedet.training.losses import get_loss_criterion
from deepanedet.utils import create_lr_scheduler, get_model

from conftest import compose_config


def optimizer():
    return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError, match="Unsupported scheduler 'Cosine'"):
        create_lr_scheduler(optimizer(), {"name": "Cosine"}, epochs=1)


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="Unsupported loss function"):
        get_loss_criterion("Dice")


@pytest.mark.parametrize(
    "override, message",
    [
        ("model.basic_module=ResBlock", "basic_module"),
        ("model.pool_type=min", "pool_type"),
        ("model.layer_order=bl", "conv"),
    ],
)
def test_invalid_model_settings_are_rejected(tmp_path, override, message):
    cfg = compose_config("train", [f"data_dir={tmp_path}", "name=exp", "model.f_maps=4", override])

    with pytest.raises(Exception, match=message):  # Hydra wraps the model's ValueError
        get_model(cfg.model)


def test_unknown_normalization_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="normalize"):
        dio.read_patient_data(str(tmp_path / "volume.nii.gz"), None, normalize="MinMax")


def test_missing_volume_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="volume.nii.gz"):
        dio.read_nii_from_file(str(tmp_path / "volume.nii.gz"))


@pytest.mark.parametrize("normalize", [dio.rescale, dio.standardize])
def test_constant_volume_cannot_be_normalized(normalize):
    with pytest.raises(ValueError, match="constant volume"):
        normalize(np.ones((4, 4, 4), dtype=np.float32))
