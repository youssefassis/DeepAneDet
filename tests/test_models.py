import pytest
import torch

from deepanedet.models.building_blocks import ExtResNetBlock
from deepanedet.utils import detection_grid_size, get_model

from conftest import compose_config


def test_resnet_block_without_activation_raises():
    with pytest.raises(ValueError, match="no activation"):
        ExtResNetBlock(1, 2, order="cb")


def test_the_detection_grid_matches_the_model_output(tmp_path):
    cfg = compose_config("train", [f"data_dir={tmp_path}", "name=exp", "model.f_maps=4", "model.num_levels=3"])

    outputs = get_model(cfg.model)(torch.zeros(1, 1, 32, 32, 32))

    assert [output[0].shape[1:4] for output in outputs] == [(8, 8, 8)]
    assert detection_grid_size(32, cfg.model.num_levels) == 8


def test_the_default_grid_has_12_cells_per_side(tmp_path):
    cfg = compose_config("train", [f"data_dir={tmp_path}", "name=exp"])

    assert detection_grid_size(cfg.data.patch_shape[0], cfg.model.num_levels) == 12
