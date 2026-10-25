import pytest

from deepanedet.models.building_blocks import ExtResNetBlock


def test_resnet_block_without_activation_raises():
    with pytest.raises(ValueError, match="no activation"):
        ExtResNetBlock(1, 2, order="cb")


def test_default_scales_give_one_grid_matching_the_model_output(tmp_path):
    import json

    import torch

    from deepanedet.utils import get_model, load_training_config

    config = {
        "patch_shape": [32, 32, 32],
        "model": "ASSIS",
        "num parameters": 4,
        "in_channels": 1,
        "n_base_filters": 4,
        "layer_order": "cbl",
        "depth": 3,
        "basic_module": "ExtResNetBlock",
    }
    (tmp_path / "ndl_config.json").write_text(json.dumps(config))

    config = load_training_config(tmp_path)
    outputs = get_model(config)(torch.zeros(1, 1, 32, 32, 32))

    assert config["scales"] == [8]
    assert [output[0].shape[1:4] for output in outputs] == [(8, 8, 8)]
