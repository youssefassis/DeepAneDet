import json

import pytest

from deepanedet.utils import load_training_config


@pytest.mark.parametrize("stored", [{}, {"scales": None}])
def test_training_config_defaults_to_one_cell_per_8_voxels(tmp_path, stored):
    (tmp_path / "ndl_config.json").write_text(json.dumps({"patch_shape": [96, 96, 100], **stored}))

    assert load_training_config(str(tmp_path))["scales"] == [12, 12, 13]


def test_training_config_keeps_its_scales(tmp_path):
    (tmp_path / "ndl_config.json").write_text(json.dumps({"patch_shape": [96, 96, 96], "scales": [6]}))

    assert load_training_config(str(tmp_path))["scales"] == [6]


def test_missing_training_config_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="ndl_config.json"):
        load_training_config(str(tmp_path))
