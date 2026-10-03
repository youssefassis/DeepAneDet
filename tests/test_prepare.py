import json

from prepare import save_config


def test_save_config_creates_the_training_directory(tmp_path):
    config = {"test_dir": str(tmp_path / "Work" / "Fold1"), "batch_size": 32}

    save_config(config)
    save_config(config)  # the directory now exists

    assert json.loads((tmp_path / "Work" / "Fold1" / "ndl_config.json").read_text()) == config
