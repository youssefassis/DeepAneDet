import pytest

from deepanedet.utils import load_training_config

from conftest import compose_config, save_training_config


def test_training_config_paths_are_taken_from_the_launch_folder(tmp_path):
    cfg = compose_config("train", ["data_dir=Data", "data.split_file=reproducibility/fold1.json", "name=fold1"])
    save_training_config(tmp_path / "fold1", cfg, launch_dir="/runs")

    loaded = load_training_config(str(tmp_path / "fold1"))

    assert (loaded.data.data_dir, loaded.data.split_file) == ("/runs/Data", "/runs/reproducibility/fold1.json")


def test_missing_training_config_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="deepanedet-train"):
        load_training_config(str(tmp_path))
