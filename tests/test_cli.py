"""The installed commands, the configs they read, and the pipeline they run."""

import json
import os
import shutil
import subprocess
from importlib.metadata import entry_points
from importlib.resources import files

import nibabel as ni
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from conftest import TINY_TRAINING, compose_config, run_command

COMMANDS = {
    "deepanedet-convert-annotations": "convert_annotations",
    "deepanedet-remove-skull": "remove_skull",
    "deepanedet-extract-points": "extract_points",
    "deepanedet-train": "train",
    "deepanedet-predict": "predict",
    "deepanedet-evaluate": "evaluate",
}


def test_every_command_is_installed():
    installed = {ep.name: ep for ep in entry_points(group="console_scripts") if ep.name.startswith("deepanedet-")}

    assert set(installed) == set(COMMANDS)
    assert all(callable(ep.load()) for ep in installed.values())


@pytest.mark.parametrize("config", sorted(COMMANDS.values()))
def test_every_command_has_its_config_in_the_package(config):
    assert (files("deepanedet.configs") / f"{config}.yaml").is_file()


@pytest.mark.parametrize("command", ["deepanedet-train", "deepanedet-predict"])
def test_commands_run_from_any_folder(command, tmp_path):
    result = subprocess.run([shutil.which(command), "--help"], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0 and "Configuration groups" in result.stdout


@pytest.mark.parametrize("name", ["train", "remove_skull", "extract_points", "convert_annotations"])
def test_configs_require_data_dir(name):
    with pytest.raises(Exception, match="data_dir"):
        OmegaConf.to_container(compose_config(name), resolve=True, throw_on_missing=True)


def test_train_config_defaults_match_the_paper_setup(tmp_path):
    cfg = compose_config("train", [f"data_dir={tmp_path}", "name=exp"])

    assert cfg.data.work_dir == f"{tmp_path}/0Work"
    assert list(cfg.data.patch_shape) == [96, 96, 96] and list(cfg.data.patch_size) == [38.4, 38.4, 38.4]
    assert cfg.loss == "YOLO_Loss"
    assert (cfg.optimizer.lr, cfg.optimizer.momentum, cfg.scheduler.name) == (1e-2, 0.98, "WarmupPoly")
    assert cfg.augmentation.positive.duplicates == 50


def write_patient(patient, aneurysm):
    patient.mkdir(parents=True)
    volume = np.random.default_rng(0).random((40, 40, 40)).astype(np.float32)
    ni.save(ni.Nifti1Image(volume, np.eye(4)), patient / "volume.nii.gz")
    (patient / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz"}))
    pd.DataFrame([[8.0, 8, 8], [30.0, 30, 30]], columns=["x", "y", "z"]).to_csv(patient / "points.csv")
    if aneurysm:
        (patient / "aneurysms.csv").write_text("x,y,z\n20,20,20\n22,20,20\n")


def test_train_predict_and_evaluate_from_a_relative_split(tmp_path):
    write_patient(tmp_path / "Data" / "P0001", aneurysm=True)
    write_patient(tmp_path / "Data" / "sub-002" / "ses-1", aneurysm=False)
    split = {"training list": ["P0001"], "validation list": ["P0001"], "testing list": ["P0001", "sub-002/ses-1"]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    train_dir = tmp_path / "Data" / "0Work" / "exp"

    trained = run_command("train", *TINY_TRAINING, "data_dir=Data", "epochs=1", cwd=tmp_path)
    assert trained.returncode == 0, trained.stderr[-2000:]
    assert (train_dir / "last_checkpoint.pytorch").is_file() and (train_dir / "run_info.json").is_file()
    assert (train_dir / "best_checkpoint.pytorch").is_file()  # validated on P0001

    # from another folder: the training's relative data folder and split file are found again
    predicted = run_command("predict", f"train_dir={train_dir}", "margin=2", "tta=false", cwd=tmp_path / "Data")
    assert predicted.returncode == 0, predicted.stderr[-2000:]
    predictions = train_dir / "prediction" / "1_epochs"
    assert sorted(p.name for p in predictions.glob("*.json")) == ["P0001.json", "sub-002_ses-1.json"]

    evaluated = run_command(
        "evaluate", f"train_dir={train_dir}", f"predictions={predictions}", env={**os.environ, "MPLBACKEND": "Agg"}
    )
    assert evaluated.returncode == 0, evaluated.stderr[-2000:]
    assert (predictions / "detection_evaluation@0.1.csv").is_file()
