import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).parents[1] / "deepanedet" / "configs")

# a tiny model trained on CPU on 16-voxel patches of 8 mm, without augmentation
TINY_TRAINING = [
    "name=exp",
    "data.split_file=split.json",
    "data.patch_shape=[16,16,16]",
    "data.patch_size=[8,8,8]",
    "model.f_maps=4",
    "batch_size=2",
    "drop_last=false",
    "negative_patches=50",
    "num_workers=0",
    "validation_batch_size=2",
    "augmentation.positive.duplicates=2",
    *[
        f"augmentation.{kind}.{aug}=null"
        for kind in ("positive", "negative")
        for aug in ("shift", "rotation", "distortion")
    ],
    "optimizer.momentum=0.9",
    "optimizer.weight_decay=0",
    "scheduler.warmup_epochs=1",
    "scheduler.start_epoch=1",
    "scheduler.stop_epoch=4",
]


def compose_config(name, overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def save_training_config(train_dir, cfg, launch_dir):
    """Writes the .hydra folder of a training launched from launch_dir, as deepanedet-train does"""
    (train_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, train_dir / ".hydra" / "config.yaml")
    OmegaConf.save({"hydra": {"runtime": {"cwd": str(launch_dir)}}}, train_dir / ".hydra" / "hydra.yaml")


def run_command(module, *overrides, **kwargs):
    """Runs a deepanedet.cli module as its installed command does"""
    return subprocess.run(
        [sys.executable, "-m", f"deepanedet.cli.{module}", *map(str, overrides)],
        capture_output=True,
        text=True,
        **kwargs,
    )


@pytest.fixture
def patient_db():
    """Two 40mm patients at 1mm resolution, each with one aneurysm (a 1mm segment) and two negative patch centers."""
    rng = np.random.default_rng(0)
    return [
        {
            "dir": f"P000{i}",
            "data": rng.random((40, 40, 40)),
            "affine": np.eye(4),
            "aneurysms": np.array([[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]]),
            "points": np.array([[8.0, 8.0, 8.0], [30.0, 30.0, 30.0]]),
        }
        for i in (1, 2)
    ]


@pytest.fixture
def evaluation_fold(tmp_path):
    """
    A trained fold ready for evaluation: two test patients, each with a 4mm aneurysm (aneurysms.csv) and two
    predictions (one on the aneurysm, one false positive); P0003 has no aneurysm and one false positive.
    Returns (training directory, predictions directory).
    """
    import json

    data = tmp_path / "Data"
    train_dir = data / "0Work" / "fold1"
    predictions_dir = train_dir / "prediction" / "10_epochs"
    predictions_dir.mkdir(parents=True)
    patients = []
    for name, has_aneurysm in (("P0001", True), ("P0002", True), ("P0003", False)):
        patient = data / name
        patient.mkdir(parents=True)
        detections = {"0": {"center": [50.0, 0.0, 0.0], "radius": 2.0, "confidence": 0.6}}
        if has_aneurysm:
            (patient / "aneurysms.csv").write_text("x,y,z\n-2,0,0\n2,0,0\n")  # center (0,0,0), radius 2
            detections["1"] = {"center": [0.5, 0.0, 0.0], "radius": 2.0, "confidence": 0.9}
        (predictions_dir / f"{name}.json").write_text(json.dumps(detections))
        patients.append(name)

    split = {"training list": [], "validation list": patients[:1], "testing list": patients[1:]}
    (tmp_path / "fold1.json").write_text(json.dumps(split))
    cfg = compose_config("train", ["data_dir=Data", "data.split_file=fold1.json", "name=fold1"])
    save_training_config(train_dir, cfg, launch_dir=tmp_path)
    return train_dir, predictions_dir
