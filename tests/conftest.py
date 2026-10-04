import numpy as np
import pytest


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
def training_config():
    """Smallest ndl_config.json used for training: 16-voxel patches of 8mm, without augmentation."""
    return {
        "patch_shape": [16, 16, 16],
        "patch_size": [0.5, 0.5, 0.5],
        "scales": [2],
        "num parameters": 4,
        "batch_size": 2,
        "validation_batch_size": 2,
        "balancedbatch": True,
        "drop_last": False,
        "positive duplicates": 2,
        "nb neg patches per patient": None,
        "percentage of negative patches": 50,
        "flip probability": 0.5,
        "flip axes": [0],
        **{
            f"{kind} sample {aug}": None
            for kind in ("positive", "negative")
            for aug in ("shift", "rotation", "distortion", "scaling")
        },
    }


@pytest.fixture
def evaluation_fold(tmp_path):
    """
    A trained fold ready for evaluation: two test patients, each with a 4mm aneurysm (aneurysms.csv) and two
    predictions (one on the aneurysm, one false positive); P0003 has no aneurysm and one false positive.
    Returns (training directory, predictions directory).
    """
    import json

    data = tmp_path / "Data"
    predictions_dir = tmp_path / "Work" / "Fold1" / "Predictions" / "10_epochs"
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
        patients.append(str(patient))

    split_file = tmp_path / "fold1.json"
    split = {"training list": [], "validation list": patients[:1], "testing list": patients[1:]}
    split_file.write_text(json.dumps(split))
    config = {"split_file": str(split_file), "truth file": "aneurysms.csv"}
    (tmp_path / "Work" / "Fold1" / "ndl_config.json").write_text(json.dumps(config))
    return tmp_path / "Work" / "Fold1", predictions_dir
