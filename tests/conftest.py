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
        **{f"{kind} sample {aug}": None for kind in ("positive", "negative") for aug in ("shift", "rotation", "distortion", "scaling")},
    }
