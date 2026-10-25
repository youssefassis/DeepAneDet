import json
import random

import numpy as np
import pytest
import torch

from deepanedet.utils import run_info, seed_everything

from conftest import TINY_TRAINING, run_command
from test_cli import write_patient


def test_seed_everything_seeds_python_numpy_and_torch():
    def draws():
        return random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(7)
    first = draws()
    seed_everything(7)

    assert draws() == first


def test_run_info_records_versions_and_commit():
    info = run_info()

    assert info["torch"] == torch.__version__
    assert info["git_commit"] != "unknown"  # the tests run from the git checkout


@pytest.mark.parametrize("num_workers", [0, 2])
def test_same_seed_gives_the_same_trained_weights(tmp_path, num_workers):
    for name in ("P0001", "P0002"):
        write_patient(tmp_path / name, aneurysm=True)
    (tmp_path / "split.json").write_text(
        json.dumps({"training list": ["P0001", "P0002"], "validation list": [], "testing list": []})
    )

    def train(name):
        overrides = [*TINY_TRAINING, f"data_dir={tmp_path}", f"name={name}", "seed=5", "epochs=1"]
        overrides += [
            f"num_workers={num_workers}",
            "augmentation.positive.distortion=3",
            "augmentation.positive.shift=2",
        ]
        result = run_command("train", *overrides, cwd=tmp_path)
        assert result.returncode == 0, result.stderr[-2000:]
        return torch.load(tmp_path / "0Work" / name / "last_checkpoint.pytorch")["model_state_dict"]

    first, second = train("a"), train("b")

    assert all(torch.equal(first[k], second[k]) for k in first)
    assert json.loads((tmp_path / "0Work" / "a" / "run_info.json").read_text())["seed"] == 5
