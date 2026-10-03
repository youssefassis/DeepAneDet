import numpy as np
import pytest
import torch

import prediction


class OneDetectionPerPatch(torch.nn.Module):
    """Detects a sphere in the first cell of a 2x2x2 grid of every patch."""

    def forward(self, x):
        confidence = torch.full((x.shape[0], 2, 2, 2, 1), -10.0)
        confidence[:, 0, 0, 0, 0] = 10.0
        return [[confidence, torch.zeros((x.shape[0], 2, 2, 2, 4))]]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_patch_wise_prediction_does_not_depend_on_batch_size(batch_size):
    volume = np.zeros((16, 8, 8), dtype=np.float32)  # two 8-voxel patches along x

    _, spheres = prediction.ndl_patch_wise_prediction(
        "cpu", OneDetectionPerPatch(), volume, [8, 8, 8], iou_threshold=0.01, detections_per_patch=None, batch_size=batch_size
    )

    # cell centers (offset sigmoid(0) = 0.5 of a 4-voxel cell) in each patch
    assert sorted(s[:3] for s in spheres) == [[2.0, 2.0, 2.0], [10.0, 2.0, 2.0]]


@pytest.mark.parametrize("tta", [False, True])
def test_validation_case_writes_the_final_predictions(tmp_path, tta):
    patient = {"data": np.zeros((16, 8, 8), dtype=np.float32), "affine": np.eye(4)}

    prediction.ndl_run_validation_case(
        patient, "cpu", size=[8, 8, 8], dim=[8, 8, 8], output_dir=str(tmp_path), patient_name="P0001",
        model=OneDetectionPerPatch(), margin=0, batch_size=1, mirror_axes=[0], iou_threshold=0.01,
        detections_per_patch=None, patient_wise_tta=tta,
    )

    assert (tmp_path / "P0001.json").is_file()


def test_validation_cases_keep_patients_with_the_same_session_apart(tmp_path):
    import json

    import nibabel as ni

    patients = []
    for subject in ("sub-001", "sub-002"):
        patient = tmp_path / subject / "ses-20100101"
        patient.mkdir(parents=True)
        ni.save(ni.Nifti1Image(np.zeros((16, 8, 8), np.float32), np.eye(4)), patient / "volume.nii.gz")
        (patient / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz"}))
        patients.append(str(patient))

    prediction.ndl_run_validation_cases(
        patients, "cpu", OneDetectionPerPatch(), patch_size=[8, 8, 8], patch_dim=[8, 8, 8],
        input_volume="init volume", normalization=None, output_dir=str(tmp_path / "out"),
    )

    assert sorted(p.name for p in (tmp_path / "out").glob("*.json")) == [
        "sub-001_ses-20100101.json",
        "sub-002_ses-20100101.json",
    ]
