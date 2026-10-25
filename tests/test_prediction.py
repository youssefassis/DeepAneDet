import numpy as np
import pytest
import torch

from deepanedet.inference import prediction


class OneDetectionPerPatch(torch.nn.Module):
    """Detects a sphere in the first cell of a 2x2x2 grid of every patch."""

    def forward(self, x):
        confidence = torch.full((x.shape[0], 2, 2, 2, 1), -10.0)
        confidence[:, 0, 0, 0, 0] = 10.0
        return [[confidence, torch.zeros((x.shape[0], 2, 2, 2, 4))]]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_patch_wise_prediction_does_not_depend_on_batch_size(batch_size):
    volume = np.zeros((16, 8, 8), dtype=np.float32)  # two 8-voxel patches along x

    spheres = prediction.ndl_patch_wise_prediction(
        "cpu",
        OneDetectionPerPatch(),
        volume,
        [8, 8, 8],
        iou_threshold=0.01,
        detections_per_patch=None,
        batch_size=batch_size,
    )

    # cell centers (offset sigmoid(0) = 0.5 of a 4-voxel cell) in each patch
    assert sorted(s[:3] for s in spheres) == [[2.0, 2.0, 2.0], [10.0, 2.0, 2.0]]


def test_detections_are_found_up_to_the_volume_borders():
    volume = np.zeros((10, 8, 8), dtype=np.float32)  # not a whole number of 6-voxel patch cores

    spheres = prediction.ndl_patch_wise_prediction(
        "cpu", OneDetectionPerPatch(), volume, [8, 8, 8], iou_threshold=0.01, detections_per_patch=None, margin=1
    )

    # one detection 1 voxel past the start of each patch core: cores start at 0 and 6 along every axis
    assert sorted(tuple(s[:3]) for s in spheres) == sorted((x, y, z) for x in (1, 7) for y in (1, 7) for z in (1, 7))


def test_a_margin_without_core_is_rejected():
    with pytest.raises(ValueError, match="no core"):
        prediction.ndl_patch_wise_prediction(
            "cpu", OneDetectionPerPatch(), np.zeros((8, 8, 8)), [8, 8, 8], 0.01, None, margin=4
        )


@pytest.mark.parametrize("tta", [False, True])
def test_validation_case_writes_the_final_predictions(tmp_path, tta):
    patient = {"data": np.zeros((16, 8, 8), dtype=np.float32), "affine": np.eye(4)}

    prediction.ndl_run_validation_case(
        patient,
        "cpu",
        size=[8, 8, 8],
        dim=[8, 8, 8],
        output_dir=str(tmp_path),
        patient_name="P0001",
        model=OneDetectionPerPatch(),
        margin=0,
        batch_size=1,
        mirror_axes=[0],
        iou_threshold=0.01,
        detections_per_patch=None,
        patient_wise_tta=tta,
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
        patients,
        "cpu",
        OneDetectionPerPatch(),
        patch_size=[8, 8, 8],
        patch_dim=[8, 8, 8],
        input_volume="init volume",
        normalization=None,
        output_dir=str(tmp_path / "out"),
    )

    assert sorted(p.name for p in (tmp_path / "out").glob("*.json")) == [
        "sub-001_ses-20100101.json",
        "sub-002_ses-20100101.json",
    ]


@pytest.mark.parametrize("position", [(20, 30, 40), (80, 15, 12), (50, 45, 30)])
def test_resampling_keeps_positions_in_the_scanner_space(position):
    spacing, target_spacing = np.array([0.35, 0.35, 0.5]), np.array([0.4, 0.4, 0.4])
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = [-10, 5, 3]
    grid = np.indices((101, 90, 61)).transpose(1, 2, 3, 0)
    blob = np.exp(-(((grid - position) * spacing) ** 2).sum(-1) / 2).astype(np.float32)  # 1 mm wide

    resampled = prediction.resample_to_spacing(blob, spacing, target_spacing)

    weights = resampled.reshape(-1)
    center = (np.indices(resampled.shape).reshape(3, -1) * weights).sum(1) / weights.sum()
    found = prediction.set_affine_spacing(affine, target_spacing) @ np.r_[center, 1]
    np.testing.assert_allclose(found[:3], (affine @ np.r_[position, 1])[:3], atol=0.01)
