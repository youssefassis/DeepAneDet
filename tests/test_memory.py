"""Volumes are kept in memory as float32 from loading to prediction."""

import json

import nibabel as ni
import numpy as np
import pytest

from deepanedet.data import io as dio
from deepanedet.inference.prediction import resample_to_spacing
from deepanedet.volume.patch import get_patch

AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])


@pytest.fixture
def patient(tmp_path):
    vol = np.random.default_rng(0).integers(0, 2000, (30, 30, 30)).astype(np.int16)
    ni.save(ni.Nifti1Image(vol, AFFINE), tmp_path / "volume.nii.gz")
    (tmp_path / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz"}))
    return tmp_path


def test_volumes_are_read_as_float32(patient):
    vol, _ = dio.read_nii_from_file(str(patient / "volume.nii.gz"))

    assert vol.dtype == np.float32


@pytest.mark.parametrize("normalize", [None, "Linear", "Normal"])
def test_patient_data_stays_float32_after_normalization(patient, normalize):
    data = dio.read_patient_data_base([str(patient)], normalize=normalize)[0]["data"]

    assert data.dtype == np.float32
    if normalize == "Linear":
        assert (data.min(), data.max()) == (0.0, 1.0)
    if normalize == "Normal":
        assert abs(data.mean()) < 1e-5 and data.std() == pytest.approx(1.0, abs=1e-5)


def test_patches_and_resampled_volumes_are_float32(patient):
    data = dio.read_patient_data_base([str(patient)], normalize="Linear")[0]["data"]

    patch, _ = get_patch(data, AFFINE, center=[7.5, 7.5, 7.5], size=[8, 8, 8], dim=[16, 16, 16])
    resampled = resample_to_spacing(data, [0.5, 0.5, 0.5], [0.4, 0.4, 0.4])

    assert patch.dtype == np.float32 and resampled.dtype == np.float32
