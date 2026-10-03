import nibabel as ni
import numpy as np
import pytest

import Data.IO as dio


@pytest.fixture
def volume_file(tmp_path):
    vol = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) + 10
    path = tmp_path / "volume.nii.gz"
    ni.save(ni.Nifti1Image(vol, np.eye(4)), path)
    return str(path)


def test_linear_normalization_rescales_to_unit_range(volume_file):
    data = dio.read_patient_data(volume_file, None, normalize="Linear")["data"]

    assert data.min() == 0 and data.max() == 1


def test_normal_normalization_standardizes(volume_file):
    data = dio.read_patient_data(volume_file, None, normalize="Normal")["data"]

    assert data.mean() == pytest.approx(0) and data.std() == pytest.approx(1)
