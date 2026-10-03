import json

import nibabel as ni
import numpy as np
import pandas as pd
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


@pytest.fixture
def patient_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # restores the working directory changed by extractPointsFromPatient
    patient = tmp_path / "P0001"
    patient.mkdir()
    vol = np.random.default_rng(0).random((30, 30, 30)).astype(np.float32)
    ni.save(ni.Nifti1Image(vol, np.eye(4)), patient / "noskull.nii.gz")
    (patient / "aneurysms.csv").write_text("x,y,z\n10,10,10\n12,10,10\n")
    config = {"noskull volume": "noskull.nii.gz", "pts aneurysm": "aneurysms.csv"}
    (patient / "config.json").write_text(json.dumps(config))
    return patient


def test_extract_points_writes_vessel_and_parenchyma_points(patient_dir):
    dio.extractPointsFromPatient(str(patient_dir), r=5, nbPoints=5)

    points = pd.read_csv(patient_dir / "points.csv")
    assert set(points["type"]) == {"Vessel", "Parenchyma"}
    assert (patient_dir / "points.fcsv").is_file()


def test_extract_points_for_a_patient_without_aneurysm(patient_dir):
    (patient_dir / "aneurysms.csv").unlink()

    dio.extractPointsFromPatient(str(patient_dir), r=5, nbPoints=5)

    assert set(pd.read_csv(patient_dir / "points.csv")["type"]) == {"Vessel", "Parenchyma"}


def test_points_to_spheres_without_points_has_sphere_columns():
    assert dio.points_to_spheres(None).shape == (0, 4)


def test_generate_masks_burns_the_aneurysm_spheres(tmp_path):
    from Data.Generators import generate_masks_nii

    patient = tmp_path / "P0001"
    patient.mkdir()
    ni.save(ni.Nifti1Image(np.zeros((20, 20, 20), np.float32), np.eye(4)), patient / "volume.nii.gz")
    (patient / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz"}))
    (patient / "F.csv").write_text("x,y,z\n8,10,10\n12,10,10\n")  # sphere of radius 2 centered on (10,10,10)

    generate_masks_nii(str(tmp_path))

    mask = ni.load(patient / "mask_spheres.nii.gz").get_fdata()
    assert mask[10, 10, 10] == 1 and mask[10, 10, 13] == 0


@pytest.mark.parametrize(
    "patient_dir, name",
    [
        ("/data/P0001", "P0001"),
        ("/data/P0001/", "P0001"),
        ("/data/sub-013/ses-20101220", "sub-013_ses-20101220"),
    ],
)
def test_patient_name(patient_dir, name):
    assert dio.patient_name(patient_dir) == name


def test_read_points_keeps_the_coordinates(tmp_path):
    points = tmp_path / "points.csv"
    points.write_text(",x,y,z,type\n0,1,2,3,Vessel\n1,4,5,6,Vessel\n2,7,8,9,Parenchyma\n")

    assert dio.read_points_from_csv(points).tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert sorted(dio.read_points_from_csv(points, nb=1)[:, 0]) in ([1, 7], [4, 7])


def test_read_points_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dio.read_points_from_csv(tmp_path / "points.csv")


def test_read_points_reports_missing_coordinates(tmp_path):
    points = tmp_path / "points.csv"
    points.write_text("x,y\n1,2\n")

    with pytest.raises(KeyError):
        dio.read_points_from_csv(points)


def test_save_split_updates_an_existing_file(tmp_path):
    split = tmp_path / "fold1.json"
    split.write_text(json.dumps({"comment": "kept"}))

    dio.saveSplit(str(split), ["P0001"], ["P0002"], ["P0003"])

    assert json.loads(split.read_text())["comment"] == "kept"
    assert dio.readSplit(str(split)) == (["P0001"], ["P0002"], ["P0003"])


def test_save_split_reports_a_corrupt_file(tmp_path):
    split = tmp_path / "fold1.json"
    split.write_text("{")

    with pytest.raises(json.JSONDecodeError):
        dio.saveSplit(str(split), [], [], [])


def test_generate_masks_of_a_patient_without_aneurysm_is_empty(tmp_path):
    from Data.Generators import generate_masks_nii

    patient = tmp_path / "P0001"
    patient.mkdir()
    ni.save(ni.Nifti1Image(np.ones((5, 5, 5), np.float32), np.eye(4)), patient / "volume.nii.gz")
    (patient / "config.json").write_text(json.dumps({"init volume": "volume.nii.gz"}))

    generate_masks_nii(str(tmp_path))

    assert ni.load(patient / "mask_spheres.nii.gz").get_fdata().sum() == 0
