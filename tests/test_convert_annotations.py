import json
import shutil
from pathlib import Path

import numpy as np

import Data.IO as dio
from convertAnnotations import convert_annotations

ANNOTATION = Path(__file__).parents[1] / "Reproducibility" / "Annotations" / "sub-026_ses-20101106.json"  # 2 aneurysms


def test_annotations_become_the_truth_file_read_for_training(tmp_path):
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    shutil.copy(ANNOTATION, annotations)
    shutil.copy(ANNOTATION, annotations / "sub-999_ses-20000101.json")  # not downloaded
    patient = tmp_path / "Data" / "sub-026" / "ses-20101106"
    patient.mkdir(parents=True)

    missing = convert_annotations(str(annotations), str(tmp_path / "Data"))

    spheres = dio.points_to_spheres(dio.read_points_from_csv(patient / "aneurysms.csv"))
    expected = [a["center"] + [a["radius"]] for a in json.loads(ANNOTATION.read_text())]
    np.testing.assert_allclose(spheres, expected)
    assert missing == ["sub-999_ses-20000101"]
