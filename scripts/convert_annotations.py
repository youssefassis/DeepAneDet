"""
Writes the aneurysm annotations of reproducibility/annotations (one sub-XXX_ses-YYYYMMDD.json per patient, each
aneurysm given by its points P1 and P2) as the truth file read for training and evaluation: a CSV of x, y, z point
pairs in each patient directory <data_dir>/sub-XXX/ses-YYYYMMDD.
Patients without an annotation file have no aneurysm and get no truth file.

Usage: uv run python scripts/convert_annotations.py path/to/reproducibility/annotations path/to/Data_dir [truth file name, aneurysms.csv by default]
"""

import glob
import json
import os
import sys
import pandas as pd


def annotation_to_points(annotation_file):
    """
    Returns the aneurysms of an annotation file as a DataFrame of x, y, z points, two consecutive rows per aneurysm (P1, P2)
    """
    with open(annotation_file, "r") as f:
        aneurysms = json.load(f)
    return pd.DataFrame([point for a in aneurysms for point in (a["P1"], a["P2"])], columns=["x", "y", "z"])


def convert_annotations(annotations_dir, data_dir, truth_file="aneurysms.csv"):
    """
    Writes the truth file of each annotated patient found in data_dir; returns the names of the annotated patients
    missing from data_dir
    """
    missing = []
    for annotation_file in sorted(glob.glob(os.path.join(annotations_dir, "sub-*_ses-*.json"))):
        name = os.path.basename(annotation_file)[: -len(".json")]
        subject, session = name.split("_", 1)
        patient_dir = os.path.join(data_dir, subject, session)
        if not os.path.isdir(patient_dir):
            missing.append(name)
            continue
        annotation_to_points(annotation_file).to_csv(os.path.join(patient_dir, truth_file), index=False)
    return missing


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(
            f"Usage: uv run python {sys.argv[0]} path/to/reproducibility/annotations path/to/Data_dir [truth file name]"
        )
    missing = convert_annotations(*sys.argv[1:])
    if missing:
        print(f"{len(missing)} annotated patients not found in {sys.argv[2]}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
