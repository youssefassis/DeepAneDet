"""Skull-strip every patient volume; configuration in deepanedet/configs/remove_skull.yaml.

Example: deepanedet-remove-skull data_dir=/data
The result is saved next to each volume and registered as 'noskull volume' in the patient's config.json.
"""

import json
import os

import hydra
import nibabel as ni
import numpy as np

from deepanedet.data import io as dio
from deepanedet.volume import selection as vs


def remove_skull(patient_dir, output_name):
    config_file = os.path.join(patient_dir, "config.json")
    with open(config_file, "r") as f:
        d = json.load(f)
    ni_vol = ni.load(os.path.join(patient_dir, d["init volume"]))
    vol = np.asarray(ni_vol.dataobj).astype(np.float32)

    mask = vs.remove_skull_mask(vol)
    vol = mask * vol
    vol /= np.max(vol)

    ni.Nifti1Image(vol, ni_vol.affine).to_filename(os.path.join(patient_dir, output_name))
    d["noskull volume"] = output_name
    with open(config_file, "w") as f:
        json.dump(d, f, indent=2)


@hydra.main(version_base="1.3", config_path="../configs", config_name="remove_skull")
def main(cfg):
    for patient_dir in dio.fetch_patient_dirs(cfg.data.data_dir):
        print(f"Removing the skull of {patient_dir}")
        remove_skull(patient_dir, cfg.output)


if __name__ == "__main__":
    main()
