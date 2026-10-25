"""Select negative patch centers (on vessels and in the parenchyma) for every patient, away from its aneurysms.

Configuration in deepanedet/configs/extract_points.yaml. Example: deepanedet-extract-points data_dir=/data
Writes <output> (CSV) and its 3D Slicer .fcsv counterpart in each patient folder.
Requires the 'noskull volume' produced by deepanedet-remove-skull.
"""

import hydra

from deepanedet.data import io as dio


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract_points")
def main(cfg):
    for patient_dir in dio.fetch_patient_dirs(cfg.data.data_dir):
        dio.extract_points_from_patient(
            patient_dir,
            r=cfg.radius,
            nb_points=cfg.n_points,
            outfile=cfg.output,
            random_points=cfg.random,
            truth_file=cfg.data.truth_file,
        )


if __name__ == "__main__":
    main()
