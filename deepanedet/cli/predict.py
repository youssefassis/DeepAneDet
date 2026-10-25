"""Predict the validation and testing patients of a training; configuration in deepanedet/configs/predict.yaml.

Example: deepanedet-predict train_dir=/data/0Work/fold1
Data and model settings come from the training's resolved configuration (<train_dir>/.hydra/config.yaml). Writes one
<patient>.json of detections per patient (center and radius in mm, confidence) to output_dir;
patients already predicted there are skipped.
"""

import os

import hydra
import torch

from deepanedet.data import io as dio
from deepanedet.inference.prediction import ndl_run_validation_cases
from deepanedet.utils import detection_grid_size, get_logger, load_model, load_training_config


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict")
def main(cfg):
    logger = get_logger("Prediction")
    train_cfg = load_training_config(cfg.train_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(train_cfg.model, os.path.join(cfg.train_dir, cfg.checkpoint), device)
    model.eval()
    logger.info(f"Model '{cfg.checkpoint}' ({epoch} epochs) loaded on '{device}' in evaluation mode")

    _, valid_list, test_list = dio.read_split(train_cfg.data.split_file, train_cfg.data.data_dir)
    patients = sorted(valid_list + test_list)
    logger.info(f"{len(patients)} patients, '{train_cfg.data.volume}' with '{train_cfg.data.normalize}' normalization")

    with torch.no_grad():
        ndl_run_validation_cases(
            patients,
            device=device,
            model=model,
            patch_size=list(train_cfg.data.patch_size),
            patch_dim=list(train_cfg.data.patch_shape),
            input_volume=train_cfg.data.volume,
            normalization=train_cfg.data.normalize,
            output_dir=cfg.output_dir or os.path.join(cfg.train_dir, "prediction", f"{epoch}_epochs"),
            margin=cfg.margin,
            mirror_axes=list(cfg.mirror_axes),
            batch_size=cfg.batch_size,
            do_tta=cfg.tta,
            iou_threshold=cfg.iou_threshold,
            scales=[detection_grid_size(train_cfg.data.patch_shape[0], train_cfg.model.num_levels)],
            detections_per_patch=cfg.detections_per_patch,
        )


if __name__ == "__main__":
    main()
