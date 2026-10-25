"""Train a model; configuration in deepanedet/configs/train.yaml.

Example: deepanedet-train data_dir=/data data.split_file=reproducibility/fold1.json name=fold1
Trains on the training patients of the split file, whose relative folders are taken from data_dir. Checkpoints,
TensorBoard logs and the resolved configuration (.hydra/) are written to <data_dir>/0Work/<name>. Running the same
command again resumes from the last checkpoint.
"""

import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from deepanedet.data.io import add_points_to_patient_data, read_patient_data_base, read_split
from deepanedet.training.dataset import get_dataloaders
from deepanedet.training.losses import get_loss_criterion
from deepanedet.training.trainer import create_trainer
from deepanedet.utils import create_lr_scheduler, detection_grid_size, get_model


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg):
    if cfg.anchors is not None and len(cfg.anchors) != cfg.model.anchors_per_scale:
        raise ValueError(f"{len(cfg.anchors)} anchors but model.anchors_per_scale={cfg.model.anchors_per_scale}")
    train_dir = HydraConfig.get().runtime.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg.model, weight_init=cfg.weight_init, device=device)

    train_list, _, _ = read_split(cfg.data.split_file, cfg.data.data_dir)
    patients = read_patient_data_base(
        train_list,
        volume=cfg.data.volume,
        pts_aneurysm=cfg.data.truth_file,
        normalize=cfg.data.normalize,
        label="training",
        workers=min(cfg.num_workers, os.cpu_count() or 1),
    )
    train_db = add_points_to_patient_data(patients, cfg.data.negative_patch_centers)
    loaders, iterations = get_dataloaders(train_db, None, cfg)

    optimizer = instantiate(cfg.optimizer, params=model.parameters(), _convert_="all")
    warmup_steps = iterations["train"] * cfg.scheduler.get("warmup_epochs", 0)
    lr_scheduler = create_lr_scheduler(optimizer, cfg.scheduler, epochs=cfg.epochs, warmup_steps=warmup_steps)

    trainer = create_trainer(
        train_dir,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=get_loss_criterion(cfg.loss),
        loaders=loaders,
        max_iterations=iterations,
        max_num_epochs=cfg.epochs,
        save_every=cfg.save_every,
        warmup_epochs=cfg.scheduler.get("warmup_epochs", 0),
        scales=[detection_grid_size(cfg.data.patch_shape[0], cfg.model.num_levels)],
        anchors=None if cfg.anchors is None else list(cfg.anchors),
        mixed_precision=cfg.mixed_precision,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
