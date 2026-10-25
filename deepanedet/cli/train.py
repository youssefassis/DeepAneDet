"""Train a model; configuration in deepanedet/configs/train.yaml.

Example: deepanedet-train data_dir=/data data.split_file=reproducibility/fold1.json name=fold1
Trains on the training patients of the split file and computes the loss on its validation patients after each epoch;
relative patient folders are taken from data_dir. Checkpoints, TensorBoard logs and the resolved configuration
(.hydra/) are written to <data_dir>/0Work/<name>. Running the same command again resumes from the last checkpoint.
"""

import json
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from deepanedet.data.io import add_points_to_patient_data, read_patient_data_base, read_split
from deepanedet.training.dataset import get_dataloaders
from deepanedet.training.losses import get_loss_criterion
from deepanedet.training.trainer import create_trainer
from deepanedet.utils import create_lr_scheduler, get_model, run_info, seed_everything


def load_patients(pat_list, data_cfg, label, workers):
    if not pat_list:
        return None
    patients = read_patient_data_base(
        pat_list,
        volume=data_cfg.volume,
        pts_aneurysm=data_cfg.truth_file,
        normalize=data_cfg.normalize,
        label=label,
        workers=workers,
    )
    return add_points_to_patient_data(patients, data_cfg.negative_patch_centers)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg):
    train_dir = HydraConfig.get().runtime.output_dir
    if cfg.seed is not None:
        seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = not cfg.deterministic
    torch.backends.cudnn.deterministic = cfg.deterministic
    with open(os.path.join(train_dir, "run_info.json"), "w") as f:
        json.dump(run_info(seed=cfg.seed), f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg.model, weight_init=cfg.weight_init, device=device)

    workers = min(cfg.num_workers, os.cpu_count() or 1)
    train_list, valid_list, _ = read_split(cfg.data.split_file, cfg.data.data_dir)
    train_db = load_patients(train_list, cfg.data, "Training", workers)
    valid_db = load_patients(valid_list, cfg.data, "Validation", workers)
    loaders, iterations = get_dataloaders(train_db, valid_db, cfg)

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
        mixed_precision=cfg.mixed_precision,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
