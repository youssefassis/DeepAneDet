import logging
import math
import os
import platform
import random
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from deepanedet.training.schedulers import LearningRateScheduler


def load_training_config(train_dir):
    """
    Reads the resolved configuration of a training (<train_dir>/.hydra, written by deepanedet-train), with its data
    folder and split file made absolute from the folder the training was launched in.
    """
    hydra_dir = os.path.join(train_dir, ".hydra")
    if not os.path.isfile(os.path.join(hydra_dir, "config.yaml")):
        raise FileNotFoundError(f"No training configuration in {hydra_dir}; is {train_dir} a deepanedet-train folder?")
    cfg = OmegaConf.load(os.path.join(hydra_dir, "config.yaml"))
    launch_dir = OmegaConf.load(os.path.join(hydra_dir, "hydra.yaml")).hydra.runtime.cwd
    cfg.data.data_dir = os.path.join(launch_dir, cfg.data.data_dir)
    cfg.data.split_file = os.path.join(launch_dir, cfg.data.split_file)
    return cfg


def detection_grid_size(patch_shape, depth):
    """Cells along each side of the detection grid: the patch is halved by each of the depth - 1 poolings"""
    return math.ceil(patch_shape / 2 ** (depth - 1))


def get_model(model_cfg, weight_init="he", device="cpu"):
    """
    Instantiates the model described by model_cfg (a Hydra config with a _target_ class, see configs/model).
    weight_init: "he" (Kaiming normal, as in the paper) or "pytorch" (PyTorch defaults)
    """
    logger = get_logger("Model")
    if weight_init not in ("he", "pytorch"):
        raise ValueError(f"Unsupported weight_init '{weight_init}', expected 'he' or 'pytorch'")
    model = instantiate(model_cfg, _convert_="all")
    logger.info(f"The model '{type(model).__name__}' is selected")

    # model properties
    nparams = get_number_of_learnable_parameters(model)
    size = (
        sum([param.nelement() * param.element_size() for param in model.parameters()])
        + sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    ) * 1e-6
    logger.info(f"Number of learnable params {nparams}")
    logger.info(f"Size allocated by the model is {size:.2f} MB")

    if weight_init == "he":
        logger.info("Apply He/Kaiming weight initialization")
        initialize_weights(model)
    else:
        logger.info("Apply PyTorch default weight initialization")

    logger.info(f"Sending the model to '{device}'")
    return model.to(device)


def load_model(model_cfg, checkpoint_path, device="cpu"):
    """Builds the model described by model_cfg and loads its weights; returns it with its number of trained epochs"""
    model = get_model(model_cfg, device=device)
    state = load_checkpoint(checkpoint_path, model)
    return model, state["epoch"]


def save_checkpoint(state, is_best, checkpoint_dir, epoch=None, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if epoch is not None:
        model_name = os.path.join(checkpoint_dir, f"{epoch}_checkpoint.pytorch")
        torch.save(state, model_name)
        log_info(f"Saving checkpoint ({epoch} epochs) to '{model_name}'")
        return False

    last_file_path = os.path.join(checkpoint_dir, "last_checkpoint.pytorch")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, "best_checkpoint.pytorch")
        shutil.copyfile(last_file_path, best_file_path)
        return True
    return False


def load_checkpoint(
    checkpoint_path, model, optimizer=None, model_key="model_state_dict", optimizer_key="optimizer_state_dict"
):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    logger = get_logger("Model Configuration")
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])
    logger.info(f"Checkpoint loaded ( {state['epoch']} epochs; metric: {state['last_eval_score']})")
    return state


def create_lr_scheduler(optimizer, config, epochs, warmup_steps=0):
    """
    Creates the learning rate scheduler described by config (see configs/scheduler), whose name is one of None,
    ReduceLROnPlateau, Poly, Warmup, WarmupPlateau or WarmupPoly. The optimizer's learning rate is the peak one:
    warmup schedulers start from 1e-8 and raise it over warmup_steps batches.
    """
    assert config["name"] in [None, "ReduceLROnPlateau", "Poly", "Warmup", "WarmupPlateau", "WarmupPoly"]
    initial_lr = optimizer.param_groups[0]["lr"]
    if str(config["name"]).startswith("Warmup"):
        LearningRateScheduler.set_lr(optimizer, 1e-8)

    logger = get_logger("Scheduler")
    if config["name"] is None:
        scheduler = None
        logger.info("No scheduler specified")

    elif config["name"] == "Warmup":
        from deepanedet.training.schedulers import WarmupLRScheduler

        scheduler = WarmupLRScheduler(
            optimizer=optimizer,
            min_lr=config["min_lr"],
            max_lr=initial_lr,
            warmup_steps=warmup_steps,
        )
        logger.info(f"WarmupLR Scheduler: warmup_steps = {warmup_steps}")

    elif config["name"] == "ReduceLROnPlateau":
        from deepanedet.training.schedulers import ReduceLROnPlateau

        factor, patience, wloss = config["factor"], config["patience"], True  # config["wloss"]
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=factor, patience=patience, wloss=wloss, start_epoch=0)
        logger.info(f"ReduceLROnPlateau Scheduler: patience={patience}, factor={factor}, wloss={wloss}")

    elif config["name"] == "Poly":
        from deepanedet.training.schedulers import Poly

        start_epoch = config["start_epoch"]
        scheduler = Poly(optimizer=optimizer, start_epoch=start_epoch, max_epochs=epochs, exponent=0.99, min_lr=1e-4)
        logger.info(f"Polynomial Scheduler: start_epoch={start_epoch}, max_epochs={epochs}")

    elif config["name"] == "WarmupPlateau":
        from deepanedet.training.schedulers import WarmupScheduler, ReduceLROnPlateau

        start_epoch = 10
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            min_lr=1e-8,
            max_lr=initial_lr,
            warmup_steps=warmup_steps,
            after_warmup_scheduler=ReduceLROnPlateau(
                optimizer=optimizer,
                factor=config["factor"],
                patience=config["patience"],
                wloss=True,
                start_epoch=start_epoch,
            ),
        )
        logger.info(
            f"WarmupPlateau scheduler : lr={initial_lr}, patience={config['patience']}, factor={config['factor']}, start_epoch={start_epoch}"
        )
    elif config["name"] == "WarmupPoly":
        from deepanedet.training.schedulers import WarmupScheduler, Poly

        start_epoch = config["start_epoch"]
        stop_epoch = config["stop_epoch"]
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            min_lr=1e-8,
            max_lr=initial_lr,
            warmup_steps=warmup_steps,
            after_warmup_scheduler=Poly(
                optimizer=optimizer,
                start_epoch=start_epoch,
                max_epochs=stop_epoch,
                exponent=0.9,
                min_lr=config["min_lr"],
                initial_lr=initial_lr,
            ),
        )
        logger.info(f"WarmupPoly scheduler : lr={initial_lr}, start_epoch={start_epoch}")
    return scheduler


def get_number_of_steps(n_samples, batch_size):
    """
    This function calculates the number of steps needed to complete an epoch in a training
    loop based on the number of samples and the batch size.
    It takes two arguments:
        - n_samples, an integer representing the total number of samples in the dataset,
        - batch_size, an integer representing the number of samples per batch.
    """
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1


class RunningAverage:
    """Computes and stores the average (loss and metric scores)"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def get_number_of_learnable_parameters(model):
    """
    Returns the number of trainable parameters of the model
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def initialize_weights(*models, neg_slope: float = 1e-2):
    """
    Initialize the model weights using Kaiming/He initialization
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                module.weight = torch.nn.init.kaiming_normal_(module.weight, a=neg_slope)
                if module.bias is not None:
                    module.bias = torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.BatchNorm3d):
                module.weight.data.fill_(1)  # gamma
                if module.bias is not None:
                    module.bias = torch.nn.init.constant_(module.bias, 0)


def get_logger(name, level=logging.INFO):
    """Returns a named logger; handlers are left to the application (Hydra configures console and file logs)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def seed_everything(seed):
    """Seeds Python, NumPy and PyTorch (CPU and CUDA). DataLoader workers derive their seeds from PyTorch's."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit():
    """Commit of the checkout the code runs from ("unknown" outside a git checkout)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
        return commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def run_info(**extra):
    """Code and environment versions of a run, to be stored with its outputs."""
    commit, dirty = _git_commit()
    return {
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
