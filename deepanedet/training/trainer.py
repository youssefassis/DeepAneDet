import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.amp import GradScaler
from deepanedet.training.schedulers import Poly, ReduceLROnPlateau, WarmupLRScheduler, WarmupScheduler
from deepanedet.utils import get_logger, RunningAverage, save_checkpoint, load_checkpoint

logger = get_logger("Trainer")


def create_trainer(
    train_dir,
    device,
    model,
    optimizer,
    lr_scheduler,
    loss_criterion,
    loaders,
    max_iterations,
    max_num_epochs,
    save_every,
    warmup_epochs,
    mixed_precision=True,
):
    """Creates a Trainer writing to train_dir, resuming from train_dir/last_checkpoint.pytorch if it exists."""
    model_path = os.path.join(train_dir, "last_checkpoint.pytorch")
    if os.path.isfile(model_path):
        logger.info(f"Continue training from checkpoint ({model_path})")
        state = load_checkpoint(model_path, model, optimizer)
        num_epoch = state["epoch"]
    else:
        logger.info("Training from scratch")
        state = None
        num_epoch = 0

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=loss_criterion,
        device=device,
        loaders=loaders,
        model_path=model_path,
        max_iterations=max_iterations,
        max_num_epochs=max_num_epochs,
        save_model_each_epochs=save_every,
        warmup_epochs=warmup_epochs,
        num_epoch=num_epoch,
        mixed_precision=mixed_precision,
    )
    if state is not None:
        trainer.resume(state)
    return trainer


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        device,
        loaders,
        model_path,
        save_model_each_epochs,
        max_num_epochs,
        max_iterations,
        warmup_epochs,
        num_epoch,
        mixed_precision=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = os.path.dirname(model_path)
        self.model_path = model_path
        self.max_num_epochs = max_num_epochs
        self.max_iterations = max_iterations

        self.num_epoch = num_epoch
        self.save_model_each_epochs = save_model_each_epochs
        self.warmup_epochs = warmup_epochs
        self.device_type = torch.device(device).type
        self.use_amp = mixed_precision and self.device_type == "cuda"  # float16 mixed precision is for GPUs
        self.mixed_precision = GradScaler(self.device_type, enabled=self.use_amp)

        self.best_valid_loss = float("inf")

        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, "logs"))

    def resume(self, state):
        """
        Restores the learning rate schedule and the loss scaling saved in a checkpoint
        """
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("mixed_precision"):  # empty when saved without CUDA
            self.mixed_precision.load_state_dict(state["mixed_precision"])
        self.best_valid_loss = state.get("best_valid_loss", float("inf"))

    def fit(self):
        logger.info(f"Training the model for {self.max_num_epochs - self.num_epoch} epochs")
        for epoch in range(self.num_epoch, self.max_num_epochs):
            self.train()

            # save_checkpoint every n epoch
            if self.num_epoch > 0 and self.num_epoch % self.save_model_each_epochs == 0:
                self._save_checkpoint(is_best=None, last_score=0, epoch=self.num_epoch)

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Training finished.")

    def train(self):
        train_losses = RunningAverage()
        torch.cuda.empty_cache()
        logger.info(f"Epoch [{self.num_epoch}/{self.max_num_epochs}]")
        self.model.train()
        train_loop = tqdm(self.loaders["train"], leave=True, unit="batch", total=self.max_iterations["train"])

        for batch in train_loop:
            input_volume, target_spheres = self._to_device(batch)

            self.optimizer.zero_grad()
            # Mixed precision training: Forward pass
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                pred_spheres = self.model(input_volume)
                loss = self.loss_criterion(predictions=pred_spheres, targets=target_spheres)

            # Back propagation
            self.mixed_precision.scale(loss).backward()
            self.mixed_precision.step(self.optimizer)
            self.mixed_precision.update()

            del pred_spheres, target_spheres

            train_losses.update(loss, input_volume.size(0))
            loss_pr = train_losses.avg.item()

            self._update_lr(warmup=True)

            # Update progress bar
            train_loop.set_description("Training")
            train_loop.set_postfix(loss=loss_pr, lr=self.optimizer.param_groups[0]["lr"])

        valid_loss = self.validate()
        is_best = valid_loss is not None and valid_loss < self.best_valid_loss
        if is_best:
            logger.info(f"Best validation loss so far: {valid_loss}")
            self.best_valid_loss = valid_loss
        self._save_checkpoint(is_best=is_best, last_score=loss_pr)
        self._log_stats(train_loss=loss_pr, valid_loss=valid_loss)
        self._update_lr()

    def validate(self):
        """Returns the mean loss on the validation patches (None without validation patients)"""
        if self.loaders.get("valid") is None:
            return None
        self.model.eval()
        losses = RunningAverage()
        loop = tqdm(self.loaders["valid"], leave=True, unit="batch", total=self.max_iterations["valid"])
        with torch.no_grad():
            for batch in loop:
                input_volume, targets = self._to_device(batch)
                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    loss = self.loss_criterion(predictions=self.model(input_volume), targets=targets)
                losses.update(loss.item(), input_volume.size(0))
                loop.set_description("Validation")
                loop.set_postfix(loss=losses.avg)
        return losses.avg

    def _to_device(self, batch):
        input_volume, targets = batch["volume"].to(self.device), batch["truth"]
        targets = [[target.to(self.device) for target in scale] for scale in targets]
        return input_volume, targets

    def _update_lr(self, warmup=False):
        if self.scheduler is None:
            return
        elif warmup:  # each step
            if self.num_epoch < self.warmup_epochs:
                # WarmUp
                if isinstance(self.scheduler, WarmupLRScheduler):
                    self.scheduler.step()
                elif isinstance(self.scheduler, WarmupScheduler):
                    self.scheduler.schedulers[0].step()

        else:  # after validation loop
            # Poly or ReduceLROnPlateau, alone or after the warmup
            scheduler = self.scheduler.schedulers[-1] if isinstance(self.scheduler, WarmupScheduler) else self.scheduler
            if isinstance(scheduler, (Poly, ReduceLROnPlateau)):
                scheduler.step(epoch=self.num_epoch)

    def _save_checkpoint(self, is_best, last_score, epoch=None):
        state = {
            "epoch": self.num_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "last_eval_score": last_score,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "num_epoch": self.num_epoch,
            "mixed_precision": self.mixed_precision.state_dict(),
            "best_valid_loss": self.best_valid_loss,
        }
        return save_checkpoint(state, is_best=is_best, checkpoint_dir=self.checkpoint_dir, epoch=epoch, logger=logger)

    def _log_stats(self, train_loss, valid_loss):
        losses = {"train": train_loss} if valid_loss is None else {"train": train_loss, "valid": valid_loss}
        self.writer.add_scalars("Loss", losses, self.num_epoch)
        self.writer.add_scalar("learningRate", self.optimizer.param_groups[0]["lr"], self.num_epoch)
        self.writer.close()
