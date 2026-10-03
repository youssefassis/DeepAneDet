import os, torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.amp import GradScaler
from schedulers import Poly, ReduceLROnPlateau, WarmupLRScheduler, WarmupScheduler
from utils import get_logger, RunningAverage, save_checkpoint, load_checkpoint

logger = get_logger('Trainer')

def create_trainer(config, device, model, optimizer, lr_scheduler, loss_criterion, loaders, max_iterations, scales, anchors):
    assert config is not None, 'Could not find trainer configuration'
    
    if os.path.isfile(config['model_file']):
        logger.info(f"Continue training from checkpoint ({config['model_file']})")
        state = load_checkpoint(config["model_file"], model, optimizer)
        num_epoch = state['epoch']    
    else:
        logger.info("Training from scratch")
        num_epoch = 0
        
    return Trainer(model = model,
                   optimizer = optimizer,
                   lr_scheduler = lr_scheduler,
                   loss_criterion = loss_criterion,
                   device = device,
                   loaders = loaders,
                   model_path= config['model_file'],
                   max_iterations = max_iterations,
                   max_num_epochs = config["n_epochs"], 
                   save_model_each_epochs = config["save model each n epoch"], 
                   warmup_epochs=config['warmEpochs'],
                   num_epoch = num_epoch,
                   scales=scales,
                   anchors=anchors)

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, device, loaders, model_path, save_model_each_epochs, 
                 max_num_epochs, max_iterations, warmup_epochs, num_epoch, scales, anchors):
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
        self.mixed_precision = GradScaler('cuda')
                    
        self.writer = SummaryWriter(log_dir = os.path.join(self.checkpoint_dir, 'logs'))
            
        self.scaled_anchors = (torch.tensor(anchors).reshape(len(scales), len(anchors)//len(scales), 1) * 
                               torch.tensor(scales).unsqueeze(1).unsqueeze(1).repeat(1, len(anchors)//len(scales), 1)).to(self.device) if anchors is not None else None
            
        
    def fit(self):
        logger.info(f"Training the model for {self.max_num_epochs - self.num_epoch} epochs")
        for epoch in range(self.num_epoch, self.max_num_epochs):
            self.train()
            
            # save_checkpoint every n epoch
            if self.num_epoch > 0 and self.num_epoch % self.save_model_each_epochs == 0:
                self._save_checkpoint(is_best = None, last_score = 0, epoch = self.num_epoch)

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Training finished.")

    def train(self):
        train_losses = RunningAverage()
        torch.cuda.empty_cache()
        logger.info(f"Epoch [{self.num_epoch}/{self.max_num_epochs}]")
        self.model.train()
        train_loop = tqdm(self.loaders['train'], leave = True, unit='batch', total = self.max_iterations['train'])
        
        for batch in train_loop:
            input_volume, target_spheres = batch['volume'], batch['truth']
            
            # TO DEVICE
            input_volume = input_volume.to(self.device)
            for i in range(len(target_spheres)):
                for j in range(len(target_spheres[i])):
                    target_spheres[i][j] = target_spheres[i][j].to(self.device)
            
            self.optimizer.zero_grad()
            # Mixed precision training: Forward pass
            with torch.amp.autocast('cuda'):
                pred_spheres = self.model(input_volume)
                loss = self.loss_criterion(predictions = pred_spheres, targets = target_spheres) if self.scaled_anchors is None else self.loss_criterion(predictions = pred_spheres, targets = target_spheres, anchors = self.scaled_anchors)
                
            # Back propagation    
            self.mixed_precision.scale(loss).backward()
            self.mixed_precision.step(self.optimizer)
            self.mixed_precision.update()

            del pred_spheres, target_spheres
    
            train_losses.update(loss, input_volume.size(0))
            loss_pr = train_losses.avg.item()

            self._updateLR(warmup = True)
    
            # Update progress bar
            train_loop.set_description(f"Training")
            train_loop.set_postfix(loss = loss_pr, lr = self.optimizer.param_groups[0]['lr'])

        self._save_checkpoint(is_best=False, last_score=loss_pr)
        self._log_stats(score=loss_pr)
        self._updateLR()

    def _updateLR(self, warmup=False):
        if self.scheduler is None:
            return
        elif warmup: # each step
            if self.num_epoch < self.warmup_epochs:
                # WarmUp
                if isinstance(self.scheduler, WarmupLRScheduler):
                        self.scheduler.step()
                elif isinstance(self.scheduler, WarmupScheduler):
                    self.scheduler.schedulers[0].step()

        else: # after validation loop
            # Poly or ReduceLROnPlateau, alone or after the warmup
            scheduler = self.scheduler.schedulers[-1] if isinstance(self.scheduler, WarmupScheduler) else self.scheduler
            if isinstance(scheduler, (Poly, ReduceLROnPlateau)):
                scheduler.step(epoch = self.num_epoch)

    
    def _save_checkpoint(self, is_best, last_score, epoch=None):
        state = {'epoch': self.num_epoch + 1,
                 'model_state_dict': self.model.state_dict(),
                 'last_eval_score': last_score,
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                 'num_epoch': self.num_epoch,
                 'mixed_precision': self.mixed_precision.state_dict() }
        return save_checkpoint(state, 
                               is_best=is_best, 
                               checkpoint_dir=self.checkpoint_dir,
                               epoch = epoch,
                               logger=logger)

    def _log_stats(self, score):
        self.writer.add_scalars("Loss", {f"train": score}, self.num_epoch)
        self.writer.add_scalar('learningRate', self.optimizer.param_groups[0]["lr"], self.num_epoch)
        self.writer.close()
