#!/usr/bin/env python3

import sys, os, json
import torch
from torch import optim

from utils import create_optimizer, create_lr_scheduler, get_model
from Dataset import getDataloadersFromDir
from losses import get_loss_criterion
from metrics import get_metric

from trainerUnique import create_trainer

def main():
    d = sys.argv[1]
    cfg = os.path.join(d,'ndl_config.json')
    try:
        with open(cfg,'r') as f:
            config = json.load(f)
    except:
        print(f'No such config file {cfg}')
        exit()

    # Model Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, default_init="keras", device=device)
    print(model)

    loss_criterion = get_loss_criterion(config["loss"])
    eval_criterion = get_metric(config["metrics"])
    optimizer = create_optimizer(model, config["initial_learning_rate"],  config["weight_decay"])
    warmup_steps=1200
    lr_scheduler = create_lr_scheduler (optimizer, config, warmup_steps = warmup_steps) # 6 first epochs warmup
#    lr_scheduler = create_lr_scheduler (optimizer, config)

    loaders, iterations = getDataloadersFromDir(config, workers=8, shuffle_train=False, shuffle_val=False)

    # Trainer
    trainer = create_trainer(config, device = device, model=model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                            eval_criterion=eval_criterion, loaders=loaders,max_iterations=iterations, warmup_steps=warmup_steps)

    trainer.fit(sanity_check=False)

if __name__ == '__main__':
    main()
