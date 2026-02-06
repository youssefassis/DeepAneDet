#!/usr/bin/env python3
import os, sys, json, torch, math

from utils import get_optimizer_scheduler, get_model
from Data.IO import add_points_to_patient_data, readSplit, read_patient_data_base
from losses import get_loss_criterion

from trainer import create_trainer
from dataset import getDataloaders

def main():
    try:
        with open(os.path.join(sys.argv[1], 'ndl_config.json'),'r') as f:
            config = json.load(f)
        if not 'scales' in config:
            config['scales'] = [math.ceil(x/8) for x in config['patch_shape']]
    except:
        print(f'No such config file for training')
        exit()    
    
    # Multiprocessing data loading
    workers = 12

    # Data Preparation
    train_list, _, _ = readSplit(config['split_file'])
    valid_list = []

    train_db = add_points_to_patient_data(read_patient_data_base(pat_list = train_list,
                                                                 volume = config['volume'],
                                                                 pts_aneurysm = config['truth file'],
                                                                 normalize = config['normalize'],
                                                                 label = "training",
                                                                 workers = workers,
                                                                ), 
                                          config['negative patch centers'],
                                          nb = None)

    valid_db = add_points_to_patient_data(read_patient_data_base(pat_list = valid_list,
                                                                 volume = config['volume'],
                                                                 pts_aneurysm = config['truth file'],
                                                                 normalize = config['normalize'],
                                                                 label = "validation",
                                                                 workers = workers,
                                                                ),
                                          config['negative patch centers'],
                                          nb = None) if len(valid_list) > 0 else None

    # Data loaders
    loaders, iterations = getDataloaders(train_db = train_db, valid_db = valid_db, config = config, workers = workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    model = get_model(config, device=device)

    # optimizer & scheduler
    warmup_steps = iterations['train'] * config["warmEpochs"] if "warmEpochs" in config else 0
    optimizer, scheduler = get_optimizer_scheduler (model = model, config=config, warmup_steps=warmup_steps)
    
    
    trainer = create_trainer( config=config, device=device, model=model, optimizer=optimizer, lr_scheduler=scheduler, loss_criterion=get_loss_criterion(config["loss"]), loaders=loaders, max_iterations=iterations, scales=config['scales'], anchors=config['anchors'])
    trainer.fit()

if __name__ == '__main__':
    main()
