import os, sys, torch

from deepanedet.utils import get_optimizer_scheduler, get_model, load_training_config
from deepanedet.data.io import add_points_to_patient_data, readSplit, read_patient_data_base
from deepanedet.training.losses import get_loss_criterion

from deepanedet.training.trainer import create_trainer
from deepanedet.training.dataset import getDataloaders

def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: uv run python {sys.argv[0]} path/to/Train001")
    config = load_training_config(sys.argv[1])
    
    # Multiprocessing data loading
    workers = min(12, os.cpu_count() or 1) # data loading processes, at most one per CPU

    # Data Preparation
    train_list, _, _ = readSplit(config['split_file'], config.get('base_dir'))
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
    
    
    trainer = create_trainer( config=config, device=device, model=model, optimizer=optimizer, lr_scheduler=scheduler, loss_criterion=get_loss_criterion(config["loss"]), loaders=loaders, max_iterations=iterations, scales=config['scales'], anchors=config['anchors'] if 'anchors' in config else None)
    trainer.fit()

if __name__ == '__main__':
    main()
