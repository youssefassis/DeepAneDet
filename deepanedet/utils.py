import json, logging, math, os, sys, shutil
import numpy as np
import torch

def load_training_config(train_dir):
    '''
    Reads the ndl_config.json of a training directory (see scripts/prepare.py). Missing scales default to one
    detection grid cell per 8 patch voxels.
    '''
    with open(os.path.join(train_dir, 'ndl_config.json'), 'r') as f:
        config = json.load(f)
    if config.get('scales') is None:
        config['scales'] = [math.ceil(x/8) for x in config['patch_shape']]
    return config

def get_model(config, default_init="He", device='cpu'):
    logger = get_logger('Model')

    # select adequate model
    model = select_model(config, logger)
    
    # model properties
    nparams = get_number_of_learnable_parameters(model)
    size = (sum([param.nelement()*param.element_size() for param in model.parameters()]) + sum([buf.nelement()*buf.element_size() for buf in model.buffers()])) * 1e-6
    logger.info(f'Number of learnable params {nparams}')
    logger.info(f'Size allocated by the model is {size:.2f} MB')

    # Model weights initialization
    if default_init == "He":
        logger.info("Apply He/Kaiming (Keras) weight initialization")
        initialize_weights(model)
    else:
        logger.info("Apply Xavier (default PyTroch) weight initialization")

    # Send model to device
    logger.info(f"Sending the model to '{device}'")
    return model.to(device)

def select_model(config, logger):
    # Instantiate the model
    if config["model"] == "ASSIS":
        from deepanedet.models.assis_model import AssisModel
        from deepanedet.models.building_blocks import DoubleConv, ExtResNetBlock
        
        basic_module = ExtResNetBlock if 'basic_module' in config and config['basic_module']=="ExtResNetBlock" else DoubleConv
        params_per_box = config['num parameters'] if 'num parameters' in config else 4
        anchors_per_scale = len(config["anchors"]) if "anchors" in config else 1

        model = AssisModel(   in_channels=config['in_channels'], #out_channels=config['out_channels'],
                        f_maps=config['n_base_filters'], layer_order=config["layer_order"], pool_type='conv', 
                        num_levels = config['depth'], basic_module=basic_module, 
                        conv_kernel_size=3, pool_kernel_size=2, conv_padding=1,
                        params_per_element = params_per_box, 
                        nb_scales=len(config['scales']),   
                        anchors_per_scale= anchors_per_scale, 
                    )

    else:
        raise ValueError(f"Unsupported model '{config['model']}'")
    logger.info(f"The model '{config['model']}' is selected")
    return model

def load_model(config, device="cpu", last=False):
    ''' Instansiate and load the appropriate (Last or Best) model weights
        And move the model to the specified device '''
    # Instantiate model
    model = get_model(config, device=device)
    if last is None:
        return model, 0

    if not os.path.exists(config["model_file"]):
        raise IOError(f"Checkpoint '{config['model_file']}' does not exist")
    
    # load pre-trained weights
    if last == True:
        state = load_checkpoint(config["model_file"], model)
    elif last == False:
        state = load_checkpoint(config["model_file"].replace("last", "best"), model)
    return model, state['epoch']

def save_checkpoint(state, is_best, checkpoint_dir, epoch = None, logger = None):
    '''Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    '''
    def log_info(message):
        if logger is not None:
            logger.info(message)
    if epoch is not None:
        model_name = os.path.join(checkpoint_dir, f"{epoch}_checkpoint.pytorch")
        torch.save(state, model_name)
        log_info(f"Saving checkpoint ({epoch} epochs) to '{model_name}'")
        return False
    
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)
        return True
    return False

def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    '''Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    '''
    logger = get_logger('Model Configuration')
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])
    logger.info(f"Checkpoint loaded ( {state['epoch']} epochs; metric: {state['last_eval_score']})")
    return state

def get_optimizer_scheduler(model, config, warmup_steps):
    '''
    Returns the optimizer and the LR scheduler
    '''
    optimizer = create_optimizer(model = model,
                                 config = config)
    scheduler = create_lr_scheduler(optimizer = optimizer,
                                       config = config,
                                       warmup_steps = warmup_steps)
    return optimizer, scheduler

def create_optimizer(model, config):
    ''' 
    Creates the learning optimizer
    '''    
    logger = get_logger('Optimizer')
    optim = config["optimizer"] if "optimizer" in config else "Adam"
    assert optim in ["Adam", "SGD"]
    
    weight_decay = config["weight_decay"]
    learning_rate = 1e-8 if str(config["lr"]).startswith("Warmup") else config["initial_learning_rate"] # warmup schedulers raise it

    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
        op = optimizer.state_dict()["param_groups"][0]
        logger.info (f"{optim} : lr = {op['lr']}; betas = {op['betas']}; eps = {op['eps']}; weight_decay = {op['weight_decay']}; amsgrad = {op['amsgrad']}")
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=config['momentum'], nesterov=True)
        op = optimizer.state_dict()["param_groups"][0]
        logger.info (f"{optim} : lr = {op['lr']}; momentum = {op['momentum']}; weight_decay = {op['weight_decay']}; nesterov = {op['nesterov']}")
    return optimizer

def create_lr_scheduler(optimizer, config, warmup_steps=0):
    ''' 
    Creates Learning rate scheduler (LR update strategy)
    '''
    assert config["lr"] in [None, "ReduceLROnPlateau", "Poly", "Warmup", "WarmupPlateau", "WarmupPoly"]

    logger = get_logger('Scheduler')
    if config["lr"] is None:
        scheduler = None
        logger.info("No scheduler specified")

    elif config["lr"] == "Warmup":
        from deepanedet.training.schedulers import WarmupLRScheduler
        scheduler = WarmupLRScheduler(optimizer = optimizer,
                                      min_lr = config['min_lr'],
                                      max_lr = config['initial_learning_rate'],
                                      warmup_steps= warmup_steps)
        logger.info(f"WarmupLR Scheduler: warmup_steps = {warmup_steps}")

    elif config["lr"] == "ReduceLROnPlateau":
        from deepanedet.training.schedulers import ReduceLROnPlateau
        factor, patience, wloss = config["factor"], config["patience"], True#config["wloss"]
        scheduler = ReduceLROnPlateau(optimizer = optimizer,
                                      factor = factor,
                                      patience = patience,
                                      wloss=wloss,
                                      start_epoch = 0)
        logger.info(f"ReduceLROnPlateau Scheduler: patience={patience}, factor={factor}, wloss={wloss}")

    elif config["lr"] == "Poly":
        from deepanedet.training.schedulers import Poly
        start_epoch = config["start_epoch"] 
        scheduler = Poly(optimizer = optimizer,
                         start_epoch = start_epoch,
                         max_epochs = config["n_epochs"],
                         exponent = 0.99,
                         min_lr = 1e-4)
        logger.info(f"Polynomial Scheduler: start_epoch={start_epoch}, max_epochs={config['n_epochs']}")

    elif config["lr"] == "WarmupPlateau":
        from deepanedet.training.schedulers import WarmupScheduler, ReduceLROnPlateau
        start_epoch = 10
        scheduler = WarmupScheduler(optimizer = optimizer, 
                                        min_lr = 1e-8,
                                        max_lr = config['initial_learning_rate'],
                                        warmup_steps = warmup_steps,
                                        after_warmup_scheduler = ReduceLROnPlateau(optimizer = optimizer,
                                                                                   factor = config["factor"], 
                                                                                   patience = config["patience"],
                                                                                   wloss = True,
                                                                                   start_epoch = start_epoch))
        logger.info(f"WarmupPlateau scheduler : lr={config['initial_learning_rate']}, patience={config['patience']}, factor={config['factor']}, start_epoch={start_epoch}")
    elif config["lr"] == "WarmupPoly":
        from deepanedet.training.schedulers import WarmupScheduler, Poly
        start_epoch = config["start_epoch"] 
        stop_epoch = config["stop_epoch"] 
        scheduler = WarmupScheduler(optimizer = optimizer, 
                                    min_lr = 1e-8,
                                    max_lr = config['initial_learning_rate'],
                                    warmup_steps = warmup_steps,
                                    after_warmup_scheduler = Poly(optimizer = optimizer,
                                                                  start_epoch = start_epoch,
                                                                  max_epochs = stop_epoch, #config["n_epochs"],
                                                                  exponent = 0.9, 
                                                                  min_lr = config['min_lr'],
                                                                  initial_lr=config['initial_learning_rate']))
        logger.info(f"WarmupPoly scheduler : lr={config['initial_learning_rate']}, start_epoch={start_epoch}")
    return scheduler

def get_number_of_steps(n_samples, batch_size):
    '''
    This function calculates the number of steps needed to complete an epoch in a training 
    loop based on the number of samples and the batch size. 
    It takes two arguments: 
        - n_samples, an integer representing the total number of samples in the dataset, 
        - batch_size, an integer representing the number of samples per batch.
    '''
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

class RunningAverage:
    '''
    Computes and stores the average (loss and metric scores)
    '''
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

def get_number_of_learnable_parameters(model):
    '''
    Returns the number of trainable parameters of the model
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def initialize_weights(*models, neg_slope: float = 1e-2):
    '''
    Initialize a model's weights to He initialization
    '''
    for model in models:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                module.weight = torch.nn.init.kaiming_normal_(module.weight, a=neg_slope)
                if module.bias is not None:
                    module.bias = torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.BatchNorm3d):
                module.weight.data.fill_(1) # gamma
                if module.bias is not None:
                    module.bias = torch.nn.init.constant_(module.bias, 0)

loggers = {}
def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        loggers[name] = logger
        return logger
               
