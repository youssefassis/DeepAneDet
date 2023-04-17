import os
import json

cfg_name = "/home/yassis/Data/0Work/BaseConfig/ndl_config.json"
with open(cfg_name, 'r') as f:
    config = json.load(f)

# Training parameters
config['test_name'] = 'ndl_Train000'

config['normalize'] = 'Linear' # patient data normalization: use 'Linear' or 'Normal' or comment line
config["patch_shape"] = (48, 48, 48)  # dimension of a patch in voxels
config["patch_size"] = (19, 19, 19) #(19, 19, 19) # 0.4 mm resolution # size of a patch in mm (ie resolution=patch_size/patch_shape)
config["batch_size"] = 16 # Training batch size
config["validation_batch_size"] = config["batch_size"]*2 #10 # Validation batch size

config['loss'] = 'BCEWithLogitsLoss' # loss function
config['metrics'] = 'Dice' # metrics

config["model"] = "unet3d" # see utils.get_model for supported models: {unet3d, nnUnet, AttentionUNet3D}
config["layer_order"] = "cbr"
config["in_channels"] = 1
config["out_channels"] = 1

config["ds"] = False # Deep supervision
config["final_sigmoid"] = True # last activation function
config["attention"] = False # Attention mechanism
config["dropout"] = False

config['depth'] = 4 # unet3d depth
config['n_base_filters'] = 32 # number of initial filters used in unet3d model

config["n_epochs"] = 150  # cutoff the training after this many epochs
config["early_stop"] = 20  # training will be stopped after this many epochs without the validation loss improving

# LR Schuduler
config["lr"] = "WarmupPlateau" # Learining rate Scheduler's name,if no scheduler use =None
config["warmEpochs"] = 5
config["patience"] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["learning_rate_drop"] = 0.7 # factor by which the learning rate will be reduced

# Optimizer
config["min_lr"] = 1e-8
config["initial_learning_rate"] = 0.0001 # initial learning rate
config["weight_decay"] = 0.0001

# augmentation parameters
## for negative samples
config["negative sample shift"] = 10  # random shift (in mm)
config["negative sample rotation"] = 180  # random rotation around patch center (in degrees)
config["negative sample distortion"] = None  # random amplitude for non rigid distortion (in mm)
config["negative patch centers"]='points.csv'

## for positive samples
config["positive sample shift"] = config["negative sample shift"]  # random shift (in mm)
config["positive sample rotation"] = config["negative sample rotation"]  # random rotation around patch center (in degrees)
config["positive sample distortion"] = 3  # random amplitude for non rigid distortion (in mm)
config["positive duplicates"] = 50 # number of duplicates for each aneurysm
config["flip_prob"] = 0.3  # Apply random mirroring to 30% max (see specified prob in Volume.getPatchAndTruth); else False

config['test_dir'] = os.path.join(config['base_dir'],'0Work',config['test_name'])
config["model_file"] = os.path.join(config['test_dir'], "last_checkpoint.pytorch")

if not os.path.exists(config['test_dir']):
    os.mkdir(config['test_dir'])

# saving configuration as json file in working directory
with open(os.path.join(config['test_dir'], 'ndl_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
