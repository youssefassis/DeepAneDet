#!/usr/bin/env python3
import os, json

config = dict()

config['base_dir'] = '/path/to/Patients/Directory'
config['working_dir'] = '/path/to/Working/Directory'
config["split_file"] = "/path/to/fold1.json"
config["test_name"] = "Fold1"

config["overwrite"] = False

config["volume"] = "init volume"

config["anchors"] = None
config["scales"] = [12]
config["normalize"] = "Normal"
config["patch_shape"] = [96, 96, 96]
config["patch_size"] = [0.4, 0.4, 0.4]
config["truth file"] = "tmi.csv"
config["balancedbatch"] = True
config["drop_last"] = True

config["batch_size"] = 32
config["validation_batch_size"] = 128

config["loss"] = "YOLO_Loss"
config["model"] = "YOLO"
config["anchors"] = None
config["num parameters"] = 4
config["layer_order"] = "cbl"
config["depth"] = 4
config["n_base_filters"] = 64
config["in_channels"] = 1
config["out_channels"] = 1
config["final_sigmoid"] = True
config["mixed_precision"] = True
config["pool_type"] = "conv"
config["basic_module"] = "ExtResNetBlock"

config["n_epochs"] = 200
config["optimizer"] = "SGD"
config["momentum"] = 0.98
config["weight_decay"] = 0.0005
config["lr"] = "WarmupPoly"
config["warmEpochs"] = 3
config["initial_learning_rate"] = 1e-2
config["min_lr"] = 1e-4
config["start_epoch"] = 50
config["stop_epoch"] =200

config["flip probability"] = 0.5
config["flip axes"] = [0]

config["nb neg patches"] = None
config["nb neg patches per patient"] = None
config["percentage of negative patches"] = 15

config["negative sample shift"] = 10
config["negative sample rotation"] = 180
config["negative sample distortion"] = None
config["negative patch centers"] = "points.csv"
config["negative sample scaling" ] = None
config["positive sample shift"] = 10
config["positive sample rotation"] = 180
config["positive sample distortion"] = 3
config["positive sample scaling" ] = None
config["large positive duplicates"] = 50
config["small positive duplicates"] = 50
config["save model each n epoch"] = 10

config['test_dir'] = os.path.join(config['working_dir'], config['test_name'])
config["model_file"] = os.path.join(config['test_dir'], "last_checkpoint.pytorch")

# Create training dir if not exists
if not os.path.exists(config['test_dir']):
    os.mkdir(config['test_dir'])

# Saving configuration as json file in working directory
with open(os.path.join(config['test_dir'], 'ndl_config.json'), 'w') as f:
    json.dump(config, f, indent = 2)