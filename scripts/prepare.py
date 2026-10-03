import argparse, os, json

def parse_args():
    parser = argparse.ArgumentParser(description="Creates a training directory <work_dir>/<name> with its ndl_config.json. "
                                                 "The other training settings are set below in this file.")
    parser.add_argument("data_dir", help="directory of the patients")
    parser.add_argument("work_dir", help="directory of the trainings")
    parser.add_argument("split_file", help="training, validation and testing patients (e.g. reproducibility/fold1.json)")
    parser.add_argument("--name", help="training directory name (default: the split file name, e.g. fold1)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = dict()
    
    config['base_dir'] = os.path.abspath(args.data_dir)
    config['working_dir'] = os.path.abspath(args.work_dir)
    config["split_file"] = os.path.abspath(args.split_file)
    config["test_name"] = args.name or os.path.splitext(os.path.basename(args.split_file))[0]
    
    config["overwrite"] = False
    
    config["volume"] = "init volume"
    
    config["normalize"] = "Normal"
    config["patch_shape"] = [96, 96, 96]
    config["patch_size"] = [0.4, 0.4, 0.4]
    config["truth file"] = "aneurysms.csv"
    config["balancedbatch"] = True
    config["drop_last"] = True
    
    config["batch_size"] = 32
    config["validation_batch_size"] = 128
    
    config["loss"] = "YOLO_Loss"
    config["model"] = "ASSIS"
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
    config["positive duplicates"] = 50
    #config["large positive duplicates"] = 50
    #config["small positive duplicates"] = 50
    config["save model each n epoch"] = 10
    
    config['test_dir'] = os.path.join(config['working_dir'], config['test_name'])
    config["model_file"] = os.path.join(config['test_dir'], "last_checkpoint.pytorch")
    save_config(config)

def save_config(config):
    '''
    Saves the configuration as ndl_config.json in the training directory, creating it if needed
    '''
    os.makedirs(config['test_dir'], exist_ok=True)
    with open(os.path.join(config['test_dir'], 'ndl_config.json'), 'w') as f:
        json.dump(config, f, indent = 2)

if __name__ == '__main__':
    main()
