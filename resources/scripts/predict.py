#!/usr/bin/env python3
import os, sys, json, torch
import Data.IO as dio
from utils import get_logger, load_model
from prediction import ndl_run_validation_cases

def main():
    try:
        with open(os.path.join(sys.argv[1], 'ndl_config.json'), 'r') as f:
            config = json.load(f)
    except:
        print(f'No such Training config file')
        exit()

    # Load Model
    logger = get_logger('Model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(config, device, last=True)
    model.eval()
    logger.info("Setting the model to evaluation mode")
    
    logger = get_logger('Data Preparation')
    train_list, valid_list, test_list = dio.readSplit(config['split_file'])
    test_list = sorted(valid_list + test_list)
    
    logger.info(f"{len(test_list)} Patients for testing")
    logger.info(f"{config['normalize']} Normalization")
    logger.info(f"from {config['volume']} volumes")
    
    margin = config['patch_shape'][0] // 6
    logger.info(f"Prediction with {margin} voxels margin")

    detections_per_patch = None
    iou_threshold = 0.01
    with torch.no_grad():
        ndl_run_validation_cases(test_list,
                                 device = device,
                                 model = model,
                                 patch_size = [x*y for x, y in zip(config['patch_size'], config['patch_shape'])],
                                 patch_dim = config['patch_shape'],
                                 
                                 input_volume=config['volume'],
                                 normalization = config['normalize'],
                                 
                                 output_dir = os.path.join(sys.argv[1], 
                                                           "Predictions", 
                                                           f"{epoch}_epochs", 
                                                           f"Validation_{iou_threshold}", 
                                                           f"{detections_per_patch}_detections_per_patch"
                                                          ),
                                 margin = margin,
                                 mirror_axes=[0, 1, 2],
                                 batch_size = 1,
                                 do_tta = True,
                                 scales=config['scales'],
                                 anchors=config['anchors'],
                                 iou_threshold=iou_threshold,
                                 detections_per_patch = detections_per_patch,
                                )

if __name__ == "__main__":
    main()