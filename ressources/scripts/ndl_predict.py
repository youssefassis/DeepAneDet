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
    healthy = ["/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-209/ses-20110724",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-255/ses-20110526",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-246/ses-20110622",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-257/ses-20110517",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-202/ses-20110912",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-107/ses-20100206",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-162/ses-20111102",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-297/ses-20110316",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-267/ses-20110618",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-112/ses-20100121",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-273/ses-20110530",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-149/ses-20111126",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-295/ses-20110505",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-253/ses-20110628",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-100/ses-20100213",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-164/ses-20111124",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-160/ses-20111010",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-252/ses-20110522",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-116/ses-20091229",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-033/ses-20101021",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-198/ses-20110922",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-175/ses-20111004",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-077/ses-20100414",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-000/ses-20110101",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-176/ses-20111113",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-167/ses-20111013",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-001/ses-20101222",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-226/ses-20110621",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-349/ses-20121111",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-250/ses-20110525",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-319/ses-20110206",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-079/ses-20100502",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-065/ses-20100703",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-279/ses-20110413",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-278/ses-20110527",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-251/ses-20110702",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-284/ses-20110515",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-197/ses-20110914",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-318/ses-20110202",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-287/ses-20110506",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-276/ses-20110609",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-482/ses-20140210",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-274/ses-20110609",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-353/ses-20121011",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-245/ses-20110709",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-224/ses-20110802",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-185/ses-20111009",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-256/ses-20110526",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-242/ses-20110710",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-264/ses-20110622",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-351/ses-20121116",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-294/ses-20110429",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-293/ses-20110407",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-248/ses-20110602",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-053/ses-20100808",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-109/ses-20100120",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-021/ses-20101111",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-183/ses-20111002",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-214/ses-20110901",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-299/ses-20110422",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-240/ses-20110701",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-187/ses-20111023",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-324/ses-20110210",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-291/ses-20110330",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-282/ses-20110528",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-005/ses-20101215",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-347/ses-20121115",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-144/ses-20111211",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-275/ses-20110612",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-172/ses-20111110",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-339/ses-20110125",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-047/ses-20100908",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-239/ses-20110628",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-178/ses-20111105",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-165/ses-20111105",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-262/ses-20110620",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-286/ses-20110530",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-073/ses-20100426",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-337/ses-20110120",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-271/ses-20110518",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-179/ses-20111101",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-222/ses-20110808",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-119/ses-20091226",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-344/ses-20121218",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-151/ses-20111130",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-111/ses-20100208",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-306/ses-20110328",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-316/ses-20110219",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-204/ses-20110727",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-189/ses-20110927",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-277/ses-20110604",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-266/ses-20110421",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-280/ses-20110411",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-091/ses-20100405",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-045/ses-20101006",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-159/ses-20111127",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-092/ses-20100313",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-354/ses-20121111",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-270/ses-20110522",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-007/ses-20101225",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-352/ses-20121128",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-243/ses-20110620",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-105/ses-20100212",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-171/ses-20111105",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-015/ses-20101111",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-036/ses-20101030",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-215/ses-20110721",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-288/ses-20110503",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-128/ses-20120124",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-035/ses-20101019",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-308/ses-20110401",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-006/ses-20101217",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-186/ses-20110925",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-134/ses-20111127",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-002/ses-20110127",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-269/ses-20110608",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-131/ses-20120103",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-261/ses-20110615",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-069/ses-20100513",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-078/ses-20100430",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-037/ses-20100921",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-345/ses-20121218",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-188/ses-20111013",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-030/ses-20101012",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-268/ses-20110522",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-228/ses-20110626",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-334/ses-20110101",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-304/ses-20110409",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-265/ses-20110507",  "/home/yassis/NeuroDL/Data/CHUV/healthy_patients/sub-039/ses-20101028" ]
    test_list = test_list+healthy
    
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