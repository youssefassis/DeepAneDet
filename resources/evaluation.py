import os, json
import numpy as np
import Data.IO as dio
from helpers import save_json, intersection_over_union

def get_CM_dict(pred_spheres, truths, iou_thr, confidence_thr, pat_name, verbose):
    '''
    This function computes the confusion matrix (CM) for the predictions. 
    Parameters:
        - The predicted aneurysms (list of lists), each prediction is a list 
        of 5 elements (3-center, 1-radius, 1-confidence)
        - The ground truth aneurysms (list of lists)
        - The Intersection over union (IoU) threshold (float)
        - The confidence threshold (float)
        - Patient name (String, e.g. P0001)
        - verbosity flag (boolean).
        
    The function first removes the predictions with a confidence score lower
    than the predefined threshold confidence_thr. Then, it loops over each 
    prediction, calculates the IoU between predicted and GT spheres.
    If the IoU score is greater than or equal to the IoU threshold, the 
    function adds the detection to a list (ious) : 
        - If there is only one ground truth sphere that matches the predicted
        sphere (len(ious)=1), then it is a true positive (TP). 
        - If there are no ground truth sphere that match the predicted sphere
        (len(ious)=0), the function marks it as a false positive (FP).
        - If a prediction covers multiple ground truth, returns and ERROR!.
    The function then computes the false negatives (FN) by iterating over the
    ground truth spheres that were not detected. 
    
    Finally, the function returns the detection table and the list of FNs. 
    The detection table is a list of dictionaries, where each dictionary 
    contains information about a detected aneurysm, such as its confidence 
    score, diameter, IoU score, TP or FP status, and the coordinates of its 
    center and endpoints. The FNs list is a list of dictionaries, where each
    dictionary contains information about a missed aneurysm, such as its 
    diameter and the coordinates of its center.    
    '''
    detection_table, detected_aneurysms_diameters = [], []    
    # Remove spheres with low confidence score than predefined confidence_thr
    pred_spheres = pred_spheres[(pred_spheres[:, -1] >= confidence_thr), :]
    if verbose: print(f"\n{pred_spheres.shape[0]} detections / {truths.shape[0]} GT aneurysms")

    # loop over detections spheres
    for det_idx, detection in enumerate(pred_spheres):
        diameter = 2 * detection[3]
        confidence = detection[-1] * 100
        if verbose: print(f"\t- Detection {det_idx+1}: confidence={round(confidence, 3)}%; diameter={round(diameter, 3)}mm")

        ious, iou_score = [], 0
        # loop over ground truth CC
        for j, truth in enumerate(truths):
            diam_gt = truth[3] * 2
            center = truth[:3].tolist()
            iou_score = intersection_over_union(prediction = detection[:4], truth = truth[:4])
            if iou_score >= iou_thr:
                ious.append([iou_score, diam_gt, center, truth[4:7].tolist(), truth[7:10].tolist()])
            if verbose: print(f"\t\t-> vs truth {j+1}: iou={round(iou_score * 100, 3)}%, gt_diam={round(diam_gt, 3)}mm")

        if len(ious) == 0:
            detection_table.append({'Confidence': confidence, 
                                    'Diameter': diameter, 
                                    'Center': detection[:3].tolist(), 
                                    'IoU': 0, 
                                    'TP': 0, 
                                    'FP': 1
                                   })
        elif len(ious) == 1:
            if len(detected_aneurysms_diameters) > 0 and (ious[0][1] in detected_aneurysms_diameters): # check if this detection is already detected                
                print(f'\t\tAneurysm already detected by another prediction ({pat_name})')
                TP, FP = 0, 1
            else:
                TP, FP = 1, 0
                detected_aneurysms_diameters.append(ious[0][1])
            detection_table.append({'Confidence': confidence, 
                                    'Diameter': diameter, 
                                    'Center': detection[:3].tolist(), 
                                    'GT diameter': ious[0][1], 
                                    'IoU': ious[0][0], 
                                    'GT Center': ious[0][2], 
                                    'TP': TP, 
                                    'FP': FP
                                   })
        else:
            raise ValueError('Large Detection that covers multiple aneurysms')
    # FNs
    FNs = []
    for gt in truths:
        diameter = gt[3]*2
        if diameter not in detected_aneurysms_diameters:
            FNs.append({'Center': gt[:3].tolist(), 'Diameter': gt[3]*2})
    return detection_table, FNs

def get_detections(patients, iou_thr=0.1, confidence_thr=0.05, dir_name="Labels", max_per_patient=None, truth_file_name='F.csv', verbose=False):
    detections, FNs_cases, FN_diams = [], [], []
    for patient in patients:
        pat_name = patient.split("/")[-1].split(".")[0]
        if verbose: print(f"{pat_name}", end = ", ")
        
        # Get Prediction Spheres
        with open(os.path.join("/".join(patient.split("/")[:-1]), dir_name, f"{pat_name}.json"), "r") as f:
            predictions = json.load(f)
        pred_spheres = np.empty((0, 5))
        for idx, key in enumerate(predictions):
            pred_spheres = np.vstack((pred_spheres, np.hstack((predictions[key]["center"], 
                                                               predictions[key]["radius"], 
                                                               predictions[key]["confidence"]))))
            if max_per_patient is not None and pred_spheres.shape[0] == max_per_patient: break
        del predictions
        
        # Get Truth Spheres
        truth_file = os.path.join(dio.get_patient_dir_from_name(pat_name), truth_file_name)        
        truths = dio.points_to_spheres(dio.read_points_from_csv(truth_file)) if os.path.isfile(truth_file) else np.empty((0, 4))
        
        detection_table, FNs = get_CM_dict(pred_spheres, truths, iou_thr = iou_thr, confidence_thr = confidence_thr, verbose = verbose, pat_name = pat_name)
        for FN in FNs:
            FN['Patient'] = pat_name
            FNs_cases.append( FN)
                    
        for det in detection_table:
            det['Patient'] = pat_name
            det['FNs diameters'] = [f['Diameter'] for f in FNs]
            detections.append(det)
    
    return sorted(detections, key=lambda d: d['Confidence'], reverse=True) , FNs_cases