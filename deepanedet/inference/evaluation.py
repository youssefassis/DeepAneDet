import os, json
import numpy as np
from deepanedet.data import io as dio
from deepanedet.helpers import intersection_over_union

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
    than the predefined threshold confidence_thr. Then, from the most to the
    least confident, it computes the IoU between each predicted sphere and
    every GT sphere:
        - If it overlaps (IoU >= iou_thr) GT spheres that are not detected yet,
        it is a true positive (TP) of the most overlapping one.
        - If it only overlaps GT spheres already detected by a more confident
        prediction, it is a false positive (FP) duplicate.
        - If it overlaps no GT sphere, it is a false positive (FP).
    The GT spheres left undetected are the false negatives (FN).
    
    Finally, the function returns the detection table and the list of FNs. 
    The detection table is a list of dictionaries, where each dictionary 
    contains information about a detected aneurysm, such as its confidence 
    score, diameter, IoU score, TP or FP status, and the coordinates of its 
    center (and those of the matched GT sphere). The FNs list is a list of
    dictionaries, where each dictionary contains information about a missed
    aneurysm, such as its diameter and the coordinates of its center.    
    '''
    detection_table = []
    detected = np.zeros(truths.shape[0], dtype=bool)
    # Remove spheres with low confidence score than predefined confidence_thr
    pred_spheres = pred_spheres[(pred_spheres[:, -1] >= confidence_thr), :]
    pred_spheres = pred_spheres[np.argsort(-pred_spheres[:, -1], kind='stable')] # most confident first
    if verbose: print(f"\n{pred_spheres.shape[0]} detections / {truths.shape[0]} GT aneurysms")

    # loop over detections spheres
    for det_idx, detection in enumerate(pred_spheres):
        diameter = 2 * detection[3]
        confidence = detection[-1] * 100
        if verbose: print(f"\t- Detection {det_idx+1}: confidence={round(confidence, 3)}%; diameter={round(diameter, 3)}mm")

        ious = np.array([intersection_over_union(prediction = detection[:4], truth = truth[:4]) for truth in truths])
        if verbose:
            for j, (iou_score, truth) in enumerate(zip(ious, truths)):
                print(f"\t\t-> vs truth {j+1}: iou={round(iou_score * 100, 3)}%, gt_diam={round(truth[3] * 2, 3)}mm")

        overlapping = ious >= iou_thr
        if not overlapping.any():
            detection_table.append({'Confidence': confidence, 
                                    'Diameter': diameter, 
                                    'Center': detection[:3].tolist(), 
                                    'IoU': 0, 
                                    'TP': 0, 
                                    'FP': 1
                                   })
            continue

        candidates = overlapping & ~detected
        if candidates.any():
            match = np.argmax(np.where(candidates, ious, -1))
            detected[match] = True
            TP, FP = 1, 0
        else:
            match = np.argmax(ious)
            print(f'\t\tAneurysm already detected by another prediction ({pat_name})')
            TP, FP = 0, 1
        detection_table.append({'Confidence': confidence, 
                                'Diameter': diameter, 
                                'Center': detection[:3].tolist(), 
                                'GT diameter': truths[match][3] * 2, 
                                'IoU': ious[match], 
                                'GT Center': truths[match][:3].tolist(), 
                                'TP': TP, 
                                'FP': FP
                               })
    # FNs
    FNs = [{'Center': gt[:3].tolist(), 'Diameter': gt[3]*2} for gt in truths[~detected]]
    return detection_table, FNs

def get_detections(predictions_dir, patient_dirs, truth_file_name, iou_thr=0.1, confidence_thr=0.05, max_per_patient=None, verbose=False):
    '''
    Matches the predictions saved in predictions_dir (<patient name>.json, see deepanedet.data.io.patient_name) with the
    ground truth aneurysms of each patient directory (truth_file_name, a CSV of point pairs; none if missing).
    Returns the detections sorted by decreasing confidence, and the missed aneurysms (FNs).
    '''
    detections, FNs_cases = [], []
    for patient_dir in patient_dirs:
        pat_name = dio.patient_name(patient_dir)
        if verbose: print(f"{pat_name}", end = ", ")
        
        # Get Prediction Spheres
        with open(os.path.join(predictions_dir, f"{pat_name}.json"), "r") as f:
            predictions = json.load(f)
        pred_spheres = np.empty((0, 5))
        for idx, key in enumerate(predictions):
            pred_spheres = np.vstack((pred_spheres, np.hstack((predictions[key]["center"], 
                                                               predictions[key]["radius"], 
                                                               predictions[key]["confidence"]))))
            if max_per_patient is not None and pred_spheres.shape[0] == max_per_patient: break
        del predictions
        
        # Get Truth Spheres
        truth_file = os.path.join(patient_dir, truth_file_name)
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