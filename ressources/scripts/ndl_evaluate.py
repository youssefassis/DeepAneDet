#!/usr/bin/env python3
import os, json, random, sys
import numpy as np
import pandas as pd
import Data.IO as dio
from sklearn.metrics import auc
from importlib import reload
import matplotlib.pyplot as plt
import evaluation
import statistics
import cmath
import scipy.stats as stats

def main():
    predictions_dir = sys.argv[1]
    print(predictions_dir)
    if not os.path.isdir(predictions_dir):
        print(f'No directory found to load predictions')
        exit()

    iou_threshold = 0.1
    confidence_threshold = 0.01


    # Only one fold
    train_dir = predictions_dir[: predictions_dir.find("Predictions")]
    out_dir = os.path.join(os.getcwd(), predictions_dir)
    try: 
        with open(os.path.join(train_dir, 'ndl_config.json'),'r') as f:
            config = json.load(f)
    except:
        with open(os.path.join("/path/to/0Work/YOLO/CHUV_final/5Folds_validation/Fold1_1_10_translation/", 'ndl_config.json'), 'r') as f:
            config = json.load(f)
    split_patients = config['split_file']
    
    if isinstance(predictions_dir, list):
        patients = []
        for d in predictions_dir:
            if os.path.isdir(d):
                print(d)
                patients += [os.path.join(d, f) for f in sorted(os.listdir(d)) if ".json" in f]
        patients = sorted (patients)
    else:
        patients = sorted([os.path.join(predictions_dir, f) for f in sorted(os.listdir(predictions_dir)) if ".json" in f])
    print(len(patients), "patients for test")

    # reload(evaluation)
    detections, FNs = evaluation.get_detections(patients,
                                                iou_thr=iou_threshold,
                                                confidence_thr=confidence_threshold,
                                                dir_name=".",
                                                truth_file_name='F.csv',
                                                max_per_patient=None,
                                                verbose=None)
    
    # Generate Dataframe
    tot_aneurysms = sum([det['TP'] for det in detections]) + len(FNs)

    AccTP, AccFP = 0, 0
    for cmp, detection in enumerate(detections):
        AccTP += detection['TP']
        AccFP += detection['FP']
        detection['id'] = cmp + 1
        detection['Accumulate TP'] = AccTP
        detection['Accumulate FP'] = AccFP
        detection['Precision'] = AccTP/(AccTP + AccFP)
        detection['Recall'] = AccTP/(tot_aneurysms)

    ## Save Detections to csv file
    df = pd.DataFrame(detections)
    df = df.fillna('')
    columns = ["Patient", "id", "Confidence", "Diameter", "Center", "IoU", "TP", "FP", "Accumulate TP", "Accumulate FP", "Precision", "Recall", "FNs diameters", "GT Center", "GT diameter"]
    df = df.reindex(columns=columns)

    df.to_csv(os.path.join(out_dir, f"detection_evaluation@{iou_threshold}.csv"), index=False)
    print(f"CSV file saved {out_dir}")

    confidences = [f['Confidence'] for f in detections]
    ious = [f['IoU'] for f in detections]

    confidence_TP = [f['Confidence'] for f in detections if f['TP']]
    diameters_TP = [f['Diameter'] for f in detections if f['TP']]
    IoU_TP = [f['IoU']*100 for f in detections if f['TP']]

    diameters_TP_GT = [f['GT diameter'] for f in detections if f['TP']]

    confidence_FP = [f['Confidence'] for f in detections if f['FP']]
    diameters_FP = [f['Diameter'] for f in detections if f['FP']]
    IoU_FP = [f['IoU'] for f in detections if f['FP'] and f['IoU']>0]

    diameters_FN = [f['Diameter'] for f in FNs]

    assert len(confidences) == len(ious) == len(diameters_TP+diameters_FP)
    assert len(diameters_TP) == len(diameters_TP_GT) == len(confidence_TP)
    assert len(confidence_FP) == len(confidence_FP)

    try:
        epoch = int([s for s in predictions_dir[0].split("/") if "epochs" in s][0].split('_')[0]) if isinstance(predictions_dir, list) else int([s for s in predictions_dir.split("/") if "epochs" in s][0].split('_')[0])
    except:
        epoch = 60

    ########################################################################################################
    # Confusion matrix
    plt.figure()
    fig, ((ax, ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax9, ax10, ax11, ax12, ax13, ax8,)) = plt.subplots(nrows=2, ncols=7, figsize=(45,10))
    #fig.subplots_adjust(top=0.82)
    fig.tight_layout(pad=6.0)
    fig.suptitle (f"Evaluation at {iou_threshold} IoU ({epoch} epochs)", fontsize=14)

    # Fist figure: Confidence vs Diameter
    ax.scatter(diameters_TP, confidence_TP, c='green', label=f"TP ({len(confidence_TP)})", alpha=1)
    ax.scatter(diameters_FP, confidence_FP, c='red', label=f"FP ({len(confidence_FP)})", alpha=0.4)
    ax.scatter(diameters_FN, [-10 for _ in range(len(diameters_FN))], c='k', marker='x', label=f"FN ({len(diameters_FN)})", alpha=0.6)
    ax.set_title(f'Sensitivity = Recall = {round(len(confidence_TP)/(len(confidence_TP)+len(FNs))*100, 2)}%\nFPs/case = {round(len(confidence_FP)/len(patients), 3)}')
    ax.set_xlabel('Diameter (mm)')
    ax.set_ylabel('Confidence score (%)')
    ax.set_xlim((-0.5, 22))
    ax.set_ylim((-18, 110))
    ax.legend(loc="center right", framealpha=0.7)
    ax.grid(True)

    # Sesitiviy vs Diameters at 0.5 detection threshold
    # Fist figure: Confidence vs Diameter

    detections_tmp = [det for det in detections if det['Confidence']>=50]
    diameters_TP_tmp = [det['Diameter'] for det in detections_tmp if det['TP']]
    confidence_TP_tmp = [det['Confidence'] for det in detections_tmp  if det['TP']]
    diameters_FP_tmp = [det['Diameter'] for det in detections_tmp  if det['FP']]
    confidence_FP_tmp = [det['Confidence'] for det in detections_tmp  if det['FP']]
    diameters_FN_tmp = FNs+[{'Center': det['GT Center'], 'Diameter': det['GT diameter'], 'Patient': det['Patient']} for det in detections if det['Confidence']<50 and det['TP']]
    diameters_FN_tmp = [det['Diameter'] for det in diameters_FN_tmp]
    ax1.scatter(diameters_TP_tmp, confidence_TP_tmp, c='green', label=f"TP ({len(confidence_TP_tmp)})", alpha=1)
    ax1.scatter(diameters_FP_tmp, confidence_FP_tmp, c='red', label=f"FP ({len(confidence_FP_tmp)})", alpha=0.4)
    ax1.scatter(diameters_FN_tmp, [-10 for _ in range(len(diameters_FN_tmp))], c='k', marker='x', label=f"FN ({len(diameters_FN_tmp)})", alpha=0.6)
    ax1.set_title(f'Detection threshold = 50%\nSensitivity = Recall = {round(len(confidence_TP_tmp)/(len(confidence_TP_tmp)+len(diameters_FN_tmp))*100, 2)}%; FPs/case = {round(len(confidence_FP_tmp)/len(patients), 3)}')
    ax1.set_xlabel('Diameter (mm)')
    ax1.axhline(50, linestyle=':')
    ax1.set_ylabel('Confidence score (%)')
    ax1.set_xlim((-0.5, 22))
    ax1.set_ylim((-18, 110))
    ax1.legend(loc="center right", framealpha=0.7)
    ax1.grid(True)

    # Second figure: TP & FPs confidence probabilities distribution
    bins = np.arange(0, 100 + 1, 1)
    ax2.hist(confidence_TP, bins, facecolor='green', alpha=0.5, edgecolor='black', density = False, linewidth=1, label="TP")
    ax2.hist(confidence_FP, bins, facecolor='red', alpha=0.3, edgecolor='black', density = False, linewidth=1, label="FP")
    ax2.set_title('Confidence distribution')
    ax2.set_xlabel('Confidence score')
    ax2.set_ylabel('Number of detections')
    ax2.legend(framealpha=0.7)

    # Third figure: Average Precision curve
    precision_scores = [1]+[f['Precision'] for f in detections] + [ 0, 0]
    recall_scores = [0]+[f['Recall'] for f in detections] 
    if len(recall_scores)>1:
        recall_scores = recall_scores + [recall_scores[-1], 1]
    else:
        recall_scores = recall_scores + [1, 1]
    # compute interpolated precision
    i = len(recall_scores) - 2
    interpolated_precision_scores = precision_scores[:]
    while i >= 0:
        if interpolated_precision_scores[i+1] > interpolated_precision_scores[i]:
            interpolated_precision_scores[i] = interpolated_precision_scores[i+1]
        i = i-1

    ax3.plot(recall_scores, precision_scores, 'b--', label='Precision')
    ax3.plot(recall_scores, interpolated_precision_scores, 'g-', label='Interpolated Precision')
    ax3.set_title(f'Average Precision = {round(auc(np.array(recall_scores), np.array(interpolated_precision_scores))*100, 3)} %')
    ax3.set_xlabel('Sensitivity (Recall)')
    ax3.set_ylabel('Precision')
    ax3.legend(loc="lower left")
    ax3.grid(True)

    # Fourth figure: FROC curve
    sensitivities, fps_per_case = [], []
    for threshold in range(1, 101, 1):
        dets_tmp = [f for f in detections if f['Confidence']>=threshold]
        confidence_TP_tmp = [f['Confidence'] for f in dets_tmp if f['TP']]
        confidence_FP_tmp = [f['Confidence'] for f in dets_tmp if f['FP']]
        fps_per_case.append(len(confidence_FP_tmp)/len(patients))
        sensitivities.append((len(confidence_TP_tmp)/tot_aneurysms)*100)

    ax4.plot(fps_per_case + [0], sensitivities + [0],  '-', color="green")
    ax4.set_title(f'FROC Curve (AUC={round( 100 - auc([100]+sensitivities, [100]+[(f/max(fps_per_case))*100 for f in fps_per_case]) /100, 3)} %)')

    ax4.set(xlabel='FP/case', ylabel='Sensitivity (Recall) ')
    ax4.grid()
    ax4.set_ylim((0, 105))

    # Calibration Diagram
    ax5.plot([d['Precision']*100 for d in detections], [d['Confidence'] for d in detections], marker='x')
    ax5.set_title(f"Reliability Curve (Calibration)" )
    ax5.set_xlabel('Precision (%)')
    ax5.set_ylabel("Confidence (%)")
    ax5.plot([0,100], [0,100], linestyle='--',color='green')
    ax5.legend(['Ours', 'Ideal curve'])

    # Detections per patient
    patients_name = sorted(list(dict.fromkeys([f['Patient'] for f in detections]+[f['Patient'] for f in FNs])))
    ax6.set_title ('Detections per patient')
    ax6.set_xlabel('Patients')
    ax6.set_ylabel('Confidence (%)')
    ax6.set_xticks([f for f in range(1, len(patients_name)+1)], [f for f in patients_name], fontsize=4, rotation=90)
    ax6.grid(True)

    for cmp, patient_name in enumerate(patients_name):
        # FNs
        v = -4
        for fn in FNs:
            if fn['Patient']==patient_name:
                ax6.plot(cmp+1, v, '*', color='black', alpha=0.6)
                v -= 4
        for detection in detections:
            if detection['Patient'] == patient_name:
                # TPs
                if detection['TP']==1:
                    ax6.plot(cmp+1, detection['Confidence'], 'x', color='green', alpha=0.9)
                else:
                    ax6.plot(cmp+1, detection['Confidence'], 'x', color='red', alpha=0.9)


    # IoU vs Confidence
    ax7.scatter(IoU_TP, confidence_TP, c='green', label=f"TP ({len(IoU_TP)})", alpha=1)
    ax7.scatter(IoU_FP, [f['Confidence'] for f in detections if f['FP'] and f['IoU']>0], c='red', label=f"FP ({len(IoU_FP)})", alpha=1)
    ax7.axvline(x=10, linestyle=':')
    ax7.set_title(f'IoU vs Confidence\navg = {round(sum(IoU_TP)/len(IoU_TP), 2)}% ± {round(np.std(np.array(IoU_TP)),2)}%; max={round(max(IoU_TP),2)}%; min = {round(min(IoU_TP), 2)}%')
    ax7.set_xlabel('IoU (%)')
    ax7.set_ylabel('Confidence score (%)')
    ax7.set_xlim((-5, 100))
    ax7.set_ylim((-5, 105))
    ax7.legend(loc="center right", framealpha=0.4)
    ax7.grid(True)

    # Last figure
    if isinstance(split_patients, list): 
        split_patients = split_patients[0]

    with open(split_patients, 'r') as f:
        config = json.load(f)
    train_pats, val_pats, test_pats = config["training list"], config["validation list"], config["testing list"]
    test_pats = val_pats+test_pats

    train_aneurysms, test_aneurysms = [], []
    for pat in train_pats:
        train_aneurysms += dio.SizeAnev(pat)
    for pat in test_pats:
        test_aneurysms += dio.SizeAnev(pat)

    ax8.set_xticks([i for i in range(0, int(np.floor(max(train_aneurysms+test_aneurysms)))+1, 2)])
    ax8.hist(train_aneurysms, bins=range(0, 20), alpha=0.3, color='red',  edgecolor='black', linewidth=1.5)
    ax8.hist(test_aneurysms, bins=range(0, 20), alpha=0.3, color='blue', edgecolor='black', linewidth=1.5)

    ax8.set_title('Training/Test aneurysm size distribution')
    ax8.set_xlabel('Diameter (mm)')
    ax8.set_ylabel('Count')
    ax8.legend([f"Train ({len(train_aneurysms)} ans/ {len(train_pats)} patients)", f"Test ({len(test_aneurysms)} ans/ {len(test_pats)} patients)"])
    ax8.grid(False)

    difference1 = [((i-j)/max(i,j))*400 for i,j in zip(diameters_TP_GT, diameters_TP) if i>j]
    difference2 = [((j-i)/max(i,j))*400 for i,j in zip(diameters_TP_GT, diameters_TP) if i<=j]
    ax9.scatter([diameters_TP_GT[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i>j], [confidence_TP[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i>j], s=difference1,  color='r', label=f'GT > Pred ({len(difference1)})', edgecolors='r', alpha=0.3) 
    ax9.scatter([diameters_TP_GT[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i<=j], [confidence_TP[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i<=j], s=difference2, color='g',  label=f'GT < Pred ({len(difference2)})', edgecolors='g', alpha=0.3) # facecolors='none'

    ax9.legend(loc='lower right')
    ax9.set_title(f"Predicted and GT diameters error\navg={round(sum([abs(i-j) for i,j in zip(diameters_TP_GT, diameters_TP)])/len(difference1+difference2),2)}mm ± {round(np.std(np.array([abs(i-j) for i,j in zip(diameters_TP_GT, diameters_TP)])),2)}mm; max={round(max([abs(i-j) for i,j in zip(diameters_TP_GT, diameters_TP)]),2)}mm; \
    min={round(min([abs(i-j) for i,j in zip(diameters_TP_GT, diameters_TP)]),2)}mm")
    ax9.set_xlabel("Diameters (mm)")
    ax9.set_ylabel("Confidence (%)")
    ax9.set_xlim((-0.5, 22))
    ax9.set_ylim((-18, 110))
    ax9.legend(loc="center right", framealpha=0.7)
    ax9.grid(True)

    difference1 = [(i-j) for i,j in zip(diameters_TP_GT, diameters_TP) if i>j]
    difference2 = [(j-i) for i,j in zip(diameters_TP_GT, diameters_TP) if i<=j]
    ax10.scatter([diameters_TP_GT[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i>j], difference1,  color='r', label=f'GT > Pred ({len(difference1)})', edgecolors='r', alpha=0.3) 
    ax10.scatter([diameters_TP_GT[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i<=j], difference2, color='g',  label=f'GT < Pred ({len(difference2)})', edgecolors='g', alpha=0.3)

    ax10.legend(loc='lower right')
    ax10.set_title(f"Diameter Errors (mm)")
    ax10.set_xlabel("Diameters (mm)")
    ax10.set_ylabel("Error (mm)")
    ax10.set_xlim((-0.5, 22))
    #ax10.set_ylim((-18, 110))
    ax10.legend(loc="upper right", framealpha=0.7)
    ax10.grid(True)
    
    try:
        centers = [det['Center'] for det in detections if det['TP']]
        gt_centers = [det['GT Center'] for det in detections if det['TP']]
        distance = [np.linalg.norm(np.array(a)-np.array(b)) for a,b in zip(centers, gt_centers)]
        ax11.scatter(diameters_TP, distance, marker='v' ,color='green', label=f'Center (avg={round(sum(distance)/len(distance), 2)} ± {round(np.std(np.array(distance)),2)}mm;\nmedian={round(statistics.median(distance),2)};\n max={round(max(distance),2)}; min={round(min(distance),2)})', alpha=0.4) 
        p1s = [det['Pred P1'] for det in detections if det['TP']]
        gt_p1s = [det['GT P1'] for det in detections if det['TP']]
        distance = [np.linalg.norm(np.array(a)-np.array(b)) for a,b in zip(p1s, gt_p1s)]
        ax11.scatter(diameters_TP, distance, marker='x', color='blue', label=f'P1 (avg={round(sum(distance)/len(distance), 2)} ± {round(np.std(np.array(distance)),2)}mm;\nmedian={round(statistics.median(distance),2)};\n max={round(max(distance),2)}; min={round(min(distance),2)})', alpha=0.4) 
        p2s = [det['Pred P2'] for det in detections if det['TP']]
        gt_p2s = [det['GT P2'] for det in detections if det['TP']]
        distance = [np.linalg.norm(np.array(a)-np.array(b)) for a,b in zip(p2s, gt_p2s)]
        ax11.scatter(diameters_TP, distance, marker='p', color='red', label=f'P2 (avg={round(sum(distance)/len(distance), 2)} ± {round(np.std(np.array(distance)),2)}mm;\nmedian={round(statistics.median(distance),2)};\n max={round(max(distance),2)}; min={round(min(distance),2)})', alpha=0.4) 
        ax11.legend(loc='center right', framealpha=0.3 )
        ax11.set_title(f"Distance between GT and predicted keypoints")
        ax11.set_xlabel("Diameter (mm)")
        ax11.set_xlim((-0.5, 22))
        ax11.set_ylabel("Centers distance Error (mm)")
        ax11.grid(True)

        degrees_error = [det['degree'] for det in detections if det['TP']]
        degrees_diameter = [det['GT diameter'] for det in detections if det['TP']]
        degrees_confidence = [det['Confidence'] for det in detections if det['TP']]
        error_median = statistics.median(degrees_error)

        def circular_mean(angles, deg=True):
            '''Circular mean of angle data(default to degree)
            '''
            a = np.deg2rad(angles) if deg else np.array(angles)
            angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
            mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
            return round(np.rad2deg(mean) if deg else mean, 7)
        circular_mean(degrees_error)

        ax12.scatter(degrees_diameter, degrees_error, color='g', label=f'{len(degrees_diameter)} TP: \n(mean={round(circular_mean(degrees_error),2)}°±{round(np.std(np.array(degrees_error)),2)}°\nmedian={round(error_median,2)}\nmax={round(max(degrees_error),2)}°\nmin={round(min(degrees_error),2)}°)',  alpha=0.6) 
        ax12.legend(loc='center right')
        ax12.set_xlim((-0.5, 22))
        ax12.set_title(f"Erorr degree between GT/predicted aneurysm axis")
        ax12.set_xlabel("Diameter (mm)")
        ax12.set_ylabel("Error degree (°)")
        ax12.grid(True)

        ax13.scatter(degrees_error, degrees_confidence, color='g', label=f'{len(degrees_diameter)} TP',  alpha=0.6) 
        ax13.legend(loc='center right')
        ax13.set_ylim((-18, 110))
        ax13.set_title(f"Erorr degree between GT/predicted aneurysm axis")
        ax13.set_ylabel("Confidence (%)")
        ax13.set_xlabel("Error degree (°)")
        ax13.grid(True)
    except:
        pass
    
    plt.savefig(os.path.join(out_dir, f'Evaluation@{iou_threshold}.png'), dpi=100)
    print(f"Evaluation figures are saved ({os.path.join(out_dir, f'Evaluation@{iou_threshold}.png')})")
    
if __name__ == "__main__":
    main()