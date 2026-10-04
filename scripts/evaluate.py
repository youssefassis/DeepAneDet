import os, json, re, sys
import matplotlib.pyplot as plt
from deepanedet.inference import evaluation
import numpy as np
import pandas as pd
from deepanedet.data import io as dio
from sklearn.metrics import auc

def summarize(values):
    '''
    Returns (mean, std, max, min) of values, all nan when there is no value (e.g. no true positive)
    '''
    if len(values) == 0:
        return (float('nan'),) * 4
    return np.mean(values), np.std(values), np.max(values), np.min(values)

def main():
    predictions_dir = sys.argv[1]
    print(predictions_dir)
    if not os.path.isdir(predictions_dir):
        sys.exit(f"No predictions directory '{predictions_dir}'")

    iou_threshold = 0.1
    confidence_threshold = 0.01

    # Only one fold
    train_dir = predictions_dir[: predictions_dir.find("Predictions")]
    out_dir = os.path.join(os.getcwd(), predictions_dir)

    with open(os.path.join(train_dir, 'ndl_config.json'),'r') as f:
        config = json.load(f)
    split_file = config['split_file'][0] if isinstance(config['split_file'], list) else config['split_file']
    train_pats, val_pats, test_pats = dio.readSplit(split_file, config.get('base_dir'))
    patients = val_pats + test_pats # predict.py predicts both
    print(len(patients), "patients for test")

    detections, FNs = evaluation.get_detections(predictions_dir,
                                                patients,
                                                truth_file_name=config['truth file'],
                                                iou_thr=iou_threshold,
                                                confidence_thr=confidence_threshold,
                                                max_per_patient=None,
                                                verbose=None)
    
    # Generate Dataframe
    tot_aneurysms = sum([det['TP'] for det in detections]) + len(FNs)
    if tot_aneurysms == 0:
        sys.exit("No aneurysm in the evaluated patients: sensitivity is undefined")

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

    # predict.py writes to Predictions/<epoch>_epochs/...
    epoch_dir = re.search(r"(\d+)_epochs", predictions_dir)
    epoch = epoch_dir.group(1) if epoch_dir else "unknown"

    ########################################################################################################
    # Confusion matrix
    plt.figure()
    fig, ((ax, ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax9, ax10, ax11, ax12, ax13, ax8,)) = plt.subplots(nrows=2, ncols=7, figsize=(45,10))
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
    # undefined (nan) without any false positive to normalize the FPs/case
    froc_auc = 100 - auc([100]+sensitivities, [100]+[(f/max(fps_per_case))*100 for f in fps_per_case]) /100 if max(fps_per_case) > 0 else float('nan')
    ax4.set_title(f'FROC Curve (AUC={round(froc_auc, 3)} %)')

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
    iou_avg, iou_std, iou_max, iou_min = summarize(IoU_TP)
    ax7.set_title(f'IoU vs Confidence\navg = {round(iou_avg, 2)}% ± {round(iou_std,2)}%; max={round(iou_max,2)}%; min = {round(iou_min, 2)}%')
    ax7.set_xlabel('IoU (%)')
    ax7.set_ylabel('Confidence score (%)')
    ax7.set_xlim((-5, 100))
    ax7.set_ylim((-5, 105))
    ax7.legend(loc="center right", framealpha=0.4)
    ax7.grid(True)

    # Last figure
    train_aneurysms, test_aneurysms = [], []
    for pat in train_pats:
        train_aneurysms += dio.SizeAnev(pat, config['truth file'])
    for pat in patients:
        test_aneurysms += dio.SizeAnev(pat, config['truth file'])

    ax8.set_xticks([i for i in range(0, int(np.floor(max(train_aneurysms+test_aneurysms, default=0)))+1, 2)])
    ax8.hist(train_aneurysms, bins=range(0, 20), alpha=0.3, color='red',  edgecolor='black', linewidth=1.5)
    ax8.hist(test_aneurysms, bins=range(0, 20), alpha=0.3, color='blue', edgecolor='black', linewidth=1.5)

    ax8.set_title('Training/Test aneurysm size distribution')
    ax8.set_xlabel('Diameter (mm)')
    ax8.set_ylabel('Count')
    ax8.legend([f"Train ({len(train_aneurysms)} ans/ {len(train_pats)} patients)", f"Test ({len(test_aneurysms)} ans/ {len(patients)} patients)"])
    ax8.grid(False)

    difference1 = [((i-j)/max(i,j))*400 for i,j in zip(diameters_TP_GT, diameters_TP) if i>j]
    difference2 = [((j-i)/max(i,j))*400 for i,j in zip(diameters_TP_GT, diameters_TP) if i<=j]
    ax9.scatter([diameters_TP_GT[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i>j], [confidence_TP[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i>j], s=difference1,  color='r', label=f'GT > Pred ({len(difference1)})', edgecolors='r', alpha=0.3) 
    ax9.scatter([diameters_TP_GT[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i<=j], [confidence_TP[idx] for idx, (i,j) in enumerate(zip(diameters_TP_GT, diameters_TP)) if i<=j], s=difference2, color='g',  label=f'GT < Pred ({len(difference2)})', edgecolors='g', alpha=0.3) # facecolors='none'

    ax9.legend(loc='lower right')
    error_avg, error_std, error_max, error_min = summarize([abs(i-j) for i,j in zip(diameters_TP_GT, diameters_TP)])
    ax9.set_title(f"Predicted and GT diameters error\navg={round(error_avg,2)}mm ± {round(error_std,2)}mm; max={round(error_max,2)}mm; \
    min={round(error_min,2)}mm")
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
    ax10.legend(loc="upper right", framealpha=0.7)
    ax10.grid(True)
    
    plt.savefig(os.path.join(out_dir, f'Evaluation@{iou_threshold}.png'), dpi=100)
    print(f"Evaluation figures are saved ({os.path.join(out_dir, f'Evaluation@{iou_threshold}.png')})")
    
if __name__ == "__main__":
    main()