import numpy as np

from evaluation import get_CM_dict


def test_confusion_matrix_counts_tp_fp_and_fn():
    truths = np.array([[0.0, 0.0, 0.0, 2.0], [50.0, 0.0, 0.0, 3.0]])
    predictions = np.array(
        [
            [0.5, 0.0, 0.0, 2.0, 0.9],  # overlaps the first aneurysm
            [20.0, 0.0, 0.0, 2.0, 0.8],  # overlaps nothing
            [50.0, 0.0, 0.0, 3.0, 0.001],  # under the confidence threshold
        ]
    )

    detections, fns = get_CM_dict(predictions, truths, iou_thr=0.1, confidence_thr=0.01, pat_name="P0001", verbose=False)

    assert [(d["TP"], d["FP"]) for d in detections] == [(1, 0), (0, 1)]
    assert [fn["Diameter"] for fn in fns] == [6.0]
