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


def test_detections_are_read_for_each_patient_of_the_split(evaluation_fold):
    from evaluation import get_detections

    _, predictions_dir = evaluation_fold
    patients = sorted(str(p) for p in (predictions_dir.parents[3] / "Data").iterdir())

    detections, fns = get_detections(str(predictions_dir), patients, truth_file_name="aneurysms.csv")

    assert [(d["Patient"], d["TP"]) for d in detections if d["TP"]] == [("P0001", 1), ("P0002", 1)]
    assert sum(d["FP"] for d in detections) == 3 and fns == []


def test_evaluate_script_writes_the_detection_table_and_figure(evaluation_fold, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    import evaluate

    _, predictions_dir = evaluation_fold
    monkeypatch.setattr("sys.argv", ["evaluate.py", str(predictions_dir) + "/"])

    evaluate.main()

    table = pd.read_csv(predictions_dir / "detection_evaluation@0.1.csv")
    assert table["TP"].sum() == 2 and table["FP"].sum() == 3
    assert (predictions_dir / "Evaluation@0.1.png").is_file()
