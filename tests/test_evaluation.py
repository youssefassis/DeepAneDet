import numpy as np
import pytest

from deepanedet.inference.evaluation import get_CM_dict


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



def match(predictions, truths):
    detections, fns = get_CM_dict(
        np.array(predictions), np.array(truths), iou_thr=0.1, confidence_thr=0.01, pat_name="P0001", verbose=False
    )
    return [(d["TP"], d["FP"]) for d in detections], len(fns)


def test_aneurysms_of_the_same_size_are_matched_separately():
    truths = [[0.0, 0.0, 0.0, 2.0], [50.0, 0.0, 0.0, 2.0]]
    predictions = [[0.0, 0.0, 0.0, 2.0, 0.9], [50.0, 0.0, 0.0, 2.0, 0.8]]

    assert match(predictions, truths) == ([(1, 0), (1, 0)], 0)


def test_a_second_detection_of_an_aneurysm_is_a_false_positive():
    truths = [[0.0, 0.0, 0.0, 2.0]]
    predictions = [[0.0, 0.0, 0.0, 2.0, 0.9], [0.5, 0.0, 0.0, 2.0, 0.8]]

    assert match(predictions, truths) == ([(1, 0), (0, 1)], 0)


def test_a_detection_overlapping_two_aneurysms_matches_the_most_overlapping():
    truths = [[0.0, 0.0, 0.0, 2.0], [3.0, 0.0, 0.0, 2.0]]
    predictions = [[2.5, 0.0, 0.0, 2.0, 0.9]]

    detections, fns = get_CM_dict(np.array(predictions), np.array(truths), 0.1, 0.01, "P0001", False)

    assert detections[0]["TP"] == 1 and detections[0]["GT Center"] == [3.0, 0.0, 0.0]
    assert [fn["Center"] for fn in fns] == [[0.0, 0.0, 0.0]]


def test_the_most_confident_detection_is_matched_first():
    truths = [[0.0, 0.0, 0.0, 2.0]]
    predictions = [[0.5, 0.0, 0.0, 2.0, 0.3], [0.0, 0.0, 0.0, 2.0, 0.9]]

    detections, _ = get_CM_dict(np.array(predictions), np.array(truths), 0.1, 0.01, "P0001", False)

    assert [(d["Confidence"], d["TP"]) for d in detections] == [(90.0, 1), (30.0, 0)]

def test_detections_are_read_for_each_patient_of_the_split(evaluation_fold):
    from deepanedet.inference.evaluation import get_detections

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


@pytest.mark.parametrize("kept", ["true positives", "nothing"])
def test_evaluate_script_without_false_positives(evaluation_fold, monkeypatch, kept):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import evaluate

    _, predictions_dir = evaluation_fold
    for prediction_file in predictions_dir.glob("*.json"):
        detections = json.loads(prediction_file.read_text())
        kept_detections = {k: d for k, d in detections.items() if d["confidence"] == 0.9} if kept == "true positives" else {}
        prediction_file.write_text(json.dumps(kept_detections))
    monkeypatch.setattr("sys.argv", ["evaluate.py", str(predictions_dir)])

    evaluate.main()

    assert (predictions_dir / "Evaluation@0.1.png").is_file()


def test_evaluate_script_without_aneurysm_reports_it(evaluation_fold, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import evaluate

    _, predictions_dir = evaluation_fold
    for truth in (predictions_dir.parents[3] / "Data").glob("*/aneurysms.csv"):
        truth.unlink()
    monkeypatch.setattr("sys.argv", ["evaluate.py", str(predictions_dir)])

    with pytest.raises(SystemExit, match="No aneurysm"):
        evaluate.main()
