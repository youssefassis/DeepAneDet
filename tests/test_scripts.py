import py_compile
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).parents[1] / "resources" / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True)


@pytest.mark.parametrize("name", ["train", "predict"])
def test_script_without_training_directory_shows_its_usage(name, monkeypatch):
    script = __import__(name)
    monkeypatch.setattr("sys.argv", [f"{name}.py"])

    with pytest.raises(SystemExit, match="Usage"):
        script.main()


def test_evaluate_reports_a_missing_predictions_directory(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import evaluate

    monkeypatch.setattr("sys.argv", ["evaluate.py", str(tmp_path / "missing")])

    with pytest.raises(SystemExit, match="No predictions directory"):
        evaluate.main()
