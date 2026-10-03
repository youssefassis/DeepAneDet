import importlib
import py_compile
from pathlib import Path

import pytest

SCRIPTS = sorted(p for p in (Path(__file__).parents[1] / "deepanedet" / "scripts").glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True)


@pytest.mark.parametrize("name", ["train", "predict"])
def test_script_without_training_directory_shows_its_usage(name, monkeypatch):
    script = importlib.import_module(f"deepanedet.scripts.{name}")
    monkeypatch.setattr("sys.argv", [f"{name}.py"])

    with pytest.raises(SystemExit, match="Usage"):
        script.main()


def test_evaluate_reports_a_missing_predictions_directory(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from deepanedet.scripts import evaluate

    monkeypatch.setattr("sys.argv", ["evaluate.py", str(tmp_path / "missing")])

    with pytest.raises(SystemExit, match="No predictions directory"):
        evaluate.main()
