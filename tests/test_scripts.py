import importlib
from importlib.metadata import entry_points

import pytest

COMMANDS = sorted(ep for ep in entry_points(group="console_scripts") if ep.name.startswith("deepanedet-"))


@pytest.mark.parametrize("command", COMMANDS, ids=lambda ep: ep.name)
def test_command_entry_point_loads(command):
    assert callable(command.load())


@pytest.mark.parametrize("name", ["train", "predict"])
def test_script_without_training_directory_shows_its_usage(name, monkeypatch):
    script = importlib.import_module(f"deepanedet.cli.{name}")
    monkeypatch.setattr("sys.argv", [f"{name}.py"])

    with pytest.raises(SystemExit, match="Usage"):
        script.main()


def test_evaluate_reports_a_missing_predictions_directory(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from deepanedet.cli import evaluate

    monkeypatch.setattr("sys.argv", ["evaluate.py", str(tmp_path / "missing")])

    with pytest.raises(SystemExit, match="No predictions directory"):
        evaluate.main()
