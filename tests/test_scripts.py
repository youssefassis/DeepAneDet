import py_compile
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).parents[1] / "resources" / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True)
