import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_notebook.py"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_notebook", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "notebooks" / "01-eda"
    folder.mkdir(parents=True)
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "someone/credit-risk-01-eda"}))
    (folder / "01-eda.ipynb").write_text('{"source": "SMOKE = False"}')
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, "notebook": (folder / "01-eda.ipynb").read_text()})
        return SimpleNamespace(returncode=0, stdout="Kernel version 1 successfully pushed.", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return module, folder, calls


def test_build_keeps_committed_notebook_without_source(runner):
    module, folder, calls = runner
    module.build("01-eda")
    assert calls == []


def test_smoke_push_restores_committed_notebook(runner):
    module, folder, calls = runner
    module.push("01-eda", smoke=True)
    assert "SMOKE = True" in calls[-1]["notebook"]
    assert (folder / "01-eda.ipynb").read_text() == '{"source": "SMOKE = False"}'
