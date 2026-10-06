import shutil

import pytest

from server import create_app, ensure_artifacts


def test_existing_artifacts_are_used_without_download(artifact_dir):
    calls = []
    assert ensure_artifacts(artifact_dir, "owner/repo", None, lambda **kw: calls.append(kw)) == artifact_dir
    assert calls == []


def test_missing_artifacts_are_downloaded(tmp_path, artifact_dir):
    target = tmp_path / "artifacts"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        shutil.copytree(artifact_dir, kwargs["local_dir"])

    ensure_artifacts(target, "owner/repo", "secret", fake_download)
    assert calls[0]["repo_id"] == "owner/repo" and calls[0]["token"] == "secret"
    assert create_app(target).test_client().get("/api/applicant/100001").status_code == 200


def test_missing_artifacts_without_repo_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ARTIFACT_REPO"):
        ensure_artifacts(tmp_path / "artifacts", None, None, None)
