import pytest

from server import create_app


@pytest.fixture(scope="module")
def client(artifact_dir):
    return create_app(artifact_dir).test_client()


def test_model_page_shows_protected_attribute_comparison(client):
    page = client.get("/").get_data(as_text=True)
    assert "Blend without gender and family status" in page
    assert "CODE_GENDER" in page


@pytest.mark.parametrize("path", ["/", "/explainability", "/portfolio", "/applicant"])
def test_pages_render(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert b"Credit Risk" in resp.data
    assert b'id="page-data"' in resp.data
