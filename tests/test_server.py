import json

import pandas as pd
import pytest

from server import create_app

FIRST = 100001


@pytest.fixture(scope="module")
def client(artifact_dir):
    return create_app(artifact_dir).test_client()


def strict(resp):
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(resp.get_data(as_text=True), parse_constant=reject)


def test_applicant_route(client):
    body = strict(client.get(f"/api/applicant/{FIRST}"))
    assert body["id"] == FIRST and body["waterfall"]


def test_unknown_applicant_404(client):
    resp = client.get("/api/applicant/1")
    assert resp.status_code == 404 and "error" in strict(resp)


def test_random_with_grade(client):
    assert strict(client.get("/api/applicant/random?grade=1"))["grade"] == 1


def test_random_empty_grade_404(client):
    assert client.get("/api/applicant/random?grade=99").status_code == 404


def test_random_bad_grade(client):
    assert client.get("/api/applicant/random?grade=abc").status_code == 400


def test_score_with_override(client):
    resp = client.post("/api/score", json={"id": FIRST, "lgd": 0.6, "overrides": {"CREDIT_TO_ANNUITY": 1e9}})
    body = strict(resp)
    assert resp.status_code == 200 and body["clipped"] == ["CREDIT_TO_ANNUITY"] and body["lgd"] == 0.6


@pytest.mark.parametrize("payload", [None, [], {}, {"id": "100001"}, {"id": True}, {"id": FIRST, "overrides": {"NOPE": 1}},
                                     {"id": FIRST, "overrides": {"EXT_MEAN": "x"}}, {"id": FIRST, "lgd": "abc"}])
def test_score_bad_input(client, payload):
    resp = client.post("/api/score", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400 and "error" in strict(resp)


@pytest.mark.parametrize("body", [
    '{"id": 100001, "overrides": false}',
    '{"id": 100001, "overrides": 0}',
    '{"id": 100001, "overrides": ""}',
    '{"id": 100001, "overrides": []}',
    '{"id": 100001, "overrides": {"EXT_MEAN": true}}',
    '{"id": 100001, "overrides": {"EXT_MEAN": NaN}}',
    '{"id": 100001, "overrides": {"EXT_MEAN": 1' + "0" * 400 + "}}",
    '{"id": 100001, "lgd": 1' + "0" * 400 + "}",
])
def test_score_rejects_malformed_values(client, body):
    resp = client.post("/api/score", data=body, content_type="application/json")
    assert resp.status_code == 400 and "error" in strict(resp)


def test_score_clips_huge_finite_override(client):
    resp = client.post("/api/score", data='{"id": 100001, "overrides": {"EXT_MEAN": 100000000000000000000}}', content_type="application/json")
    assert resp.status_code == 200 and strict(resp)["clipped"] == ["EXT_MEAN"]


def test_score_without_overrides_key(client):
    assert client.post("/api/score", json={"id": FIRST}).status_code == 200


def test_portfolio_route(client):
    body = strict(client.get("/api/portfolio?segment=AGE_BAND&lgd=0.1"))
    assert body["lgd"] == 0.30 and body["segment"] == "AGE_BAND" and body["grades"]


@pytest.mark.parametrize("query", ["segment=NOPE", "lgd=abc", "lgd=nan", "lgd=inf"])
def test_portfolio_bad_input(client, query):
    assert client.get(f"/api/portfolio?{query}").status_code == 400


def test_missing_values_are_valid_json(client, artifact_dir):
    sample = pd.read_parquet(artifact_dir / "sample.parquet")
    applicant = int(sample.loc[sample["INS_LATE_MEAN"].isna(), "SK_ID_CURR"].iloc[0])
    body = strict(client.get(f"/api/applicant/{applicant}"))
    assert any(item["value"] is None for item in body["waterfall"])
