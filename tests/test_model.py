import numpy as np
import pytest

import riskcalc
from model import NotFoundError, RiskModel, ValidationError

FIRST = 100001


@pytest.fixture(scope="module")
def model(artifact_dir):
    return RiskModel(artifact_dir)


def test_contributions_sum_to_margin(model):
    X = model.matrix[:20]
    assert np.allclose(model.contributions(X).sum(axis=1), model.margins(X), atol=1e-6)


def test_assess_payload(model):
    out = model.assess(FIRST, 0.45)
    assert {"pd", "score", "grade", "grade_pd", "el", "rwa", "waterfall", "drivers", "other", "base", "margin", "segments"} <= set(out)
    assert 0.0005 <= out["pd"] < 1
    total = out["base"] + sum(w["contribution"] for w in out["waterfall"]) + out["other"]["contribution"]
    assert total == pytest.approx(out["margin"], abs=1e-6)
    assert out["el"] == pytest.approx(out["grade_pd"] * 0.45 * out["ead"])


def test_grade_matches_portfolio(model):
    ids = [int(i) for i in model.sample.index[:50]]
    expected = model.portfolio.set_index("SK_ID_CURR").loc[ids, "GRADE"].astype(int).tolist()
    assert [model.assess(i, 0.45)["grade"] for i in ids] == expected


def test_override_is_clipped(model):
    j = model.index["CREDIT_TO_ANNUITY"]
    out = model.assess(FIRST, 0.45, {"CREDIT_TO_ANNUITY": 1e9})
    assert out["clipped"] == ["CREDIT_TO_ANNUITY"]
    assert out["overrides"]["CREDIT_TO_ANNUITY"] == pytest.approx(model.high[j])


def test_override_moves_pd(model):
    j = model.index["EXT_MEAN"]
    worse = model.assess(FIRST, 0.45, {"EXT_MEAN": float(model.low[j])})["pd"]
    better = model.assess(FIRST, 0.45, {"EXT_MEAN": float(model.high[j])})["pd"]
    assert worse > better


def test_categorical_override_rounds(model):
    out = model.assess(FIRST, 0.45, {"NAME_EDUCATION_TYPE": 1.6})
    assert out["overrides"]["NAME_EDUCATION_TYPE"] == 2.0


@pytest.mark.parametrize("overrides", [{"NOT_A_FEATURE": 1.0}, {"EXT_MEAN": "high"}, {"EXT_MEAN": True}, {"EXT_MEAN": float("nan")}, ["EXT_MEAN"]])
def test_bad_overrides_rejected(model, overrides):
    with pytest.raises(ValidationError):
        model.assess(FIRST, 0.45, overrides)


def test_missing_values_serialise_as_none(model):
    j = model.index["INS_LATE_MEAN"]
    row = int(np.flatnonzero(np.isnan(model.matrix[:, j]))[0])
    out = model.assess(int(model.sample.index[row]), 0.45)
    values = {item["feature"]: item["value"] for item in out["waterfall"] + out["drivers"]}
    assert values["INS_LATE_MEAN"] is None


def test_categorical_values_carry_their_label(model):
    out = model.assess(FIRST, 0.45, {"NAME_EDUCATION_TYPE": 1.0})
    item = next(w for w in out["waterfall"] if w["feature"] == "NAME_EDUCATION_TYPE")
    assert item["display"] == "Higher education"
    other = next(w for w in out["waterfall"] if w["feature"] == "EXT_MEAN")
    assert other["display"] is None


def test_unknown_applicant(model):
    with pytest.raises(NotFoundError):
        model.assess(1, 0.45)


def test_random_from_empty_grade(model):
    with pytest.raises(NotFoundError):
        model.random_id(99, np.random.default_rng(0))


def test_random_from_grade(model):
    grade = int(model.sample_grade[0])
    applicant = model.random_id(grade, np.random.default_rng(0))
    assert model.assess(applicant, 0.45)["grade"] == grade


def test_portfolio_view_clamps_and_balances(model):
    out = model.portfolio_view("NAME_CONTRACT_TYPE", 0.1)
    assert out["lgd"] == 0.30
    assert sum(g["count"] for g in out["grades"]) == len(model.portfolio)
    assert out["totals"]["rwa"] == pytest.approx(sum(s["rwa"] for s in out["segments"]))


def test_portfolio_unknown_segment(model):
    with pytest.raises(ValidationError):
        model.portfolio_view("NOPE", 0.45)


def test_extreme_margins_give_finite_scores(model):
    pd_ = model.calibrate(np.array([60.0, -60.0]))
    assert np.isfinite(riskcalc.pd_to_score(pd_, model.card["factor"], model.card["offset"])).all()
