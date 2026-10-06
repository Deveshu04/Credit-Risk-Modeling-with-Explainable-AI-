import numpy as np
import pytest

import riskcalc

K_REF = 0.03661818


def test_base_odds_give_base_score():
    factor, offset = riskcalc.score_constants()
    assert float(riskcalc.pd_to_score(1 / 51, factor, offset)) == pytest.approx(600.0)


def test_doubling_odds_adds_pdo():
    factor, offset = riskcalc.score_constants()
    gain = riskcalc.pd_to_score(1 / 101, factor, offset) - riskcalc.pd_to_score(1 / 51, factor, offset)
    assert float(gain) == pytest.approx(20.0)


def test_pd_floor_and_cap():
    out = riskcalc.floor_pd(np.array([0.0, 0.0001, 0.2, 1.0]), 0.0005)
    assert out.tolist() == [0.0005, 0.0005, 0.2, riskcalc.PD_CAP]


def test_capped_pd_has_finite_score():
    factor, offset = riskcalc.score_constants()
    scores = riskcalc.pd_to_score(riskcalc.floor_pd(np.array([0.0, 1.0]), 0.0005), factor, offset)
    assert np.isfinite(scores).all()


def test_grade_boundaries():
    cutoffs = [500.0, 550.0, 600.0]
    scores = np.array([499.9, 500.0, 549.9, 550.0, 600.0, 700.0])
    assert riskcalc.grade_for_score(scores, cutoffs).tolist() == [4, 3, 3, 2, 1, 1]


def test_correlation_limits():
    assert float(riskcalc.irb_correlation(1e-9)) == pytest.approx(0.16, abs=1e-6)
    assert float(riskcalc.irb_correlation(0.99)) == pytest.approx(0.03, abs=1e-6)


def test_capital_matches_hand_computation():
    assert float(riskcalc.capital_k(0.01, 0.45)) == pytest.approx(K_REF, abs=1e-7)


def test_rwa_and_expected_loss():
    assert float(riskcalc.rwa(0.01, 0.45, 1000.0)) == pytest.approx(12.5 * K_REF * 1000.0, rel=1e-6)
    assert float(riskcalc.expected_loss(0.02, 0.45, 1000.0)) == pytest.approx(9.0)


def test_lgd_clamp():
    assert riskcalc.clamp_lgd(0.1) == 0.25
    assert riskcalc.clamp_lgd(1.5) == 1.0
    assert riskcalc.clamp_lgd(0.45) == 0.45
