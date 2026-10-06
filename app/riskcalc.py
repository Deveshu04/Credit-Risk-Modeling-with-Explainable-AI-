import math

import numpy as np
from scipy.stats import norm

PD_CAP = 0.9999
G_999 = float(norm.ppf(0.999))


def score_constants(base_score=600.0, base_odds=50.0, pdo=20.0):
    factor = pdo / math.log(2)
    return factor, base_score - factor * math.log(base_odds)


def floor_pd(pd_, floor):
    return np.clip(np.asarray(pd_, dtype=float), floor, PD_CAP)


def pd_to_score(pd_, factor, offset):
    pd_ = np.asarray(pd_, dtype=float)
    return offset + factor * np.log((1 - pd_) / pd_)


def grade_for_score(score, cutoffs):
    return len(cutoffs) + 1 - np.searchsorted(np.asarray(cutoffs, dtype=float), score, side="right")


def irb_correlation(pd_):
    weight = (1 - np.exp(-35 * np.asarray(pd_, dtype=float))) / (1 - math.exp(-35))
    return 0.03 * weight + 0.16 * (1 - weight)


def capital_k(pd_, lgd):
    pd_ = np.asarray(pd_, dtype=float)
    r = irb_correlation(pd_)
    stressed = norm.cdf((norm.ppf(pd_) + np.sqrt(r) * G_999) / np.sqrt(1 - r))
    return lgd * stressed - pd_ * lgd


def rwa(pd_, lgd, ead):
    return 12.5 * capital_k(pd_, lgd) * np.asarray(ead, dtype=float)


def expected_loss(pd_, lgd, ead):
    return np.asarray(pd_, dtype=float) * lgd * np.asarray(ead, dtype=float)


def clamp_lgd(lgd, floor=0.25):
    return float(min(max(float(lgd), floor), 1.0))
