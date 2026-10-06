import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import riskcalc

FEATURES = ["EXT_MEAN", "CREDIT_TO_ANNUITY", "DAYS_BIRTH", "INS_LATE_MEAN", "NAME_EDUCATION_TYPE"]
EDUCATION = ["Academic degree", "Higher education", "Incomplete higher", "Secondary"]
SEGMENTS = {"SEG_NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"], "SEG_AGE_BAND": ["<30", "30-39", "40-49", "50-59", "60+"]}
GROUPS = {"EXT_MEAN": "application", "CREDIT_TO_ANNUITY": "application", "DAYS_BIRTH": "application", "INS_LATE_MEAN": "installments", "NAME_EDUCATION_TYPE": "application"}


def synthetic_frame(n, rng):
    X = pd.DataFrame({
        "EXT_MEAN": rng.uniform(0.05, 0.9, n),
        "CREDIT_TO_ANNUITY": rng.uniform(5, 40, n),
        "DAYS_BIRTH": rng.integers(-25000, -7000, n).astype(float),
        "INS_LATE_MEAN": rng.uniform(0, 0.5, n),
        "NAME_EDUCATION_TYPE": rng.integers(0, 4, n).astype(float),
    })
    X.loc[rng.random(n) < 0.1, "INS_LATE_MEAN"] = np.nan
    logit = -2.5 - 4 * (X["EXT_MEAN"] - 0.5) + 3 * X["INS_LATE_MEAN"].fillna(0.1) + 0.03 * (X["CREDIT_TO_ANNUITY"] - 20)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def synthetic_charts(grades):
    scores = {"auc": 0.75, "gini": 0.5, "ks": 0.38, "pr_auc": 0.2, "brier": 0.07}
    models = ("lgbm", "xgb", "blend")
    return {
        "metrics": {
            "cv": {m: scores for m in models}, "holdout": {m: scores for m in models}, "cv_fold_auc_std": 0.003,
            "baselines": {"logistic_cv_auc": 0.74, "logistic_cv_std": 0.004, "lgbm_default_cv_auc": 0.76, "lgbm_default_cv_std": 0.003, "lgbm_default_rounds": 200},
            "n_train": 2400, "n_holdout": 600, "n_features": len(FEATURES), "target_auc": 0.797,
            "holdout_brier_calibrated": 0.068, "calibration_method": "platt",
            "without_protected": {"removed": ["CODE_GENDER", "NAME_FAMILY_STATUS"], "w_lgbm": 0.7, "cv": scores, "holdout": scores},
        },
        "roc": {m: {"fpr": [0.0, 0.1, 0.4, 1.0], "tpr": [0.0, 0.3, 0.7, 1.0]} for m in models},
        "pr": {"blend": {"recall": [1.0, 0.5, 0.0], "precision": [0.08, 0.2, 1.0]}},
        "calibration": {"predicted": [0.02, 0.08, 0.2], "observed": [0.025, 0.07, 0.21], "count": [200, 200, 200]},
        "folds": [{"model": m, "fold": f, "auc": 0.75, "rounds": None if m == "blend" else 100} for m in models for f in range(5)],
        "shap_global": [{"feature": f, "group": GROUPS[f], "mean_abs": 0.1} for f in FEATURES],
        "beeswarm": [{"feature": f, "shap": [0.1, -0.2], "color": [0.9, None]} for f in FEATURES],
        "dependence": {f: {"x": [0.1, 0.5], "shap": [0.2, -0.1]} for f in FEATURES},
        "segment_drivers": {col: {level: FEATURES[:3] for level in levels} for col, levels in SEGMENTS.items()},
        "master_scale": [{"grade": g["grade"], "score_min": g["score_min"], "score_max": g["score_max"], "pd": g["pd"], "oof_count": 300,
                          "oof_rate": g["pd"], "holdout_count": 100, "holdout_rate": g["pd"], "jeffreys_p": 0.5} for g in grades],
        "psi": 0.01,
    }


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(7)
    X, y = synthetic_frame(3000, rng)
    lgbm = lgb.train({"objective": "binary", "num_leaves": 8, "learning_rate": 0.1, "verbose": -1, "seed": 7}, lgb.Dataset(X, y), 40)
    lgbm.save_model(str(root / "lgbm_model.txt"))
    xgbm = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "seed": 7}, xgb.DMatrix(X, label=y), 40)
    xgbm.save_model(str(root / "xgb_model.json"))
    w = 0.6
    margin = w * lgbm.predict(X, raw_score=True) + (1 - w) * xgbm.predict(xgb.DMatrix(X), output_margin=True)
    factor, offset = riskcalc.score_constants()
    pd_ = riskcalc.floor_pd(1 / (1 + np.exp(-margin)), 0.0005)
    score = riskcalc.pd_to_score(pd_, factor, offset)
    cutoffs = sorted(set(np.round(np.quantile(score, np.linspace(0, 1, 6)[1:-1])).tolist()))
    grade = riskcalc.grade_for_score(score, cutoffs)
    rates = pd.Series(y).groupby(grade).mean()
    bounds = [None] + cutoffs + [None]
    grades = [{"grade": g, "pd": float(max(rates.get(g, 0.0), 0.0005)), "score_min": bounds[len(cutoffs) + 1 - g], "score_max": bounds[len(cutoffs) + 2 - g]}
              for g in range(1, len(cutoffs) + 2)]
    scorecard = {"base_score": 600.0, "base_odds": 50.0, "pdo": 20.0, "factor": factor, "offset": offset, "pd_floor": 0.0005,
                 "lgd_default": 0.45, "lgd_floor": 0.30, "cutoffs": cutoffs, "grades": grades}
    low, high = X.quantile(0.005), X.quantile(0.995)
    low["NAME_EDUCATION_TYPE"], high["NAME_EDUCATION_TYPE"] = 0.0, 3.0
    features = {"features": FEATURES, "low": low.to_dict(), "high": high.to_dict(), "groups": GROUPS, "categorical": {"NAME_EDUCATION_TYPE": EDUCATION}}
    base = pd.DataFrame({"SK_ID_CURR": np.arange(100001, 100001 + len(X)), "TARGET": y, "EAD": rng.uniform(50_000, 1_500_000, len(X)),
                         **{col: rng.choice(levels, len(X)) for col, levels in SEGMENTS.items()}})
    pd.concat([base, X], axis=1).iloc[:500].to_parquet(root / "sample.parquet", index=False)
    base.assign(PD=pd_, SCORE=score, GRADE=grade).to_parquet(root / "portfolio.parquet", index=False)
    for name, payload in {"blend.json": {"w_lgbm": w}, "calibrator.json": {"method": "platt", "a": 1.0, "b": 0.0},
                          "scorecard.json": scorecard, "features.json": features, "charts.json": synthetic_charts(grades)}.items():
        (root / name).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return root
