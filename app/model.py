import json
import math
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import riskcalc

TOP_WATERFALL = 15
TOP_DRIVERS = 8


class ValidationError(ValueError):
    pass


class NotFoundError(Exception):
    pass


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


class RiskModel:
    def __init__(self, artifact_dir):
        root = Path(artifact_dir)
        self.lgbm = lgb.Booster(model_file=str(root / "lgbm_model.txt"))
        self.xgbm = xgb.Booster()
        self.xgbm.load_model(str(root / "xgb_model.json"))
        self.weight = _read_json(root / "blend.json")["w_lgbm"]
        self.calibrator = _read_json(root / "calibrator.json")
        self.card = _read_json(root / "scorecard.json")
        meta = _read_json(root / "features.json")
        self.features = meta["features"]
        self.index = {name: i for i, name in enumerate(self.features)}
        self.low = np.array([meta["low"][f] for f in self.features], dtype=float)
        self.high = np.array([meta["high"][f] for f in self.features], dtype=float)
        self.categorical = meta.get("categorical", {})
        self.groups = meta.get("groups", {})
        self.cutoffs = np.asarray(self.card["cutoffs"], dtype=float)
        self.grade_pd = {int(g["grade"]): float(g["pd"]) for g in self.card["grades"]}
        self.sample = pd.read_parquet(root / "sample.parquet").set_index("SK_ID_CURR")
        self.matrix = self.sample[self.features].to_numpy(dtype=float)
        self.row_of = {int(i): n for n, i in enumerate(self.sample.index)}
        self.portfolio = pd.read_parquet(root / "portfolio.parquet")
        self.lock = threading.Lock()
        self.sample_grade = self.portfolio.set_index("SK_ID_CURR").loc[self.sample.index, "GRADE"].to_numpy()

    def _dmatrix(self, X):
        return xgb.DMatrix(X, feature_names=self.features)

    def margins(self, X):
        with self.lock:
            m_lgb = self.lgbm.predict(X, raw_score=True)
            m_xgb = self.xgbm.predict(self._dmatrix(X), output_margin=True)
        return self.weight * m_lgb + (1 - self.weight) * m_xgb

    def contributions(self, X):
        with self.lock:
            c_lgb = self.lgbm.predict(X, pred_contrib=True)
            c_xgb = self.xgbm.predict(self._dmatrix(X), pred_contribs=True)
        return self.weight * c_lgb + (1 - self.weight) * c_xgb

    def calibrate(self, margin):
        margin = np.asarray(margin, dtype=float)
        if self.calibrator["method"] == "platt":
            pd_ = 1 / (1 + np.exp(-(self.calibrator["a"] * margin + self.calibrator["b"])))
        else:
            pd_ = np.interp(margin, self.calibrator["x"], self.calibrator["y"])
        return riskcalc.floor_pd(pd_, self.card["pd_floor"])

    def grades(self, X):
        pd_ = self.calibrate(self.margins(X))
        return riskcalc.grade_for_score(riskcalc.pd_to_score(pd_, self.card["factor"], self.card["offset"]), self.cutoffs)

    def segment_names(self):
        return [c[4:] for c in self.portfolio.columns if c.startswith("SEG_")]

    def random_id(self, grade, rng):
        ids = self.sample.index.to_numpy()
        if grade is not None:
            ids = ids[self.sample_grade == grade]
        if len(ids) == 0:
            raise NotFoundError(f"no sample applicants in grade {grade}")
        return int(rng.choice(ids))

    def _row(self, applicant_id):
        if applicant_id not in self.row_of:
            raise NotFoundError(f"applicant {applicant_id} is not in the sample")
        return self.matrix[self.row_of[applicant_id]].copy()

    def _apply_overrides(self, row, overrides):
        if not isinstance(overrides, dict):
            raise ValidationError("overrides must be an object of feature names to numbers")
        clipped = []
        for name, value in overrides.items():
            if name not in self.index:
                raise ValidationError(f"unknown feature: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"value for {name} must be a finite number")
            try:
                value = float(value)
            except OverflowError:
                raise ValidationError(f"value for {name} must be a finite number") from None
            if not math.isfinite(value):
                raise ValidationError(f"value for {name} must be a finite number")
            j = self.index[name]
            new = float(np.clip(value, self.low[j], self.high[j]))
            if name in self.categorical:
                new = float(round(new))
            if new != float(value):
                clipped.append(name)
            row[j] = new
        return row, clipped

    def _display(self, j, value):
        labels = self.categorical.get(self.features[j])
        if labels is None or not np.isfinite(value):
            return None
        code = int(round(value))
        return labels[code] if 0 <= code < len(labels) else "Missing"

    def _driver(self, j, value):
        name = self.features[j]
        item = {"feature": name, "value": _number(value), "low": float(self.low[j]), "high": float(self.high[j]), "group": self.groups.get(name, "application")}
        if name in self.categorical:
            item["labels"] = self.categorical[name]
        return item

    def assess(self, applicant_id, lgd, overrides=None):
        lgd = riskcalc.clamp_lgd(lgd, self.card["lgd_floor"])
        row, clipped = self._apply_overrides(self._row(applicant_id), overrides if overrides is not None else {})
        contrib = self.contributions(row.reshape(1, -1))[0]
        values, base = contrib[:-1], float(contrib[-1])
        margin = float(contrib.sum())
        pd_ = float(self.calibrate(margin))
        score = float(riskcalc.pd_to_score(pd_, self.card["factor"], self.card["offset"]))
        grade = int(riskcalc.grade_for_score(score, self.cutoffs))
        grade_pd = self.grade_pd[grade]
        ead = float(self.sample.at[applicant_id, "EAD"])
        order = np.argsort(-np.abs(values), kind="stable")
        rest = order[TOP_WATERFALL:]
        return {
            "id": applicant_id,
            "target": int(self.sample.at[applicant_id, "TARGET"]),
            "segments": {c[4:]: str(self.sample.at[applicant_id, c]) for c in self.sample.columns if c.startswith("SEG_")},
            "ead": ead,
            "lgd": lgd,
            "base": base,
            "margin": margin,
            "pd": pd_,
            "score": score,
            "grade": grade,
            "grade_pd": grade_pd,
            "el": float(riskcalc.expected_loss(grade_pd, lgd, ead)),
            "rwa": float(riskcalc.rwa(grade_pd, lgd, ead)),
            "waterfall": [{"feature": self.features[j], "value": _number(row[j]), "display": self._display(j, row[j]), "contribution": float(values[j])}
                          for j in order[:TOP_WATERFALL]],
            "other": {"count": int(len(rest)), "contribution": float(values[rest].sum())},
            "drivers": [self._driver(j, row[j]) for j in order[:TOP_DRIVERS]],
            "overrides": {name: _number(row[self.index[name]]) for name in (overrides or {})},
            "clipped": clipped,
        }

    def portfolio_view(self, segment, lgd):
        lgd = riskcalc.clamp_lgd(lgd, self.card["lgd_floor"])
        column = f"SEG_{segment}"
        if column not in self.portfolio.columns:
            raise ValidationError(f"unknown segment: {segment}")
        grade_pd = self.portfolio["GRADE"].map(self.grade_pd).to_numpy()
        ead = self.portfolio["EAD"].to_numpy()
        frame = self.portfolio.assign(EL=riskcalc.expected_loss(grade_pd, lgd, ead), RWA=riskcalc.rwa(grade_pd, lgd, ead))

        def table(by):
            grouped = frame.groupby(by).agg(count=("SK_ID_CURR", "size"), exposure=("EAD", "sum"), mean_pd=("PD", "mean"),
                                            observed=("TARGET", "mean"), el=("EL", "sum"), rwa=("RWA", "sum"))
            return [{"level": str(level), "count": int(r["count"]), "exposure": float(r["exposure"]), "mean_pd": float(r["mean_pd"]),
                     "observed": float(r["observed"]), "el": float(r["el"]), "rwa": float(r["rwa"]),
                     "rwa_density": float(r["rwa"] / r["exposure"])} for level, r in grouped.iterrows()]

        totals = {"count": int(len(frame)), "exposure": float(frame["EAD"].sum()), "el": float(frame["EL"].sum()), "rwa": float(frame["RWA"].sum())}
        totals["rwa_density"] = totals["rwa"] / totals["exposure"]
        return {"segment": segment, "lgd": lgd, "totals": totals, "grades": table("GRADE"), "segments": table(column)}
