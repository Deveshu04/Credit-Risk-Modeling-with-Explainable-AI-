from pathlib import Path

import numpy as np
import pytest

from model import RiskModel

REAL = Path(__file__).resolve().parents[1] / "app" / "artifacts"

pytestmark = pytest.mark.skipif(not (REAL / "scorecard.json").exists(), reason="real artifacts not present")


@pytest.fixture(scope="module")
def model():
    return RiskModel(REAL)


def test_app_reproduces_notebook_pd_and_grade(model):
    ids = model.sample.index[:300]
    port = model.portfolio.set_index("SK_ID_CURR").loc[ids]
    pd_app = model.calibrate(model.margins(model.matrix[:300]))
    assert np.allclose(pd_app, port["PD"].to_numpy(), atol=1e-5)
    assert (model.grades(model.matrix[:300]) == port["GRADE"].to_numpy()).mean() >= 0.995


def test_assess_runs_on_real_applicant(model):
    out = model.assess(int(model.sample.index[0]), 0.45)
    assert len(out["waterfall"]) == 15 and len(out["drivers"]) == 8
