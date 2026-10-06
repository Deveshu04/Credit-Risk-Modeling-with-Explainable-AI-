import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from model import NotFoundError, RiskModel, ValidationError

ROOT = Path(__file__).resolve().parent
SEGMENT_LABELS = {
    "NAME_CONTRACT_TYPE": "Contract type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education",
    "NAME_FAMILY_STATUS": "Family status",
    "REGION_RATING_CLIENT": "Region rating",
    "AGE_BAND": "Age band",
}
PAGE_KEYS = {
    "model.html": ["metrics", "roc", "pr", "calibration", "folds"],
    "explain.html": ["shap_global", "beeswarm", "dependence"],
    "portfolio.html": ["master_scale", "psi", "segment_drivers"],
    "applicant.html": [],
}


def ensure_artifacts(artifact_dir, repo_id, token, download):
    artifact_dir = Path(artifact_dir)
    if (artifact_dir / "scorecard.json").exists():
        return artifact_dir
    if not repo_id:
        raise FileNotFoundError(f"no artifacts in {artifact_dir}; set ARTIFACT_REPO to download them")
    download(repo_id=repo_id, repo_type="model", local_dir=str(artifact_dir), token=token)
    return artifact_dir


def hub_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def create_app(artifact_dir=None):
    artifact_dir = Path(artifact_dir or os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
    ensure_artifacts(artifact_dir, os.environ.get("ARTIFACT_REPO"), os.environ.get("HF_TOKEN"), hub_download)
    app = Flask(__name__)
    model = RiskModel(artifact_dir)
    charts = json.loads((artifact_dir / "charts.json").read_text(encoding="utf-8"))
    segments = [(name, SEGMENT_LABELS.get(name, name)) for name in model.segment_names()]
    rng = np.random.default_rng()

    def read_lgd(raw):
        if raw is None or raw == "":
            return model.card["lgd_default"]
        if isinstance(raw, bool):
            raise ValidationError("lgd must be a number")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValidationError("lgd must be a number") from None
        if not np.isfinite(value):
            raise ValidationError("lgd must be a finite number")
        return value

    def page(template, active, **context):
        data = {key: charts[key] for key in PAGE_KEYS[template]}
        return render_template(template, active=active, data=data, metrics=charts["metrics"], card=model.card, segments=segments, **context)

    @app.get("/")
    def model_page():
        return page("model.html", "model")

    @app.get("/explainability")
    def explain_page():
        return page("explain.html", "explain")

    @app.get("/portfolio")
    def portfolio_page():
        return page("portfolio.html", "portfolio")

    @app.get("/applicant")
    def applicant_page():
        return page("applicant.html", "applicant", grades=sorted(model.grade_pd), sample_size=len(model.sample))

    @app.get("/api/applicant/<int:applicant_id>")
    def applicant(applicant_id):
        return jsonify(model.assess(applicant_id, read_lgd(request.args.get("lgd"))))

    @app.get("/api/applicant/random")
    def random_applicant():
        raw = request.args.get("grade")
        grade = None
        if raw not in (None, ""):
            try:
                grade = int(raw)
            except ValueError:
                raise ValidationError("grade must be an integer") from None
        return jsonify(model.assess(model.random_id(grade, rng), read_lgd(request.args.get("lgd"))))

    @app.post("/api/score")
    def score():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "id" not in body:
            raise ValidationError("body must be a JSON object with an id")
        applicant_id = body["id"]
        if isinstance(applicant_id, bool) or not isinstance(applicant_id, int):
            raise ValidationError("id must be an integer")
        overrides = body.get("overrides")
        return jsonify(model.assess(applicant_id, read_lgd(body.get("lgd")), {} if overrides is None else overrides))

    @app.get("/api/portfolio")
    def portfolio():
        segment = request.args.get("segment") or segments[0][0]
        return jsonify(model.portfolio_view(segment, read_lgd(request.args.get("lgd"))))

    @app.errorhandler(ValidationError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(NotFoundError)
    def not_found(error):
        return jsonify(error=str(error)), 404

    return app
