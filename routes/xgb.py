from flask import Blueprint, render_template  # type: ignore


xgb_bp = Blueprint("xgb", __name__)


@xgb_bp.get("/xgb_models")
def xgb_models_page():
    """Страница обучения XGB моделей."""
    return render_template("xgb_models.html")

