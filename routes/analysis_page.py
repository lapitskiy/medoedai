from flask import Blueprint, render_template

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/analitika')
def analytics_page():
    """Страница аналитики результатов обучения"""
    return render_template('analitika/index.html')
