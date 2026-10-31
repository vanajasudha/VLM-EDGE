"""Dashboard views package for VLM-Edge."""
from src.views.overview_view import render_overview
from src.views.live_monitoring_view import render_live_monitoring
from src.views.ai_assistant_view import render_ai_assistant
from src.views.performance_view import render_performance
from src.views.evaluation_view import render_evaluation
from src.views.system_settings_view import render_system_settings

__all__ = [
    "render_overview",
    "render_live_monitoring",
    "render_ai_assistant",
    "render_performance",
    "render_evaluation",
    "render_system_settings",
]

