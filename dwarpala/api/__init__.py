"""
Dwarpala REST API (Phase 3).

Exposes the existing DwarpalaPipeline as a FastAPI service. The API is a thin
transport layer — all verification, matching, and liveness logic lives in
``dwarpala.yantra.pipeline`` (single source of truth shared with the Gradio demo).
"""

from dwarpala.api.config import APISettings, load_api_settings
from dwarpala.api.server import create_app

__all__ = ["APISettings", "load_api_settings", "create_app"]
