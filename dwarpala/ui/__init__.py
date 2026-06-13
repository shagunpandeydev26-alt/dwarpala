"""
Dwarpala Gradio demo UI (Phase 4).

Pure presentation over the proven pipeline: the UI calls the SAME
``DwarpalaPipeline.verify`` / ``liveness_only`` methods the REST API uses,
in-process. No verification/matching/liveness logic lives here.
"""

from dwarpala.ui.config import DemoSettings, load_demo_settings
from dwarpala.ui.app import build_demo

__all__ = ["DemoSettings", "load_demo_settings", "build_demo"]
