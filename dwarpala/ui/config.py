"""
Gradio demo configuration loader.

Reads the ``demo:`` block from ``configs/inference_config.yaml`` with env-var
overrides, so host/port/share are config-driven (nothing hardcoded).

Env overrides:
    DWARPALA_DEMO_HOST   -> host
    DWARPALA_DEMO_PORT   -> port
    DWARPALA_DEMO_SHARE  -> share  ("1"/"true"/"yes" enable a public link)
    DWARPALA_MODEL_DIR   -> model_dir (passed to the pipeline)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("ui.config")

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "inference_config.yaml"


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DemoSettings:
    """Resolved Gradio demo settings."""

    host: str = "127.0.0.1"
    port: int = 7860
    # share=False by default: do NOT expose a public gradio.live tunnel unless
    # explicitly enabled (it routes traffic — including uploaded faces — through
    # a third party). Enable only for trusted, ephemeral demos.
    share: bool = False
    model_dir: Optional[str] = None


def load_demo_settings(config_path: Optional[Path] = None) -> DemoSettings:
    """
    Build :class:`DemoSettings` from the YAML config, then apply env overrides.

    Args:
        config_path: Optional path to the inference YAML. Defaults to
            ``configs/inference_config.yaml``. Missing file is non-fatal.

    Returns:
        Fully resolved DemoSettings.
    """
    settings = DemoSettings()

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if path.exists():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(str(path))
            demo_cfg = OmegaConf.to_container(cfg.get("demo", {}), resolve=True) or {}
            if "host" in demo_cfg:
                settings.host = str(demo_cfg["host"])
            if "port" in demo_cfg:
                settings.port = int(demo_cfg["port"])
            if "share" in demo_cfg:
                settings.share = bool(demo_cfg["share"])
            if cfg.get("model_dir") is not None:
                settings.model_dir = str(cfg.get("model_dir"))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to read demo config from {path}: {e}")
    else:
        logger.info(f"Demo config not found at {path}; using defaults + env.")

    if os.getenv("DWARPALA_DEMO_HOST"):
        settings.host = os.environ["DWARPALA_DEMO_HOST"]
    if os.getenv("DWARPALA_DEMO_PORT"):
        settings.port = int(os.environ["DWARPALA_DEMO_PORT"])
    if os.getenv("DWARPALA_DEMO_SHARE"):
        settings.share = _as_bool(os.environ["DWARPALA_DEMO_SHARE"])
    if os.getenv("DWARPALA_MODEL_DIR"):
        settings.model_dir = os.environ["DWARPALA_MODEL_DIR"]

    if settings.model_dir:
        settings.model_dir = str(Path(settings.model_dir).expanduser())

    return settings
