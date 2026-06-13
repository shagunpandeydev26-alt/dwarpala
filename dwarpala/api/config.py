"""
API configuration loader.

Settings come from ``configs/inference_config.yaml`` (the ``api:`` block) with
per-field environment-variable overrides, so nothing is hardcoded in the server.
Env vars always win over the YAML file.

Env overrides:
    DWARPALA_HOST            -> host
    DWARPALA_PORT            -> port
    DWARPALA_MAX_UPLOAD_MB   -> max_upload_mb
    DWARPALA_CORS_ORIGINS    -> cors_origins (comma-separated)
    AUDIT_LOG                -> audit_log ("off" | "sqlite")
    DWARPALA_AUDIT_DB        -> audit_db (sqlite path)
    DWARPALA_MODEL_DIR       -> model_dir (passed to the pipeline)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("api.config")

# Project-root default for the inference config.
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "inference_config.yaml"


@dataclass
class APISettings:
    """Resolved API server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 15
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    audit_log: str = "off"  # "off" | "sqlite"
    audit_db: str = "audit.db"
    model_dir: Optional[str] = None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    @property
    def audit_enabled(self) -> bool:
        return str(self.audit_log).lower() == "sqlite"


def _coerce_origins(value) -> List[str]:
    if isinstance(value, str):
        return [o.strip() for o in value.split(",") if o.strip()]
    if isinstance(value, (list, tuple)):
        return [str(o) for o in value]
    return ["*"]


def load_api_settings(config_path: Optional[Path] = None) -> APISettings:
    """
    Build :class:`APISettings` from the YAML config, then apply env overrides.

    Args:
        config_path: Optional path to the inference YAML. Defaults to
            ``configs/inference_config.yaml`` at the project root. Missing file
            is non-fatal — built-in defaults are used.

    Returns:
        Fully resolved APISettings.
    """
    settings = APISettings()

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if path.exists():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(str(path))
            api_cfg = OmegaConf.to_container(cfg.get("api", {}), resolve=True) or {}
            if "host" in api_cfg:
                settings.host = str(api_cfg["host"])
            if "port" in api_cfg:
                settings.port = int(api_cfg["port"])
            if "max_upload_mb" in api_cfg:
                settings.max_upload_mb = int(api_cfg["max_upload_mb"])
            if "cors_origins" in api_cfg:
                settings.cors_origins = _coerce_origins(api_cfg["cors_origins"])
            if "audit_log" in api_cfg:
                settings.audit_log = str(api_cfg["audit_log"])
            if "audit_db" in api_cfg:
                settings.audit_db = str(api_cfg["audit_db"])
            # model_dir lives at the top level of the inference config.
            if cfg.get("model_dir") is not None:
                settings.model_dir = str(cfg.get("model_dir"))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to read API config from {path}: {e}")
    else:
        logger.info(f"API config not found at {path}; using defaults + env.")

    # ── Environment overrides (highest priority) ──
    if os.getenv("DWARPALA_HOST"):
        settings.host = os.environ["DWARPALA_HOST"]
    if os.getenv("DWARPALA_PORT"):
        settings.port = int(os.environ["DWARPALA_PORT"])
    if os.getenv("DWARPALA_MAX_UPLOAD_MB"):
        settings.max_upload_mb = int(os.environ["DWARPALA_MAX_UPLOAD_MB"])
    if os.getenv("DWARPALA_CORS_ORIGINS"):
        settings.cors_origins = _coerce_origins(os.environ["DWARPALA_CORS_ORIGINS"])
    if os.getenv("AUDIT_LOG"):
        settings.audit_log = os.environ["AUDIT_LOG"]
    if os.getenv("DWARPALA_AUDIT_DB"):
        settings.audit_db = os.environ["DWARPALA_AUDIT_DB"]
    if os.getenv("DWARPALA_MODEL_DIR"):
        settings.model_dir = os.environ["DWARPALA_MODEL_DIR"]

    # Expand "~" in model_dir for the pipeline.
    if settings.model_dir:
        settings.model_dir = str(Path(settings.model_dir).expanduser())

    return settings
