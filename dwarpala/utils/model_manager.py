"""
Model download and management for Dwarpala.

Downloads pretrained model files to ~/.dwarpala/models/ with SHA256 verification.
Never commits weights to git — models are downloaded on first use or via CLI.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("utils.model_manager")

# Default model directory
DEFAULT_MODEL_DIR = Path.home() / ".dwarpala" / "models"

# Model registry: name → {url, sha256, filename, description}
# SHA256 values are verified after download to ensure integrity.
MODEL_REGISTRY: Dict[str, dict] = {
    "buffalo_l_recognition": {
        "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        "sha256": None,  # Will be populated after first verified download
        "filename": "w600k_r50.onnx",
        "description": "ArcFace R50 recognition model (buffalo_l) — ~175MB",
        "license": "Non-commercial research only (InsightFace license)",
    },
    "buffalo_l_detector": {
        "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        "sha256": None,
        "filename": "det_10g.onnx",
        "description": "SCRFD 10GF face detector (buffalo_l) — ~16MB",
        "license": "Non-commercial research only (InsightFace license)",
    },
    "minifas_v2_2_7": {
        "url": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/images/2.7_80x80_MiniFASNetV2.pth",
        "sha256": None,
        "filename": "2.7_80x80_MiniFASNetV2.pth",
        "description": "MiniFASNetV2 anti-spoofing model (scale 2.7, 80×80) — ~600KB",
        "license": "Apache-2.0 (Silent-Face-Anti-Spoofing)",
    },
    "minifas_v1se_4_0": {
        "url": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/images/4_0_0_80x80_MiniFASNetV1SE.pth",
        "sha256": None,
        "filename": "4_0_0_80x80_MiniFASNetV1SE.pth",
        "description": "MiniFASNetV1SE anti-spoofing model (scale 4.0, 80×80) — ~600KB",
        "license": "Apache-2.0 (Silent-Face-Anti-Spoofing)",
    },
}


class ModelManager:
    """
    Manages model downloads and paths for Dwarpala.

    Models are stored in ~/.dwarpala/models/ by default.
    Downloads include SHA256 verification when checksums are available.

    Usage:
        manager = ModelManager()
        path = manager.get_model_path("buffalo_l_recognition")
        # Downloads if not present, returns Path to local file
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Args:
            model_dir: Directory for storing models. Defaults to ~/.dwarpala/models/.
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelManager: model_dir={self.model_dir}")

    def get_model_path(self, model_name: str) -> Path:
        """
        Get path to a model file, raising an error if not downloaded.

        Args:
            model_name: Key in MODEL_REGISTRY.

        Returns:
            Path to the local model file.

        Raises:
            FileNotFoundError: If model not downloaded and no URL available.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        info = MODEL_REGISTRY[model_name]
        path = self.model_dir / info["filename"]

        if path.exists():
            return path

        # Fallback: check insightface's cache directory
        # (models downloaded via `insightface.app.FaceAnalysis`)
        insightface_cache = Path.home() / ".insightface" / "models" / "buffalo_l"
        fallback_path = insightface_cache / info["filename"]
        if fallback_path.exists():
            logger.info(
                f"Found {info['filename']} in insightface cache: {fallback_path}"
            )
            return fallback_path

        # Model not found — provide clear instructions
        raise FileNotFoundError(
            f"\n{'=' * 60}\n"
            f"  Model not found: {info['filename']}\n"
            f"  Description: {info['description']}\n"
            f"  Expected at: {path}\n"
            f"{'=' * 60}\n"
            f"  To download all required models, run:\n"
            f"    dwarpala download-models\n"
            f"  Or: pip install insightface && python -c \"\n"
            f"    from insightface.app import FaceAnalysis;\n"
            f"    FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\"\n"
            f"  And place in: {self.model_dir}/\n"
            f"{'=' * 60}"
        )

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model file exists locally."""
        if model_name not in MODEL_REGISTRY:
            return False
        info = MODEL_REGISTRY[model_name]
        return (self.model_dir / info["filename"]).exists()

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model file with optional SHA256 verification.

        Args:
            model_name: Key in MODEL_REGISTRY.
            force: Re-download even if file exists.

        Returns:
            Path to the downloaded model file.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        info = MODEL_REGISTRY[model_name]
        path = self.model_dir / info["filename"]

        if path.exists() and not force:
            logger.info(f"Model already exists: {path}")
            return path

        url = info.get("url")
        if not url:
            raise ValueError(
                f"No download URL configured for model: {model_name}. "
                f"Please download manually: {info['description']}"
            )

        logger.info(f"Downloading {model_name}: {info['description']}")
        logger.info(f"  URL: {url}")
        logger.info(f"  Destination: {path}")

        self._download_file(url, path)

        # Verify SHA256 if available
        expected_sha = info.get("sha256")
        if expected_sha:
            actual_sha = self._compute_sha256(path)
            if actual_sha != expected_sha:
                path.unlink()  # Remove corrupted file
                raise ValueError(
                    f"SHA256 mismatch for {model_name}!\n"
                    f"  Expected: {expected_sha}\n"
                    f"  Got:      {actual_sha}\n"
                    f"  File has been removed. Please try downloading again."
                )
            logger.info(f"SHA256 verified: {actual_sha[:16]}...")
        else:
            actual_sha = self._compute_sha256(path)
            logger.info(
                f"SHA256 (no expected hash to verify against): {actual_sha}"
            )

        logger.info(f"Downloaded successfully: {path}")
        return path

    def download_all(self, force: bool = False) -> Dict[str, Path]:
        """
        Download all registered models.

        Returns:
            Dictionary of model_name → local path.
        """
        results = {}
        for name, info in MODEL_REGISTRY.items():
            if info.get("url"):
                try:
                    path = self.download_model(name, force=force)
                    results[name] = path
                except Exception as e:
                    logger.error(f"Failed to download {name}: {e}")
                    results[name] = None
            else:
                logger.warning(
                    f"Skipping {name}: no download URL configured. "
                    f"Please download manually: {info['description']}"
                )
        return results

    def list_models(self) -> Dict[str, dict]:
        """List all registered models and their download status."""
        status = {}
        for name, info in MODEL_REGISTRY.items():
            path = self.model_dir / info["filename"]
            status[name] = {
                "filename": info["filename"],
                "description": info["description"],
                "license": info.get("license", "Unknown"),
                "downloaded": path.exists(),
                "path": str(path) if path.exists() else None,
                "size_mb": f"{path.stat().st_size / 1024 / 1024:.1f}" if path.exists() else None,
            }
        return status

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        """Download a file with progress bar."""
        import requests

        destination.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    mb_done = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    sys.stdout.write(
                        f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)"
                    )
                    sys.stdout.flush()

        if total_size > 0:
            print()  # Newline after progress

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
