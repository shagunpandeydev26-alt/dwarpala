"""
Dwarpala command-line interface.

Subcommands:
    serve            Launch the FastAPI REST server (config/env-driven host/port).
    demo             Launch the Gradio demo UI (config/env-driven host/port).
    download-models  Download required model weights (InsightFace + MiniFASNet).

Examples:
    dwarpala serve
    dwarpala serve --host 127.0.0.1 --port 9000
    dwarpala demo
    dwarpala demo --port 7861 --share
    dwarpala download-models
"""

import argparse
import sys
from typing import Optional, Sequence

from dwarpala.api.config import load_api_settings
from dwarpala.utils.logger import get_logger

logger = get_logger("cli")


def _cmd_serve(args: argparse.Namespace) -> int:
    """Launch uvicorn with config/env-driven host and port (CLI flags win)."""
    import uvicorn

    from dwarpala.api.server import create_app

    settings = load_api_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port

    app = create_app(settings=settings)
    logger.info(
        f"Starting Dwarpala API on http://{settings.host}:{settings.port} "
        f"(docs at /docs, audit={settings.audit_log}, "
        f"max_upload={settings.max_upload_mb}MB)"
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Launch the Gradio demo UI with config/env-driven host/port (CLI flags win)."""
    from dwarpala.ui.app import build_demo
    from dwarpala.ui.config import load_demo_settings

    settings = load_demo_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.share:
        settings.share = True

    demo = build_demo(model_dir=settings.model_dir)
    logger.info(
        f"Starting Dwarpala demo on http://{settings.host}:{settings.port} "
        f"(share={settings.share})"
    )
    demo.launch(server_name=settings.host, server_port=settings.port, share=settings.share)
    return 0


def _cmd_download_models(args: argparse.Namespace) -> int:
    """Download all registered model weights with SHA256 logging."""
    from dwarpala.utils.model_manager import ModelManager

    manager = ModelManager()
    results = manager.download_all(force=args.force)
    failed = [name for name, path in results.items() if path is None]
    for name, path in results.items():
        status = "OK" if path else "FAILED"
        logger.info(f"  [{status}] {name}: {path}")
    if failed:
        logger.error(f"Some models failed to download: {failed}")
        return 1
    logger.info("All models downloaded.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dwarpala",
        description="Dwarpala — Unified Biometric Verification Engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Launch the FastAPI REST server.")
    p_serve.add_argument("--host", default=None, help="Bind host (overrides config/env).")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (overrides config/env).")
    p_serve.set_defaults(func=_cmd_serve)

    p_demo = sub.add_parser("demo", help="Launch the Gradio demo UI.")
    p_demo.add_argument("--host", default=None, help="Bind host (overrides config/env).")
    p_demo.add_argument("--port", type=int, default=None, help="Bind port (overrides config/env).")
    p_demo.add_argument(
        "--share", action="store_true", help="Expose a public gradio.live link (off by default)."
    )
    p_demo.set_defaults(func=_cmd_demo)

    p_dl = sub.add_parser("download-models", help="Download model weights.")
    p_dl.add_argument(
        "--force", action="store_true", help="Re-download even if files already exist."
    )
    p_dl.set_defaults(func=_cmd_download_models)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
