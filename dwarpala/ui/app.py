"""
Dwarpala Gradio demo — Verify + Liveness Lab tabs.

This is pure presentation. Every number and plot comes from real pipeline
output: the callbacks call the SAME ``DwarpalaPipeline.verify`` /
``liveness_only`` methods the REST API uses, in-process. No verification,
matching, or liveness logic is implemented here.

The display helpers (verdict colors, score bars, breakdown rows, rPPG figure)
are pure functions with no Gradio or pipeline dependency, so they can be unit
tested directly with synthetic inputs (no browser/webcam needed in CI).
"""

from typing import Callable, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless / thread-safe; we build Figures, not pyplot state

import numpy as np
from matplotlib.figure import Figure

from dwarpala.utils.logger import get_logger

logger = get_logger("ui.app")

LIVENESS_LAYERS = ("minifas", "texture", "temporal", "rppg")

# Verdict → (hex color, icon, label). Both verification verdicts (ACCEPT /
# REJECT / MANUAL_REVIEW) and liveness verdicts (LIVE / SPOOF) are mapped.
_GREEN = "#1a7f37"
_RED = "#c1121f"
_AMBER = "#b45309"
_GREY = "#6b7280"

_VERDICT_STYLE = {
    "ACCEPT": (_GREEN, "✅", "ACCEPT"),
    "LIVE": (_GREEN, "✅", "LIVE"),
    "REJECT": (_RED, "❌", "REJECT"),
    "SPOOF": (_RED, "❌", "SPOOF"),
    "MANUAL_REVIEW": (_AMBER, "⚠️", "MANUAL REVIEW"),
}


# ── pure display helpers (unit-tested) ──────────────────────────────────────
def verdict_color(verdict: str) -> str:
    """Return the banner hex color for a verdict (green/red/amber/grey)."""
    return _VERDICT_STYLE.get((verdict or "").upper(), (_GREY, "❓", verdict))[0]


def verdict_banner_html(verdict: str, subtitle: str = "") -> str:
    """Render a colored verdict banner (green ACCEPT/LIVE, red REJECT/SPOOF, amber review)."""
    color, icon, label = _VERDICT_STYLE.get((verdict or "").upper(), (_GREY, "❓", verdict or "—"))
    sub = (
        f"<div style='font-size:0.85em;opacity:0.95;margin-top:4px'>{subtitle}</div>"
        if subtitle
        else ""
    )
    return (
        f"<div style='background:{color};color:white;padding:14px 18px;"
        f"border-radius:10px;font-weight:700;font-size:1.25em'>"
        f"{icon} {label}{sub}</div>"
    )


def score_bar_html(label: str, score: Optional[float]) -> str:
    """Render a labeled score bar in [0,1]; shows '—' when the score is absent."""
    if not isinstance(score, (int, float)) or score != score:  # None or NaN
        return f"<div style='margin:6px 0'><b>{label}:</b> —</div>"
    pct = max(0.0, min(100.0, float(score) * 100.0))
    color = _GREEN if score >= 0.5 else _RED
    return (
        f"<div style='margin:8px 0'><b>{label}:</b> {score:.3f}"
        f"<div style='background:#e5e7eb;border-radius:6px;height:14px;width:100%;margin-top:3px'>"
        f"<div style='width:{pct:.0f}%;background:{color};height:14px;border-radius:6px'></div>"
        f"</div></div>"
    )


def format_breakdown(
    breakdown: dict,
    signal_status: dict,
    rppg_valid: Optional[bool] = None,
) -> List[List[str]]:
    """
    Build the per-layer liveness table rows: [layer, score, signal_status].

    Absent signals (e.g. temporal/rppg on a single image) render their score as
    '—' and their status as NOT_APPLICABLE — never a misleading 0. When rPPG ran
    (status OK) but produced no confident heartbeat, the status is surfaced as
    LOW_CONFIDENCE so the user understands the signal was weak, not missing.

    Args:
        breakdown: liveness_breakdown dict (per-layer score or None).
        signal_status: per-layer status dict ("OK"/"NOT_APPLICABLE"/...).
        rppg_valid: whether rPPG found a valid heartbeat (None if unknown).

    Returns:
        Rows of [layer, score_str, status_str].
    """
    breakdown = breakdown or {}
    signal_status = signal_status or {}
    rows: List[List[str]] = []
    for layer in LIVENESS_LAYERS:
        score = breakdown.get(layer)
        status = signal_status.get(layer, "NOT_APPLICABLE")
        if layer == "rppg" and status == "OK" and rppg_valid is False:
            status = "LOW_CONFIDENCE"
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        rows.append([layer, score_str, status])
    return rows


def fft_to_image(spectrum: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert a [0,1] FFT magnitude spectrum to a uint8 grayscale image for display."""
    if spectrum is None:
        return None
    arr = np.clip(np.asarray(spectrum, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def rppg_figure(
    time_axis: Optional[np.ndarray],
    signal: Optional[np.ndarray],
    bpm: Optional[float],
    has_valid: bool,
    message: str = "",
) -> Figure:
    """
    Build the rPPG pulse figure.

    When a confident heartbeat exists, plots the REAL filtered pulse trace with
    the estimated BPM. Otherwise draws an honest "insufficient signal" message
    in the plot area instead of a fabricated trace.
    """
    fig = Figure(figsize=(7.0, 3.0), dpi=100)
    ax = fig.subplots()
    if has_valid and time_axis is not None and signal is not None and len(signal) > 1:
        ax.plot(np.asarray(time_axis), np.asarray(signal), color="#b91c1c", linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pulse (a.u.)")
        title = (
            f"rPPG pulse — {bpm:.0f} BPM"
            if isinstance(bpm, (int, float)) and bpm and bpm > 0
            else "rPPG pulse"
        )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            message or "Insufficient signal — need ≥5s stable video.",
            ha="center",
            va="center",
            wrap=True,
            fontsize=11,
            color=_GREY,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("rPPG pulse — unavailable")
    fig.tight_layout()
    return fig


# ── callbacks (call the real pipeline; tested with a mocked pipeline) ────────
def run_verify(pipeline, id_image: Optional[np.ndarray], selfie: Optional[np.ndarray]):
    """
    Tab 1 callback: run the same pipeline.verify the API uses on RGB arrays.

    Gradio Image inputs are RGB numpy arrays — the pipeline is RGB-native, so we
    pass them straight through (no channel swap here). Returns display-ready
    banner / score bars / breakdown rows / explanation / latency.
    """
    empty_rows = format_breakdown({}, {}, None)
    if id_image is None or selfie is None:
        return (
            verdict_banner_html("MANUAL_REVIEW", "Provide both an ID photo and a selfie."),
            score_bar_html("Match score", None),
            score_bar_html("Liveness score", None),
            empty_rows,
            "Waiting for inputs.",
            0.0,
        )

    result = pipeline.verify(id_image, selfie)
    d = result.to_dict()
    rows = format_breakdown(d.get("liveness_breakdown") or {}, d.get("signal_status") or {}, None)
    return (
        verdict_banner_html(d["verdict"]),
        score_bar_html("Match score", d.get("match_score")),
        score_bar_html("Liveness score", d.get("liveness_score")),
        rows,
        d.get("explanation", ""),
        d.get("latency_ms", 0.0),
    )


def _fft_for_frame(pipeline, frame: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort real FFT texture map for one frame (reuses pipeline detect/align)."""
    try:
        det = pipeline.detector.detect_largest(frame)
        if det is None:
            return None
        aligned = pipeline.aligner.align(frame, det.landmarks)
        spectrum = pipeline.liveness.texture_analyzer.get_fft_spectrum(aligned)
        return fft_to_image(spectrum)
    except Exception as e:  # viz is optional — never break the callback
        logger.debug(f"FFT texture viz unavailable: {e}")
        return None


def run_liveness_lab(pipeline, video):
    """
    Tab 2 callback: run pipeline.liveness_only on the video, then visualize the
    REAL rPPG waveform and per-layer breakdown.

    Frames are extracted ONCE via the pipeline (load_selfie_frames) and the same
    list is passed to liveness_only AND to the rPPG analyzer's waveform accessor,
    so the plotted pulse is exactly the scored signal.
    """
    if video is None:
        return (
            rppg_figure(None, None, None, False, "Provide a webcam clip or upload a video."),
            format_breakdown({}, {}, None),
            None,
            verdict_banner_html("MANUAL_REVIEW", "No video provided."),
        )

    frames = pipeline.load_selfie_frames(video)
    if not frames:
        return (
            rppg_figure(None, None, None, False, "Could not read any frames from the video."),
            format_breakdown({}, {}, None),
            None,
            verdict_banner_html("REJECT", "No decodable frames in the video."),
        )

    if len(frames) < 150:  # ~5s @ 30fps
        logger.warning(f"Only {len(frames)} usable frames (<5s @30fps); rPPG may be weak.")

    result = pipeline.liveness_only(frames)
    lv = result.liveness_verdict
    d = result.to_dict()

    rppg_result = getattr(lv, "rppg_result", None) if lv else None
    rppg_valid = bool(rppg_result and rppg_result.has_valid_heartbeat)
    bpm = rppg_result.heart_rate_bpm if rppg_result else None
    rppg_status = (d.get("signal_status") or {}).get("rppg", "NOT_APPLICABLE")

    if rppg_valid:
        time_axis, signal = pipeline.liveness.rppg_analyzer.get_rppg_waveform(frames)
        fig = rppg_figure(time_axis, signal, bpm, True, "")
    else:
        fig = rppg_figure(
            None,
            None,
            None,
            False,
            f"Insufficient signal — need ≥5s stable video (rPPG status: {rppg_status}).",
        )

    rows = format_breakdown(
        d.get("liveness_breakdown") or {},
        d.get("signal_status") or {},
        (rppg_result.has_valid_heartbeat if rppg_result else None),
    )
    fft_img = _fft_for_frame(pipeline, frames[0])
    verdict = "LIVE" if d.get("is_live") else "SPOOF"
    banner = verdict_banner_html(verdict, d.get("explanation", ""))
    return fig, rows, fft_img, banner


def _default_pipeline_factory(model_dir: Optional[str] = None) -> Callable[[], object]:
    def factory():
        from pathlib import Path

        from dwarpala.yantra.pipeline import DwarpalaPipeline

        return DwarpalaPipeline(model_dir=Path(model_dir) if model_dir else None)

    return factory


def build_demo(
    pipeline_factory: Optional[Callable[[], object]] = None,
    model_dir: Optional[str] = None,
):
    """
    Build the Gradio Blocks demo. Loads the pipeline ONCE (here) and closes over
    it in the callbacks — no per-interaction model reload.

    Args:
        pipeline_factory: Zero-arg callable returning a pipeline. Defaults to the
            real DwarpalaPipeline; tests inject a mock.
        model_dir: Optional model directory for the default factory.

    Returns:
        A gradio.Blocks app (call .launch() to serve).
    """
    import gradio as gr

    factory = pipeline_factory or _default_pipeline_factory(model_dir)
    logger.info("Loading pipeline for Gradio demo (once)...")
    pipeline = factory()

    with gr.Blocks(title="Dwarpala — Biometric Verification") as demo:
        gr.Markdown(
            "# 🛕 Dwarpala — The Celestial Gatekeeper\n"
            "Identity match + multi-modal liveness. Every score and plot is real "
            "pipeline output."
        )

        with gr.Tab("Verify"):
            with gr.Row():
                with gr.Column():
                    id_in = gr.Image(type="numpy", sources=["upload"], label="ID photo")
                    selfie_in = gr.Image(type="numpy", sources=["upload", "webcam"], label="Selfie")
                    verify_btn = gr.Button("Verify", variant="primary")
                with gr.Column():
                    v_banner = gr.HTML()
                    v_match = gr.HTML()
                    v_live = gr.HTML()
                    v_table = gr.Dataframe(
                        headers=["Layer", "Score", "Signal status"],
                        datatype=["str", "str", "str"],
                        label="Liveness breakdown",
                        interactive=False,
                    )
                    v_expl = gr.Textbox(label="Explanation", lines=3, interactive=False)
                    v_latency = gr.Number(label="Latency (ms)", interactive=False)
            verify_btn.click(
                fn=lambda a, b: run_verify(pipeline, a, b),
                inputs=[id_in, selfie_in],
                outputs=[v_banner, v_match, v_live, v_table, v_expl, v_latency],
            )

        with gr.Tab("Liveness Lab"):
            gr.Markdown(
                "Record ~5s of webcam video (or upload a clip). The rPPG plot shows "
                "your **real extracted pulse** with estimated BPM — or an honest "
                "'insufficient signal' note for short/unstable clips."
            )
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(sources=["webcam", "upload"], label="Selfie video")
                    live_btn = gr.Button("Analyze liveness", variant="primary")
                    l_banner = gr.HTML()
                with gr.Column():
                    l_plot = gr.Plot(label="rPPG pulse waveform")
                    l_table = gr.Dataframe(
                        headers=["Layer", "Score", "Signal status"],
                        datatype=["str", "str", "str"],
                        label="Liveness breakdown",
                        interactive=False,
                    )
                    l_fft = gr.Image(label="FFT texture map (real artifact)", height=200)
            live_btn.click(
                fn=lambda v: run_liveness_lab(pipeline, v),
                inputs=[vid_in],
                outputs=[l_plot, l_table, l_fft, l_banner],
            )

    return demo
