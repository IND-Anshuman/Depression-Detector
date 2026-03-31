from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from omegaconf import DictConfig

from mmds.inference import BufferedInferenceService

from .styles import glassmorphic_css


def _running_in_container() -> bool:
    if os.name == "nt":
        return False
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and any(token in cgroup.read_text(encoding="utf-8") for token in ["docker", "kubepods", "containerd"]):
            return True
    except Exception:
        return False
    return False


def _apply_gradio_client_schema_bool_patch() -> None:
    """Work around gradio_client JSON schema parsing failing on boolean schemas.

    JSON Schema allows boolean schemas (e.g. `additionalProperties: true`). Some
    gradio_client versions crash when converting those schemas to python types.
    This only affects the API info display, not actual inference.
    """

    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    if getattr(client_utils, "_mmds_bool_schema_patch", False):
        return

    original = getattr(client_utils, "_json_schema_to_python_type", None)
    if original is None:
        return

    def patched(schema: Any, defs: Any) -> str:
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    setattr(client_utils, "_json_schema_to_python_type", patched)
    setattr(client_utils, "_mmds_bool_schema_patch", True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dashboard_js() -> str:
    js_path = _repo_root() / "app" / "static" / "js" / "dashboard.js"
    if not js_path.exists():
        return ""
    return f"<script>{js_path.read_text(encoding='utf-8')}</script>"


def _blank_image() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def launch_gradio(cfg: DictConfig) -> None:
    configured_ckpt = getattr(cfg.paths, "checkpoint_path", None)
    ckpt = Path(str(configured_ckpt)) if configured_ckpt else Path(str(cfg.paths.artifacts_dir)) / "checkpoint.pt"
    service = BufferedInferenceService(cfg, ckpt_path=ckpt if ckpt.exists() else ckpt)
    css = glassmorphic_css()

    with gr.Blocks(css=css, head=_dashboard_js(), theme=gr.themes.Soft()) as demo:
        gr.HTML(
            f"""
            <section class="mmds-shell">
              <div class="mmds-hero">
                <div class="mmds-kicker">Live Multimodal Depression Screening</div>
                <h1>{cfg.ui.title}</h1>
                <p>{cfg.ui.privacy_note}</p>
                <div class="mmds-banner">{cfg.ui.disclaimer}</div>
              </div>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["mmds-card"]):
                    webcam = gr.Image(label="Live Webcam", sources=["webcam"], streaming=True, every=0.2, type="numpy")
                    mic = gr.Audio(label="Microphone (optional)", sources=["microphone"], streaming=True, type="numpy")
                    reset = gr.Button("Reset Session")
                with gr.Group(elem_classes=["mmds-card"]):
                    overlay = gr.Image(label="Holistic Overlay", type="numpy")
                    risk_text = gr.Markdown()
                    runtime_text = gr.Markdown()
                    model_text = gr.Markdown(value=service.status_markdown())
            with gr.Column(scale=1):
                with gr.Row():
                    risk_gauge = gr.Image(label="Risk Gauge", type="numpy")
                    severity_bars = gr.Image(label="3-Class Bars", type="numpy")
                with gr.Row():
                    modality_bars = gr.Image(label="Modality Bars", type="numpy")
                    attention_heatmap = gr.Image(label="Attention Heatmap", type="numpy")
                with gr.Row():
                    rolling = gr.Image(label="Rolling Risk", type="numpy")
                    au = gr.Image(label="Facial Trend", type="numpy")
                signal_text = gr.Markdown()

        def on_reset():
            service.reset()
            blank = _blank_image()
            return (
                blank,
                blank,
                blank,
                blank,
                blank,
                blank,
                blank,
                "Awaiting frames.",
                "FPS -",
                service.status_markdown(),
                "Signal quality pending.",
            )

        def _render_result(res):
            sev_names = ["low", "moderate", "high"]
            risk_md = (
                f"### Risk Summary\n"
                f"- Risk probability: `{res.risk_prob:.3f}`\n"
                f"- Severity: `{sev_names[int(np.argmax(np.array(res.severity_probs)))]}`\n"
                f"- Continuous score: `{res.continuous_score:.3f}`\n"
                f"- Uncertainty: `{res.uncertainty:.3f}`"
            )
            runtime_md = (
                f"### Runtime\n"
                f"- FPS: `{res.fps:.1f}`\n"
                f"- Latency: `{res.latency_ms:.1f} ms`\n"
                f"- Last model refresh: `{res.result_age_s:.2f} s ago`\n"
                f"- Abstain: `{'yes' if res.abstain else 'no'}`\n"
                f"- Runtime notes: `{', '.join(res.model_notes) if res.model_notes else 'none'}`"
            )
            model_md = (
                "### Model Status\n"
                f"- Checkpoint: `{'loaded' if res.checkpoint_loaded else 'not loaded'}`\n"
                f"- Path: `{res.checkpoint_path or 'none'}`\n"
                f"- Extractor: `{res.extractor_backend}`\n"
                f"- Trained modalities: `{', '.join(res.trained_modalities) if res.trained_modalities else 'unknown'}`\n"
                f"- Used live modalities: `{', '.join(res.used_modalities) if res.used_modalities else 'none'}`\n"
                f"- Skipped modalities: `{', '.join(res.skipped_modalities) if res.skipped_modalities else 'none'}`\n"
                f"- Notes: `{', '.join(res.model_notes) if res.model_notes else 'none'}`"
            )
            signal_md = (
                f"### Signal & Privacy\n"
                f"- Warnings: `{', '.join(res.quality_warnings) if res.quality_warnings else 'none'}`\n"
                f"- Brightness: `{res.quality_summary.get('brightness', 0.0):.2f}`\n"
                f"- Blur: `{res.quality_summary.get('blur', 0.0):.2f}`\n"
                f"- Motion: `{res.quality_summary.get('motion', 0.0):.2f}`\n"
                f"- BDD auxiliary: `{res.bdd_variability:.3f}`"
            )
            return (
                res.overlay_img,
                res.risk_gauge_img,
                res.severity_bars_img,
                res.modality_bars_img,
                res.attention_heatmap_img,
                res.rolling_score_img,
                res.au_trend_img,
                risk_md,
                runtime_md,
                model_md,
                signal_md,
            )

        def _audio_to_f32(audio_value: Any) -> np.ndarray | None:
            if audio_value is None or audio_value is False:
                return None
            if isinstance(audio_value, (tuple, list)) and len(audio_value) == 2:
                _, data = audio_value
            elif isinstance(audio_value, dict):
                data = audio_value.get("data") or audio_value.get("array") or audio_value.get("samples")
            elif isinstance(audio_value, (bytes, bytearray)):
                return None
            else:
                data = audio_value

            if data is None:
                return None

            if isinstance(data, (bytes, bytearray, str, Path)):
                return None

            arr = np.asarray(data)
            if arr.size == 0 or arr.ndim == 0:
                return None
            if arr.dtype.kind not in {"i", "u", "f"}:
                return None
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            return arr.astype(np.float32)

        def on_stream(frame: Any, audio: Any):
            try:
                bgr = None
                if frame is not None and frame is not False:
                    arr = np.asarray(frame)
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        bgr = arr[:, :, :3][:, :, ::-1].copy()

                audio_chunk = _audio_to_f32(audio)
                service.push(bgr, audio_chunk)
                return _render_result(service.run_live())
            except Exception as exc:
                blank = _blank_image()
                err = f"### Runtime\n- Error: `{type(exc).__name__}: {exc}`"
                return (
                    blank,
                    blank,
                    blank,
                    blank,
                    blank,
                    blank,
                    blank,
                    "Awaiting stable input.",
                    err,
                    service.status_markdown(),
                    "Signal quality unavailable.",
                )

        webcam.stream(
            on_stream,
            inputs=[webcam, mic],
            outputs=[overlay, risk_gauge, severity_bars, modality_bars, attention_heatmap, rolling, au, risk_text, runtime_text, model_text, signal_text],
            stream_every=0.2,
        )
        reset.click(
            on_reset,
            outputs=[overlay, risk_gauge, severity_bars, modality_bars, attention_heatmap, rolling, au, risk_text, runtime_text, model_text, signal_text],
        )

    demo.queue(max_size=32)
    _apply_gradio_client_schema_bool_patch()

    server_name = os.getenv("GRADIO_SERVER_NAME")
    if not server_name:
        server_name = "0.0.0.0" if _running_in_container() else "127.0.0.1"

    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    share_env = os.getenv("GRADIO_SHARE")
    share = share_env.lower() in {"1", "true", "yes"} if share_env is not None else False

    demo.launch(server_name=server_name, server_port=server_port, share=share)
