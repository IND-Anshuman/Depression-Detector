from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from omegaconf import DictConfig

from mmds.inference import BufferedInferenceService

from .styles import glassmorphic_css


def _sev_label(p: list[float]) -> str:
    names = ["low", "moderate", "high"]
    i = int(np.argmax(np.array(p)))
    return f"{names[i]}"


def launch_gradio(cfg: DictConfig) -> None:
    ckpt = Path(str(cfg.paths.artifacts_dir)) / "checkpoint.pt"
    service = BufferedInferenceService(cfg, ckpt_path=ckpt if ckpt.exists() else None)

    css = glassmorphic_css()

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            f"""
            <div class='mmds-shell' style='padding:18px'>
              <h2 style='margin:0'>{cfg.ui.title}</h2>
              <div class='mmds-sub'>{cfg.ui.disclaimer}</div>
              <div class='mmds-sub'>{cfg.ui.privacy_note}</div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["mmds-card"]):
                    webcam = gr.Image(label="Webcam", sources=["webcam"], streaming=True)
                    mic = gr.Audio(label="Microphone (optional)", sources=["microphone"], streaming=True, type="numpy")
                    reset = gr.Button("Reset")

                with gr.Group(elem_classes=["mmds-card"]):
                    file_vid = gr.Video(label="Sample file mode (optional)")
                    file_run = gr.Button("Run on file")

            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Group(elem_classes=["mmds-card"], scale=1):
                        risk = gr.Markdown("**Binary risk**\n\n—")
                    with gr.Group(elem_classes=["mmds-card"], scale=1):
                        sev = gr.Markdown("**Severity (3-class)**\n\n—")
                with gr.Row():
                    with gr.Group(elem_classes=["mmds-card"], scale=1):
                        cont = gr.Markdown("**Continuous score**\n\n—")
                    with gr.Group(elem_classes=["mmds-card"], scale=1):
                        unc = gr.Markdown("**Uncertainty / confidence**\n\n—")

                with gr.Group(elem_classes=["mmds-card"]):
                    bdd = gr.Markdown("**Behavioral variability (auxiliary)**\n\n—")
                    qwarn = gr.Markdown("**Signal quality**\n\n—")

                with gr.Row():
                    with gr.Group(elem_classes=["mmds-card"], scale=1):
                        rolling = gr.Image(label="Rolling risk", type="numpy")
                    with gr.Group(elem_classes=["mmds-card"], scale=1):
                        imp = gr.Image(label="Temporal importance", type="numpy")

                with gr.Group(elem_classes=["mmds-card"]):
                    au = gr.Image(label="AU trend", type="numpy")

        def on_reset() -> tuple[str, str, str, str, str, str, np.ndarray, np.ndarray, np.ndarray]:
            service.reset()
            blank = np.zeros((10, 10, 3), dtype=np.uint8)
            return (
                "**Binary risk**\n\n—",
                "**Severity (3-class)**\n\n—",
                "**Continuous score**\n\n—",
                "**Uncertainty / confidence**\n\n—",
                "**Behavioral variability (auxiliary)**\n\n—",
                "**Signal quality**\n\n—",
                blank,
                blank,
                blank,
            )

        def on_stream(frame: np.ndarray | None, audio: tuple[int, np.ndarray] | None):
            # frame: RGB uint8
            if frame is not None:
                bgr = frame[:, :, ::-1].copy()
            else:
                bgr = None

            audio_chunk = None
            if audio is not None:
                sr, data = audio
                if data is not None and data.size > 0:
                    # mono float32
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    audio_chunk = data.astype(np.float32)

            service.push(bgr, audio_chunk)
            try:
                res = service.run_once()
            except Exception as e:
                # Keep the demo resilient.
                return (
                    f"**Binary risk**\n\nError: {e}",
                    "**Severity (3-class)**\n\n—",
                    "**Continuous score**\n\n—",
                    "**Uncertainty / confidence**\n\n—",
                    "**Behavioral variability (auxiliary)**\n\n—",
                    "**Signal quality**\n\n—",
                    None,
                    None,
                    None,
                )

            risk_md = f"**Binary risk**\n\n<div class='mmds-kpi'>{res.risk_prob:.3f}</div>\n\n<span class='mmds-badge'>screening risk score</span>"
            sev_md = (
                f"**Severity (3-class)**\n\n<div class='mmds-kpi'>{_sev_label(res.severity_probs)}</div>\n\n"
                f"low={res.severity_probs[0]:.2f}, moderate={res.severity_probs[1]:.2f}, high={res.severity_probs[2]:.2f}"
            )
            cont_md = f"**Continuous score**\n\n<div class='mmds-kpi'>{res.continuous_score:.3f}</div>"
            unc_md = (
                f"**Uncertainty / confidence**\n\nuncertainty={res.uncertainty:.3f} | confidence={res.confidence:.3f}"
                + ("\n\n<span class='mmds-badge'>abstain recommended</span>" if res.abstain else "")
            )
            bdd_md = f"**Behavioral variability (auxiliary)**\n\n<div class='mmds-kpi'>{res.bdd_variability:.3f}</div>\n\n(separate from depression outputs)"
            q_md = (
                f"**Signal quality**\n\nbrightness={res.quality_summary.get('brightness', 0.0):.2f}, "
                f"blur={res.quality_summary.get('blur', 0.0):.2f}, motion={res.quality_summary.get('motion', 0.0):.2f}\n\n"
                f"warnings: {', '.join(res.quality_warnings) if res.quality_warnings else 'none'}"
            )

            return (risk_md, sev_md, cont_md, unc_md, bdd_md, q_md, res.rolling_score_img, res.importance_img, res.au_trend_img)

        webcam.stream(on_stream, inputs=[webcam, mic], outputs=[risk, sev, cont, unc, bdd, qwarn, rolling, imp, au])
        reset.click(on_reset, outputs=[risk, sev, cont, unc, bdd, qwarn, rolling, imp, au])

        def on_file_run(video_path: str | None):
            if not video_path:
                return gr.update(value="No file provided")
            # File mode is intentionally minimal in this first scaffold.
            return gr.update(value="File mode not implemented yet; use webcam mode.")

        file_run.click(on_file_run, inputs=[file_vid], outputs=[risk])

    demo.queue(max_size=32)
    demo.launch(server_name="0.0.0.0", server_port=7860)
