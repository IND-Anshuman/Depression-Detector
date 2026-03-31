from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mmds.features import build_extractor
from mmds.features.extractors.base import ExtractionResult
from mmds.features.quality import compute_behavioral_variability
from mmds.features.quality.signals import summarize_quality
from mmds.inference.mc_dropout import UncertaintySummary, mc_forward, summarize_mc
from mmds.inference.viz import (
    plot_attention_heatmap,
    plot_au_trend,
    plot_importance,
    plot_probability_bars,
    plot_risk_gauge,
    plot_rolling_score,
)
from mmds.models.mmds_model import model_from_cfg
from mmds.utils.device import get_default_device


@dataclass(frozen=True)
class InferenceResult:
    risk_prob: float
    severity_probs: list[float]
    continuous_score: float
    uncertainty: float
    confidence: float
    abstain: bool
    bdd_variability: float
    quality_warnings: list[str]
    quality_summary: dict[str, float]
    au_trend_img: np.ndarray
    importance_img: np.ndarray
    rolling_score_img: np.ndarray
    overlay_img: np.ndarray
    risk_gauge_img: np.ndarray
    severity_bars_img: np.ndarray
    modality_bars_img: np.ndarray
    attention_heatmap_img: np.ndarray
    fps: float
    latency_ms: float
    modality_contributions: dict[str, float]
    checkpoint_loaded: bool
    checkpoint_path: str | None
    extractor_backend: str
    trained_modalities: list[str]
    used_modalities: list[str]
    skipped_modalities: list[str]
    model_notes: list[str]
    result_age_s: float


class BufferedInferenceService:
    """Near-real-time buffered inference with async scheduling and modality alignment."""

    def __init__(self, cfg: DictConfig, ckpt_path: Path | None = None) -> None:
        self.cfg = cfg
        self.device = get_default_device()
        self.extractor = build_extractor(cfg)
        self.checkpoint_path = self._resolve_checkpoint_path(ckpt_path)
        self.checkpoint_loaded = False
        self.checkpoint_error: str | None = None
        self.runtime_error: str | None = None
        self.trained_modalities: list[str] = []

        ckpt = None
        model_cfg = cfg
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            try:
                try:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                except Exception:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                if isinstance(ckpt, dict):
                    ckpt_cfg = ckpt.get("cfg")
                    if ckpt_cfg is not None:
                        model_cfg = OmegaConf.merge(
                            ckpt_cfg,
                            {
                                "features": cfg.features,
                                "inference": cfg.inference,
                                "ui": cfg.ui,
                                "paths": cfg.paths,
                                "mode": cfg.mode,
                                "modalities": cfg.modalities,
                            },
                        )
                    state_dict = ckpt.get("model", {})
                    self.trained_modalities = self._trained_modalities_from_state_dict(state_dict)
            except Exception as exc:  # pragma: no cover
                self.checkpoint_error = repr(exc)

        self.model = model_from_cfg(model_cfg).to(self.device)
        self.model.eval()
        if ckpt is not None:
            try:
                state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                self.model.load_state_dict(state_dict, strict=False)
                self.checkpoint_loaded = True
            except Exception as exc:
                self.checkpoint_error = repr(exc)

        self.buffer_seconds = float(cfg.inference.buffer_seconds)
        self.step_seconds = float(cfg.inference.step_seconds)
        self.fps = float(cfg.features.fps)
        self.audio_sr = int(cfg.features.audio_sr)
        self.max_frame_latency_ms = float(getattr(cfg.inference, "max_frame_latency_ms", 15.0))

        self._frame_buf: deque[np.ndarray] = deque()
        self._audio_buf: deque[np.ndarray] = deque()
        self._rolling_scores: deque[float] = deque(maxlen=50)
        self._last_result: InferenceResult | None = None
        self._last_run_time = 0.0
        self._skip_budget = 0
        self._run_counter = 0

        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmds-live")
        self._future: Future[tuple[int, InferenceResult]] | None = None
        self._generation = 0

    def _resolve_checkpoint_path(self, ckpt_path: Path | None) -> Path | None:
        if ckpt_path is not None:
            return ckpt_path
        configured = getattr(self.cfg.paths, "checkpoint_path", None)
        if configured is None:
            return None
        return Path(str(configured))

    @staticmethod
    def _trained_modalities_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[str]:
        modalities = {
            key.split(".")[1]
            for key in state_dict.keys()
            if key.startswith("encoders.") and len(key.split(".")) > 2
        }
        return sorted(modalities)

    def status_markdown(self) -> str:
        status = "loaded" if self.checkpoint_loaded else "not loaded"
        ckpt_path = str(self.checkpoint_path) if self.checkpoint_path is not None else "none"
        trained = ", ".join(self.trained_modalities) if self.trained_modalities else "unknown"
        lines = [
            "### Model Status",
            f"- Checkpoint: `{status}`",
            f"- Path: `{ckpt_path}`",
            f"- Backend: `{self.cfg.features.backend}`",
            f"- Trained modalities: `{trained}`",
        ]
        if self.checkpoint_error:
            lines.append(f"- Load warning: `{self.checkpoint_error}`")
        if self.runtime_error:
            lines.append(f"- Runtime warning: `{self.runtime_error}`")
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._frame_buf.clear()
            self._audio_buf.clear()
            self._rolling_scores.clear()
            self._last_result = None
            self._last_run_time = 0.0
            self._skip_budget = 0
            self._run_counter = 0
            self._generation += 1
            self.runtime_error = None

    def push(self, frame_bgr: np.ndarray | None, audio_chunk_f32: np.ndarray | None) -> None:
        with self._lock:
            if frame_bgr is not None:
                self._frame_buf.append(frame_bgr)
            if audio_chunk_f32 is not None and audio_chunk_f32.size > 0:
                self._audio_buf.append(audio_chunk_f32.astype(np.float32))

            max_frames = int(self.buffer_seconds * self.fps)
            while len(self._frame_buf) > max_frames:
                self._frame_buf.popleft()

            max_audio = int(self.buffer_seconds * self.audio_sr)
            while self._audio_buf and sum(a.size for a in self._audio_buf) > max_audio:
                self._audio_buf.popleft()

    def _snapshot_inputs(self) -> tuple[int, list[np.ndarray], np.ndarray | None]:
        with self._lock:
            generation = self._generation
            frames = list(self._frame_buf)
            audio = np.concatenate(list(self._audio_buf), axis=0) if self._audio_buf else None
        return generation, frames, audio

    def _modality_maps(
        self,
        last_out,
        time_importance: np.ndarray | None,
    ) -> tuple[dict[str, float], np.ndarray | None, list[str]]:
        modality_contrib: dict[str, float] = {}
        modality_time = None
        modality_labels: list[str] = []
        if last_out.token_importance is None or last_out.token_modality is None or last_out.token_time_index is None:
            return modality_contrib, modality_time, modality_labels

        imp = last_out.token_importance[0].detach().cpu().numpy()
        tids = last_out.token_time_index.detach().cpu().numpy()
        mods = last_out.token_modality
        modality_labels = sorted(set(mods))
        if tids.size == 0:
            return modality_contrib, modality_time, modality_labels
        modality_time = np.zeros((len(modality_labels), int(tids.max()) + 1), dtype=np.float32)
        mod_to_idx = {m: i for i, m in enumerate(modality_labels)}
        for idx, mod in enumerate(mods):
            value = float(imp[idx]) if idx < len(imp) else 0.0
            modality_time[mod_to_idx[mod], int(tids[idx])] += value
            modality_contrib[mod] = modality_contrib.get(mod, 0.0) + value
        total = sum(modality_contrib.values()) + 1e-6
        modality_contrib = {k: float(v / total) for k, v in modality_contrib.items()}
        if time_importance is not None and modality_time.sum() > 0:
            modality_time = modality_time / (modality_time.sum(axis=1, keepdims=True) + 1e-6)
        return modality_contrib, modality_time, modality_labels

    def _build_model_inputs(
        self,
        ex: ExtractionResult,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str], list[str]]:
        x: dict[str, torch.Tensor] = {}
        x_mask: dict[str, torch.Tensor] = {}
        used_modalities: list[str] = []
        skipped_modalities: list[str] = []

        for mod, arr in ex.modality_payloads.items():
            if self.trained_modalities and mod not in self.trained_modalities:
                skipped_modalities.append(mod)
                continue
            a = np.asarray(arr)
            if a.ndim == 1:
                a = a[:, None]
            x[mod] = torch.from_numpy(a[None, :, :]).to(dtype=torch.float32, device=self.device)
            x_mask[mod] = torch.ones((1, a.shape[0]), dtype=torch.bool, device=self.device)
            used_modalities.append(mod)

        if not x:
            raise RuntimeError(
                "No live modalities matched the trained checkpoint. "
                f"Extractor emitted {sorted(ex.modality_payloads.keys())}, checkpoint expects {self.trained_modalities or ['unknown']}."
            )
        return x, x_mask, used_modalities, skipped_modalities

    @staticmethod
    def _facial_trend_source(ex: ExtractionResult) -> np.ndarray | None:
        face_au = ex.modality_payloads.get("face_au")
        if face_au is not None and face_au.size > 0:
            return face_au
        face_landmarks = ex.modality_payloads.get("face_landmarks")
        if face_landmarks is None or face_landmarks.size == 0:
            return None
        arr = np.asarray(face_landmarks, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return arr
        delta = np.diff(arr, axis=0, prepend=arr[:1])
        motion = np.linalg.norm(delta, axis=1, keepdims=True)
        return motion

    def _blank_overlay(self, frame_bgr: np.ndarray | None) -> np.ndarray:
        if frame_bgr is None:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        if frame_bgr.ndim == 3:
            return frame_bgr[:, :, ::-1].copy()
        return np.repeat(frame_bgr[:, :, None], 3, axis=2)

    def _preview_quality(self, frames: list[np.ndarray]) -> tuple[dict[str, float], list[str]]:
        if not frames:
            return {"brightness": 0.0, "blur": 0.0, "motion": 0.0}, ["warming_up"]
        latest = frames[-1]
        gray = latest if latest.ndim == 2 else latest[:, :, :3].mean(axis=2)
        brightness = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
        blur_raw = float(np.var(np.diff(gray.astype(np.float32), axis=0))) if gray.shape[0] > 1 else 0.0
        blur = float(blur_raw / (blur_raw + 100.0))
        motion = 0.0
        if len(frames) >= 2:
            prev = frames[-2]
            prev_gray = prev if prev.ndim == 2 else prev[:, :, :3].mean(axis=2)
            motion = float(np.clip(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0, 0.0, 1.0))
        warnings: list[str] = []
        if brightness < 0.2:
            warnings.append("low_light")
        if blur < 0.12:
            warnings.append("blurry")
        if motion > 0.6:
            warnings.append("high_motion")
        return {"brightness": brightness, "blur": blur, "motion": motion}, warnings

    def _placeholder_result(self, frame_bgr: np.ndarray | None, notes: list[str]) -> InferenceResult:
        _, frames, _ = self._snapshot_inputs()
        preview_quality, preview_warnings = self._preview_quality(frames)
        age = max(time.perf_counter() - self._last_run_time, 0.0) if self._last_run_time else 0.0
        if self._last_result is not None:
            merged_notes = sorted(set(self._last_result.model_notes + notes))
            return replace(
                self._last_result,
                model_notes=merged_notes,
                overlay_img=self._blank_overlay(frame_bgr),
                quality_summary=preview_quality,
                quality_warnings=preview_warnings or self._last_result.quality_warnings,
                result_age_s=age,
            )

        overlay = self._blank_overlay(frame_bgr)
        return InferenceResult(
            risk_prob=0.0,
            severity_probs=[1.0, 0.0, 0.0],
            continuous_score=0.0,
            uncertainty=1.0,
            confidence=0.0,
            abstain=True,
            bdd_variability=0.0,
            quality_warnings=preview_warnings or ["warming_up"],
            quality_summary=preview_quality,
            au_trend_img=plot_au_trend(None),
            importance_img=plot_importance(None),
            rolling_score_img=plot_rolling_score(list(self._rolling_scores), title="Rolling risk score"),
            overlay_img=overlay,
            risk_gauge_img=plot_risk_gauge(0.0),
            severity_bars_img=plot_probability_bars([1.0, 0.0, 0.0], ["low", "moderate", "high"], "3-class severity"),
            modality_bars_img=plot_probability_bars([0.0], ["pending"], "Modality contributions", color="#4ecdc4"),
            attention_heatmap_img=plot_attention_heatmap(None, []),
            fps=0.0,
            latency_ms=0.0,
            modality_contributions={},
            checkpoint_loaded=self.checkpoint_loaded,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            extractor_backend=str(self.cfg.features.backend),
            trained_modalities=list(self.trained_modalities),
            used_modalities=[],
            skipped_modalities=[],
            model_notes=notes,
            result_age_s=age,
        )

    def _compute_result(self, frames: list[np.ndarray], audio: np.ndarray | None) -> InferenceResult:
        if not frames:
            raise RuntimeError("No frames buffered yet")

        t0 = time.perf_counter()
        ex: ExtractionResult = self.extractor.extract_window(frames, self.fps, audio, self.audio_sr)
        bdd = float(
            ex.debug.get(
                "bdd_score",
                compute_behavioral_variability(ex.modality_payloads.get("face_au"), ex.modality_payloads.get("body_pose")),
            )
        )
        x, x_mask, used_modalities, skipped_modalities = self._build_model_inputs(ex)

        passes = int(self.cfg.inference.mc_dropout_passes)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                outs = mc_forward(self.model, x, x_mask, passes=passes)
        u: UncertaintySummary = summarize_mc(outs, batch_index=0)
        abstain = bool(u.uncertainty >= float(self.cfg.inference.abstain_uncertainty_threshold))
        risk = float(u.risk_prob_mean)
        sev = [float(x) for x in u.severity_probs_mean]
        cont = float(u.continuous_mean_mean)

        last = outs[-1]
        time_importance = None
        if last.token_importance is not None and last.token_time_index is not None:
            imp = last.token_importance[0].detach().cpu().numpy()
            tidx = last.token_time_index.detach().cpu().numpy()
            if tidx.size:
                time_importance = np.zeros((int(tidx.max()) + 1,), dtype=np.float32)
                for i in range(tidx.size):
                    time_importance[int(tidx[i])] += float(imp[i])
                if time_importance.sum() > 0:
                    time_importance = time_importance / time_importance.sum()

        modality_contributions, modality_time, modality_labels = self._modality_maps(last, time_importance)
        q_summary = summarize_quality(ex.modality_payloads.get("quality", np.zeros((0, 3), dtype=np.float32)))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        effective_fps = 1000.0 / max(latency_ms, 1.0)
        model_notes: list[str] = []
        if not self.checkpoint_loaded:
            model_notes.append("checkpoint_not_loaded")
        if skipped_modalities:
            model_notes.append(f"skipped={','.join(sorted(skipped_modalities))}")
        if latency_ms > self.max_frame_latency_ms:
            model_notes.append("background_refresh")

        overlay = ex.debug.get("overlay_frame", frames[-1])
        rolling_values = list(self._rolling_scores) + [risk]
        return InferenceResult(
            risk_prob=risk,
            severity_probs=sev,
            continuous_score=cont,
            uncertainty=float(u.uncertainty),
            confidence=float(u.confidence),
            abstain=abstain,
            bdd_variability=bdd,
            quality_warnings=list(ex.quality_warnings),
            quality_summary=q_summary,
            au_trend_img=plot_au_trend(self._facial_trend_source(ex)),
            importance_img=plot_importance(time_importance),
            rolling_score_img=plot_rolling_score(rolling_values, title="Rolling risk score"),
            overlay_img=overlay[:, :, ::-1].copy() if overlay.ndim == 3 else np.repeat(overlay[:, :, None], 3, axis=2),
            risk_gauge_img=plot_risk_gauge(risk),
            severity_bars_img=plot_probability_bars(sev, ["low", "moderate", "high"], "3-class severity"),
            modality_bars_img=plot_probability_bars(
                [modality_contributions[k] for k in sorted(modality_contributions.keys())] or [0.0],
                sorted(modality_contributions.keys()) or ["n/a"],
                "Modality contributions",
                color="#4ecdc4",
            ),
            attention_heatmap_img=plot_attention_heatmap(modality_time, modality_labels),
            fps=effective_fps,
            latency_ms=latency_ms,
            modality_contributions=modality_contributions,
            checkpoint_loaded=self.checkpoint_loaded,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            extractor_backend=str(ex.debug.get("source_backend", self.cfg.features.backend)),
            trained_modalities=list(self.trained_modalities),
            used_modalities=used_modalities,
            skipped_modalities=skipped_modalities,
            model_notes=model_notes,
            result_age_s=0.0,
        )

    def run_once(self) -> InferenceResult:
        _, frames, audio = self._snapshot_inputs()
        result = self._compute_result(frames, audio)
        with self._lock:
            self._last_result = result
            self._rolling_scores.append(result.risk_prob)
            self._last_run_time = time.perf_counter()
            self._skip_budget = 1 if result.latency_ms > self.max_frame_latency_ms else 0
            self._run_counter += 1
        return result

    def _consume_future(self) -> None:
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        try:
            generation, result = future.result()
            with self._lock:
                if generation != self._generation:
                    return
                self._last_result = result
                self._rolling_scores.append(result.risk_prob)
                self._last_run_time = time.perf_counter()
                self._skip_budget = 1 if result.latency_ms > self.max_frame_latency_ms else 0
                self._run_counter += 1
        except Exception as exc:
            self.runtime_error = repr(exc)

    def _schedule_async(self, now: float) -> bool:
        if self._future is not None:
            return False
        if self._last_run_time and (now - self._last_run_time) < self.step_seconds:
            return False
        generation, frames, audio = self._snapshot_inputs()
        if not frames:
            return False
        self._future = self._executor.submit(lambda: (generation, self._compute_result(frames, audio)))
        return True

    def run_live(self) -> InferenceResult:
        self._consume_future()
        now = time.perf_counter()
        _, frames, _ = self._snapshot_inputs()
        latest_frame = frames[-1] if frames else None

        if self._future is None:
            self._schedule_async(now)

        if self._future is not None:
            return self._placeholder_result(latest_frame, ["background_inference"])

        if self._last_result is not None:
            return self._last_result

        if latest_frame is None:
            raise RuntimeError("No frames buffered yet")
        return self._placeholder_result(latest_frame, ["warming_up"])
