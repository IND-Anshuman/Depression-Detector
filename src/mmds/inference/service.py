from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from mmds.features import build_extractor
from mmds.features.extractors.base import ExtractionResult
from mmds.features.quality import compute_behavioral_variability
from mmds.features.quality.signals import summarize_quality
from mmds.inference.mc_dropout import UncertaintySummary, mc_forward, summarize_mc
from mmds.inference.viz import plot_au_trend, plot_importance, plot_rolling_score
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


class BufferedInferenceService:
    """Near-real-time buffered inference.

    The service maintains rolling buffers of frames and audio chunks and runs inference
    every `step_seconds` over the last `buffer_seconds`.
    """

    def __init__(self, cfg: DictConfig, ckpt_path: Path | None = None) -> None:
        self.cfg = cfg
        self.device = get_default_device()
        self.extractor = build_extractor(cfg)

        self.model = model_from_cfg(cfg).to(self.device)
        self.model.eval()

        if ckpt_path is not None and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=False)

        self.buffer_seconds = float(cfg.inference.buffer_seconds)
        self.step_seconds = float(cfg.inference.step_seconds)
        self.fps = float(cfg.features.fps)
        self.audio_sr = int(cfg.features.audio_sr)

        self._frame_buf: deque[np.ndarray] = deque()
        self._audio_buf: deque[np.ndarray] = deque()
        self._rolling_scores: deque[float] = deque(maxlen=50)

    def reset(self) -> None:
        self._frame_buf.clear()
        self._audio_buf.clear()
        self._rolling_scores.clear()

    def push(self, frame_bgr: np.ndarray | None, audio_chunk_f32: np.ndarray | None) -> None:
        if frame_bgr is not None:
            self._frame_buf.append(frame_bgr)
        if audio_chunk_f32 is not None and audio_chunk_f32.size > 0:
            self._audio_buf.append(audio_chunk_f32.astype(np.float32))

        max_frames = int(self.buffer_seconds * self.fps)
        while len(self._frame_buf) > max_frames:
            self._frame_buf.popleft()

        max_audio = int(self.buffer_seconds * self.audio_sr)
        # keep audio as chunks; drop oldest until within budget
        while self._audio_buf and sum(a.size for a in self._audio_buf) > max_audio:
            self._audio_buf.popleft()

    def _current_audio(self) -> np.ndarray | None:
        if not self._audio_buf:
            return None
        return np.concatenate(list(self._audio_buf), axis=0)

    def run_once(self) -> InferenceResult:
        frames = list(self._frame_buf)
        audio = self._current_audio()

        ex: ExtractionResult = self.extractor.extract_window(frames, self.fps, audio, self.audio_sr)

        # Post-hoc BDD-style variability (separate from depression outputs)
        bdd = compute_behavioral_variability(
            ex.modality_payloads.get("face_au"), ex.modality_payloads.get("body_pose")
        )

        # Build a 1-sample batch for model.
        x: dict[str, torch.Tensor] = {}
        x_mask: dict[str, torch.Tensor] = {}
        for mod, arr in ex.modality_payloads.items():
            a = np.asarray(arr)
            if a.ndim == 1:
                a = a[:, None]
            x[mod] = torch.from_numpy(a[None, :, :]).to(dtype=torch.float32, device=self.device)
            x_mask[mod] = torch.ones((1, a.shape[0]), dtype=torch.bool, device=self.device)

        passes = int(self.cfg.inference.mc_dropout_passes)
        outs = mc_forward(self.model, x, x_mask, passes=passes)
        u: UncertaintySummary = summarize_mc(outs, batch_index=0)

        abstain = bool(u.uncertainty >= float(self.cfg.inference.abstain_uncertainty_threshold))

        # Use mean scores as main outputs.
        risk = u.risk_prob_mean
        sev = u.severity_probs_mean
        cont = u.continuous_mean_mean

        self._rolling_scores.append(float(risk))

        # Interpretability: AU trend + time importance
        last = outs[-1]
        time_importance = None
        if last.token_importance is not None and last.token_time_index is not None:
            imp = last.token_importance[0].detach().cpu().numpy()  # (N,)
            tidx = last.token_time_index.detach().cpu().numpy()  # (N,)
            tmax = int(tidx.max()) + 1 if tidx.size else 0
            time_importance = np.zeros((tmax,), dtype=np.float32)
            for i in range(tidx.size):
                time_importance[int(tidx[i])] += float(imp[i])
            if time_importance.sum() > 0:
                time_importance = time_importance / time_importance.sum()

        au_img = plot_au_trend(ex.modality_payloads.get("face_au"))
        imp_img = plot_importance(time_importance)
        roll_img = plot_rolling_score(list(self._rolling_scores), title="Rolling risk score")

        q_summary = summarize_quality(ex.modality_payloads.get("quality", np.zeros((0, 3), dtype=np.float32)))

        return InferenceResult(
            risk_prob=float(risk),
            severity_probs=[float(x) for x in sev],
            continuous_score=float(cont),
            uncertainty=float(u.uncertainty),
            confidence=float(u.confidence),
            abstain=abstain,
            bdd_variability=float(bdd),
            quality_warnings=list(ex.quality_warnings),
            quality_summary=q_summary,
            au_trend_img=au_img,
            importance_img=imp_img,
            rolling_score_img=roll_img,
        )
