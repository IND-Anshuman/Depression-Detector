from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mmds.features.compact_audio import build_compact_audio_features
from mmds.features.compact_visual import build_compact_visual_modalities

from .base import ExtractionResult, FeatureExtractor


DVLOG_FACE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 195, 5, 4, 1, 19, 94, 2,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 81, 13, 311, 308, 402, 14, 178,
]


def _quality_vector(gray: np.ndarray, prev_gray: np.ndarray | None, face_found: bool) -> np.ndarray:
    brightness = float(gray.mean()) / 255.0
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_norm = blur / (blur + 50.0)
    motion = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean()) / 255.0
    return np.array([brightness, blur_norm, motion, float(face_found)], dtype=np.float32)


def _downsample_ts(array: np.ndarray, max_steps: int) -> np.ndarray:
    if array.ndim != 2 or max_steps <= 0 or array.shape[0] <= max_steps:
        return array
    idx = np.linspace(0, array.shape[0] - 1, num=max_steps, dtype=np.int64)
    return array[idx]


class MediaPipeDVlogExtractor(FeatureExtractor):
    """Compact live extractor aligned to the DVlog importer schema.

    Outputs:
    - `face_landmarks`: 68 selected MediaPipe points with x/y only -> 136 dims
    - `audio`: MFCC-like 25-d timestep features
    - `quality`: lightweight signal diagnostics for the dashboard only
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._drawing = mp.solutions.drawing_utils
            self._styles = mp.solutions.drawing_styles
            self._available = True
        except Exception:
            self._mp = None
            self._mesh = None
            self._drawing = None
            self._styles = None
            self._available = False

    @property
    def modality_dims(self) -> dict[str, int]:
        return {
            "audio": 25,
            "face_landmarks": 136,
            "face_au": 16,
            "head_pose": 6,
            "gaze": 4,
            "blink": 2,
            "behavioral_stats": 3,
            "quality": 4,
        }

    def _extract_audio(
        self,
        audio_f32: np.ndarray | None,
        audio_sr: int | None,
        max_steps: int,
    ) -> tuple[np.ndarray | None, str | None]:
        if audio_f32 is None or audio_sr is None or audio_f32.size == 0:
            return None, None
        try:
            import torch
            import torchaudio  # type: ignore

            audio = torch.from_numpy(audio_f32.astype(np.float32)).view(1, -1)
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=int(audio_sr),
                n_mfcc=25,
                melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40},
            )(audio)
            features = mfcc.squeeze(0).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
            features = _downsample_ts(features, max_steps=max_steps)
            return build_compact_audio_features(features), None
        except Exception as exc:  # pragma: no cover
            return None, repr(exc)

    def extract_window(
        self,
        frames_bgr: list[np.ndarray],
        fps: float,
        audio_f32: np.ndarray | None,
        audio_sr: int | None,
    ) -> ExtractionResult:
        if not frames_bgr:
            return ExtractionResult({}, {}, ["no_frames"], {})

        stream_fps = float(getattr(self.cfg.features, "fps", fps) or fps or 5.0)
        target_hz = float(getattr(self.cfg.features, "dvlog_target_hz", 1.0))
        if fps <= 0:
            fps = stream_fps
        step = max(int(round(stream_fps / max(target_hz, 1.0e-6))), 1)
        frames = frames_bgr[::step]

        face_l: list[np.ndarray] = []
        quality_l: list[np.ndarray] = []
        warnings: list[str] = []
        debug: dict[str, Any] = {}
        prev_gray = None
        overlay_bgr = frames[-1].copy()

        for idx, frame_bgr in enumerate(frames):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            face_found = False
            face_vec = np.zeros((136,), dtype=np.float32)

            if self._available:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = self._mesh.process(frame_rgb)
                multi = getattr(result, "multi_face_landmarks", None)
                if multi:
                    face_found = True
                    lm = multi[0].landmark
                    pts = np.array([[float(lm[i].x), float(lm[i].y)] for i in DVLOG_FACE_INDICES], dtype=np.float32)
                    face_vec = pts.reshape(-1)
                    if idx == len(frames) - 1:
                        self._drawing.draw_landmarks(
                            overlay_bgr,
                            multi[0],
                            self._mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self._styles.get_default_face_mesh_contours_style(),
                        )

            face_l.append(face_vec)
            quality_l.append(_quality_vector(gray, prev_gray, face_found))
            prev_gray = gray

        face_landmarks = np.stack(face_l, axis=0).astype(np.float32)
        visual_payloads = build_compact_visual_modalities(
            face_landmarks,
            confidence=np.stack(quality_l, axis=0)[:, 0],
            success=np.stack(quality_l, axis=0)[:, 3],
        )
        audio_features, audio_error = self._extract_audio(audio_f32, audio_sr, max_steps=max(len(frames), 1))
        if audio_error is not None:
            warnings.append("audio_backend_error")
            debug["audio_error"] = audio_error

        quality = np.stack(quality_l, axis=0).astype(np.float32)
        if float(quality[:, 0].mean()) < 0.2:
            warnings.append("low_light")
        if float(quality[:, 1].mean()) < 0.15:
            warnings.append("blurry")
        if float(quality[:, 3].mean()) < 0.2:
            warnings.append("face_not_detected")

        payloads: dict[str, np.ndarray] = dict(visual_payloads)
        masks = {k: True for k in payloads.keys()}
        if audio_features is not None:
            payloads["audio"] = audio_features
            masks["audio"] = True
        else:
            masks["audio"] = False

        debug["overlay_frame"] = overlay_bgr
        debug["source_backend"] = "mediapipe_dvlog"
        debug["dvlog_face_indices"] = DVLOG_FACE_INDICES
        return ExtractionResult(payloads, masks, warnings, debug)
