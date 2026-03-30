from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from .base import ExtractionResult, FeatureExtractor


@dataclass(frozen=True)
class SimpleDims:
    audio: int = 32
    face_au: int = 16
    body_pose: int = 24
    head_pose: int = 6
    gaze: int = 4
    blink: int = 2
    quality: int = 6


class SimpleExtractor(FeatureExtractor):
    """Pure-Python/OpenCV fallback backend used for the public demo path.

    It intentionally produces *proxy* behavioral indicators rather than real AUs/pose.
    Swap points:
    - For research-grade AUs: use OpenFace or a dedicated AU model.
    - For landmarks/pose: enable the MediaPipe backend.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._dims = SimpleDims()
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @property
    def modality_dims(self) -> dict[str, int]:
        return {
            "audio": self._dims.audio,
            "face_au": self._dims.face_au,
            "body_pose": self._dims.body_pose,
            "head_pose": self._dims.head_pose,
            "gaze": self._dims.gaze,
            "blink": self._dims.blink,
            "quality": self._dims.quality,
        }

    def extract_window(
        self,
        frames_bgr: list[np.ndarray],
        fps: float,
        audio_f32: np.ndarray | None,
        audio_sr: int | None,
    ) -> ExtractionResult:
        warnings: list[str] = []
        debug: dict[str, Any] = {}

        if not frames_bgr:
            return ExtractionResult({}, {}, ["no_frames"], debug)

        # Normalize frame count to a manageable sequence length.
        target_fps = float(getattr(self.cfg.features, "fps", fps) or fps)
        if fps <= 0:
            fps = target_fps
        step = max(int(round(fps / target_fps)), 1)
        frames = frames_bgr[::step]

        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        h, w = gray[0].shape[:2]

        # Quality features per frame.
        brightness = np.array([float(g.mean()) / 255.0 for g in gray], dtype=np.float32)
        blur = np.array([
            float(cv2.Laplacian(g, cv2.CV_32F).var()) / (50.0 + float(cv2.Laplacian(g, cv2.CV_32F).var()))
            for g in gray
        ], dtype=np.float32)

        motion = [0.0]
        for i in range(1, len(gray)):
            diff = cv2.absdiff(gray[i], gray[i - 1])
            motion.append(float(diff.mean()) / 255.0)
        motion_arr = np.array(motion, dtype=np.float32)

        # Attempt a crude face ROI for proxy facial activity.
        face_activity: list[np.ndarray] = []
        blink_proxy: list[np.ndarray] = []
        gaze_proxy: list[np.ndarray] = []
        head_proxy: list[np.ndarray] = []

        for g in gray:
            faces = self._face_cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                face_activity.append(np.zeros((self._dims.face_au,), dtype=np.float32))
                blink_proxy.append(np.zeros((self._dims.blink,), dtype=np.float32))
                gaze_proxy.append(np.zeros((self._dims.gaze,), dtype=np.float32))
                head_proxy.append(np.zeros((self._dims.head_pose,), dtype=np.float32))
                continue

            x, y, fw, fh = max(faces, key=lambda t: t[2] * t[3])
            roi = g[y : y + fh, x : x + fw]

            # AU-like: histogram bins + edge magnitude summaries.
            hist = cv2.calcHist([roi], [0], None, [8], [0, 256]).flatten().astype(np.float32)
            hist = hist / (hist.sum() + 1e-6)
            edges = cv2.Canny(roi, 50, 150)
            edge_mean = float(edges.mean()) / 255.0
            edge_var = float(edges.var()) / (255.0 * 255.0)
            au = np.concatenate(
                [hist, np.array([edge_mean, edge_var], dtype=np.float32)], axis=0
            )
            if au.shape[0] < self._dims.face_au:
                au = np.pad(au, (0, self._dims.face_au - au.shape[0]))
            face_activity.append(au[: self._dims.face_au])

            # Blink/gaze/head pose proxies are placeholders in simple backend.
            # They are derived from ROI aspect ratio and centroid location.
            cx = (x + fw / 2.0) / max(w, 1)
            cy = (y + fh / 2.0) / max(h, 1)
            ar = (fw / max(fh, 1))

            gaze_proxy.append(np.array([cx, cy, 1.0 - cx, 1.0 - cy], dtype=np.float32))
            blink_proxy.append(np.array([float(ar < 0.85), float(ar > 1.25)], dtype=np.float32))
            head_proxy.append(np.array([cx, cy, ar, fw / w, fh / h, 1.0], dtype=np.float32))

        face_au = np.stack(face_activity, axis=0)
        blink = np.stack(blink_proxy, axis=0)
        gaze = np.stack(gaze_proxy, axis=0)
        head_pose = np.stack(head_proxy, axis=0)

        # Body-motion proxy: global motion statistics + image moment summaries.
        body_feats: list[np.ndarray] = []
        for g in gray:
            m = cv2.moments(g)
            cx = float(m["m10"] / (m["m00"] + 1e-6)) / max(w, 1)
            cy = float(m["m01"] / (m["m00"] + 1e-6)) / max(h, 1)
            var = float(g.var()) / (255.0 * 255.0)
            body = np.concatenate(
                [
                    np.array([cx, cy, var], dtype=np.float32),
                    np.zeros((self._dims.body_pose - 3,), dtype=np.float32),
                ]
            )
            body_feats.append(body)
        body_pose = np.stack(body_feats, axis=0)

        # Quality vector packs simple signal diagnostics.
        quality = np.stack(
            [
                brightness,
                blur,
                motion_arr,
                np.clip(brightness - 0.5, -0.5, 0.5),
                np.clip(blur - 0.5, -0.5, 0.5),
                np.clip(motion_arr - 0.1, -0.1, 0.9),
            ],
            axis=1,
        ).astype(np.float32)

        if float(brightness.mean()) < 0.2:
            warnings.append("low_light")
        if float(blur.mean()) < 0.15:
            warnings.append("blurry")

        payloads: dict[str, np.ndarray] = {
            "face_au": face_au,
            "blink": blink,
            "gaze": gaze,
            "head_pose": head_pose,
            "body_pose": body_pose,
            "quality": quality,
        }
        masks: dict[str, bool] = {k: True for k in payloads.keys()}

        # Audio frontend: log-mel summary per time step.
        if audio_f32 is not None and audio_sr is not None and audio_f32.size > 0:
            try:
                audio = torch.from_numpy(audio_f32.astype(np.float32)).view(1, -1)
                mel = torch.nn.Sequential(
                    torch.nn.Identity(),
                )
                # Avoid torchaudio hard dependency behavior differences at runtime.
                import torchaudio  # type: ignore

                spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=int(audio_sr), n_fft=512, hop_length=160, n_mels=32
                )(audio)
                logmel = torch.log(spec + 1e-6).squeeze(0).transpose(0, 1)  # (T, 32)
                payloads["audio"] = logmel.detach().cpu().numpy().astype(np.float32)
                masks["audio"] = True
            except Exception as e:  # pragma: no cover
                warnings.append("audio_backend_error")
                debug["audio_error"] = repr(e)
        else:
            masks["audio"] = False

        return ExtractionResult(payloads, masks, warnings, debug)
