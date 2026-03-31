from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mmds.features.emonet import CleanRoomEmoNetExtractor
from mmds.features.quality.entropy_bdd import action_entropy, bdd_score, expression_entropy

from .base import ExtractionResult, FeatureExtractor
from .simple_backend import SimpleExtractor


FACE_COUNT = 468
POSE_COUNT = 33
HAND_COUNT = 21


def _flatten_landmarks(landmarks: Any, count: int, include_visibility: bool = False) -> np.ndarray:
    dim = 4 if include_visibility else 3
    out = np.zeros((count, dim), dtype=np.float32)
    if landmarks is None:
        return out.reshape(-1)
    for idx, lm in enumerate(getattr(landmarks, "landmark", [])[:count]):
        out[idx, 0] = float(lm.x)
        out[idx, 1] = float(lm.y)
        out[idx, 2] = float(lm.z)
        if include_visibility:
            out[idx, 3] = float(getattr(lm, "visibility", 0.0))
    return out.reshape(-1)


def _face_crop(frame_rgb: np.ndarray, face_flat: np.ndarray) -> np.ndarray:
    if face_flat.size == 0 or np.allclose(face_flat, 0.0):
        return frame_rgb
    pts = face_flat.reshape(-1, 3)[:, :2]
    x1 = int(np.clip(np.min(pts[:, 0]) * frame_rgb.shape[1], 0, frame_rgb.shape[1] - 1))
    y1 = int(np.clip(np.min(pts[:, 1]) * frame_rgb.shape[0], 0, frame_rgb.shape[0] - 1))
    x2 = int(np.clip(np.max(pts[:, 0]) * frame_rgb.shape[1], x1 + 1, frame_rgb.shape[1]))
    y2 = int(np.clip(np.max(pts[:, 1]) * frame_rgb.shape[0], y1 + 1, frame_rgb.shape[0]))
    crop = frame_rgb[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame_rgb


def _eye_aspect(face_pts: np.ndarray, upper: int, lower: int, left: int, right: int) -> float:
    vert = float(np.linalg.norm(face_pts[upper, :2] - face_pts[lower, :2]))
    horiz = float(np.linalg.norm(face_pts[left, :2] - face_pts[right, :2])) + 1e-6
    return vert / horiz


def _derive_face_au(face_flat: np.ndarray) -> np.ndarray:
    if face_flat.size == 0 or np.allclose(face_flat, 0.0):
        return np.zeros((16,), dtype=np.float32)
    pts = face_flat.reshape(-1, 3)
    feats = np.array(
        [
            np.linalg.norm(pts[13, :2] - pts[14, :2]),
            np.linalg.norm(pts[61, :2] - pts[291, :2]),
            np.linalg.norm(pts[70, :2] - pts[105, :2]),
            np.linalg.norm(pts[336, :2] - pts[334, :2]),
            np.linalg.norm(pts[159, :2] - pts[145, :2]),
            np.linalg.norm(pts[386, :2] - pts[374, :2]),
            np.linalg.norm(pts[78, :2] - pts[308, :2]),
            np.linalg.norm(pts[1, :2] - pts[9, :2]),
        ],
        dtype=np.float32,
    )
    derived = np.concatenate([feats, feats - feats.mean(), np.array([pts[:, 2].mean(), pts[:, 2].std()], dtype=np.float32)], axis=0)
    if derived.shape[0] < 16:
        derived = np.pad(derived, (0, 16 - derived.shape[0]))
    return derived[:16].astype(np.float32)


def _derive_head_pose(pose_flat: np.ndarray) -> np.ndarray:
    if pose_flat.size == 0 or np.allclose(pose_flat, 0.0):
        return np.zeros((6,), dtype=np.float32)
    pts = pose_flat.reshape(-1, 4)
    nose = pts[0]
    left_eye = pts[2]
    right_eye = pts[5]
    yaw = float(right_eye[0] - left_eye[0])
    pitch = float(nose[1] - (left_eye[1] + right_eye[1]) / 2.0)
    roll = float(right_eye[1] - left_eye[1])
    return np.array([yaw, pitch, roll, nose[0], nose[1], nose[2]], dtype=np.float32)


class MediaPipeFullExtractor(FeatureExtractor):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._fallback = SimpleExtractor(cfg)
        emonet_cfg = getattr(cfg.features, "emonet", {})
        self._emonet = CleanRoomEmoNetExtractor(
            weights_path=getattr(emonet_cfg, "weights_path", None),
            embedding_dim=int(getattr(emonet_cfg, "embedding_dim", 10)),
            strict_weights=bool(getattr(emonet_cfg, "strict_weights", False)),
        )
        try:
            import mediapipe as mp  # type: ignore

            hcfg = getattr(cfg.features, "holistic", {})
            self._mp = mp
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=int(getattr(hcfg, "model_complexity", 1)),
                smooth_landmarks=bool(getattr(hcfg, "smooth_landmarks", True)),
                min_detection_confidence=float(getattr(hcfg, "min_detection_confidence", 0.5)),
                min_tracking_confidence=float(getattr(hcfg, "min_tracking_confidence", 0.5)),
            )
            self._drawing = mp.solutions.drawing_utils
            self._drawing_styles = mp.solutions.drawing_styles
            self._available = True
        except Exception:
            self._mp = None
            self._holistic = None
            self._drawing = None
            self._drawing_styles = None
            self._available = False

    @property
    def modality_dims(self) -> dict[str, int]:
        return {
            "audio": 32,
            "face_au": 16,
            "face_landmarks": FACE_COUNT * 3,
            "body_pose": POSE_COUNT * 4,
            "hand_pose": HAND_COUNT * 3 * 2,
            "head_pose": 6,
            "gaze": 4,
            "blink": 2,
            "emotion": 10,
            "behavioral_stats": 3,
            "quality": 8,
        }

    def _expand_simple(self, res: ExtractionResult) -> ExtractionResult:
        payloads = dict(res.modality_payloads)
        t = next((v.shape[0] for v in payloads.values() if v.ndim == 2), 1)
        payloads.setdefault("face_landmarks", np.zeros((t, FACE_COUNT * 3), dtype=np.float32))
        payloads.setdefault("hand_pose", np.zeros((t, HAND_COUNT * 3 * 2), dtype=np.float32))
        payloads.setdefault("emotion", np.zeros((t, 10), dtype=np.float32))
        payloads.setdefault("behavioral_stats", np.zeros((t, 3), dtype=np.float32))
        if "body_pose" in payloads and payloads["body_pose"].shape[1] != POSE_COUNT * 4:
            current = payloads["body_pose"]
            padded = np.zeros((current.shape[0], POSE_COUNT * 4), dtype=np.float32)
            padded[:, : current.shape[1]] = current
            payloads["body_pose"] = padded
        masks = {k: True for k in payloads.keys()}
        masks["audio"] = res.modality_masks.get("audio", False)
        return ExtractionResult(payloads, masks, ["mediapipe_full_not_installed"] + res.quality_warnings, res.debug)

    def extract_window(self, frames_bgr, fps, audio_f32, audio_sr) -> ExtractionResult:
        if not self._available:
            return self._expand_simple(self._fallback.extract_window(frames_bgr, fps, audio_f32, audio_sr))
        if not frames_bgr:
            return ExtractionResult({}, {}, ["no_frames"], {})

        face_au_l: list[np.ndarray] = []
        face_landmarks_l: list[np.ndarray] = []
        body_pose_l: list[np.ndarray] = []
        hand_pose_l: list[np.ndarray] = []
        head_pose_l: list[np.ndarray] = []
        gaze_l: list[np.ndarray] = []
        blink_l: list[np.ndarray] = []
        emotion_l: list[np.ndarray] = []
        quality_l: list[np.ndarray] = []

        prev_gray = None
        overlay_bgr = frames_bgr[-1].copy()
        for idx, frame_bgr in enumerate(frames_bgr):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._holistic.process(frame_rgb)

            face_flat = _flatten_landmarks(getattr(result, "face_landmarks", None), FACE_COUNT)
            pose_flat = _flatten_landmarks(getattr(result, "pose_landmarks", None), POSE_COUNT, include_visibility=True)
            left_hand = _flatten_landmarks(getattr(result, "left_hand_landmarks", None), HAND_COUNT)
            right_hand = _flatten_landmarks(getattr(result, "right_hand_landmarks", None), HAND_COUNT)
            hand_flat = np.concatenate([left_hand, right_hand], axis=0).astype(np.float32)
            face_landmarks_l.append(face_flat.astype(np.float32))
            body_pose_l.append(pose_flat.astype(np.float32))
            hand_pose_l.append(hand_flat)
            face_au_l.append(_derive_face_au(face_flat))
            head_pose_l.append(_derive_head_pose(pose_flat))

            if face_flat.size > 0 and not np.allclose(face_flat, 0.0):
                pts = face_flat.reshape(-1, 3)
                left_ear = _eye_aspect(pts, 159, 145, 33, 133)
                right_ear = _eye_aspect(pts, 386, 374, 362, 263)
                blink_l.append(np.array([left_ear, right_ear], dtype=np.float32))
                left_eye_center = pts[[33, 133, 159, 145], :2].mean(axis=0)
                right_eye_center = pts[[362, 263, 386, 374], :2].mean(axis=0)
                gaze_vec = np.concatenate([left_eye_center, right_eye_center], axis=0).astype(np.float32)
                gaze_l.append(gaze_vec)
                emotion_l.append(self._emonet.extract(_face_crop(frame_rgb, face_flat), face_landmarks=face_flat))
            else:
                blink_l.append(np.zeros((2,), dtype=np.float32))
                gaze_l.append(np.zeros((4,), dtype=np.float32))
                emotion_l.append(np.zeros((10,), dtype=np.float32))

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean()) / 255.0
            blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            blur_norm = blur / (blur + 50.0)
            motion = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean()) / 255.0
            prev_gray = gray
            quality_l.append(
                np.array(
                    [
                        brightness,
                        blur_norm,
                        motion,
                        float(not np.allclose(face_flat, 0.0)),
                        float(not np.allclose(pose_flat, 0.0)),
                        float(not np.allclose(hand_flat, 0.0)),
                        float(np.count_nonzero(gray) / max(gray.size, 1)),
                        float(idx / max(len(frames_bgr) - 1, 1)),
                    ],
                    dtype=np.float32,
                )
            )

            if idx == len(frames_bgr) - 1:
                if getattr(result, "face_landmarks", None) is not None:
                    self._drawing.draw_landmarks(
                        overlay_bgr,
                        result.face_landmarks,
                        self._mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self._drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                if getattr(result, "pose_landmarks", None) is not None:
                    self._drawing.draw_landmarks(
                        overlay_bgr,
                        result.pose_landmarks,
                        self._mp.solutions.holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
                    )
                for hand in [getattr(result, "left_hand_landmarks", None), getattr(result, "right_hand_landmarks", None)]:
                    if hand is not None:
                        self._drawing.draw_landmarks(
                            overlay_bgr,
                            hand,
                            self._mp.solutions.holistic.HAND_CONNECTIONS,
                            self._drawing_styles.get_default_hand_landmarks_style(),
                            self._drawing_styles.get_default_hand_connections_style(),
                        )

        payloads = {
            "face_au": np.stack(face_au_l, axis=0).astype(np.float32),
            "face_landmarks": np.stack(face_landmarks_l, axis=0).astype(np.float32),
            "body_pose": np.stack(body_pose_l, axis=0).astype(np.float32),
            "hand_pose": np.stack(hand_pose_l, axis=0).astype(np.float32),
            "head_pose": np.stack(head_pose_l, axis=0).astype(np.float32),
            "gaze": np.stack(gaze_l, axis=0).astype(np.float32),
            "blink": np.stack(blink_l, axis=0).astype(np.float32),
            "emotion": np.stack(emotion_l, axis=0).astype(np.float32),
            "quality": np.stack(quality_l, axis=0).astype(np.float32),
        }
        expr_ent = expression_entropy(payloads["face_au"]) if bool(getattr(self.cfg.features, "include_entropy", False)) else 0.0
        action_ent = action_entropy(payloads["body_pose"]) if bool(getattr(self.cfg.features, "include_entropy", False)) else 0.0
        bdd = bdd_score(payloads["face_au"], payloads["body_pose"]) if bool(getattr(self.cfg.features, "include_entropy", False)) else 0.0
        payloads["behavioral_stats"] = np.tile(
            np.array([[expr_ent, action_ent, bdd]], dtype=np.float32), (payloads["quality"].shape[0], 1)
        )
        if audio_f32 is not None and audio_sr is not None and audio_f32.size > 0:
            audio_res = self._fallback.extract_window([frames_bgr[0]], fps, audio_f32, audio_sr)
            if "audio" in audio_res.modality_payloads:
                payloads["audio"] = audio_res.modality_payloads["audio"]

        masks = {k: True for k in payloads.keys()}
        if "audio" not in payloads:
            masks["audio"] = False

        warnings: list[str] = []
        if float(payloads["quality"][:, 0].mean()) < 0.2:
            warnings.append("low_light")
        if float(payloads["quality"][:, 1].mean()) < 0.15:
            warnings.append("blurry")
        if float(payloads["quality"][:, 3].mean()) < 0.2:
            warnings.append("face_not_detected")

        debug = {
            "source_backend": "mediapipe_full",
            "expression_entropy": expr_ent,
            "action_entropy": action_ent,
            "bdd_score": bdd,
            "overlay_frame": overlay_bgr,
        }
        return ExtractionResult(payloads, masks, warnings, debug)
