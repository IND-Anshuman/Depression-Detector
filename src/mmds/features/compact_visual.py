from __future__ import annotations

import numpy as np

from mmds.features.quality.entropy_bdd import action_entropy, bdd_score, expression_entropy


LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
LEFT_BROW = [17, 18, 19, 20, 21]
RIGHT_BROW = [22, 23, 24, 25, 26]
MOUTH = list(range(48, 68))


def _safe_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(points: np.ndarray, idxs: list[int]) -> float:
    p = points[idxs]
    vert = _safe_distance(p[1], p[5]) + _safe_distance(p[2], p[4])
    horiz = 2.0 * _safe_distance(p[0], p[3]) + 1.0e-6
    return float(vert / horiz)


def normalize_landmarks_xy(landmarks_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks_xy, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 136:
        raise ValueError(f"Expected landmark shape (T,136), got {arr.shape}")

    out = np.zeros_like(arr, dtype=np.float32)
    for idx in range(arr.shape[0]):
        pts = arr[idx].reshape(68, 2)
        if np.allclose(pts, 0.0):
            continue
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        scale = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1.0))
        norm = (pts - center) / scale
        out[idx] = np.clip(norm.reshape(-1), -1.0, 1.0)
    return out


def derive_face_au_from_landmarks(landmarks_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks_xy, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    feats: list[np.ndarray] = []
    for frame in arr:
        pts = frame.reshape(68, 2)
        if np.allclose(pts, 0.0):
            feats.append(np.zeros((16,), dtype=np.float32))
            continue
        left_ear = _eye_aspect_ratio(pts, LEFT_EYE)
        right_ear = _eye_aspect_ratio(pts, RIGHT_EYE)
        mouth_open = _safe_distance(pts[62], pts[66])
        mouth_width = _safe_distance(pts[48], pts[54])
        brow_left = float(np.mean([_safe_distance(pts[i], pts[37]) for i in LEFT_BROW]))
        brow_right = float(np.mean([_safe_distance(pts[i], pts[44]) for i in RIGHT_BROW]))
        jaw_width = _safe_distance(pts[0], pts[16])
        nose_height = _safe_distance(pts[27], pts[33])
        lip_curve = float((pts[54, 1] - pts[48, 1]) - (pts[57, 1] - pts[51, 1]))
        face_height = _safe_distance(pts[8], pts[27])
        x_mean = float(pts[:, 0].mean())
        y_mean = float(pts[:, 1].mean())
        x_std = float(pts[:, 0].std())
        y_std = float(pts[:, 1].std())
        feats.append(
            np.array(
                [
                    left_ear,
                    right_ear,
                    mouth_open,
                    mouth_width,
                    brow_left,
                    brow_right,
                    jaw_width,
                    nose_height,
                    lip_curve,
                    face_height,
                    x_mean,
                    y_mean,
                    x_std,
                    y_std,
                    brow_left - brow_right,
                    mouth_open / (mouth_width + 1.0e-6),
                ],
                dtype=np.float32,
            )
        )
    return np.stack(feats, axis=0).astype(np.float32)


def derive_head_pose_from_landmarks(landmarks_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks_xy, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    feats: list[np.ndarray] = []
    for frame in arr:
        pts = frame.reshape(68, 2)
        if np.allclose(pts, 0.0):
            feats.append(np.zeros((6,), dtype=np.float32))
            continue
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        nose = pts[30]
        mouth = pts[48:68].mean(axis=0)
        eye_mid = (left_eye + right_eye) / 2.0
        yaw = float(nose[0] - eye_mid[0])
        pitch = float(mouth[1] - eye_mid[1])
        roll = float(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0] + 1.0e-6))
        face_width = _safe_distance(pts[0], pts[16])
        face_height = _safe_distance(pts[8], pts[27])
        feats.append(np.array([yaw, pitch, roll, eye_mid[0], eye_mid[1], face_width / (face_height + 1.0e-6)], dtype=np.float32))
    return np.stack(feats, axis=0).astype(np.float32)


def derive_gaze_from_landmarks(landmarks_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks_xy, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    feats: list[np.ndarray] = []
    for frame in arr:
        pts = frame.reshape(68, 2)
        if np.allclose(pts, 0.0):
            feats.append(np.zeros((4,), dtype=np.float32))
            continue
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        nose = pts[30]
        feats.append(np.array([left_eye[0] - nose[0], left_eye[1] - nose[1], right_eye[0] - nose[0], right_eye[1] - nose[1]], dtype=np.float32))
    return np.stack(feats, axis=0).astype(np.float32)


def derive_blink_from_landmarks(landmarks_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks_xy, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    feats: list[np.ndarray] = []
    for frame in arr:
        pts = frame.reshape(68, 2)
        if np.allclose(pts, 0.0):
            feats.append(np.zeros((2,), dtype=np.float32))
            continue
        feats.append(np.array([_eye_aspect_ratio(pts, LEFT_EYE), _eye_aspect_ratio(pts, RIGHT_EYE)], dtype=np.float32))
    return np.stack(feats, axis=0).astype(np.float32)


def derive_behavioral_stats(face_au_ts: np.ndarray, head_pose_ts: np.ndarray) -> np.ndarray:
    expr = float(expression_entropy(face_au_ts))
    act = float(action_entropy(head_pose_ts))
    bdd = float(bdd_score(face_au_ts, head_pose_ts))
    t = face_au_ts.shape[0] if face_au_ts.ndim == 2 else 1
    return np.tile(np.array([[expr, act, bdd]], dtype=np.float32), (t, 1))


def build_compact_visual_modalities(
    landmarks_xy: np.ndarray,
    confidence: np.ndarray | None = None,
    success: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    face_landmarks = normalize_landmarks_xy(landmarks_xy)
    face_au = derive_face_au_from_landmarks(face_landmarks)
    head_pose = derive_head_pose_from_landmarks(face_landmarks)
    gaze = derive_gaze_from_landmarks(face_landmarks)
    blink = derive_blink_from_landmarks(face_landmarks)
    behavioral_stats = derive_behavioral_stats(face_au, head_pose)

    t = face_landmarks.shape[0]
    conf = np.ones((t,), dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32).reshape(-1)[:t]
    succ = np.ones((t,), dtype=np.float32) if success is None else np.asarray(success, dtype=np.float32).reshape(-1)[:t]
    if conf.shape[0] < t:
        conf = np.pad(conf, (0, t - conf.shape[0]), constant_values=float(conf[-1]) if conf.size else 1.0)
    if succ.shape[0] < t:
        succ = np.pad(succ, (0, t - succ.shape[0]), constant_values=float(succ[-1]) if succ.size else 1.0)
    motion = np.linalg.norm(np.diff(face_landmarks, axis=0, prepend=face_landmarks[:1]), axis=1)
    quality = np.stack([conf, succ, motion, np.linalg.norm(face_landmarks.reshape(t, 68, 2), axis=2).mean(axis=1)], axis=1).astype(np.float32)

    return {
        "face_landmarks": face_landmarks.astype(np.float32),
        "face_au": face_au.astype(np.float32),
        "head_pose": head_pose.astype(np.float32),
        "gaze": gaze.astype(np.float32),
        "blink": blink.astype(np.float32),
        "behavioral_stats": behavioral_stats.astype(np.float32),
        "quality": quality.astype(np.float32),
    }
