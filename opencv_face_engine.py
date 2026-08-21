# coding=utf-8
"""OpenCV YuNet + SFace lightweight face pipeline.

The module is deliberately independent from Flask so it can be benchmarked and
unit-tested in isolation.  Images accepted by the public methods are RGB numpy
arrays, matching the existing dlib API used by this project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YUNET_MODEL = (
    PROJECT_ROOT / 'models' / 'opencv' / 'face_detection_yunet_2023mar.onnx'
)
DEFAULT_SFACE_MODEL = (
    PROJECT_ROOT / 'models' / 'opencv' / 'face_recognition_sface_2021dec.onnx'
)


class FaceModelUnavailable(RuntimeError):
    """Raised when the configured OpenCV model files cannot be loaded."""


@dataclass(frozen=True)
class FaceObservation:
    """One YuNet detection and the quality information derived from it."""

    location: tuple[int, int, int, int]  # top, right, bottom, left
    detector_row: np.ndarray
    landmarks: tuple[tuple[float, float], ...]
    confidence: float
    quality: float
    quality_parts: dict[str, float]


def normalize_feature(feature):
    """Return a finite L2-normalized 128-D face feature."""
    value = np.asarray(feature, dtype=np.float32).reshape(-1)
    if value.shape != (128,) or not np.all(np.isfinite(value)):
        raise ValueError('SFace feature must contain 128 finite values')
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError('SFace feature norm must be positive')
    return value / norm


def cosine_scores(known_features, probe):
    """Vectorized cosine similarity for already or not-yet normalized vectors."""
    if not known_features:
        return np.empty((0,), dtype=np.float32)
    known = np.asarray([normalize_feature(item) for item in known_features],
                       dtype=np.float32)
    return known @ normalize_feature(probe)


def find_best_cosine_match(known_features, probe, threshold=0.40,
                           minimum_margin=0.08):
    """Return ``(index, best_score, margin)`` using open-set dual thresholds.

    A candidate is accepted only when the best cosine score reaches ``threshold``
    and is sufficiently separated from the runner-up.  With a one-person gallery
    the margin is infinite, because no second candidate exists.
    """
    scores = cosine_scores(known_features, probe)
    if scores.size == 0:
        return None, None, None
    order = np.argsort(scores)[::-1]
    best_index = int(order[0])
    best_score = float(scores[best_index])
    margin = (float('inf') if scores.size == 1 else
              best_score - float(scores[int(order[1])]))
    if best_score < threshold or margin < minimum_margin:
        return None, best_score, margin
    return best_index, best_score, margin


def _clip_box(row, shape):
    height, width = shape[:2]
    x, y, box_width, box_height = [float(value) for value in row[:4]]
    left = max(0, min(width, int(round(x))))
    top = max(0, min(height, int(round(y))))
    right = max(left, min(width, int(round(x + box_width))))
    bottom = max(top, min(height, int(round(y + box_height))))
    return top, right, bottom, left


def assess_face_quality(image_rgb, detector_row):
    """Calculate a transparent 0..1 quality score for key-frame scheduling.

    The score uses five inexpensive and explainable signals: sharpness,
    exposure, relative face size, approximate frontal pose, and YuNet confidence.
    It is a scheduling score, not a biometric confidence score.
    """
    row = np.asarray(detector_row, dtype=np.float32).reshape(-1)
    if row.size < 15:
        raise ValueError('YuNet detection row must contain 15 values')

    top, right, bottom, left = _clip_box(row, image_rgb.shape)
    crop = image_rgb[top:bottom, left:right]
    if crop.size == 0:
        parts = {key: 0.0 for key in
                 ('sharpness', 'exposure', 'size', 'pose', 'confidence')}
        return 0.0, parts

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.clip(sharpness_raw / 160.0, 0.0, 1.0))

    mean_light = float(gray.mean())
    exposure = float(np.clip(1.0 - abs(mean_light - 127.5) / 105.0,
                             0.0, 1.0))

    image_short_side = max(1.0, float(min(image_rgb.shape[:2])))
    face_short_side = max(0.0, float(min(right - left, bottom - top)))
    size = float(np.clip(face_short_side / (image_short_side * 0.28),
                         0.0, 1.0))

    right_eye_x, left_eye_x, nose_x = float(row[4]), float(row[6]), float(row[8])
    eye_span = abs(left_eye_x - right_eye_x)
    if eye_span <= 1e-6:
        pose = 0.0
    else:
        eye_mid_x = (right_eye_x + left_eye_x) / 2.0
        normalized_offset = abs(nose_x - eye_mid_x) / eye_span
        pose = float(np.clip(1.0 - normalized_offset / 0.38, 0.0, 1.0))

    confidence = float(np.clip((float(row[14]) - 0.60) / 0.40,
                               0.0, 1.0))
    parts = {
        'sharpness': sharpness,
        'exposure': exposure,
        'size': size,
        'pose': pose,
        'confidence': confidence,
    }
    quality = (0.30 * sharpness + 0.20 * exposure + 0.20 * size +
               0.15 * pose + 0.15 * confidence)
    return round(float(np.clip(quality, 0.0, 1.0)), 4), parts


class OpenCVFaceEngine:
    """Thread-safe wrapper around stateful YuNet and SFace OpenCV objects."""

    def __init__(self, detector_model=DEFAULT_YUNET_MODEL,
                 recognizer_model=DEFAULT_SFACE_MODEL,
                 detector_threshold=0.75, nms_threshold=0.30,
                 top_k=5000, detection_max_side=640):
        detector_model = Path(detector_model)
        recognizer_model = Path(recognizer_model)
        missing = [str(path) for path in (detector_model, recognizer_model)
                   if not path.is_file()]
        if missing:
            raise FaceModelUnavailable(
                'Missing OpenCV face model(s): ' + ', '.join(missing))
        try:
            self._detector = cv2.FaceDetectorYN_create(
                str(detector_model), '', (320, 320), detector_threshold,
                nms_threshold, top_k)
            self._recognizer = cv2.FaceRecognizerSF_create(
                str(recognizer_model), '')
        except cv2.error as exc:
            raise FaceModelUnavailable(
                'Unable to load OpenCV face models: %s' % exc) from exc
        self._lock = threading.RLock()
        self._detection_max_side = max(160, int(detection_max_side))

    def detect(self, image_rgb):
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('image must be an RGB HxWx3 array')
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scale = min(1.0, self._detection_max_side / float(max(height, width)))
        if scale < 1.0:
            detect_bgr = cv2.resize(
                image_bgr, (max(2, round(width * scale)),
                            max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA)
        else:
            detect_bgr = image_bgr
        detect_height, detect_width = detect_bgr.shape[:2]
        with self._lock:
            self._detector.setInputSize((detect_width, detect_height))
            _, faces = self._detector.detect(detect_bgr)
        if faces is None:
            return []

        observations = []
        for row in faces:
            row = np.asarray(row, dtype=np.float32).copy()
            if scale < 1.0:
                # x/y/w/h and five landmark pairs occupy columns 0..13.
                row[:14] /= scale
            quality, parts = assess_face_quality(image, row)
            landmarks = tuple((float(row[index]), float(row[index + 1]))
                              for index in range(4, 14, 2))
            observations.append(FaceObservation(
                location=_clip_box(row, image.shape),
                detector_row=row,
                landmarks=landmarks,
                confidence=float(row[14]),
                quality=quality,
                quality_parts=parts,
            ))
        return observations

    def encode(self, image_rgb, observation):
        image = np.asarray(image_rgb)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        row = (observation.detector_row if isinstance(observation, FaceObservation)
               else np.asarray(observation, dtype=np.float32))
        with self._lock:
            aligned = self._recognizer.alignCrop(image_bgr, row)
            feature = self._recognizer.feature(aligned)
        return normalize_feature(feature)

    def detect_and_encode(self, image_rgb, minimum_quality=0.0):
        """Return observations plus features; low-quality faces get ``None``."""
        result = []
        for observation in self.detect(image_rgb):
            feature = (self.encode(image_rgb, observation)
                       if observation.quality >= minimum_quality else None)
            result.append((observation, feature))
        return result


_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_opencv_face_engine():
    """Lazily initialize the relatively expensive OpenCV DNN objects once."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = OpenCVFaceEngine()
    return _ENGINE
