# coding=utf-8
"""CPU-only passive RGB anti-spoofing with MiniFASNetV2 ONNX.

The checkpoint is an Apache-2.0 ONNX conversion of Minivision's
Silent-Face-Anti-Spoofing MiniFASNetV2. Runtime inference uses OpenCV DNN, so
the deployed Flask application does not require PyTorch or ONNX Runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / 'models' / 'antispoof' / 'minifasnet_v2.onnx'


class AntiSpoofModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AntiSpoofResult:
    live_probability: float
    attack_probability: float
    predicted_class: int
    inference_ms: float


def _softmax(logits):
    value = np.asarray(logits, dtype=np.float32).reshape(1, -1)
    shifted = value - np.max(value, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _expanded_crop(image, location, scale=2.7, output_size=(80, 80)):
    """Crop with the scale and boundary-shift behavior used by Silent-Face."""
    top, right, bottom, left = [int(value) for value in location]
    src_height, src_width = image.shape[:2]
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    scale = min((src_height - 1) / box_height,
                (src_width - 1) / box_width, float(scale))
    new_width = box_width * scale
    new_height = box_height * scale
    center_x = left + box_width / 2.0
    center_y = top + box_height / 2.0

    x1, y1 = center_x - new_width / 2.0, center_y - new_height / 2.0
    x2, y2 = center_x + new_width / 2.0, center_y + new_height / 2.0
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > src_width - 1:
        x1 -= x2 - src_width + 1
        x2 = src_width - 1
    if y2 > src_height - 1:
        y1 -= y2 - src_height + 1
        y2 = src_height - 1

    crop = image[max(0, int(y1)):min(src_height, int(y2) + 1),
                 max(0, int(x1)):min(src_width, int(x2) + 1)]
    if crop.size == 0:
        raise ValueError('empty face crop')
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)


class PassiveAntiSpoofEngine:
    """Thread-safe MiniFASNetV2 inference using OpenCV's ONNX importer."""

    def __init__(self, model_path=DEFAULT_MODEL, crop_scale=2.7):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise AntiSpoofModelUnavailable(
                'Missing anti-spoof model: %s' % model_path)
        try:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
        except cv2.error as exc:
            raise AntiSpoofModelUnavailable(
                'Unable to load anti-spoof model: %s' % exc) from exc
        self._crop_scale = float(crop_scale)
        self._lock = threading.Lock()

    def predict(self, image_rgb, location):
        # The converted checkpoint's reference inference uses raw 0..255 BGR
        # CHW float input without mean/std normalization.
        image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
        crop = _expanded_crop(image_bgr, location, self._crop_scale, (80, 80))
        blob = np.transpose(crop.astype(np.float32), (2, 0, 1))[None, ...]
        started = time.perf_counter()
        with self._lock:
            self._net.setInput(blob)
            logits = self._net.forward()
        inference_ms = (time.perf_counter() - started) * 1000.0
        probabilities = _softmax(logits)[0]
        if probabilities.shape[0] != 3:
            raise ValueError('MiniFASNet output must contain three classes')
        # Original Silent-Face convention: class 1 is bona fide; 0 and 2 are
        # different presentation-attack categories.
        live_probability = float(probabilities[1])
        return AntiSpoofResult(
            live_probability=live_probability,
            attack_probability=float(probabilities[0] + probabilities[2]),
            predicted_class=int(np.argmax(probabilities)),
            inference_ms=round(inference_ms, 3),
        )


_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_anti_spoof_engine():
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = PassiveAntiSpoofEngine()
    return _ENGINE
