# coding=utf-8
import numpy as np

from opencv_face_engine import (assess_face_quality, cosine_scores,
                                find_best_cosine_match, normalize_feature)


def _unit(index):
    value = np.zeros(128, dtype=np.float32)
    value[index] = 1.0
    return value


def _detector_row(confidence=0.95):
    # box + right eye/left eye/nose/right mouth/left mouth + confidence
    return np.asarray([
        30, 20, 80, 90,
        52, 50, 88, 50, 70, 70, 56, 91, 84, 91,
        confidence,
    ], dtype=np.float32)


def test_normalize_and_cosine_match_with_margin():
    known = [_unit(0), _unit(1)]
    probe = normalize_feature(_unit(0) * 7.0)
    assert np.allclose(cosine_scores(known, probe), [1.0, 0.0])
    index, score, margin = find_best_cosine_match(
        known, probe, threshold=0.4, minimum_margin=0.08)
    assert index == 0
    assert score == 1.0
    assert margin == 1.0


def test_cosine_match_rejects_ambiguous_identity():
    first = normalize_feature(np.asarray([1.0, 0.0] + [0.0] * 126))
    second = normalize_feature(np.asarray([0.99, 0.01] + [0.0] * 126))
    index, score, margin = find_best_cosine_match(
        [first, second], first, threshold=0.4, minimum_margin=0.08)
    assert index is None
    assert score > 0.99
    assert margin < 0.08


def test_quality_score_penalizes_blur_and_darkness():
    rng = np.random.RandomState(7)
    textured = rng.randint(60, 210, size=(140, 160, 3), dtype=np.uint8)
    dark_flat = np.full((140, 160, 3), 5, dtype=np.uint8)
    row = _detector_row()

    good, good_parts = assess_face_quality(textured, row)
    poor, poor_parts = assess_face_quality(dark_flat, row)

    assert 0.0 <= poor < good <= 1.0
    assert good_parts['sharpness'] > poor_parts['sharpness']
    assert good_parts['exposure'] > poor_parts['exposure']


def test_invalid_feature_is_rejected():
    for value in ([1.0, 2.0], np.zeros(128), [float('nan')] * 128):
        try:
            normalize_feature(value)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid feature should be rejected')
