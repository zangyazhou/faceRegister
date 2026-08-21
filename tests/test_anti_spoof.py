# coding=utf-8
import numpy as np

from anti_spoof import _expanded_crop, _softmax


def test_softmax_returns_probabilities():
    values = _softmax([[1.0, 2.0, 3.0]])
    assert values.shape == (1, 3)
    assert np.isclose(values.sum(), 1.0)
    assert int(np.argmax(values)) == 2


def test_expanded_crop_handles_image_edges():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[0:60, 0:60] = 200
    crop = _expanded_crop(image, (0, 60, 60, 0), scale=2.7)
    assert crop.shape == (80, 80, 3)
    assert crop.dtype == np.uint8
