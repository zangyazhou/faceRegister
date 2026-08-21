# coding=utf-8
"""
find_best_match 匹配逻辑单测：纯 numpy 构造，不依赖 app/数据库。
与 test_web_signin.py 配合覆盖「识别出对的人、其他人不误签」的核心。
"""
import math

import numpy as np

import api


def _unit_vec(seed):
    """128 维单位随机向量（归一化去掉尺度干扰，距离只反映方向差异）。"""
    rng = np.random.RandomState(seed)
    v = rng.randn(128)
    return v / np.linalg.norm(v)


def test_exact_match_returns_zero_distance():
    """同一向量应匹配自身且距离为 0"""
    v = _unit_vec(1)
    idx, dist = api.find_best_match([v], v, tolerance=0.55)
    assert idx == 0
    assert dist == 0.0


def test_best_index_picks_nearest():
    """应返回库中距离最近的（第二个）"""
    a, b, probe = _unit_vec(2), _unit_vec(3), _unit_vec(4)
    idx, _ = api.find_best_match([a, b], probe, tolerance=10.0)
    # 距 probe 近的应该是 a 或 b 中距离更小的那个
    d_a = float(np.linalg.norm(a - probe))
    d_b = float(np.linalg.norm(b - probe))
    assert idx == (0 if d_a < d_b else 1)


def test_all_far_returns_none_with_distance():
    """全部超阈值 → (None, dist)，dist 仍返回最近距离供调试"""
    v = _unit_vec(1)
    far = _unit_vec(99)  # 与 v 距离约 sqrt(2)≈1.41，远超 0.55
    idx, dist = api.find_best_match([v], far, tolerance=0.55)
    assert idx is None
    assert dist is not None and dist > 0.55


def test_empty_library_returns_none_none():
    """空库 → (None, None)"""
    v = _unit_vec(1)
    assert api.find_best_match([], v, tolerance=0.55) == (None, None)


def test_tolerance_boundary_is_inclusive():
    """距离恰好等于阈值应视为匹配（<= 语义）"""
    # 构造 probe 与 known 距离恰为 0.6
    known = np.zeros(128)
    probe = np.full(128, 0.6 / math.sqrt(128))
    idx, dist = api.find_best_match([known], probe, tolerance=0.6)
    assert idx == 0
    assert abs(dist - 0.6) < 1e-9
    # 阈值再收紧一点则拒绝
    idx2, _ = api.find_best_match([known], probe, tolerance=0.6 - 1e-6)
    assert idx2 is None


def test_compare_faces_regression():
    """find_best_match 与历史 compare_faces 语义一致"""
    v, probe, far = _unit_vec(1), _unit_vec(2), _unit_vec(98)
    # 匹配时 compare_faces 至少一个 True
    idx, _ = api.find_best_match([v, probe], v, tolerance=0.55)
    assert idx is not None
    assert True in api.compare_faces([v, probe], v, tolerance=0.55)
    # 未匹配时全部 False
    idx2, _ = api.find_best_match([v], far, tolerance=0.55)
    assert idx2 is None
    assert not any(api.compare_faces([v], far, tolerance=0.55))


def test_multi_face_library_distinguishes():
    """库内两人（距离~1.41）互不误认：各自的 probe 只命中自己"""
    a, b = _unit_vec(1), _unit_vec(2)
    idx_a, _ = api.find_best_match([a, b], a, tolerance=0.55)
    idx_b, _ = api.find_best_match([a, b], b, tolerance=0.55)
    assert idx_a == 0
    assert idx_b == 1
