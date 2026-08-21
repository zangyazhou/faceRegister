# coding=utf-8
"""读取可复现实验产物，为算法实验中心提供统一指标口径。"""
from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
PERFORMANCE_REPORT = ROOT / 'docs' / 'performance_baseline.md'
SFACE_REPORT = ROOT / 'evaluation_report' / 'sface_metrics.json'


def load_experiment_dashboard():
    performance = _load_performance_baseline()
    identity = _load_sface_report()
    return {
        'performance': performance,
        'identity': identity,
        'evidence_matrix': [
            {
                'name': '推理速度对比', 'status': 'completed',
                'source': '45 个图片样本 × 两条流水线',
                'result': 'YuNet + SFace 中位耗时降低 %.1f%%' %
                          ((1 - performance['opencv']['median'] /
                            performance['dlib']['median']) * 100),
            },
            {
                'name': 'SFace 开放集阈值', 'status': 'limited',
                'source': '%d 正样本 + %d 陌生人负样本' % (
                    identity['dataset']['positive_samples'],
                    identity['dataset']['negative_samples']),
                'result': '阈值 %.2f 下 FAR %.1f%% / FRR %.1f%%' % (
                    identity['threshold'], identity['metrics']['far'] * 100,
                    identity['metrics']['frr'] * 100),
            },
            {
                'name': '多人并发时延', 'status': 'pending',
                'source': '仍需 2 / 4 / 6 人视频或多路回放',
                'result': '系统功能已支持，尚缺正式实验数据',
            },
            {
                'name': '活体攻击评估', 'status': 'pending',
                'source': '仍需打印照片、屏幕重放与真人样本',
                'result': '具备风险审计，尚不能报告 ACER',
            },
            {
                'name': '时序特征融合消融', 'status': 'pending',
                'source': '仍需同一视频关闭/开启融合对照',
                'result': '避免使用无来源的模拟提升百分比',
            },
        ],
        'sources': [
            {'name': '性能基线', 'path': 'docs/performance_baseline.md',
             'updated_at': _modified_time(PERFORMANCE_REPORT)},
            {'name': 'SFace 准确性结果',
             'path': 'evaluation_report/sface_metrics.json',
             'updated_at': _modified_time(SFACE_REPORT)},
            {'name': '评估图片集', 'path': 'data/evaluation/',
             'updated_at': _modified_time(ROOT / 'data' / 'evaluation')},
        ],
    }


def _load_performance_baseline():
    text = PERFORMANCE_REPORT.read_text(encoding='utf-8')
    row_pattern = re.compile(
        r'^\|\s*(dlib[^|]*|YuNet[^|]*)\s*\|\s*([\d.]+)\s*ms\s*\|'
        r'\s*([\d.]+)\s*ms\s*\|\s*([\d.]+)\s*ms\s*\|\s*(\d+)\s*\|$',
        re.MULTILINE | re.IGNORECASE)
    rows = {}
    for name, median, p95, mean, faces in row_pattern.findall(text):
        key = 'dlib' if name.lower().startswith('dlib') else 'opencv'
        rows[key] = {
            'name': name.strip(), 'median': float(median),
            'p95': float(p95), 'mean': float(mean), 'faces': int(faces),
        }
    if 'dlib' not in rows or 'opencv' not in rows:
        raise ValueError('性能基线表格格式无法识别')
    speedup_match = re.search(r'中位耗时加速为 \*\*([\d.]+) 倍\*\*', text)
    sample_match = re.search(r'共 (\d+) 个样本', text)
    return {
        **rows,
        'speedup': (float(speedup_match.group(1)) if speedup_match else
                    round(rows['dlib']['median'] / rows['opencv']['median'], 2)),
        'samples': int(sample_match.group(1)) if sample_match else 0,
        'source': 'docs/performance_baseline.md',
        'environment': {
            'cpu': _bullet_value(text, 'CPU'),
            'system': _bullet_value(text, '系统'),
            'python': _bullet_value(text, 'Python'),
            'opencv': _bullet_value(text, 'OpenCV'),
        },
    }


def _load_sface_report():
    payload = json.loads(SFACE_REPORT.read_text(encoding='utf-8'))
    genuine = payload['scores']['genuine']
    impostor = payload['scores']['impostor']
    payload['score_summary'] = {
        'genuine_min': min(genuine) if genuine else None,
        'genuine_mean': sum(genuine) / len(genuine) if genuine else None,
        'genuine_max': max(genuine) if genuine else None,
        'impostor_min': min(impostor) if impostor else None,
        'impostor_mean': sum(impostor) / len(impostor) if impostor else None,
        'impostor_max': max(impostor) if impostor else None,
        'separation': ((min(genuine) - max(impostor))
                       if genuine and impostor else None),
    }
    return payload


def _bullet_value(text, label):
    match = re.search(r'^-\s*%s：(.+)$' % re.escape(label), text, re.MULTILINE)
    return match.group(1).strip() if match else '未记录'


def _modified_time(path):
    if not path.exists():
        return None
    from datetime import datetime
    return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
