# coding=utf-8
"""在本项目评估集上生成 YuNet + SFace 同源准确性证据。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import load_image_file  # noqa: E402
from opencv_face_engine import get_opencv_face_engine  # noqa: E402


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def image_paths(folder):
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def feature_for(engine, path):
    image = load_image_file(str(path))
    observations = engine.detect(image)
    if len(observations) != 1:
        return None, '检测到 %d 张人脸' % len(observations)
    return engine.encode(image, observations[0]), None


def rates(genuine, impostor, threshold):
    far = (sum(score >= threshold for score in impostor) / len(impostor)
           if impostor else 0.0)
    frr = (sum(score < threshold for score in genuine) / len(genuine)
           if genuine else 0.0)
    return far, frr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'data' / 'evaluation')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'evaluation_report' / 'sface_metrics.json')
    parser.add_argument('--threshold', type=float, default=0.40)
    args = parser.parse_args()

    engine = get_opencv_face_engine()
    enrolled_dir = args.data_dir / 'enrolled'
    impostor_dir = args.data_dir / 'impostor'
    gallery = []
    positives = []
    skipped = []

    for person_index, person_dir in enumerate(sorted(
            path for path in enrolled_dir.iterdir() if path.is_dir()), start=1):
        images = image_paths(person_dir)
        if not images:
            continue
        feature, reason = feature_for(engine, images[0])
        if feature is None:
            skipped.append({'sample': 'enrolled_%03d_gallery' % person_index,
                            'reason': reason})
            continue
        gallery.append((person_dir.name, feature))
        for probe_index, path in enumerate(images[1:], start=1):
            probe, reason = feature_for(engine, path)
            if probe is None:
                skipped.append({'sample': 'enrolled_%03d_probe_%03d' %
                                (person_index, probe_index),
                                'reason': reason})
            else:
                positives.append((person_dir.name, probe))

    negatives = []
    if impostor_dir.exists():
        impostor_index = 0
        for path in sorted(impostor_dir.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            impostor_index += 1
            feature, reason = feature_for(engine, path)
            if feature is None:
                skipped.append({'sample': 'impostor_%03d' % impostor_index,
                                'reason': reason})
            else:
                negatives.append(feature)

    if not gallery:
        raise SystemExit('没有可用的 SFace 注册模板')

    labels = [label for label, _ in gallery]
    gallery_features = np.asarray([feature for _, feature in gallery])
    genuine, impostor = [], []
    rank1_correct = 0
    for label, probe in positives:
        scores = gallery_features @ probe
        predicted = int(np.argmax(scores))
        if labels[predicted] == label:
            rank1_correct += 1
        genuine.append(float(scores[labels.index(label)]))
    for probe in negatives:
        impostor.append(float(np.max(gallery_features @ probe)))

    far, frr = rates(genuine, impostor, args.threshold)
    thresholds = np.linspace(0.10, 0.80, 71)
    curve = []
    for threshold in thresholds:
        curve_far, curve_frr = rates(genuine, impostor, float(threshold))
        curve.append({'threshold': round(float(threshold), 3),
                      'far': round(curve_far, 6),
                      'frr': round(curve_frr, 6),
                      'tpr': round(1.0 - curve_frr, 6)})
    eer_point = min(curve, key=lambda item: abs(item['far'] - item['frr']))

    payload = {
        'schema_version': 1,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'engine': 'YuNet + SFace',
        'score_type': 'cosine_similarity',
        'threshold': args.threshold,
        'dataset': {
            'registered_identities': len(gallery),
            'positive_samples': len(positives),
            'negative_samples': len(negatives),
            'skipped_samples': len(skipped),
            # 报告只保留匿名编号，避免泄露注册目录名或真实身份。
            'labels': ['identity_%03d' % index
                       for index in range(1, len(labels) + 1)],
        },
        'metrics': {
            'rank1': round(rank1_correct / len(positives), 6) if positives else 0,
            'far': round(far, 6),
            'frr': round(frr, 6),
            'tpr': round(1.0 - frr, 6),
            'eer_estimate': round((eer_point['far'] + eer_point['frr']) / 2, 6),
            'eer_threshold': eer_point['threshold'],
        },
        'scores': {
            'genuine': [round(value, 6) for value in genuine],
            'impostor': [round(value, 6) for value in impostor],
        },
        'curve': curve,
        'skipped': skipped,
        'limitations': [
            '当前仅包含 1 个注册身份，Rank-1 不能代表多人身份分类能力。',
            '正样本仅 2 张、负样本仅 11 张，错误率只能作为功能性基线。',
            '该数据集不包含视频重放、打印照片或三维面具，不能评估活体 ACER。',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'registered': len(gallery),
        'positives': len(positives),
        'negatives': len(negatives),
        'far': far,
        'frr': frr,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
