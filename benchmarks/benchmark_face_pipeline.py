# coding=utf-8
"""Repeatable local latency comparison for dlib and YuNet/SFace.

Usage:
    python benchmarks/benchmark_face_pipeline.py --images lfw-deepfunneled --runs 5
"""
from pathlib import Path
import argparse
import statistics
import sys
import time

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api  # noqa: E402
from opencv_face_engine import get_opencv_face_engine  # noqa: E402


def load_rgb(path, max_side):
    image = api.load_image_file(path)
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def percentile(values, probability):
    ordered = sorted(values)
    index = min(len(ordered) - 1,
                max(0, round((len(ordered) - 1) * probability)))
    return ordered[index]


def summarize(name, timings, detected):
    print('\n%s' % name)
    print('  samples: %d, detected faces: %d' % (len(timings), detected))
    print('  median: %.2f ms' % statistics.median(timings))
    print('  p95:    %.2f ms' % percentile(timings, 0.95))
    print('  mean:   %.2f ms' % statistics.mean(timings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=str(ROOT / 'lfw-deepfunneled'))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--max-side', type=int, default=640)
    args = parser.parse_args()

    paths = sorted(path for path in Path(args.images).rglob('*')
                   if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise SystemExit('No images found: %s' % args.images)
    images = [load_rgb(path, args.max_side) for path in paths]
    engine = get_opencv_face_engine()

    # Warm up OpenCV DNN lazy initialization outside the measured section.
    engine.detect_and_encode(images[0])

    dlib_times, opencv_times = [], []
    dlib_faces = opencv_faces = 0
    for _ in range(args.runs):
        for image in images:
            started = time.perf_counter()
            locations = api.face_locations(image)
            api.face_encodings(image, known_face_locations=locations)
            dlib_times.append((time.perf_counter() - started) * 1000.0)
            dlib_faces += len(locations)

            started = time.perf_counter()
            observations = engine.detect_and_encode(image)
            opencv_times.append((time.perf_counter() - started) * 1000.0)
            opencv_faces += len(observations)

    summarize('dlib HOG + ResNet', dlib_times, dlib_faces)
    summarize('YuNet + SFace', opencv_times, opencv_faces)
    speedup = statistics.median(dlib_times) / statistics.median(opencv_times)
    print('\nmedian speed-up: %.2fx' % speedup)


if __name__ == '__main__':
    main()
