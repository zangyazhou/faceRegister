# coding=utf-8
"""Download the two official OpenCV Zoo face models used by the project."""
from pathlib import Path
import hashlib
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'opencv'
MODELS = {
    'face_detection_yunet_2023mar.onnx': (
        'https://github.com/opencv/opencv_zoo/raw/main/models/'
        'face_detection_yunet/face_detection_yunet_2023mar.onnx',
        '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    ),
    'face_recognition_sface_2021dec.onnx': (
        'https://github.com/opencv/opencv_zoo/raw/main/models/'
        'face_recognition_sface/face_recognition_sface_2021dec.onnx',
        '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
    ),
}

ANTI_SPOOF_MODELS = {
    'minifasnet_v2.onnx': (
        'https://github.com/yakhyo/face-anti-spoofing/releases/download/'
        'weights/MiniFASNetV2.onnx',
        'b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907',
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_hash) in MODELS.items():
        destination = MODEL_DIR / filename
        if destination.exists() and sha256(destination) == expected_hash:
            print('[ok] %s' % filename)
            continue
        print('[download] %s' % filename)
        urllib.request.urlretrieve(url, destination)
        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            raise RuntimeError('%s checksum mismatch: %s' %
                               (filename, actual_hash))
        print('[ok] %s' % filename)
    anti_spoof_dir = ROOT / 'models' / 'antispoof'
    anti_spoof_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_hash) in ANTI_SPOOF_MODELS.items():
        destination = anti_spoof_dir / filename
        if destination.exists() and sha256(destination) == expected_hash:
            print('[ok] %s' % filename)
            continue
        print('[download] %s' % filename)
        urllib.request.urlretrieve(url, destination)
        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            raise RuntimeError('%s checksum mismatch: %s' %
                               (filename, actual_hash))
        print('[ok] %s' % filename)


if __name__ == '__main__':
    main()
