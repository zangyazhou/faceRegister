# Third-party model notices

## OpenCV Zoo YuNet and SFace

- Source: https://github.com/opencv/opencv_zoo
- YuNet license: MIT
- SFace license: Apache License 2.0

The runtime model files are downloaded by
`scripts/download_opencv_face_models.py` and are not committed to this repo.

## MiniFASNetV2 face anti-spoofing

- Original project: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- ONNX conversion used here: https://github.com/yakhyo/face-anti-spoofing
- License: Apache License 2.0

The ONNX file is downloaded at setup time and is not committed to this repo.

## Apache ECharts 6.1.0

- Source: https://github.com/apache/echarts
- License: Apache License 2.0

The minified browser build is vendored in `static/vendor/echarts/` so the
analytics dashboard remains available during an offline graduation defense.
