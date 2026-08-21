# 人脸流水线性能基线（2026-08-20）

## 测试环境

- CPU：Intel Core i7-12650H
- 系统：Windows 11（build 26200）
- Python：3.11.15
- OpenCV：4.10.0
- 输入：`lfw-deepfunneled` 中 15 张图片，最长边缩放至 640px
- 重复：每张图片 3 次，共 45 个样本

运行命令：

```powershell
python benchmarks/benchmark_face_pipeline.py `
  --images lfw-deepfunneled --runs 3 --max-side 640
```

## 结果

| 流水线 | 中位耗时 | P95 | 平均耗时 | 检出人脸数 |
|---|---:|---:|---:|---:|
| dlib HOG + dlib ResNet | 286.61 ms | 390.25 ms | 299.18 ms | 42 |
| YuNet + SFace | 16.65 ms | 32.85 ms | 19.97 ms | 45 |

YuNet + SFace 的中位耗时加速为 **17.22 倍**。

## Web 端到端补充测试

使用一张 480×640、约 74KB 的 JPEG，通过 Flask 测试客户端连续请求：

- 首帧（模型冷启动）：约 174–202ms；
- 预热后完整请求：约 42–47ms；
- MiniFASNetV2 静默活体：约 1.4–1.6ms/脸；
- 高置信真人：累计 2 帧后无感通过。

这些数值是当前电脑上的工程基线，不替代后续多人视频、照片攻击和不同光照下的正式实验。
