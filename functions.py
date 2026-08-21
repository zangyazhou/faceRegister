import ast
import json

import numpy as np

import api
from opencv_face_engine import FaceModelUnavailable, get_opencv_face_engine


def parse_face_encoding(encoding_str):
    """
    将数据库中存储的人脸编码字符串安全地还原为 numpy 数组。
    :param encoding_str: 形如 "[0.123, 0.456, ...]" 的字符串，或 None
    :return: numpy 数组；解析失败或为空时返回 None
    """
    if not encoding_str:
        return None
    try:
        values = ast.literal_eval(encoding_str)
        arr = np.asarray(values, dtype=np.float64)
        # 人脸编码应为 128 维向量
        if arr.ndim != 1 or arr.shape[0] != 128:
            return None
        return arr
    except (ValueError, SyntaxError, TypeError):
        return None


def get_face_encoding_in_image(file_stream):
    # 载入上传的图片
    img = api.load_image_file(file_stream)
    # 获取图片中所有人脸的特征编码
    unknown_face_encodings = api.face_encodings(img)
    face_count = len(unknown_face_encodings)
    face_encoding = None

    if face_count > 0:
        # 只取第一张人脸；以 JSON 数组格式存储，便于安全解析
        # （多脸照片由调用方拒绝，避免绑定到错误的人脸）
        face_encoding = str(list(unknown_face_encodings[0]))

    # 同一张注册照同时生成 SFace 特征。迁移期间保留 dlib 特征，旧数据和
    # 桌面端仍可使用；Web 端在 SFace 特征存在时自动走高速流水线。
    sface_encoding = None
    sface_face_count = 0
    try:
        observations = get_opencv_face_engine().detect(img)
        sface_face_count = len(observations)
        if sface_face_count == 1:
            feature = get_opencv_face_engine().encode(img, observations[0])
            sface_encoding = json.dumps(feature.tolist(), separators=(',', ':'))
    except FaceModelUnavailable:
        # 模型尚未下载时允许旧 dlib 流程继续工作，启动页会显示回退状态。
        pass

    result = {
        "face_found_in_image": face_count > 0,
        "face_count": face_count,
        "unknown_face_encodings": face_encoding,
        "sface_face_count": sface_face_count,
        "sface_encoding": sface_encoding,
    }
    return result
