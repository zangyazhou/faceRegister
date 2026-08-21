__author__ = '孟汉君'
__email__ = 'menghanjun001@gmail.com'

import os

_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _model_location(filename):
    """返回 models 目录下指定模型文件的绝对路径"""
    return os.path.join(_MODELS_DIR, filename)


def pose_predictor_model_location():
    return _model_location('shape_predictor_68_face_landmarks.dat')


def pose_predictor_five_point_model_location():
    return _model_location("shape_predictor_5_face_landmarks.dat")


def face_recognition_model_location():
    return _model_location("dlib_face_recognition_resnet_model_v1.dat")


def cnn_face_detector_model_location():
    return _model_location("mmod_human_face_detector.dat")
