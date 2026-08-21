import PIL

import dlib
import numpy as np
from PIL import Image, ImageOps

import faceRecognitonModels

face_detector = dlib.get_frontal_face_detector()
# <dlib.fhog_object_detector object at 0x7fafd8f955e0>

predictor_68_point_model = faceRecognitonModels.pose_predictor_model_location()
predictor_5_point_model = faceRecognitonModels.pose_predictor_five_point_model_location()
cnn_face_detection_model = faceRecognitonModels.cnn_face_detector_model_location()
face_recognition_model = faceRecognitonModels.face_recognition_model_location()

pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def load_image_file(file, mode='RGB'):
    """
    将图像文件（.jpg，.png等）加载到numpy数组中
    :param file:要加载的图像文件名或文件对象
    :param mode:将图像转换为的格式。仅支持“RGB”（8位RGB，3个通道）和“L”（黑色和白色）。
    :return:图像内容为numpy数组
    """
    im = Image.open(file)
    # 修正 EXIF 方向：手机等设备拍摄的竖拍照片常带 EXIF 旋转标记，
    # 若不处理，人脸会被 90° 翻转，导致关键点/编码错乱
    im = ImageOps.exif_transpose(im)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _css_to_rect(css):
    """
    将（顶部，右侧，底部，左侧）顺序中的元组转换为dlib“rect”对象
    :param css:rect（top，right，bottom，left）顺序的简单元组表示
    :return:一个dlib`rect`对象
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _rect_to_css(rect):
    """
    将dlib'rect'对象转换为（顶部，右侧，底部，左侧）顺序的普通元组
    :param rect:一个dlib'rect'对象
    :return:以（顶部，右侧，底部，左侧）顺序显示矩形的简单元组表示
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _trim_css_to_bounds(css, img_shape):
    return max(css[0], 0), min(css[1], img_shape[1]), min(css[2], img_shape[0]), max(css[3], 0)


def _raw_face_locations(img, number_of_times_to_sample=1, model='hog'):
    """
    返回图像中人脸边界框的数组
    :param img:一个图像（作为一个numpy数组）
    :param number_of_times_to_sample:对图像进行上采样以查找面部的次数。数字越大，面部越小。
    :param model:使用哪种人脸检测模型。“hog”不太准确，但在CPU上更快。“cnn”更准确
                  深度学习模型，GPU / CUDA加速（如果可用）。默认为“hog”。
    :return:找到的面部位置的dlib'rect'对象列表
    """
    if model == 'cnn':
        return cnn_face_detector(img, number_of_times_to_sample)
    else:
        return face_detector(img, number_of_times_to_sample)
    # rectangles[[(52, 66) (114, 129)]]


def face_locations(img, number_of_times_to_sample=1, model='hog'):
    if model == 'cnn':
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in
                _raw_face_locations(img, number_of_times_to_sample, 'cnn')]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in
                _raw_face_locations(img, number_of_times_to_sample, model)]


def _raw_face_landmarks(face_image, face_locations=None, model='large'):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    pose_predictor = pose_predictor_68_point
    if model == 'small':
        pose_predictor = pose_predictor_5_point
    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None, model='large'):
    landmarks = _raw_face_landmarks(face_image, face_locations, model)
    # dlib-bin 的 pybind11 版本只提供 part(index)，不提供无参数 part()。
    return [[(landmark.part(i).x, landmark.part(i).y)
             for i in range(landmark.num_parts)]
            for landmark in landmarks]


def face_encodings_with_landmarks(face_image, known_face_locations=None, num=1):
    """一次关键点定位同时返回人脸编码与 68 点坐标。

    Web 签到既需要 128 维编码做身份匹配，也需要眼睛/鼻子关键点做
    服务端活体检测。两者共用同一份 dlib full_object_detection，避免
    对同一帧重复运行 shape predictor。

    :return: ``(encodings, landmarks)``；两者顺序与人脸框顺序一致。
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model='large')
    encodings = [
        np.array(face_encoder.compute_face_descriptor(face_image, shape, num))
        for shape in raw_landmarks
    ]
    landmarks = [
        [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
        for shape in raw_landmarks
    ]
    return encodings, landmarks


def face_encodings(face_image, known_face_locations=None, num=1):
    """

    :param face_image: 输入包含脸的图像
    :param known_face_locations: 已认识的脸部位置
    :param num: 计算编码时重新采样的次数，越大越准确也越慢
    """
    encodings, _ = face_encodings_with_landmarks(
        face_image, known_face_locations=known_face_locations, num=num)
    return encodings


def face_distance(face_encoding, face_to_compare):
    """
    给定面部编码列表，将它们与已知的面部编码进行比较并获得欧几里德距离
    对于每个比较面。距离告诉您脸部的相似程度。
    :param face_encoding:要比较的面部编码列表
    :param face_to_compare:要与之进行比较的面部编码
    :return:一个numpy ndarray，每个面的距离与'faces'数组的顺序相同 
    """
    if len(face_encoding) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encoding - face_to_compare, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
     将面部编码列表与候选编码进行比较，看它们是否匹配。
    :param known_face_encodings:已知面部编码的列表
    :param face_encoding_to_check:单个面部编码，用于与列表进行比较
    :param tolerance:将面之间的距离视为匹配。越低越严格。0.6是典型的最佳性能
                     （face_recognition 官方默认值），历史数据与新版本 dlib
                     比对建议使用 0.6 而非 0.5。
    :return:True / False值的列表，指示哪个known_face_encodings与要检查的面部编码匹配
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def find_best_match(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    返回库中最接近的人脸下标与其欧氏距离。
    :param known_face_encodings: 已知人脸编码列表
    :param face_encoding_to_check: 待匹配的单个编码
    :param tolerance: 匹配距离上限；最近距离超过它则视为「未注册」
    :return: (best_index, best_distance)；best_index 为 None 表示没有匹配
             （库为空或最近距离超阈值）；known 为空时 best_distance 也为 None
    """
    if len(known_face_encodings) == 0:
        return None, None
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    best = int(np.argmin(distances))
    dist = float(distances[best])
    if dist <= tolerance:
        return best, dist
    return None, dist
