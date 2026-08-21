# coding=utf-8
# OpenCV 图像上绘制中文，解决中文乱码

import os

import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont

# 字体文件与脚本同目录，避免硬编码绝对路径导致换机器后崩溃
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simsun.ttc')


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(_FONT_PATH, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
