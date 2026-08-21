# coding=utf-8
"""
摄像头实时人脸识别签到演示

用法：
    python test.py                # 打开摄像头窗口，识别到学生自动写入签到记录
    python test.py --no-display   # 无显示器环境（如服务器）：只识别与落库，不弹窗

说明：
    - 同一名学生 60 秒内只写入一次签到，且数据库层面保证同日仅一条记录
    - 按 q 键退出
"""
import argparse
import sys
import time

import cv2
from flask import current_app

import api
from app import app, User
from attendance import record_sign_in
from font.fontToImg import cv2ImgAddText
from functions import parse_face_encoding

# 同一学生两次签到之间的最短间隔（秒），避免连续帧重复写入
SIGN_IN_INTERVAL = 60


def main():
    parser = argparse.ArgumentParser(description='摄像头人脸识别签到')
    parser.add_argument('--no-display', action='store_true',
                        help='无显示器模式：只识别并写入签到记录，不弹出视频窗口')
    args = parser.parse_args()

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print('无法打开摄像头（设备 0），请检查摄像头连接后重试')
        sys.exit(1)

    with app.app_context():
        # 从数据库加载已录入人脸特征的学生
        known_face_names = []
        known_user_ids = []
        known_face_encodings = []
        for u in User.query.filter(User.face_encoding.isnot(None)).all():
            encoding = parse_face_encoding(u.face_encoding)
            if encoding is not None:
                known_face_names.append(u.username)
                known_user_ids.append(u.id)
                known_face_encodings.append(encoding)
        print('已加载 %d 名学生的人脸特征：%s' % (len(known_face_names), known_face_names))

        # 内存节流：以用户 id 为键记录最近一次尝试签到的时间
        # （不能以姓名为键，否则同名学生会互相干扰节流判定）
        last_sign_in = {}

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        face_sign_status = []
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print('读取摄像头画面失败')
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which api uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = api.face_locations(rgb_small_frame)
                face_encodings = api.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                face_sign_status = []
                now = time.time()
                for face_encoding in face_encodings:
                    # 与 Web 签到同一阈值语义：找最近匹配（最佳距离 + FACE_MATCH_TOLERANCE）
                    idx, dist = api.find_best_match(
                        known_face_encodings, face_encoding,
                        current_app.config['FACE_MATCH_TOLERANCE'])
                    name = '无匹配'
                    status = None

                    if idx is not None:
                        name = known_face_names[idx]
                        # 节流：间隔内不重复写库
                        if now - last_sign_in.get(known_user_ids[idx], 0) > SIGN_IN_INTERVAL:
                            record = record_sign_in(known_user_ids[idx])
                            last_sign_in[known_user_ids[idx]] = now
                            status = '已签到' if record else '今日已签到'
                            if args.no_display:
                                print('%s %s（%s）' % (time.strftime('%H:%M:%S'),
                                                      status, name))
                        else:
                            status = '已签到'

                    face_names.append(name)
                    face_sign_status.append(status)

            process_this_frame = not process_this_frame

            if args.no_display:
                # 无显示器模式：不渲染画面，直接进入下一帧
                continue

            # Display the results
            for (top, right, bottom, left), name, status in \
                    zip(face_locations, face_names, face_sign_status):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
                frame = cv2ImgAddText(frame, name, left + 60, bottom - 30, (255, 0, 0), 30)
                if status:
                    color = (0, 200, 0) if status == '已签到' else (0, 165, 255)
                    frame = cv2ImgAddText(frame, status, left + 60, bottom + 10,
                                          color, 24)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
