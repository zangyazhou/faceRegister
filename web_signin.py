# coding=utf-8
"""
Web 端人脸识别签到 API 核心逻辑：输入校验、最佳匹配、多帧确认、落库。

支持两种输入（共用同一套比对/投票/落库核心）：
  1. POST /api/sign-in        —— 前端算好的 128 维描述子（旧模式，face-api.js）
  2. POST /api/sign-in-image  —— 前端摄像头截图，后端 dlib 识别（新模式，性能好）

数据流（新模式）：
    前端摄像头定时截图 -> POST /api/sign-in-image (JPEG base64)
    -> 后端 dlib 一次提取 68 点关键点与 128 维描述子
    -> 空间位置关联同一张脸的跨帧轨迹
    -> 每条轨迹独立完成身份多帧确认 + 眨眼/转头活体 -> 落库

人脸编码永不下载到浏览器；识别判定唯一权威在后端 dlib（C++ 层，快）。
"""
import base64
import binascii
import io
import math
import secrets
import time

import numpy as np
from flask import current_app, request, session

from anti_spoof import (AntiSpoofModelUnavailable,
                        get_anti_spoof_engine)
from api import (find_best_match, face_encodings_with_landmarks,
                 face_landmarks, face_locations, load_image_file)
from app import User
from attendance_platform import record_recognition_decision
from attendance import record_sign_in
from functions import parse_face_encoding
from opencv_face_engine import (FaceModelUnavailable, find_best_cosine_match,
                                get_opencv_face_engine, normalize_feature)

# 128 维描述子的 JSON 约 1.5~2KB，64KB 足够宽松；截图 base64 较大，单独放宽。
MAX_JSON_BODY = 65536
MAX_IMAGE_BODY = 8 * 1024 * 1024  # 截图 base64 较大，放宽到 8MB（前端 canvas 压缩后通常仅几十 KB）


def _json(status, code, message, **extra):
    payload = {'status': status, 'code': code, 'message': message}
    payload.update(extra)
    return payload


def _load_registered(field='face_encoding'):
    """返回 [(user, np.ndarray 128维), ...]，跳过 face_encoding 为空或解析失败的学生。"""
    pairs = []
    column = getattr(User, field)
    for u in User.query.filter(column.isnot(None)).all():
        enc = parse_face_encoding(getattr(u, field))
        if enc is not None:
            pairs.append((u, enc))
    return pairs


def _bump_vote(user_id, votes_required, window):
    """
    会话级多帧确认：同一 user_id 在窗口（秒）内累计 votes_required 票才通过。

    票存在签名 session cookie（重启不丢、跨 worker 一致）；键是 user_id，
    换一个人识别到别的用户时自动重置 → kiosk 共用一台浏览器也正确。
    :return: (confirmed, votes)
    """
    now = time.time()
    st = session.get('votes')
    if st and st.get('user_id') == user_id and now - st.get('t', 0) <= window:
        st['count'] += 1
    else:
        st = {'user_id': user_id, 'count': 1}
    st['t'] = now
    session['votes'] = st
    if st['count'] >= votes_required:
        session.pop('votes', None)
        return True, st['count']
    return False, st['count']


def _box(loc):
    """css 坐标 (top,right,bottom,left) -> {x,y,w,h}"""
    top, right, bottom, left = loc
    return {'x': left, 'y': top, 'w': right - left, 'h': bottom - top}


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]),
                      float(a[1]) - float(b[1]))


def _eye_aspect_ratio(points, indices):
    """计算一只眼睛的 EAR（Eye Aspect Ratio）。"""
    p1, p2, p3, p4, p5, p6 = [points[i] for i in indices]
    width = _distance(p1, p4)
    if width <= 1e-6:
        return None
    return (_distance(p2, p6) + _distance(p3, p5)) / (2.0 * width)


def _liveness_metrics(landmarks):
    """从 68 点关键点计算双眼 EAR 与归一化水平头部姿态。"""
    if not landmarks or len(landmarks) < 68:
        return None, None

    left_ear = _eye_aspect_ratio(landmarks, (36, 37, 38, 39, 40, 41))
    right_ear = _eye_aspect_ratio(landmarks, (42, 43, 44, 45, 46, 47))
    if left_ear is None or right_ear is None:
        return None, None
    ear = (left_ear + right_ear) / 2.0

    left_eye_x = sum(landmarks[i][0] for i in range(36, 42)) / 6.0
    right_eye_x = sum(landmarks[i][0] for i in range(42, 48)) / 6.0
    eye_span = abs(right_eye_x - left_eye_x)
    if eye_span <= 1e-6:
        return ear, None
    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    # 正脸约为 0；左右转头时鼻尖相对双眼中点产生水平位移。
    yaw = (float(landmarks[30][0]) - eye_mid_x) / eye_span
    return ear, yaw


def _fresh_liveness(now):
    policy = current_app.config.get('LIVENESS_POLICY', 'passive_adaptive')
    challenge = None
    if policy == 'single_action':
        challenge = secrets.choice(('blink', 'head_turn'))
    return {
        'started_at': now,
        'open_seen': False,
        'closed_seen': False,
        'blink': False,
        'yaw_min': None,
        'yaw_max': None,
        'head_turn': False,
        'passed': False,
        'challenge': challenge,
        'mode': 'evaluating' if policy == 'passive_adaptive' else 'challenge',
        'risk': 'medium',
        'risk_score': None,
        'passive_score': None,
        'passive_samples': 0,
        'blocked': False,
        'ear': None,
        'yaw': None,
    }


def _update_liveness(state, landmarks, now, passive_score=None,
                     identity_confidence=None, quality=1.0):
    """静默优先；仅对风险不确定的轨迹要求一次随机动作。"""
    window = current_app.config['LIVENESS_WINDOW']
    if not state or (not state.get('passed') and
                     now - float(state.get('started_at', 0)) > window):
        state = _fresh_liveness(now)

    policy = current_app.config.get('LIVENESS_POLICY', 'passive_adaptive')
    effective_policy = policy
    if policy == 'passive_adaptive' and passive_score is None:
        # 模型缺失、旧 dlib 对照模式或推理失败时，安全降级为单动作，
        # 绝不因为无法计算静默分数而直接放行。
        effective_policy = 'single_action'
    if effective_policy == 'single_action' and not state.get('challenge'):
        state['challenge'] = secrets.choice(('blink', 'head_turn'))
        state['mode'] = 'challenge'

    ear, yaw = _liveness_metrics(landmarks)
    if ear is not None:
        state['ear'] = round(float(ear), 4)
        if ear >= current_app.config['LIVENESS_EAR_OPEN']:
            if state.get('closed_seen'):
                state['blink'] = True
            state['open_seen'] = True
        elif (ear <= current_app.config['LIVENESS_EAR_CLOSED'] and
              state.get('open_seen')):
            state['closed_seen'] = True

    if yaw is not None:
        yaw = float(yaw)
        state['yaw'] = round(yaw, 4)
        state['yaw_min'] = yaw if state.get('yaw_min') is None else min(state['yaw_min'], yaw)
        state['yaw_max'] = yaw if state.get('yaw_max') is None else max(state['yaw_max'], yaw)
        if state['yaw_max'] - state['yaw_min'] >= current_app.config['LIVENESS_HEAD_DELTA']:
            state['head_turn'] = True

    if effective_policy == 'passive_adaptive':
        previous_score = state.get('passive_score')
        sample_count = int(state.get('passive_samples', 0))
        if previous_score is None:
            fused_passive = float(passive_score)
        else:
            # Quality-weighted EMA: good frames have more influence while an
            # isolated frame cannot instantly flip the decision.
            weight = float(np.clip(0.35 + 0.40 * quality, 0.40, 0.75))
            fused_passive = ((1.0 - weight) * float(previous_score) +
                             weight * float(passive_score))
        state['passive_score'] = round(fused_passive, 4)
        state['passive_samples'] = min(sample_count + 1, 99)

        identity_confidence = float(identity_confidence or 0.0)
        quality = float(np.clip(quality, 0.0, 1.0))
        stability = min(1.0, float(state.get('passive_samples', 0)) /
                        max(1, current_app.config['LIVENESS_PASSIVE_MIN_FRAMES']))
        identity_risk = max(
            0.0,
            (current_app.config['LIVENESS_IDENTITY_STRONG'] - identity_confidence) /
            max(current_app.config['LIVENESS_IDENTITY_STRONG'], 1e-6))
        risk_score = (0.55 * (1.0 - fused_passive) +
                      0.20 * identity_risk +
                      0.15 * (1.0 - quality) +
                      0.10 * (1.0 - stability))
        state['risk_score'] = round(float(np.clip(risk_score, 0.0, 1.0)), 4)

        enough_samples = (state['passive_samples'] >=
                          current_app.config['LIVENESS_PASSIVE_MIN_FRAMES'])
        strong_identity = (identity_confidence >=
                           current_app.config['LIVENESS_IDENTITY_STRONG'])
        strong_quality = quality >= current_app.config['LIVENESS_QUALITY_STRONG']
        if not enough_samples:
            state['mode'] = 'evaluating'
            state['risk'] = 'medium'
            state['blocked'] = False
            state['passed'] = False
        elif (fused_passive >= current_app.config['LIVENESS_PASSIVE_ACCEPT'] and
              strong_identity and strong_quality):
            state['mode'] = 'passive'
            state['risk'] = 'low'
            state['blocked'] = False
            state['passed'] = True
            state['challenge'] = None
        elif fused_passive <= current_app.config['LIVENESS_PASSIVE_REJECT']:
            state['mode'] = 'rejected'
            state['risk'] = 'high'
            state['blocked'] = True
            state['passed'] = False
        else:
            state['mode'] = 'challenge'
            state['risk'] = 'medium'
            state['blocked'] = False
            if not state.get('challenge'):
                # Challenge starts now; discard movements observed before the
                # random instruction to reduce replay predictability.
                state['challenge'] = secrets.choice(('blink', 'head_turn'))
                state['open_seen'] = False
                state['closed_seen'] = False
                state['blink'] = False
                state['yaw_min'] = None
                state['yaw_max'] = None
                state['head_turn'] = False
                state['passed'] = False
            else:
                state['passed'] = bool(state.get(state['challenge']))
    elif effective_policy == 'single_action':
        challenge = state.get('challenge')
        state['passed'] = bool(state.get(challenge))
    else:
        state['passed'] = bool(state.get('blink') and state.get('head_turn'))
    return state


def _public_liveness(state):
    state = state or {}
    return {
        'blink': bool(state.get('blink')),
        'head_turn': bool(state.get('head_turn')),
        'passed': bool(state.get('passed')),
        'challenge': state.get('challenge'),
        'mode': state.get('mode'),
        'risk': state.get('risk'),
        'risk_score': state.get('risk_score'),
        'passive_score': state.get('passive_score'),
        'passive_samples': int(state.get('passive_samples', 0)),
        'blocked': bool(state.get('blocked')),
        'ear': state.get('ear'),
        'yaw': state.get('yaw'),
    }


def _box_iou(a, b):
    ax2, ay2 = a['x'] + a['w'], a['y'] + a['h']
    bx2, by2 = b['x'] + b['w'], b['y'] + b['h']
    x1, y1 = max(a['x'], b['x']), max(a['y'], b['y'])
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a['w'] * a['h'] + b['w'] * b['h'] - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(a, b):
    ac = (a['x'] + a['w'] / 2.0, a['y'] + a['h'] / 2.0)
    bc = (b['x'] + b['w'] / 2.0, b['y'] + b['h'] / 2.0)
    scale = max(math.hypot(a['w'], a['h']), math.hypot(b['w'], b['h']), 1.0)
    return _distance(ac, bc) / scale


def _new_track(track_id, box, now):
    return {
        'id': str(track_id),
        'box': box,
        'last_seen': now,
        'user_id': None,
        'votes': 0,
        'completed': False,
        'liveness': _fresh_liveness(now),
        'feature_fused': None,
        'feature_samples': 0,
        'final_event_id': None,
        'risk_event_recorded': False,
    }


def _encode_compact_feature(feature):
    """Store a fused vector compactly in the signed cookie session (~344 B)."""
    value = normalize_feature(feature).astype(np.float16)
    return base64.b64encode(value.tobytes()).decode('ascii')


def _decode_compact_feature(value):
    if not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
        feature = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        return normalize_feature(feature)
    except (ValueError, TypeError, binascii.Error):
        return None


def _fuse_track_feature(track, feature, quality):
    """Quality-weighted exponential fusion for one person's temporal track."""
    current = normalize_feature(feature)
    previous = _decode_compact_feature(track.get('feature_fused'))
    if previous is None:
        fused = current
    else:
        # Clearer frames replace more of the accumulated state; even the lowest
        # accepted frame gets enough weight to recover after a track hand-over.
        weight = float(np.clip(0.35 + 0.50 * quality, 0.45, 0.85))
        fused = normalize_feature((1.0 - weight) * previous + weight * current)
    track['feature_fused'] = _encode_compact_feature(fused)
    track['feature_samples'] = min(int(track.get('feature_samples', 0)) + 1, 99)
    return fused


def _touch_low_quality_track(track, box, now):
    """Keep spatial continuity without letting a poor frame reset identity."""
    track['box'] = box
    track['last_seen'] = now
    return track


def _associate_tracks(boxes, now):
    """用 IoU/中心距离把当前帧人脸与 session 中的历史轨迹做一一关联。"""
    ttl = current_app.config['FACE_TRACK_TTL']
    raw_tracks = session.get('face_tracks') or {}
    tracks = {
        str(tid): track for tid, track in raw_tracks.items()
        if isinstance(track, dict) and
        now - float(track.get('last_seen', 0)) <= ttl
    }

    candidates = []
    for face_index, box in enumerate(boxes):
        for track_id, track in tracks.items():
            old_box = track.get('box')
            if not isinstance(old_box, dict):
                continue
            iou = _box_iou(box, old_box)
            center_dist = _normalized_center_distance(box, old_box)
            if iou >= 0.10 or center_dist <= 0.70:
                candidates.append((iou * 2.0 + max(0.0, 1.0 - center_dist),
                                   face_index, track_id))

    assignments = [None] * len(boxes)
    used_faces, used_tracks = set(), set()
    for _, face_index, track_id in sorted(candidates, reverse=True):
        if face_index in used_faces or track_id in used_tracks:
            continue
        assignments[face_index] = track_id
        used_faces.add(face_index)
        used_tracks.add(track_id)

    next_id = int(session.get('next_face_track_id', 1))
    for face_index, box in enumerate(boxes):
        if assignments[face_index] is None:
            track_id = str(next_id)
            next_id += 1
            tracks[track_id] = _new_track(track_id, box, now)
            assignments[face_index] = track_id

    session['next_face_track_id'] = next_id
    return tracks, assignments


def _update_track(track, user_id, box, landmarks, now, passive_score=None,
                  identity_confidence=None, quality=1.0):
    """在一条空间轨迹内独立累计身份票数与活体动作。"""
    previous_seen = float(track.get('last_seen', 0))
    same_identity = user_id is not None and track.get('user_id') == user_id

    if user_id is None:
        track['user_id'] = None
        track['votes'] = 0
        track['completed'] = False
        track['liveness'] = _fresh_liveness(now)
        track['final_event_id'] = None
        track['risk_event_recorded'] = False
    elif not same_identity:
        track['user_id'] = user_id
        track['votes'] = 1
        track['completed'] = False
        track['liveness'] = _fresh_liveness(now)
        track['final_event_id'] = None
        track['risk_event_recorded'] = False
    else:
        if now - previous_seen <= current_app.config['FACE_MATCH_WINDOW']:
            track['votes'] = min(int(track.get('votes', 0)) + 1,
                                 current_app.config['FACE_MATCH_VOTES'])
        else:
            track['votes'] = 1
            track['completed'] = False
            track['liveness'] = _fresh_liveness(now)
            track['final_event_id'] = None
            track['risk_event_recorded'] = False

    if user_id is not None and not track.get('completed'):
        if current_app.config['LIVENESS_REQUIRED']:
            track['liveness'] = _update_liveness(
                track.get('liveness'), landmarks, now,
                passive_score=passive_score,
                identity_confidence=identity_confidence,
                quality=quality)
        else:
            track['liveness'] = _fresh_liveness(now)
            track['liveness']['passed'] = True
            track['liveness']['mode'] = 'disabled'
            track['liveness']['risk'] = 'disabled'

    track['box'] = box
    track['last_seen'] = now
    return track


def _recognize_and_sign_in(probe):
    """核心识别：加载人脸库 → 最佳匹配 → 多帧投票 → 落库。返回 payload dict。"""
    pairs = _load_registered()
    if not pairs:
        return _json('unregistered', 'EMPTY_DATABASE',
                     '系统暂无已录入人脸，请联系管理员')

    tolerance = current_app.config['FACE_MATCH_TOLERANCE']
    best, dist = find_best_match([e for _, e in pairs], probe, tolerance)
    if best is None:
        session.pop('votes', None)
        return _json('unregistered', 'NO_MATCH', '未识别到已注册学生', distance=dist)

    user = pairs[best][0]
    confirmed, votes = _bump_vote(
        user.id,
        current_app.config['FACE_MATCH_VOTES'],
        current_app.config['FACE_MATCH_WINDOW'])
    if not confirmed:
        return _json('pending', 'NEED_MORE_FRAMES', '识别中，请保持正对摄像头',
                     user={'id': user.id, 'number': user.number,
                           'username': user.username},
                     distance=dist, votes=votes)

    record = record_sign_in(user.id)
    event, attendance_record, attendance_created = record_recognition_decision(
        user_id=user.id, outcome='success' if record is not None else 'duplicate',
        engine='legacy-browser', liveness_mode='client', risk_level='low')
    if record is None and not attendance_created:
        return _json('duplicate', 'DUPLICATE', '今日已签到',
                     user={'id': user.id, 'number': user.number,
                           'username': user.username},
                     distance=dist)
    return _json('success', 'SIGNED_IN', '签到成功',
                 user={'id': user.id, 'number': user.number,
                       'username': user.username},
                 distance=dist,
                 sign_time=(record.sign_time if record else attendance_record.checked_at)
                 .strftime('%Y-%m-%d %H:%M:%S'))


def handle_sign_in_request():
    """处理 POST /api/sign-in：前端算好 128 维描述子。返回 (payload, status)。"""
    if request.content_length and request.content_length > MAX_JSON_BODY:
        return _json('error', 'BODY_TOO_LARGE', '请求体过大'), 413

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return _json('error', 'BAD_JSON', '请求体必须是 JSON 对象'), 400

    encoding = data.get('encoding')
    if not isinstance(encoding, list) or len(encoding) != 128:
        return _json('error', 'INVALID_ENCODING', 'encoding 必须是 128 维数组'), 400
    for v in encoding:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return _json('error', 'INVALID_ENCODING', 'encoding 含非有限数值（NaN/Inf）'), 400

    if data.get('liveness') is not True:
        return _json('error', 'LIVENESS_FAILED', '活体检测未通过'), 400

    probe = np.asarray(encoding, dtype=np.float64)
    return _recognize_and_sign_in(probe), 200


def handle_sign_in_image():
    """处理截图：多人跟踪、身份投票、服务端活体检测并分别签到。"""
    request_started = time.perf_counter()
    if request.content_length and request.content_length > MAX_IMAGE_BODY:
        return _json('error', 'BODY_TOO_LARGE', '请求体过大'), 413

    # 新页面直接上传二进制 JPEG，省去约 33% 的 base64 膨胀和 JSON 解析。
    # 旧页面与测试仍可发送 data URL，保持接口向后兼容。
    if request.mimetype and request.mimetype.startswith('image/'):
        img_bytes = request.get_data(cache=False)
        if not img_bytes:
            return _json('error', 'BAD_IMAGE', '缺少图片数据'), 400
    else:
        data = request.get_json(silent=True)
        if data is None or not isinstance(data, dict):
            return _json('error', 'BAD_JSON', '请求体必须是图片或 JSON 对象'), 400
        img_b64 = data.get('image')
        if not img_b64:
            return _json('error', 'BAD_IMAGE', '缺少图片数据'), 400
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(img_b64, validate=True)
        except (ValueError, TypeError, binascii.Error):
            return _json('error', 'BAD_IMAGE', '图片解码失败'), 400

    try:
        img = load_image_file(io.BytesIO(img_bytes))
    except Exception:
        return _json('error', 'BAD_IMAGE', '图片读取失败'), 400

    # auto 模式只在数据库已经有 SFace 特征时切换；历史 dlib 用户不会
    # 因为升级而突然无法签到。opencv 模式用于强制性能/论文对照实验。
    requested_engine = current_app.config.get('FACE_ENGINE', 'auto')
    sface_pairs = (_load_registered('face_encoding_sface')
                   if requested_engine != 'dlib' else [])
    use_opencv = requested_engine == 'opencv' or (
        requested_engine == 'auto' and bool(sface_pairs))
    pairs = sface_pairs if use_opencv else _load_registered()
    qualities = []
    match_engine = 'dlib'
    try:
        if use_opencv:
            engine = get_opencv_face_engine()
            observations = engine.detect(img)
            locations = [observation.location for observation in observations]
            qualities = [observation.quality for observation in observations]
            minimum_quality = current_app.config['FACE_MIN_QUALITY']
            encs = [engine.encode(img, observation)
                    if observation.quality >= minimum_quality else None
                    for observation in observations]
            # 68 点只承担一次随机动作挑战，不再承担检测和身份编码。
            landmarks = face_landmarks(img, locations)
            match_engine = 'opencv'
        else:
            locations = face_locations(img)
            if locations:
                encs, landmarks = face_encodings_with_landmarks(
                    img, known_face_locations=locations)
            else:
                encs, landmarks = [], []
            qualities = [1.0] * len(locations)
    except FaceModelUnavailable:
        if requested_engine == 'opencv':
            return _json('error', 'MODEL_UNAVAILABLE',
                         'YuNet/SFace 模型不可用，请运行模型下载脚本'), 503
        # auto 模式模型异常时安全回退 dlib。
        pairs = _load_registered()
        locations = face_locations(img)
        if locations:
            encs, landmarks = face_encodings_with_landmarks(
                img, known_face_locations=locations)
        else:
            encs, landmarks = [], []
        qualities = [1.0] * len(locations)
        match_engine = 'dlib-fallback'
    except Exception:
        current_app.logger.exception('face pipeline failed')
        return _json('error', 'BAD_IMAGE', '人脸特征提取失败'), 400

    now = time.time()
    if not locations:
        # 保留短暂漏检的人脸轨迹，但及时清理已经离开画面的轨迹。
        tracks, _ = _associate_tracks([], now)
        session['face_tracks'] = tracks
        return _json('error', 'NO_FACE', '未检测到人脸',
                     engine=match_engine,
                     processing_ms=round((time.perf_counter() - request_started) * 1000, 1)), 200

    if (len(encs) != len(locations) or len(landmarks) != len(locations)):
        return _json('error', 'NO_FACE', '未检测到人脸'), 200

    boxes = [_box(loc) for loc in locations]
    tracks, assignments = _associate_tracks(boxes, now)
    if not pairs:
        faces = []
        for box, track_id, points, quality in zip(
                boxes, assignments, landmarks, qualities):
            track = _update_track(tracks[track_id], None, box, points, now)
            faces.append({'face_box': box, 'track_id': track_id,
                          'name': None, 'distance': None, 'quality': quality})
        session['face_tracks'] = tracks
        code = 'REENROLL_REQUIRED' if use_opencv else 'EMPTY_DATABASE'
        message = ('当前没有 SFace 人脸特征，请在学生资料中重新上传照片'
                   if use_opencv else '系统暂无已录入人脸')
        return _json('unregistered', code, message, faces=faces,
                     engine=match_engine), 200

    # 每张脸独立比对，并把结果写入各自的空间轨迹。
    known_encodings = [encoding for _, encoding in pairs]
    users_by_id = {user.id: user for user, _ in pairs}
    infos = []
    identity_counts = {}
    for box, track_id, enc, points, quality in zip(
            boxes, assignments, encs, landmarks, qualities):
        track = tracks[track_id]
        if enc is None:
            _touch_low_quality_track(track, box, now)
            user = users_by_id.get(track.get('user_id'))
            infos.append({'box': box, 'track_id': track_id, 'track': track,
                          'user': user, 'distance': None, 'similarity': None,
                          'margin': None, 'quality': quality,
                          'quality_pending': True})
            continue

        similarity = margin = None
        if match_engine == 'opencv':
            probe = _fuse_track_feature(track, enc, quality)
            threshold = current_app.config['SFACE_COSINE_THRESHOLD']
            # Poor-but-accepted frames may only tighten the threshold, never
            # loosen it. This prevents quality adaptation from increasing FAR.
            threshold += max(0.0, 0.60 - float(quality)) * 0.10
            best, similarity, margin = find_best_cosine_match(
                known_encodings, probe, threshold,
                current_app.config['SFACE_MINIMUM_MARGIN'])
            dist = None if similarity is None else 1.0 - similarity
        else:
            probe = np.asarray(enc, dtype=np.float64)
            best, dist = find_best_match(
                known_encodings, probe,
                current_app.config['FACE_MATCH_TOLERANCE'])
        user = pairs[best][0] if best is not None else None
        user_id = user.id if user is not None else None
        passive_score = None
        anti_spoof_ms = None
        if (user is not None and match_engine == 'opencv' and
                current_app.config.get('ANTI_SPOOF_ENABLED', True) and
                not track.get('completed')):
            try:
                anti_result = get_anti_spoof_engine().predict(img, locations[len(infos)])
                passive_score = anti_result.live_probability
                anti_spoof_ms = anti_result.inference_ms
            except AntiSpoofModelUnavailable:
                # _update_liveness will fall back to a single active challenge.
                passive_score = None
            except Exception:
                current_app.logger.exception('passive anti-spoof inference failed')
                passive_score = None
        track = _update_track(
            track, user_id, box, points, now,
            passive_score=passive_score,
            identity_confidence=similarity,
            quality=quality)
        if user_id is not None:
            identity_counts[user_id] = identity_counts.get(user_id, 0) + 1
        infos.append({'box': box, 'track_id': track_id, 'track': track,
                      'user': user, 'distance': dist,
                      'similarity': similarity,
                      # A one-person gallery has an infinite mathematical
                      # margin. JSON has no Infinity value, so expose null.
                      'margin': (margin if margin is None or
                                 math.isfinite(margin) else None),
                      'quality': quality, 'quality_pending': False,
                      'anti_spoof_ms': anti_spoof_ms})

    # 同一身份若在同一帧出现两次（例如真人与其照片同时入镜），无法可靠判断
    # 哪条轨迹是真人，两个轨迹都不允许落库。
    ambiguous_ids = {uid for uid, count in identity_counts.items() if count > 1}
    signed = []
    faces = []
    votes_required = current_app.config['FACE_MATCH_VOTES']
    for info in infos:
        box = info['box']
        track = info['track']
        user = info['user']
        face = {'face_box': box, 'track_id': info['track_id'],
                'distance': info['distance'], 'similarity': info['similarity'],
                'margin': info['margin'], 'quality': info['quality'],
                'quality_pending': info['quality_pending'],
                'anti_spoof_ms': info.get('anti_spoof_ms')}

        if user is None:
            face.update({'name': None, 'votes': 0, 'liveness': None})
            faces.append(face)
            continue

        face.update({'name': user.username, 'number': user.number,
                     'user_id': user.id})
        if user.id in ambiguous_ids:
            track['votes'] = 0
            track['completed'] = False
            track['liveness'] = _fresh_liveness(now)
            if not track.get('risk_event_recorded'):
                event, _, _ = record_recognition_decision(
                    user_id=user.id, track_id=info['track_id'],
                    outcome='rejected', engine=match_engine,
                    quality=info['quality'], similarity=info['similarity'],
                    margin=info['margin'], risk_score=.88,
                    risk_level='high',
                    latency_ms=(time.perf_counter() - request_started) * 1000,
                    liveness_mode='ambiguous',
                    risk_type='identity_collision',
                    risk_message=u'同一身份在同一画面出现多次，疑似照片或屏幕重放',
                    risk_severity='high')
                track['risk_event_recorded'] = True
                track['final_event_id'] = event.id
            face.update({'votes': 0,
                         'liveness': _public_liveness(track['liveness']),
                         'ambiguous_identity': True})
            faces.append(face)
            continue

        liveness = _public_liveness(track.get('liveness'))
        face.update({'votes': int(track.get('votes', 0)),
                     'votes_required': votes_required,
                     'liveness': liveness,
                     'completed': bool(track.get('completed'))})

        if liveness['blocked'] and not track.get('risk_event_recorded'):
            event, _, _ = record_recognition_decision(
                user_id=user.id, track_id=info['track_id'],
                outcome='rejected', engine=match_engine,
                quality=info['quality'], similarity=info['similarity'],
                margin=info['margin'], passive_score=liveness['passive_score'],
                risk_score=liveness['risk_score'], risk_level='high',
                latency_ms=(time.perf_counter() - request_started) * 1000,
                liveness_mode=liveness['mode'], risk_type='spoof_attack',
                risk_message=u'静默活体判定为疑似照片/屏幕翻拍，已阻断签到',
                risk_severity='high')
            track['risk_event_recorded'] = True
            track['final_event_id'] = event.id
            face['risk_event_id'] = event.id

        confirmed = (track.get('votes', 0) >= votes_required and
                     liveness['passed'])
        if confirmed and not track.get('completed'):
            record = record_sign_in(user.id)
            outcome = 'success' if record is not None else 'duplicate'
            event, attendance_record, attendance_created = record_recognition_decision(
                user_id=user.id, track_id=info['track_id'], outcome=outcome,
                engine=match_engine, quality=info['quality'],
                similarity=info['similarity'], margin=info['margin'],
                passive_score=liveness['passive_score'],
                risk_score=liveness['risk_score'],
                risk_level=liveness['risk'] or 'low',
                latency_ms=(time.perf_counter() - request_started) * 1000,
                liveness_mode=liveness['mode'])
            duplicate = record is None and not attendance_created
            signed_item = {'name': user.username, 'number': user.number,
                           'duplicate': duplicate,
                           'event_id': event.id}
            sign_time = (record.sign_time if record is not None else
                         attendance_record.checked_at if attendance_record else None)
            if sign_time:
                signed_item['sign_time'] = sign_time.strftime('%Y-%m-%d %H:%M:%S')
            if attendance_record:
                signed_item['attendance_status'] = attendance_record.status
                signed_item['session_id'] = attendance_record.session_id
            signed.append(signed_item)
            track['final_event_id'] = event.id
            track['completed'] = True
            face['completed'] = True
        faces.append(face)

    # 所有键均使用字符串 track_id，避免 Flask Cookie session 把整数 dict key
    # 序列化成字符串后导致下一帧无法读取、票数永远停在 1。
    session['face_tracks'] = tracks
    session.modified = True

    processing_ms = round((time.perf_counter() - request_started) * 1000, 1)
    return _json('multi', 'MULTI', '多人识别结果', faces=faces, signed=signed,
                 engine=match_engine, processing_ms=processing_ms), 200
