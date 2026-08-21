# coding=utf-8
"""当前 Web 页面所用截图接口的多人跟踪与服务端活体测试。"""
import numpy as np

from app import Role, SignIn, User, db
import web_signin


def _unit_vec(seed):
    rng = np.random.RandomState(seed)
    value = rng.randn(128)
    return value / np.linalg.norm(value)


def _register(app, username, number, encoding):
    with app.app_context():
        role = Role.query.filter_by(name='Student').first()
        user = User(username=username, number=number,
                    face_encoding=str(list(encoding)), role=role)
        db.session.add(user)
        db.session.commit()
        return user.id


def _landmarks(offset_x=0, closed=False, yaw=0.0):
    """生成只关心眼睛与鼻尖的 68 点测试关键点。"""
    points = [(offset_x, 0) for _ in range(68)]
    upper = 9.7 if closed else 8.0
    lower = 10.3 if closed else 12.0

    points[36:42] = [
        (offset_x + 10, 10), (offset_x + 12, upper),
        (offset_x + 14, upper), (offset_x + 16, 10),
        (offset_x + 14, lower), (offset_x + 12, lower),
    ]
    points[42:48] = [
        (offset_x + 22, 10), (offset_x + 24, upper),
        (offset_x + 26, upper), (offset_x + 28, 10),
        (offset_x + 26, lower), (offset_x + 24, lower),
    ]
    # 双眼中心横坐标为 offset_x + 19，眼间距为 12。
    points[30] = (offset_x + 19 + yaw * 12, 18)
    return points


def _install_frames(monkeypatch, frames):
    """把 dlib 图像流水线替换为确定性的多帧检测结果。"""
    state = {'index': 0}

    monkeypatch.setattr(
        web_signin, 'load_image_file',
        lambda _: np.zeros((360, 480, 3), dtype=np.uint8))

    def fake_locations(_img):
        return frames[state['index']]['locations']

    def fake_features(_img, known_face_locations=None):
        frame = frames[state['index']]
        state['index'] += 1
        return frame['encodings'], frame['landmarks']

    monkeypatch.setattr(web_signin, 'face_locations', fake_locations)
    monkeypatch.setattr(
        web_signin, 'face_encodings_with_landmarks', fake_features)


def _post_image(client):
    # 图像读取已在测试中替换；这里只需提供合法 base64 请求结构。
    return client.post('/api/sign-in-image',
                       json={'image': 'data:image/jpeg;base64,AA=='})


def _configure(app):
    app.config.update(
        FACE_MATCH_VOTES=3,
        FACE_MATCH_WINDOW=6.0,
        FACE_TRACK_TTL=2.0,
        LIVENESS_REQUIRED=True,
        LIVENESS_WINDOW=15.0,
        LIVENESS_EAR_CLOSED=0.20,
        LIVENESS_EAR_OPEN=0.24,
        LIVENESS_HEAD_DELTA=0.12,
    )


def test_two_people_track_and_sign_independently(app, client, monkeypatch):
    """两张脸分别完成眨眼/转头和三帧确认后，在同一响应中签到。"""
    _configure(app)
    vector_a, vector_b = _unit_vec(10), _unit_vec(20)
    _register(app, '甲同学', 'B20260001001', vector_a)
    _register(app, '乙同学', 'B20260001002', vector_b)

    frames = [
        {
            'locations': [(20, 120, 120, 20), (30, 320, 130, 220)],
            'encodings': [vector_a, vector_b],
            'landmarks': [_landmarks(yaw=-0.15),
                          _landmarks(offset_x=200, yaw=-0.15)],
        },
        {
            'locations': [(22, 122, 122, 22), (31, 321, 131, 221)],
            'encodings': [vector_a, vector_b],
            'landmarks': [_landmarks(closed=True, yaw=0.0),
                          _landmarks(offset_x=200, closed=True, yaw=0.0)],
        },
        {
            'locations': [(24, 124, 124, 24), (32, 322, 132, 222)],
            'encodings': [vector_a, vector_b],
            'landmarks': [_landmarks(yaw=0.15),
                          _landmarks(offset_x=200, yaw=0.15)],
        },
    ]
    _install_frames(monkeypatch, frames)

    first = _post_image(client).get_json()
    assert [face['votes'] for face in first['faces']] == [1, 1]
    track_ids = [face['track_id'] for face in first['faces']]
    assert len(set(track_ids)) == 2
    assert first['signed'] == []

    second = _post_image(client).get_json()
    assert [face['votes'] for face in second['faces']] == [2, 2]
    assert [face['track_id'] for face in second['faces']] == track_ids
    assert second['signed'] == []

    third = _post_image(client).get_json()
    assert [face['track_id'] for face in third['faces']] == track_ids
    assert all(face['liveness']['passed'] for face in third['faces'])
    assert {item['name'] for item in third['signed']} == {'甲同学', '乙同学'}
    assert all(not item['duplicate'] for item in third['signed'])

    with client.session_transaction() as session_state:
        assert all(isinstance(key, str)
                   for key in session_state['face_tracks'].keys())
    with app.app_context():
        assert SignIn.query.count() == 2


def test_static_photo_never_passes_liveness(app, client, monkeypatch):
    """身份即使连续命中，静止睁眼照片也不能落库。"""
    _configure(app)
    vector = _unit_vec(30)
    _register(app, '真人', 'B20260002001', vector)
    frames = [{
        'locations': [(20, 120, 120, 20)],
        'encodings': [vector],
        'landmarks': [_landmarks(yaw=0.0)],
    } for _ in range(5)]
    _install_frames(monkeypatch, frames)

    last = None
    for _ in frames:
        last = _post_image(client).get_json()
    face = last['faces'][0]
    assert face['votes'] == 3
    assert face['liveness']['blink'] is False
    assert face['liveness']['head_turn'] is False
    assert last['signed'] == []
    with app.app_context():
        assert SignIn.query.count() == 0


def test_single_action_policy_does_not_require_both_actions(
        app, client, monkeypatch):
    """随机动作模式只需当前挑战，避免用户同时完成眨眼和转头。"""
    _configure(app)
    app.config['LIVENESS_POLICY'] = 'single_action'
    vector = _unit_vec(35)
    _register(app, '快速签到', 'B20260002501', vector)
    frames = [{
        'locations': [(20, 120, 120, 20)],
        'encodings': [vector],
        'landmarks': [_landmarks(yaw=value)],
    } for value in (-0.2, 0.0, 0.2)]
    _install_frames(monkeypatch, frames)

    # 固定为转头挑战，第三帧达到身份票数时应直接完成，不要求眨眼。
    monkeypatch.setattr(web_signin.secrets, 'choice',
                        lambda _: 'head_turn')
    for _ in frames:
        last = _post_image(client).get_json()
    face = last['faces'][0]
    assert face['liveness']['challenge'] == 'head_turn'
    assert face['liveness']['head_turn'] is True
    assert face['liveness']['blink'] is False
    assert len(last['signed']) == 1


def test_passive_adaptive_liveness_has_three_risk_paths(app, monkeypatch):
    """高置信无感通过、中间区间单动作、明显攻击直接拒绝。"""
    app.config.update(
        LIVENESS_POLICY='passive_adaptive',
        LIVENESS_PASSIVE_MIN_FRAMES=2,
        LIVENESS_PASSIVE_ACCEPT=0.80,
        LIVENESS_PASSIVE_REJECT=0.15,
        LIVENESS_IDENTITY_STRONG=0.55,
        LIVENESS_QUALITY_STRONG=0.55,
    )
    monkeypatch.setattr(web_signin.secrets, 'choice', lambda _: 'blink')
    with app.app_context():
        live = web_signin._fresh_liveness(1.0)
        live = web_signin._update_liveness(
            live, _landmarks(), 1.1, passive_score=0.95,
            identity_confidence=0.90, quality=0.90)
        assert live['mode'] == 'evaluating'
        live = web_signin._update_liveness(
            live, _landmarks(), 1.2, passive_score=0.96,
            identity_confidence=0.90, quality=0.90)
        assert live['mode'] == 'passive'
        assert live['passed'] is True
        assert live['challenge'] is None

        uncertain = web_signin._fresh_liveness(2.0)
        for now in (2.1, 2.2):
            uncertain = web_signin._update_liveness(
                uncertain, _landmarks(), now, passive_score=0.50,
                identity_confidence=0.90, quality=0.90)
        assert uncertain['mode'] == 'challenge'
        assert uncertain['challenge'] == 'blink'
        assert uncertain['passed'] is False

        attack = web_signin._fresh_liveness(3.0)
        for now in (3.1, 3.2):
            attack = web_signin._update_liveness(
                attack, _landmarks(), now, passive_score=0.02,
                identity_confidence=0.90, quality=0.90)
        assert attack['mode'] == 'rejected'
        assert attack['blocked'] is True
        assert attack['passed'] is False


def test_same_identity_twice_in_one_frame_is_rejected(app, client, monkeypatch):
    """真人与其照片同时出现时，同一身份的两条轨迹都不能签到。"""
    _configure(app)
    vector = _unit_vec(40)
    _register(app, '甲同学', 'B20260003001', vector)
    frames = [{
        'locations': [(20, 120, 120, 20), (20, 320, 120, 220)],
        'encodings': [vector, vector],
        'landmarks': [_landmarks(yaw=0.2),
                      _landmarks(offset_x=200, yaw=0.2)],
    } for _ in range(3)]
    _install_frames(monkeypatch, frames)

    last = None
    for _ in frames:
        last = _post_image(client).get_json()
    assert all(face['ambiguous_identity'] for face in last['faces'])
    assert all(face['votes'] == 0 for face in last['faces'])
    assert last['signed'] == []
    with app.app_context():
        assert SignIn.query.count() == 0


def test_unknown_face_does_not_block_registered_person(app, client, monkeypatch):
    """陌生人与已注册学生同框时，学生仍可独立完成签到。"""
    _configure(app)
    known, stranger = _unit_vec(50), _unit_vec(999)
    _register(app, '已注册', 'B20260004001', known)
    actions = [
        (_landmarks(yaw=-0.15), _landmarks(offset_x=200)),
        (_landmarks(closed=True), _landmarks(offset_x=200)),
        (_landmarks(yaw=0.15), _landmarks(offset_x=200)),
    ]
    frames = [{
        'locations': [(20, 120, 120, 20), (20, 320, 120, 220)],
        'encodings': [known, stranger],
        'landmarks': list(frame_landmarks),
    } for frame_landmarks in actions]
    _install_frames(monkeypatch, frames)

    for _ in frames:
        last = _post_image(client).get_json()
    assert len(last['signed']) == 1
    assert last['signed'][0]['name'] == '已注册'
    assert [face['name'] for face in last['faces']] == ['已注册', None]
    with app.app_context():
        assert SignIn.query.count() == 1
