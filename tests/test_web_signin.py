# coding=utf-8
"""
Web 端人脸签到 API 测试：公开页面、多帧确认、误识拒绝、活体/输入校验、CSRF。
核心诉求验证：对的人能签入、其他人绝不误签、同日不重复。
"""
import re

import numpy as np

from app import SignIn, User, db, Role


def _unit_vec(seed):
    """128 维单位随机向量作为人脸描述子（归一化去尺度干扰）。"""
    rng = np.random.RandomState(seed)
    v = rng.randn(128)
    return v / np.linalg.norm(v)


def _register(app, username, number, vec):
    """插入一个带人脸编码的学生，返回其 id。"""
    with app.app_context():
        role = Role.query.filter_by(name='Student').first()
        u = User(username=username, number=number,
                 face_encoding=str(list(vec)), role=role)
        db.session.add(u)
        db.session.commit()
        return u.id


def _post(client, vec, liveness=True):
    return client.post('/api/sign-in',
                       json={'encoding': list(vec), 'liveness': liveness})


def _sign_in_ok(client, vec):
    """连投 3 票直至成功，返回最后一次响应。"""
    resp = None
    for _ in range(3):
        resp = _post(client, vec)
        assert resp.status_code == 200
    assert resp.get_json()['status'] == 'success'
    return resp


# ---------- 公开页面 ----------

def test_web_signin_page_is_public(app, client):
    """未登录也能访问签到页，且包含 CSRF token 与截图识别接口"""
    r = client.get('/web-sign-in')
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'id="csrf-token"' in html
    assert 'sign-in-image' in html


def test_api_signin_requires_post(app, client):
    """GET /api/sign-in 应为 405（路由仅允许 POST）"""
    assert client.get('/api/sign-in').status_code == 405


# ---------- 多帧确认与落库 ----------

def test_three_votes_then_success(app, client):
    """同一人连续 3 票才落库；中途只显示 pending"""
    vec = _unit_vec(1)
    _register(app, '张三', 'B20260001001', vec)
    with app.app_context():
        assert SignIn.query.count() == 0

    r = _post(client, vec)
    assert r.get_json()['status'] == 'pending'
    assert r.get_json()['votes'] == 1
    r = _post(client, vec)
    assert r.get_json()['status'] == 'pending'
    assert r.get_json()['votes'] == 2
    r = _post(client, vec)
    d = r.get_json()
    assert d['status'] == 'success'
    assert d['user']['username'] == '张三'
    assert d['user']['number'] == 'B20260001001'

    with app.app_context():
        assert SignIn.query.count() == 1


def test_same_day_duplicate(app, client):
    """同日重复签到 → duplicate，不再新增记录"""
    vec = _unit_vec(1)
    _register(app, '张三', 'B20260001001', vec)
    _sign_in_ok(client, vec)
    with app.app_context():
        assert SignIn.query.count() == 1

    # 已签过，再投满 3 票返回 duplicate
    for _ in range(3):
        r = _post(client, vec)
    assert r.get_json()['status'] == 'duplicate'
    with app.app_context():
        assert SignIn.query.count() == 1


# ---------- 误识拒绝（核心诉求） ----------

def test_unknown_face_rejected_no_db_write(app, client):
    """陌生人：全部超阈值 → unregistered，绝不落库"""
    vec = _unit_vec(1)
    _register(app, '张三', 'B20260001001', vec)
    stranger = _unit_vec(999)  # 与库内距离约 sqrt(2)≈1.41，远超 0.55

    r = _post(client, stranger)
    d = r.get_json()
    assert d['status'] == 'unregistered'
    assert d['code'] == 'NO_MATCH'
    with app.app_context():
        assert SignIn.query.count() == 0


def test_unregistered_clears_votes(app, client):
    """命中后遇到陌生人，累计票数被清空：本人重投需从 1 重新开始（而非累加）"""
    va = _unit_vec(1)
    _register(app, '甲', 'B20260011001', va)
    stranger = _unit_vec(999)

    r = _post(client, va)
    assert r.get_json()['votes'] == 1        # 甲第 1 票
    r = _post(client, stranger)
    assert r.get_json()['status'] == 'unregistered'  # 陌生人 → 清票
    r = _post(client, va)
    assert r.get_json()['votes'] == 1        # 重投从 1 开始，而非 2


def test_empty_database_hints_admin(app, client):
    """库为空 → unregistered/EMPTY_DATABASE，提示联系管理员"""
    vec = _unit_vec(1)
    r = _post(client, vec)
    d = r.get_json()
    assert d['status'] == 'unregistered'
    assert d['code'] == 'EMPTY_DATABASE'


def test_votes_reset_when_face_switches(app, client):
    """换一个人识别时投票自动重置（kiosk 共用浏览器也正确）"""
    va, vb = _unit_vec(1), _unit_vec(2)
    _register(app, '甲', 'B20260011001', va)
    _register(app, '乙', 'B20260012001', vb)

    r = _post(client, va)
    assert r.get_json()['votes'] == 1  # 甲第 1 票
    r = _post(client, vb)
    assert r.get_json()['votes'] == 1  # 换乙，票数重置为 1
    r = _post(client, vb)
    assert r.get_json()['votes'] == 2  # 乙第 2 票
    r = _post(client, vb)
    assert r.get_json()['status'] == 'success'
    assert r.get_json()['user']['username'] == '乙'
    with app.app_context():
        assert SignIn.query.count() == 1
        signed = SignIn.query.first().user_id == User.query.filter_by(username='乙').first().id


# ---------- 输入校验 ----------

def test_invalid_encoding_dimension(app, client):
    """127/129 维 → 400"""
    for n in (127, 129):
        r = _post(client, np.zeros(n))
        assert r.status_code == 400
        assert r.get_json()['code'] == 'INVALID_ENCODING'


def test_nan_encoding_rejected(app, client):
    """描述子含 NaN → 400（Python json 默认接受 JS 的 NaN，必须显式拒绝）"""
    body = '{"encoding":[' + ','.join(['NaN'] * 128) + '],"liveness":true}'
    r = client.post('/api/sign-in', data=body, content_type='application/json')
    assert r.status_code == 400
    assert r.get_json()['code'] == 'INVALID_ENCODING'


def test_liveness_required(app, client):
    """liveness 必须为 true，否则 400"""
    vec = _unit_vec(1)
    _register(app, '张三', 'B20260001001', vec)
    r = _post(client, vec, liveness=False)
    assert r.status_code == 400
    assert r.get_json()['code'] == 'LIVENESS_FAILED'
    # 缺失字段同样拒绝
    r2 = client.post('/api/sign-in', json={'encoding': list(vec)})
    assert r2.status_code == 400
    assert r2.get_json()['code'] == 'LIVENESS_FAILED'


def test_body_too_large(app, client):
    """>64KB 请求体 → 413（不影响 add-user 照片上传的全局限制）"""
    big_num = '0.' + '1' * 600  # 单个数字 600+ 字符
    body = '{"encoding":[' + ','.join([big_num] * 128) + '],"liveness":true}'
    assert len(body) > 65536
    r = client.post('/api/sign-in', data=body, content_type='application/json')
    assert r.status_code == 413


# ---------- CSRF ----------

def test_csrf_required_for_api_signin(app, client):
    """开启 CSRF 后：无 token 头 400；带 token 头通过"""
    vec = _unit_vec(1)
    _register(app, '张三', 'B20260001001', vec)
    app.config['WTF_CSRF_ENABLED'] = True
    try:
        r = client.post('/api/sign-in',
                        json={'encoding': list(vec), 'liveness': True})
        assert r.status_code == 400

        # 从公开页面取 CSRF token
        html = client.get('/web-sign-in').get_data(as_text=True)
        m = re.search(r'id="csrf-token"\s+value="([^"]+)"', html)
        assert m, '签到页应渲染 csrf token'
        token = m.group(1)

        r2 = client.post('/api/sign-in',
                         json={'encoding': list(vec), 'liveness': True},
                         headers={'X-CSRFToken': token})
        assert r2.status_code == 200
        assert r2.get_json()['status'] == 'pending'
    finally:
        app.config['WTF_CSRF_ENABLED'] = False
