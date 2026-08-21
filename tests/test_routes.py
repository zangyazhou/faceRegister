# coding=utf-8
"""路由与权限测试"""
from datetime import datetime, date

from werkzeug.security import generate_password_hash

from app import db, Role, User, SignIn


def _add_user(number, username, role_name, password='pw'):
    role = Role.query.filter_by(name=role_name).first()
    u = User(number=number, username=username,
             password=generate_password_hash(password), role=role)
    db.session.add(u)
    db.session.commit()
    return u


def test_index_requires_login(client):
    r = client.get('/')
    assert r.status_code == 302
    assert '/login' in r.headers['Location']


def test_admin_login_and_pages(client, app):
    with app.app_context():
        _add_user('0', '测试管理员', 'Admin')
    r = client.post('/login', data={'username': '测试管理员', 'password': 'pw', 'submit': '登录'})
    assert r.status_code == 302
    for path in ['/', '/add-user', '/sign-ins', '/sign-ins/statistics']:
        r = client.get(path)
        assert r.status_code == 200, path


def test_student_cannot_login(client, app):
    """系统仅对管理员开放"""
    with app.app_context():
        _add_user('B20150004001', '普通学生', 'Student')
    r = client.post('/login', data={'username': '普通学生', 'password': 'pw', 'submit': '登录'})
    assert '系统只对管理员开放' in r.get_data(as_text=True)


def test_wrong_password_rejected(client, app):
    with app.app_context():
        _add_user('0', '测试管理员', 'Admin')
    r = client.post('/login', data={'username': '测试管理员', 'password': 'bad', 'submit': '登录'})
    assert '账号或密码错误' in r.get_data(as_text=True)


def test_remove_user_requires_post(client):
    """删除接口仅接受 POST（防 GET 链接触发删除）"""
    r = client.get('/remove-user/1')
    assert r.status_code == 405


def test_manual_sign_in_requires_login(client):
    r = client.post('/sign-ins/manual', data={'user_id': '1', 'sign_date': '2026-08-18'})
    assert r.status_code == 302
    assert '/login' in r.headers['Location']


def test_admin_required_rejects_student_session(client, app):
    """防御纵深：即使会话被伪造为学生身份，也不能访问管理员页面"""
    with app.app_context():
        _add_user('B20150005001', '学生A', 'Student')
        uid = User.query.filter_by(number='B20150005001').first().id
    with client.session_transaction() as sess:
        sess['_user_id'] = str(uid)  # 模拟一个已登录但非管理员会话
    r = client.get('/add-user')
    assert r.status_code == 302
    assert r.headers['Location'].endswith('/')


def test_sign_ins_filter_and_statistics(client, app):
    """签到记录按日期筛选 + 考勤统计"""
    with app.app_context():
        _add_user('0', '管理员', 'Admin')
        stu = _add_user('B20150006001', '张三', 'Student')
        db.session.add(SignIn(user_id=stu.id, sign_time=datetime(2026, 8, 18, 9, 0),
                              sign_date=date(2026, 8, 18)))
        db.session.commit()

    r = client.post('/login', data={'username': '管理员', 'password': 'pw', 'submit': '登录'})
    assert r.status_code == 302

    # 按日期筛选：命中当天记录
    r = client.get('/sign-ins?date=2026-08-18')
    assert r.status_code == 200
    assert '张三' in r.get_data(as_text=True)

    # 按日期筛选：无记录日期 → 空列表
    r = client.get('/sign-ins?date=2026-08-17')
    assert r.status_code == 200
    assert '张三' not in r.get_data(as_text=True)

    # 考勤统计：学生列表 + 已签到标记
    r = client.get('/sign-ins/statistics')
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert '张三' in html
    assert '1' in html  # 已签到人数
