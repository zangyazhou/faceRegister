# coding=utf-8
"""CSRF 保护测试：无 token 的 POST 应被拒绝"""
import re

from app import db, Role, User
from werkzeug.security import generate_password_hash


def _add_admin():
    role = Role.query.filter_by(name='Admin').first()
    u = User(number='0', username='测试管理员',
             password=generate_password_hash('pw'), role=role)
    db.session.add(u)
    db.session.commit()


def test_csrf_rejects_post_without_token(app, client):
    """开启 CSRF 后，无 token 的 POST 请求应返回 400"""
    with app.app_context():
        _add_admin()
    # 先（CSRF 关闭状态下）登录，拿到会话
    r = client.post('/login', data={'username': '测试管理员', 'password': 'pw', 'submit': '登录'})
    assert r.status_code == 302

    # 开启 CSRF，模拟一个不含 token 的 POST（注销）
    app.config['WTF_CSRF_ENABLED'] = True
    try:
        r = client.post('/logout')
        assert r.status_code == 400, '无 CSRF token 的 POST 应被拒绝'
    finally:
        app.config['WTF_CSRF_ENABLED'] = False


def test_login_page_renders_csrf_token(app, client):
    """登录页应包含 CSRF token 隐藏字段（生产环境 CSRF 开启时）"""
    app.config['WTF_CSRF_ENABLED'] = True  # fixture 默认关闭，此处按生产配置开启
    try:
        r = client.get('/login')
        assert r.status_code == 200
        assert 'csrf_token' in r.get_data(as_text=True)
    finally:
        app.config['WTF_CSRF_ENABLED'] = False


def test_csrf_renders_as_hidden_input_not_raw_text(app, client):
    """{{ csrf_token() }} 应渲染成 hidden input，而非裸 token 文本（防回归）"""
    with app.app_context():
        _add_admin()
    client.post('/login', data={'username': '测试管理员', 'password': 'pw', 'submit': '登录'})
    html = client.get('/').get_data(as_text=True)

    # 每个 CSRF token 都应包裹在 hidden input 的 value 里
    tokens = re.findall(r'name="csrf_token"[^>]*value="([^"]+)"', html)
    assert tokens, '页面应包含 hidden csrf input'

    # 去掉所有 input 标签后，token 值不应再作为可见文本出现
    body_no_input = re.sub(r'<input[^>]*>', '', html)
    for t in tokens:
        assert t not in body_no_input, 'csrf token 被渲染成裸文本'
