# coding=utf-8
"""安全相关测试：密码哈希、明文密码惰性升级、人脸编码安全解析"""
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

from app import User, Role, db
from functions import parse_face_encoding


def test_password_hash_roundtrip():
    hashed = generate_password_hash('123456')
    assert check_password_hash(hashed, '123456')
    assert not check_password_hash(hashed, 'wrong')


def test_legacy_plaintext_password_upgrade(app):
    """历史明文密码：验证通过后应自动升级为哈希存储"""
    with app.app_context():
        admin_role = Role.query.filter_by(name='Admin').first()
        u = User(number=99990001, username='明文用户',
                 password='plaintext123', role=admin_role)
        db.session.add(u)
        db.session.commit()

        assert u.verify_password('plaintext123') is True
        # 已升级为哈希
        assert u.password.startswith(('pbkdf2:', 'scrypt:', 'sha256:'))
        # 升级后再次验证仍然通过
        assert u.verify_password('plaintext123') is True
        # 错误密码被拒绝
        assert u.verify_password('wrong') is False


def test_parse_face_encoding():
    vec = list(np.random.rand(128))
    # 正常 128 维编码
    parsed = parse_face_encoding(str(vec))
    assert parsed is not None and parsed.shape == (128,)
    # 空值 / 非法输入
    assert parse_face_encoding(None) is None
    assert parse_face_encoding('') is None
    assert parse_face_encoding('[0.1, 0.2') is None       # 语法错误
    assert parse_face_encoding('[1, 2, 3]') is None       # 维度不足
    assert parse_face_encoding('__import__("os")') is None  # 代码注入被拒绝
    assert parse_face_encoding('{"a": 1}') is None        # 非列表
