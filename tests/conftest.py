# coding=utf-8
"""
pytest 公共 fixtures：使用独立临时数据库，避免污染演示数据库 data.sqlite。

重要：必须在 import app 之前设置 DATABASE_URL 环境变量——
flask-sqlalchemy 会按 app 缓存 engine（修改配置不会重建），
因此要让应用从一开始就指向测试库，而不是事后改配置。
"""
import os
import tempfile

import pytest

TEST_DB_PATH = os.path.join(tempfile.gettempdir(), 'face_register_test.sqlite')
os.environ['DATABASE_URL'] = 'sqlite:///' + TEST_DB_PATH

from app import app as flask_app, db, Role  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def _clean_test_db():
    """每个测试会话开始时删除遗留的测试数据库文件，
    避免上一次异常退出留下的脏数据污染本轮结果。"""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield


@pytest.fixture
def app():
    flask_app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,  # 测试中默认禁用 CSRF，简化表单提交
        SECRET_KEY='test-secret',
    )
    with flask_app.app_context():
        db.create_all()
        Role.insert_roles()
    yield flask_app
    with flask_app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()
