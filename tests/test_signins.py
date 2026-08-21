# coding=utf-8
"""签到模块测试：同日防重复、跨日可签、仅日期参数"""
from datetime import datetime, date, timedelta

from app import db, Role, User, SignIn
from attendance import record_sign_in


def _make_student():
    student_role = Role.query.filter_by(name='Student').first()
    u = User(number='B20150003001', username='测试学生', password='x', role=student_role)
    db.session.add(u)
    db.session.commit()
    return u.id


def test_sign_in_same_day_dedup(app):
    """同一学生同一天重复签到应被唯一约束拦截"""
    with app.app_context():
        uid = _make_student()
        r1 = record_sign_in(uid)
        assert r1 is not None
        r2 = record_sign_in(uid)  # 同日重复
        assert r2 is None
        assert SignIn.query.count() == 1


def test_sign_in_next_day_allowed(app):
    """不同日期可以分别签到"""
    with app.app_context():
        uid = _make_student()
        t = datetime(2026, 8, 18, 9, 0, 0)
        assert record_sign_in(uid, t) is not None
        assert record_sign_in(uid, t + timedelta(days=1)) is not None
        assert SignIn.query.count() == 2


def test_sign_in_date_only(app):
    """手动补签只传日期时，签到时刻记为当日 00:00"""
    with app.app_context():
        uid = _make_student()
        r = record_sign_in(uid, date(2026, 8, 18))
        assert r is not None
        assert r.sign_date == date(2026, 8, 18)
        assert r.sign_time == datetime(2026, 8, 18, 0, 0)
