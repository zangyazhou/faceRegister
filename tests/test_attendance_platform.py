# coding=utf-8
"""新版课程考勤、识别审计、风险中心与演示数据测试。"""
from datetime import datetime, timedelta

from werkzeug.security import generate_password_hash

from app import (AttendanceRecord, AttendanceSession, Course,
                 CourseEnrollment, RecognitionEvent, RiskEvent,
                 Role, User, db)
from attendance_platform import (build_dashboard, close_attendance_session,
                                 build_session_detail,
                                 record_recognition_decision, seed_demo_data)
from experiment_service import load_experiment_dashboard


def _student(number='B20260000001'):
    role = Role.query.filter_by(name='Student').first()
    user = User(number=number, username='平台测试学生', password='!', role=role)
    db.session.add(user)
    db.session.commit()
    return user


def _active_session(user, now):
    course = Course(code='TEST-CV', name='视觉测试课', teacher='测试教师')
    db.session.add(course)
    db.session.flush()
    db.session.add(CourseEnrollment(course_id=course.id, user_id=user.id))
    item = AttendanceSession(
        course_id=course.id, title='测试考勤',
        starts_at=now - timedelta(minutes=5),
        late_after=now + timedelta(minutes=5),
        ends_at=now + timedelta(minutes=55),
        status='active', activated_at=now, is_demo=False)
    db.session.add(item)
    db.session.commit()
    return item


def test_recognition_decision_creates_one_session_record(app):
    with app.app_context():
        now = datetime(2026, 8, 20, 9, 0)
        user = _student()
        item = _active_session(user, now)
        event, record, created = record_recognition_decision(
            user_id=user.id, track_id='7', outcome='success',
            engine='opencv', quality=.82, similarity=.71,
            passive_score=.94, risk_score=.08, latency_ms=108,
            liveness_mode='passive', occurred_at=now)
        assert created is True
        assert event.session_id == item.id
        assert record.status == 'on_time'

        _, same_record, created_again = record_recognition_decision(
            user_id=user.id, track_id='8', outcome='duplicate',
            engine='opencv', occurred_at=now + timedelta(seconds=10))
        assert created_again is False
        assert same_record.id == record.id
        assert AttendanceRecord.query.count() == 1
        assert RecognitionEvent.query.count() == 2


def test_risk_decision_is_auditable_but_does_not_sign_in(app):
    with app.app_context():
        now = datetime(2026, 8, 20, 9, 0)
        user = _student()
        _active_session(user, now)
        event, record, _ = record_recognition_decision(
            user_id=user.id, outcome='rejected', engine='opencv',
            passive_score=.04, risk_score=.94, risk_level='high',
            liveness_mode='rejected', risk_type='spoof_attack',
            risk_message='测试照片攻击', occurred_at=now)
        assert record is None
        assert event.risk_events[0].severity == 'high'
        assert RiskEvent.query.count() == 1
        assert AttendanceRecord.query.count() == 0


def test_closing_session_marks_missing_students_absent(app):
    with app.app_context():
        now = datetime(2026, 8, 20, 9, 0)
        user = _student()
        item = _active_session(user, now)
        close_attendance_session(item, now + timedelta(hours=1))
        assert item.status == 'closed'
        assert item.records[0].status == 'absent'
        detail = build_session_detail(item)
        assert detail['counts']['absent'] == 1
        assert detail['attendance_rate'] == 0


def test_demo_seed_is_isolated_idempotent_and_dashboard_ready(app):
    with app.app_context():
        now = datetime(2026, 8, 20, 9, 0)
        first = seed_demo_data(now)
        second = seed_demo_data(now)
        assert first['created'] is True
        assert second['created'] is False
        demo_students = User.query.filter_by(is_demo=True).all()
        assert len(demo_students) == 28
        assert all(not user.face_encoding and not user.face_encoding_sface
                   for user in demo_students)
        dashboard = build_dashboard(True, now)
        assert dashboard['student_count'] == 28
        assert dashboard['signed_count'] == 23
        assert dashboard['avg_latency'] > 0
        assert dashboard['recent_risks']


def test_new_admin_pages_and_demo_switch(client, app):
    with app.app_context():
        admin_role = Role.query.filter_by(name='Admin').first()
        db.session.add(User(number='0', username='平台管理员',
                            password=generate_password_hash('pw'), role=admin_role))
        db.session.commit()
    client.post('/login', data={'username': '平台管理员', 'password': 'pw'})
    for path in ('/', '/students', '/courses', '/attendance-sessions',
                 '/experiments',
                 '/recognition-center', '/risk-events', '/api/dashboard/summary'):
        response = client.get(path)
        assert response.status_code == 200, path
    response = client.post('/demo/seed', follow_redirects=True)
    assert response.status_code == 200
    assert '演示数据' in response.get_data(as_text=True)
    with app.app_context():
        session_id = AttendanceSession.query.filter_by(is_demo=True).first().id
    detail = client.get('/attendance-sessions/%d' % session_id)
    assert detail.status_code == 200
    assert '学生考勤与识别证据' in detail.get_data(as_text=True)
    exported = client.get('/attendance-sessions/%d/export.csv' % session_id)
    assert exported.status_code == 200
    assert exported.data.startswith(b'\xef\xbb\xbf')
    assert '课程编号' in exported.data.decode('utf-8-sig')


def test_experiment_dashboard_uses_real_artifacts():
    experiment = load_experiment_dashboard()
    assert experiment['performance']['samples'] == 45
    assert experiment['performance']['speedup'] == 17.22
    assert experiment['identity']['engine'] == 'YuNet + SFace'
    assert experiment['identity']['dataset']['negative_samples'] == 11
    assert any(item['status'] == 'pending'
               for item in experiment['evidence_matrix'])
