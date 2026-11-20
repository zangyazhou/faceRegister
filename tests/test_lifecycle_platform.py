# coding=utf-8
"""实验室人员生命周期、区域授权与隐私清理测试。"""
from datetime import date, datetime, timedelta

from werkzeug.security import generate_password_hash

from app import (AccessEvent, AccessPoint, ErasureTask, FaceTemplate,
                 OrganizationUnit, RecognitionEvent, Role, User, db)
from lifecycle_service import (create_membership, evaluate_access,
                               offboard_and_erase, sync_face_template,
                               upsert_access_grant)


def _person(number='LAB001', name='测试成员', demo=False):
    role = Role.query.filter_by(name='Student').first()
    item = User(number=number, username=name,
                password=generate_password_hash('pw'), role=role,
                person_type='student', lifecycle_status='active',
                joined_at=date.today(), is_demo=demo)
    db.session.add(item)
    db.session.flush()
    return item


def _lab_with_gate(demo=False):
    lab = OrganizationUnit(code='LAB-A' if not demo else 'DEMO-LAB-A',
                           name='视觉实验室', location='A305', is_demo=demo)
    db.session.add(lab)
    db.session.flush()
    gate = AccessPoint(organization=lab,
                       code='LAB-A-DOOR' if not demo else 'DEMO-LAB-A-DOOR',
                       name='主入口', is_demo=demo)
    db.session.add(gate)
    db.session.flush()
    return lab, gate


def test_access_requires_active_membership_and_explicit_grant(app):
    with app.app_context():
        person = _person()
        lab, gate = _lab_with_gate()
        create_membership(person, lab.id, grant_all=False)
        denied = evaluate_access(person, gate.code)
        assert denied['allowed'] is False
        assert denied['code'] == 'NO_ACCESS_GRANT'

        grant = upsert_access_grant(person, gate)
        db.session.commit()
        allowed = evaluate_access(person, gate.code)
        assert allowed['allowed'] is True
        assert allowed['code'] == 'ACCESS_GRANTED'

        grant.ends_at = datetime.now() - timedelta(seconds=1)
        db.session.commit()
        expired = evaluate_access(person, gate.code)
        assert expired['allowed'] is False
        assert expired['code'] == 'GRANT_OUTSIDE_WINDOW'


def test_suspended_person_is_denied_even_with_face_and_grant(app):
    with app.app_context():
        person = _person()
        lab, gate = _lab_with_gate()
        create_membership(person, lab.id, grant_all=False)
        upsert_access_grant(person, gate)
        sync_face_template(person, '[0.0]', '[1.0]', consent=True)
        person.lifecycle_status = 'suspended'
        db.session.commit()
        result = evaluate_access(person, gate.code)
        assert result['allowed'] is False
        assert result['code'] == 'PERSON_INACTIVE'


def test_offboarding_physically_erases_profile_and_biometric_template(app):
    with app.app_context():
        person = _person()
        lab, gate = _lab_with_gate()
        create_membership(person, lab.id, grant_all=False)
        upsert_access_grant(person, gate)
        sync_face_template(person, '[0.0]', '[1.0]', consent=True)
        event = RecognitionEvent(
            user=person, outcome='success', engine='test', risk_level='low',
            is_demo=False, created_at=datetime.now())
        db.session.add(event)
        db.session.flush()
        db.session.add(AccessEvent(
            user=person, access_point=gate, recognition_event=event,
            decision='allowed', reason_code='ACCESS_GRANTED',
            reason_text='测试放行', is_demo=False, occurred_at=datetime.now()))
        db.session.commit()
        person_id = person.id
        event_id = event.id

        task = offboard_and_erase(person)
        assert User.query.filter_by(id=person_id).first() is None
        assert FaceTemplate.query.filter_by(user_id=person_id).first() is None
        assert ErasureTask.query.filter_by(receipt_code=task.receipt_code).count() == 1
        assert db.session.get(RecognitionEvent, event_id).user_id is None
        assert AccessEvent.query.filter_by(recognition_event_id=event_id).first().user_id is None
        assert task.status == 'completed'
        assert task.templates_deleted == 1


def test_lifecycle_admin_pages_render(client, app):
    with app.app_context():
        admin = Role.query.filter_by(name='Admin').first()
        db.session.add(User(number='0', username='管理员',
                            password=generate_password_hash('pw'), role=admin))
        db.session.commit()
    response = client.post('/login', data={
        'username': '管理员', 'password': 'pw', 'submit': '登录'})
    assert response.status_code == 302
    for path in ('/students', '/laboratories', '/access-events',
                 '/privacy-center'):
        response = client.get(path)
        assert response.status_code == 200, path

    response = client.post('/laboratories', data={
        'code': 'CVLAB', 'name': '计算机视觉实验室', 'location': 'A305',
        'manager': '负责人', 'description': '视觉研究', 'submit': '创建实验室'})
    assert response.status_code == 302
    with app.app_context():
        assert OrganizationUnit.query.filter_by(code='CVLAB').count() == 1
