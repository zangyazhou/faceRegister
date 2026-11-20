# coding=utf-8
"""实验室人员生命周期、区域权限与隐私清理业务服务。"""
from datetime import date, datetime, timedelta
import secrets

from app import (AccessEvent, AccessGrant, AccessPoint, AttendanceRecord,
                 CourseEnrollment, ErasureTask, FaceTemplate, LabMembership,
                 LifecycleEvent, OrganizationUnit, RecognitionEvent,
                 RiskEvent, SignIn, User, db)


ACTIVE_PERSON_STATUS = 'active'


def sync_face_template(user, dlib_encoding=None, sface_encoding=None,
                       consent=True, now=None):
    """写入独立模板表，同时保留旧字段用于平滑迁移。"""
    now = now or datetime.now()
    template = user.face_template
    if template is None:
        template = FaceTemplate(user=user, is_demo=bool(user.is_demo),
                                enrolled_at=now)
        db.session.add(template)
    template.dlib_encoding = dlib_encoding
    template.sface_encoding = sface_encoding
    template.status = 'active'
    template.updated_at = now
    template.destroyed_at = None
    user.face_encoding = dlib_encoding
    user.face_encoding_sface = sface_encoding
    if consent:
        user.privacy_consent_at = user.privacy_consent_at or now
    return template


def template_value(user, field):
    """优先读取独立模板，兼容尚未迁移的历史用户。"""
    template = user.face_template
    if template is not None and template.status == 'active':
        value = (template.sface_encoding if field == 'face_encoding_sface'
                 else template.dlib_encoding)
        if value:
            return value
    return getattr(user, field, None)


def create_membership(user, organization_id, role_title=None, joined_at=None,
                      ends_at=None, grant_all=True, now=None):
    """建立或恢复成员关系，并按实验室默认规则授权现有门禁点。"""
    if not organization_id:
        return None
    now = now or datetime.now()
    organization = db.session.get(OrganizationUnit, int(organization_id))
    if organization is None or bool(organization.is_demo) != bool(user.is_demo):
        raise ValueError(u'实验室不存在或数据空间不一致')
    membership = LabMembership.query.filter_by(
        organization_id=organization.id, user_id=user.id).first()
    if membership is None:
        membership = LabMembership(
            organization=organization, user=user,
            role_title=(role_title or u'实验室成员').strip(),
            joined_at=joined_at or user.joined_at or date.today(),
            ends_at=ends_at, status='active')
        db.session.add(membership)
    else:
        membership.role_title = (role_title or membership.role_title or
                                 u'实验室成员').strip()
        membership.joined_at = joined_at or membership.joined_at
        membership.ends_at = ends_at
        membership.status = 'active'
        membership.ended_at = None

    if grant_all:
        for point in organization.access_points:
            if point.is_active:
                upsert_access_grant(user, point, ends_at=None, now=now)
    return membership


def upsert_access_grant(user, access_point, ends_at=None, source='manual',
                        now=None):
    now = now or datetime.now()
    if bool(access_point.is_demo) != bool(user.is_demo):
        raise ValueError(u'门禁点与人员不在同一数据空间')
    grant = AccessGrant.query.filter_by(
        user_id=user.id, access_point_id=access_point.id).first()
    if grant is None:
        grant = AccessGrant(user=user, access_point=access_point,
                            starts_at=now, source=source)
        db.session.add(grant)
    grant.status = 'active'
    grant.ends_at = ends_at
    grant.revoked_at = None
    grant.granted_at = now
    return grant


def change_person_status(user, target_status, summary=None, now=None):
    """暂停或恢复人员；权限同时冻结/恢复。"""
    if target_status not in ('active', 'suspended', 'offboarding'):
        raise ValueError(u'不支持的人员状态')
    now = now or datetime.now()
    previous = user.lifecycle_status or 'active'
    if previous == target_status:
        return None
    user.lifecycle_status = target_status
    if target_status == 'suspended':
        user.suspended_at = now
        for grant in user.access_grants:
            if grant.status == 'active':
                grant.status = 'suspended'
    elif target_status == 'active':
        user.suspended_at = None
        for grant in user.access_grants:
            if grant.status == 'suspended':
                grant.status = 'active'
    event = LifecycleEvent(
        user=user, event_type=target_status, from_status=previous,
        to_status=target_status,
        summary=summary or {
            'active': u'人员状态恢复，在有效授权范围内可正常通行',
            'suspended': u'人员权限已暂停，人脸模板保留但不进入识别人脸库',
            'offboarding': u'人员进入离组处理流程，等待隐私清理',
        }[target_status],
        is_demo=bool(user.is_demo), created_at=now)
    db.session.add(event)
    db.session.commit()
    return event


def evaluate_access(user, access_point_code=None, occurred_at=None):
    """联合人员状态、成员关系、区域和时间授权作出通行决定。"""
    occurred_at = occurred_at or datetime.now()
    if user.lifecycle_status != ACTIVE_PERSON_STATUS:
        return {'allowed': False, 'code': 'PERSON_INACTIVE',
                'message': u'人员当前不是在组状态，已拒绝通行',
                'access_point': None}
    if user.expected_leave_at and user.expected_leave_at < occurred_at.date():
        return {'allowed': False, 'code': 'MEMBERSHIP_EXPIRED',
                'message': u'人员有效期已结束，请联系管理员续期',
                'access_point': None}
    if not access_point_code:
        return {'allowed': True, 'code': 'ATTENDANCE_ONLY',
                'message': u'身份核验通过', 'access_point': None}

    point = AccessPoint.query.filter_by(code=access_point_code.strip().upper()).first()
    if point is None or not point.is_active:
        return {'allowed': False, 'code': 'ACCESS_POINT_UNAVAILABLE',
                'message': u'门禁点不存在或已停用', 'access_point': point}
    if bool(point.is_demo) != bool(user.is_demo):
        return {'allowed': False, 'code': 'DATA_SPACE_MISMATCH',
                'message': u'人员与门禁点不在同一数据空间', 'access_point': point}

    membership = LabMembership.query.filter_by(
        organization_id=point.organization_id, user_id=user.id,
        status='active').first()
    if (membership is None or
            (membership.ends_at and membership.ends_at < occurred_at.date())):
        return {'allowed': False, 'code': 'NO_ACTIVE_MEMBERSHIP',
                'message': u'当前不是该实验室的有效成员', 'access_point': point}

    grant = AccessGrant.query.filter_by(
        access_point_id=point.id, user_id=user.id, status='active').first()
    if grant is None:
        return {'allowed': False, 'code': 'NO_ACCESS_GRANT',
                'message': u'身份核验通过，但没有该区域通行权限',
                'access_point': point}
    if grant.starts_at > occurred_at or (grant.ends_at and grant.ends_at < occurred_at):
        return {'allowed': False, 'code': 'GRANT_OUTSIDE_WINDOW',
                'message': u'当前不在授权有效时间内', 'access_point': point}
    return {'allowed': True, 'code': 'ACCESS_GRANTED',
            'message': u'身份、活体与区域权限均已通过', 'access_point': point}


def record_access_event(user, recognition_event, decision, occurred_at=None):
    occurred_at = occurred_at or datetime.now()
    event = AccessEvent(
        user=user, access_point=decision.get('access_point'),
        recognition_event=recognition_event,
        decision='allowed' if decision['allowed'] else 'denied',
        reason_code=decision['code'], reason_text=decision['message'],
        is_demo=bool(user.is_demo), occurred_at=occurred_at)
    db.session.add(event)
    db.session.commit()
    return event


def offboard_and_erase(user, reason=None, now=None):
    """离组后即时撤权并物理删除人员与生物模板，留下匿名销毁凭证。"""
    now = now or datetime.now()
    if user.role and user.role.name == 'Admin':
        raise ValueError(u'不能离组或删除管理员账号')

    receipt = 'ER-%s-%s' % (now.strftime('%Y%m%d%H%M%S'),
                            secrets.token_hex(4).upper())
    templates_deleted = int(bool(
        user.face_encoding or user.face_encoding_sface or user.face_template))
    grants_revoked = sum(1 for item in user.access_grants
                         if item.status in ('active', 'suspended'))
    records_deleted = (
        SignIn.query.filter_by(user_id=user.id).count() +
        AttendanceRecord.query.filter_by(user_id=user.id).count() +
        CourseEnrollment.query.filter_by(user_id=user.id).count() +
        LabMembership.query.filter_by(user_id=user.id).count() +
        AccessGrant.query.filter_by(user_id=user.id).count()
    )

    # 审计算法性能与风险趋势可保留，但必须先解除与自然人的关联。
    RecognitionEvent.query.filter_by(user_id=user.id).update(
        {'user_id': None}, synchronize_session=False)
    RiskEvent.query.filter_by(user_id=user.id).update(
        {'user_id': None}, synchronize_session=False)
    AccessEvent.query.filter_by(user_id=user.id).update(
        {'user_id': None}, synchronize_session=False)
    LifecycleEvent.query.filter_by(user_id=user.id).update(
        {'user_id': None}, synchronize_session=False)

    task = ErasureTask(
        receipt_code=receipt, status='completed', trigger_type='offboarding',
        templates_deleted=templates_deleted, grants_revoked=grants_revoked,
        records_deleted=records_deleted, profile_action='physical_delete',
        message=(reason or u'离组触发自动清理：权限、人脸模板和人员档案已删除'),
        is_demo=bool(user.is_demo), requested_at=now, completed_at=now)
    db.session.add(task)
    db.session.delete(user)
    db.session.commit()
    return task


def ensure_demo_laboratory(now=None, users=None):
    """幂等创建匿名实验室演示空间，不写入任何人脸特征。"""
    now = now or datetime.now().replace(second=0, microsecond=0)
    lab = OrganizationUnit.query.filter_by(code='DEMO-LAB-AI').first()
    created = lab is None
    if lab is None:
        lab = OrganizationUnit(
            code='DEMO-LAB-AI', name=u'智能感知与视觉计算实验室',
            location=u'创新楼 A305', manager=u'演示负责人',
            description=u'面向计算机视觉、边缘智能与可信身份研究的演示实验室',
            is_demo=True, created_at=now - timedelta(days=180))
        db.session.add(lab)
        db.session.flush()
    points = []
    for code, name, location in (
            ('DEMO-A305-DOOR', u'A305 主入口', u'创新楼 A305'),
            ('DEMO-A305-EQUIP', u'设备区入口', u'A305 内部设备区')):
        point = AccessPoint.query.filter_by(code=code).first()
        if point is None:
            point = AccessPoint(organization=lab, code=code, name=name,
                                location=location, is_demo=True,
                                created_at=now - timedelta(days=120))
            db.session.add(point)
            db.session.flush()
        points.append(point)

    if users is None:
        users = User.query.filter_by(is_demo=True).all()
    for index, user in enumerate(users):
        user.person_type = user.person_type or 'student'
        user.lifecycle_status = user.lifecycle_status or 'active'
        user.joined_at = user.joined_at or (now.date() - timedelta(days=90))
        user.expected_leave_at = user.expected_leave_at or (
            now.date() + timedelta(days=180 + index))
        membership = LabMembership.query.filter_by(
            organization_id=lab.id, user_id=user.id).first()
        if membership is None:
            db.session.add(LabMembership(
                organization=lab, user=user, role_title=u'课题组成员',
                status='active', joined_at=user.joined_at,
                ends_at=user.expected_leave_at))
        # 主入口全员授权，设备区只授权部分成员，便于演示“识别成功但无权限”。
        upsert_access_grant(user, points[0], source='demo', now=now)
        if index % 3 != 0:
            upsert_access_grant(user, points[1], source='demo', now=now)
    db.session.commit()
    return {'created': created, 'organization_id': lab.id,
            'access_points': len(points)}


def build_lifecycle_summary(demo_mode=False, now=None):
    now = now or datetime.now()
    people = User.query.filter_by(is_demo=bool(demo_mode)).filter(
        User.role.has(name='Student'))
    active = people.filter_by(lifecycle_status='active').count()
    suspended = people.filter_by(lifecycle_status='suspended').count()
    expiring = people.filter(
        User.expected_leave_at.isnot(None),
        User.expected_leave_at >= now.date(),
        User.expected_leave_at <= now.date() + timedelta(days=30)).count()
    template_count = (FaceTemplate.query.filter_by(
        is_demo=bool(demo_mode), status='active').count())
    access_today = AccessEvent.query.filter_by(
        is_demo=bool(demo_mode)).filter(
        AccessEvent.occurred_at >= datetime.combine(now.date(), datetime.min.time())
    ).count()
    denied_today = AccessEvent.query.filter_by(
        is_demo=bool(demo_mode), decision='denied').filter(
        AccessEvent.occurred_at >= datetime.combine(now.date(), datetime.min.time())
    ).count()
    return {
        'active_people': active, 'suspended_people': suspended,
        'expiring_people': expiring, 'template_count': template_count,
        'laboratory_count': OrganizationUnit.query.filter_by(
            is_demo=bool(demo_mode), is_active=True).count(),
        'access_today': access_today, 'denied_today': denied_today,
        'erasure_count': ErasureTask.query.filter_by(
            is_demo=bool(demo_mode), status='completed').count(),
    }
