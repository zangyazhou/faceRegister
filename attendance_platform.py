# coding=utf-8
"""课程考勤、识别审计、风险分析与演示数据服务。"""
from collections import Counter
from datetime import date, datetime, time as date_time, timedelta
import math
import random

from sqlalchemy import func

from app import (AttendanceRecord, AttendanceSession, Course,
                 CourseEnrollment, RecognitionEvent, RiskEvent, Role,
                 SignIn, User, db)


PRESENT_STATUSES = ('on_time', 'late', 'manual')


def get_active_session(demo_mode=False, now=None):
    """返回当前模式下最近激活且尚未结束的考勤场次。"""
    now = now or datetime.now()
    return (AttendanceSession.query
            .filter_by(status='active', is_demo=bool(demo_mode))
            .filter(AttendanceSession.starts_at <= now,
                    AttendanceSession.ends_at >= now)
            .order_by(AttendanceSession.activated_at.desc(),
                      AttendanceSession.starts_at.desc())
            .first())


def record_recognition_decision(user_id=None, track_id=None, outcome='unknown',
                                engine='opencv', quality=None, similarity=None,
                                margin=None, passive_score=None, risk_score=None,
                                risk_level='low', latency_ms=None,
                                liveness_mode=None, occurred_at=None,
                                risk_type=None, risk_message=None,
                                risk_severity=None):
    """
    持久化一条轨迹的最终决策，并在存在活动场次时同步生成考勤结果。

    返回 ``(recognition_event, attendance_record, record_created)``。
    """
    occurred_at = occurred_at or datetime.now()
    user = db.session.get(User, user_id) if user_id else None
    is_demo = bool(user.is_demo) if user is not None else False
    active = get_active_session(is_demo, occurred_at)

    event = RecognitionEvent(
        session_id=active.id if active else None,
        user_id=user.id if user else None,
        track_id=str(track_id) if track_id is not None else None,
        outcome=outcome,
        engine=engine,
        quality=_finite_or_none(quality),
        similarity=_finite_or_none(similarity),
        margin=_finite_or_none(margin),
        passive_score=_finite_or_none(passive_score),
        risk_score=_finite_or_none(risk_score),
        risk_level=risk_level or 'low',
        latency_ms=_finite_or_none(latency_ms),
        liveness_mode=liveness_mode,
        is_demo=is_demo,
        created_at=occurred_at,
    )
    db.session.add(event)
    db.session.flush()

    attendance_record = None
    record_created = False
    if user is not None and active is not None and outcome in ('success', 'duplicate'):
        enrolled = CourseEnrollment.query.filter_by(
            course_id=active.course_id, user_id=user.id).first()
        if enrolled is not None:
            attendance_record = AttendanceRecord.query.filter_by(
                session_id=active.id, user_id=user.id).first()
            status = 'on_time' if occurred_at <= active.late_after else 'late'
            if attendance_record is None:
                attendance_record = AttendanceRecord(
                    session_id=active.id,
                    user_id=user.id,
                    checked_at=occurred_at,
                    status=status,
                    method='face',
                    recognition_event_id=event.id,
                    created_at=occurred_at,
                )
                db.session.add(attendance_record)
                record_created = True
            elif attendance_record.status == 'absent':
                attendance_record.checked_at = occurred_at
                attendance_record.status = status
                attendance_record.method = 'face'
                attendance_record.recognition_event_id = event.id
                record_created = True

    if risk_type:
        db.session.add(RiskEvent(
            recognition_event_id=event.id,
            session_id=active.id if active else None,
            user_id=user.id if user else None,
            event_type=risk_type,
            severity=risk_severity or risk_level or 'medium',
            message=risk_message or u'识别风险事件',
            is_demo=is_demo,
            created_at=occurred_at,
        ))

    db.session.commit()
    return event, attendance_record, record_created


def close_attendance_session(attendance_session, now=None):
    """关闭考勤场次，并为未到学生生成缺勤结果。"""
    now = now or datetime.now()
    existing_ids = {record.user_id for record in attendance_session.records}
    for enrollment in attendance_session.course.enrollments:
        if enrollment.user_id not in existing_ids:
            db.session.add(AttendanceRecord(
                session_id=attendance_session.id,
                user_id=enrollment.user_id,
                checked_at=None,
                status='absent',
                method='system',
                created_at=now,
            ))
    attendance_session.status = 'closed'
    attendance_session.closed_at = now
    db.session.commit()


def build_session_detail(attendance_session):
    """生成场次详情和导出共用的逐学生最终状态，保证页面与 CSV 对账。"""
    record_by_user = {record.user_id: record
                      for record in attendance_session.records}
    rows = []
    counts = Counter()
    enrollments = sorted(attendance_session.course.enrollments,
                         key=lambda item: item.user.number)
    for enrollment in enrollments:
        user = enrollment.user
        record = record_by_user.get(user.id)
        if record is None:
            status = ('absent' if attendance_session.status == 'closed'
                      else 'pending')
            event = None
        else:
            status = record.status
            event = record.recognition_event
        counts[status] += 1
        rows.append({
            'user_id': user.id,
            'number': user.number,
            'name': user.username,
            'status': status,
            'checked_at': (record.checked_at if record else None),
            'method': record.method if record else None,
            'event_id': event.id if event else None,
            'engine': event.engine if event else None,
            'quality': event.quality if event else None,
            'similarity': event.similarity if event else None,
            'passive_score': event.passive_score if event else None,
            'risk_score': event.risk_score if event else None,
            'latency_ms': event.latency_ms if event else None,
        })
    total = len(rows)
    present = sum(counts[name] for name in PRESENT_STATUSES)
    return {
        'rows': rows,
        'counts': dict(counts),
        'total': total,
        'present': present,
        'attendance_rate': round(present / total * 100, 1) if total else 0,
    }


def build_dashboard(demo_mode=False, now=None):
    """生成驾驶舱与实时识别中心共用的可序列化指标。"""
    demo_mode = bool(demo_mode)
    now = now or datetime.now()
    today = now.date()
    student_role = Role.query.filter_by(name='Student').first()
    students_query = User.query.filter_by(is_demo=demo_mode)
    if student_role:
        students_query = students_query.filter_by(role=student_role)
    student_count = students_query.count()

    active = get_active_session(demo_mode, now)
    status_counts = Counter()
    total = student_count
    if active:
        total = CourseEnrollment.query.filter_by(course_id=active.course_id).count()
        for status, count in (db.session.query(AttendanceRecord.status, func.count())
                              .filter(AttendanceRecord.session_id == active.id)
                              .group_by(AttendanceRecord.status).all()):
            status_counts[status] = int(count)
    elif not demo_mode:
        status_counts['on_time'] = (SignIn.query.join(User)
                                    .filter(SignIn.sign_date == today,
                                            User.is_demo.is_(False)).count())

    signed_count = sum(status_counts[s] for s in PRESENT_STATUSES)
    absent_count = max(0, total - signed_count)
    attendance_rate = round(signed_count / total * 100, 1) if total else 0.0

    trend = []
    latency_trend = []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        start = datetime.combine(day, date_time.min)
        end = start + timedelta(days=1)
        day_sessions = AttendanceSession.query.filter(
            AttendanceSession.is_demo == demo_mode,
            AttendanceSession.starts_at >= start,
            AttendanceSession.starts_at < end).all()
        session_ids = [item.id for item in day_sessions]
        if session_ids:
            present = AttendanceRecord.query.filter(
                AttendanceRecord.session_id.in_(session_ids),
                AttendanceRecord.status.in_(PRESENT_STATUSES)).count()
            possible = sum(len(item.course.enrollments) for item in day_sessions)
        elif not demo_mode:
            present = (SignIn.query.join(User)
                       .filter(SignIn.sign_date == day,
                               User.is_demo.is_(False)).count())
            possible = student_count
        else:
            present, possible = 0, 0
        trend.append({
            'date': day.strftime('%m-%d'),
            'weekday': u'周' + u'一二三四五六日'[day.weekday()],
            'count': present,
            'rate': round(present / possible * 100, 1) if possible else 0,
        })
        latencies = [row[0] for row in db.session.query(RecognitionEvent.latency_ms)
                     .filter(RecognitionEvent.is_demo == demo_mode,
                             RecognitionEvent.created_at >= start,
                             RecognitionEvent.created_at < end,
                             RecognitionEvent.latency_ms.isnot(None)).all()]
        latency_trend.append({
            'date': day.strftime('%m-%d'),
            'avg': round(sum(latencies) / len(latencies), 1) if latencies else 0,
        })

    window_start = now - timedelta(days=30)
    event_query = RecognitionEvent.query.filter(
        RecognitionEvent.is_demo == demo_mode,
        RecognitionEvent.created_at >= window_start)
    latencies = [row[0] for row in db.session.query(RecognitionEvent.latency_ms)
                 .filter(RecognitionEvent.is_demo == demo_mode,
                         RecognitionEvent.created_at >= window_start,
                         RecognitionEvent.latency_ms.isnot(None)).all()]
    latencies.sort()
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
    p95_latency = round(_percentile(latencies, 0.95), 1) if latencies else 0
    recent_events = [_serialize_recognition_event(item) for item in
                     event_query.order_by(RecognitionEvent.created_at.desc()).limit(10)]

    risk_query = RiskEvent.query.filter(
        RiskEvent.is_demo == demo_mode,
        RiskEvent.created_at >= window_start)
    risks = [_serialize_risk_event(item) for item in
             risk_query.order_by(RiskEvent.created_at.desc()).limit(8)]
    risk_count_today = risk_query.filter(RiskEvent.created_at >=
                                         datetime.combine(today, date_time.min)).count()

    hour_distribution = [0] * 24
    weekday_heatmap = [[hour, weekday, 0] for weekday in range(7)
                       for hour in range(6, 23)]
    heat_lookup = {(item[0], item[1]): item for item in weekday_heatmap}
    success_events = event_query.filter(
        RecognitionEvent.outcome.in_(('success', 'duplicate'))).all()
    for event in success_events:
        hour_distribution[event.created_at.hour] += 1
        key = (event.created_at.hour, event.created_at.weekday())
        if key in heat_lookup:
            heat_lookup[key][2] += 1

    outcome_rows = (db.session.query(RecognitionEvent.outcome, func.count())
                    .filter(RecognitionEvent.is_demo == demo_mode,
                            RecognitionEvent.created_at >= window_start)
                    .group_by(RecognitionEvent.outcome).all())
    outcome_counts = {name: int(count) for name, count in outcome_rows}
    engine_rows = (db.session.query(RecognitionEvent.engine, func.count())
                   .filter(RecognitionEvent.is_demo == demo_mode,
                           RecognitionEvent.created_at >= window_start)
                   .group_by(RecognitionEvent.engine).all())

    return {
        'demo_mode': demo_mode,
        'generated_at': now.strftime('%Y-%m-%d %H:%M:%S'),
        'student_count': student_count,
        'total': total,
        'signed_count': signed_count,
        'late_count': status_counts['late'],
        'absent_count': absent_count,
        'attendance_rate': attendance_rate,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'risk_count_today': risk_count_today,
        'trend': trend,
        'latency_trend': latency_trend,
        'hour_distribution': hour_distribution,
        'heatmap': weekday_heatmap,
        'outcome_counts': outcome_counts,
        'engines': [{'name': name, 'count': int(count)}
                    for name, count in engine_rows],
        'recent_events': recent_events,
        'recent_risks': risks,
        'active_session': _serialize_session(active) if active else None,
    }


def seed_demo_data(now=None):
    """幂等地创建隔离演示数据；不生成任何人脸特征。"""
    now = now or datetime.now().replace(second=0, microsecond=0)
    existing = Course.query.filter_by(code='DEMO-CV-01').first()
    if existing:
        return {'created': False, 'course_id': existing.id}

    student_role = Role.query.filter_by(name='Student').first()
    if student_role is None:
        Role.insert_roles()
        student_role = Role.query.filter_by(name='Student').first()

    students = []
    surnames = [u'陈', u'林', u'黄', u'周', u'吴', u'徐', u'孙']
    given = [u'晨', u'宇', u'思远', u'嘉宁', u'子涵', u'雨桐', u'浩然']
    for index in range(1, 29):
        number = 'B99%09d' % index
        student = User.query.filter_by(number=number).first()
        if student is None:
            student = User(number=number,
                           username=surnames[(index - 1) % len(surnames)] +
                           given[(index * 3) % len(given)],
                           password='!', role=student_role, is_demo=True)
            db.session.add(student)
        students.append(student)
    db.session.flush()

    course = Course(code='DEMO-CV-01', name=u'计算机视觉与智能感知',
                    teacher=u'李老师', classroom=u'人工智能实验室 A305',
                    is_demo=True, created_at=now - timedelta(days=60))
    db.session.add(course)
    db.session.flush()
    for student in students:
        db.session.add(CourseEnrollment(course_id=course.id, user_id=student.id,
                                        enrolled_at=now - timedelta(days=45)))
    db.session.flush()

    rng = random.Random(20260820)
    sessions = []
    for offset in range(13, -1, -1):
        day = now.date() - timedelta(days=offset)
        starts = datetime.combine(day, date_time(8, 30))
        if offset == 0:
            starts = now - timedelta(minutes=18)
        late_after = starts + timedelta(minutes=10)
        ends = starts + timedelta(minutes=55)
        item = AttendanceSession(
            course_id=course.id,
            title=u'第 %02d 次课堂考勤' % (14 - offset),
            starts_at=starts,
            late_after=late_after,
            ends_at=ends,
            status='active' if offset == 0 else 'closed',
            is_demo=True,
            created_at=starts - timedelta(days=1),
            activated_at=starts,
            closed_at=None if offset == 0 else ends,
        )
        db.session.add(item)
        sessions.append(item)
    db.session.flush()

    for day_index, attendance_session in enumerate(sessions):
        present_count = 22 + (day_index * 3) % 7
        if day_index == len(sessions) - 1:
            present_count = 23
        selected = students[:]
        rng.shuffle(selected)
        selected_ids = {item.id for item in selected[:present_count]}
        for student_index, student in enumerate(students):
            if student.id not in selected_ids:
                if attendance_session.status == 'closed':
                    db.session.add(AttendanceRecord(
                        session_id=attendance_session.id, user_id=student.id,
                        status='absent', method='system', checked_at=None,
                        created_at=attendance_session.ends_at))
                continue
            delay = rng.randint(1, 16)
            checked_at = attendance_session.starts_at + timedelta(minutes=delay)
            status = ('on_time' if checked_at <= attendance_session.late_after
                      else 'late')
            event = RecognitionEvent(
                session_id=attendance_session.id, user_id=student.id,
                track_id=str(student_index + 1), outcome='success',
                engine='opencv-yunet-sface', quality=round(rng.uniform(.61, .94), 3),
                similarity=round(rng.uniform(.58, .89), 3),
                margin=round(rng.uniform(.12, .41), 3),
                passive_score=round(rng.uniform(.82, .98), 3),
                risk_score=round(rng.uniform(.02, .19), 3), risk_level='low',
                latency_ms=round(rng.uniform(72, 168), 1),
                liveness_mode='passive', is_demo=True, created_at=checked_at)
            db.session.add(event)
            db.session.flush()
            db.session.add(AttendanceRecord(
                session_id=attendance_session.id, user_id=student.id,
                checked_at=checked_at, status=status, method='face',
                recognition_event_id=event.id, created_at=checked_at))

        # 每隔几堂课加入一条被拦截的照片攻击，专供风险页展示。
        if day_index % 3 == 1 or day_index == len(sessions) - 1:
            event_time = attendance_session.starts_at + timedelta(minutes=4)
            event = RecognitionEvent(
                session_id=attendance_session.id, user_id=None,
                track_id='risk-%s' % day_index, outcome='rejected',
                engine='opencv-yunet-sface', quality=.78, similarity=.66,
                margin=.18, passive_score=.06, risk_score=.91,
                risk_level='high', latency_ms=116.4,
                liveness_mode='rejected', is_demo=True, created_at=event_time)
            db.session.add(event)
            db.session.flush()
            db.session.add(RiskEvent(
                recognition_event_id=event.id, session_id=attendance_session.id,
                event_type='spoof_attack', severity='high',
                message=u'静默活体判定为疑似照片/屏幕翻拍，已阻断签到',
                is_demo=True, created_at=event_time))

    db.session.commit()
    return {'created': True, 'course_id': course.id,
            'students': len(students), 'sessions': len(sessions)}


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values, fraction):
    if not values:
        return 0
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _serialize_session(item):
    return {
        'id': item.id,
        'title': item.title,
        'course': item.course.name,
        'course_code': item.course.code,
        'teacher': item.course.teacher,
        'classroom': item.course.classroom,
        'starts_at': item.starts_at.strftime('%Y-%m-%d %H:%M'),
        'late_after': item.late_after.strftime('%H:%M'),
        'ends_at': item.ends_at.strftime('%H:%M'),
        'status': item.status,
        'enrolled': len(item.course.enrollments),
    }


def _serialize_recognition_event(item):
    return {
        'id': item.id,
        'time': item.created_at.strftime('%H:%M:%S'),
        'date': item.created_at.strftime('%Y-%m-%d'),
        'name': item.user.username if item.user else u'未知访客',
        'number': item.user.number if item.user else '--',
        'outcome': item.outcome,
        'engine': item.engine,
        'quality': item.quality,
        'similarity': item.similarity,
        'margin': item.margin,
        'passive_score': item.passive_score,
        'risk_score': item.risk_score,
        'risk_level': item.risk_level,
        'latency_ms': item.latency_ms,
        'liveness_mode': item.liveness_mode,
    }


def _serialize_risk_event(item):
    return {
        'id': item.id,
        'time': item.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'name': item.user.username if item.user else u'未知访客',
        'type': item.event_type,
        'severity': item.severity,
        'message': item.message,
        'course': (item.attendance_session.course.name
                   if item.attendance_session else u'无活动场次'),
    }
