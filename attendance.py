# coding=utf-8
"""
签到写入助手：供 Web 手动补签与 test.py 摄像头识别共用。
同一学生同一天只允许一条签到记录（sign_ins 表唯一约束兜底）。
"""
from datetime import datetime, date

from sqlalchemy.exc import IntegrityError

from app import db, SignIn


def record_sign_in(user_id, sign_time=None):
    """
    记录一次签到。

    :param user_id: 学生的 users.id
    :param sign_time: 签到时间（datetime）；传 date 对象时视为该日期的 00:00；缺省为当前本地时间
    :return: 新建的 SignIn 记录；若该学生同日已签到（或参数非法）则返回 None
    """
    if sign_time is None:
        sign_time = datetime.now()
    elif isinstance(sign_time, date) and not isinstance(sign_time, datetime):
        # 手动补签只指定日期时，签到时刻记为当日 00:00
        sign_time = datetime.combine(sign_time, datetime.min.time())

    record = SignIn(user_id=user_id, sign_time=sign_time, sign_date=sign_time.date())
    db.session.add(record)
    try:
        db.session.commit()
        return record
    except IntegrityError as e:
        db.session.rollback()
        # 仅当触发的是 (user_id, sign_date) 唯一约束时才视为「同日重复签到」：
        # SQLite 报错不含约束名，只报冲突列，按特征判断。
        # 其它完整性错误（外键失败、NOT NULL 等）属于真实问题，应抛出而不是静默吞掉。
        msg = str(e.orig)
        if 'UNIQUE constraint failed' in msg and 'sign_ins.user_id' in msg:
            return None
        raise
