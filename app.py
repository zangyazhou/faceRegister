# coding=utf-8
import csv
import functools
import io
import os
from datetime import datetime, date, timedelta

import click
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, \
    url_for, flash, current_app, request, jsonify, session as flask_session, \
    Response, abort
from flask_login import UserMixin, LoginManager, login_required, \
    login_user, logout_user, current_user
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_wtf.file import FileField, FileAllowed
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import StringField, PasswordField, SubmitField, SelectField, \
    BooleanField, IntegerField, DateField, ValidationError
from wtforms.fields import DateTimeLocalField
from wtforms.validators import DataRequired, InputRequired, Length, Regexp, Optional

from functions import get_face_encoding_in_image, parse_face_encoding

'''
Config
'''
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv()  # 从 .env 读取 SECRET_KEY / ADMIN_PASSWORD / DATABASE_URL 等配置

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(32)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or \
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
# 初始管理员密码仅从本机环境读取，避免源码仓库携带公开默认密码。
app.config['AdminPassword'] = os.environ.get('ADMIN_PASSWORD')
# Web 人脸签到：后端 dlib 匹配、多人轨迹与服务端活体检测参数。
app.config['FACE_MATCH_TOLERANCE'] = float(os.environ.get('FACE_MATCH_TOLERANCE', '0.55'))
app.config['FACE_MATCH_VOTES'] = int(os.environ.get('FACE_MATCH_VOTES', '3'))
app.config['FACE_MATCH_WINDOW'] = float(os.environ.get('FACE_MATCH_WINDOW', '6.0'))
app.config['FACE_TRACK_TTL'] = float(os.environ.get('FACE_TRACK_TTL', '2.0'))
app.config['FACE_ENGINE'] = os.environ.get('FACE_ENGINE', 'auto').strip().lower()
app.config['SFACE_COSINE_THRESHOLD'] = float(os.environ.get(
    'SFACE_COSINE_THRESHOLD', '0.40'))
app.config['SFACE_MINIMUM_MARGIN'] = float(os.environ.get(
    'SFACE_MINIMUM_MARGIN', '0.08'))
app.config['FACE_MIN_QUALITY'] = float(os.environ.get('FACE_MIN_QUALITY', '0.38'))
app.config['LIVENESS_REQUIRED'] = os.environ.get(
    'LIVENESS_REQUIRED', 'true').strip().lower() not in ('0', 'false', 'no', 'off')
# passive_adaptive：静默活体优先，不确定时随机要求一个动作；
# single_action：总是随机一个动作；legacy：眨眼和转头都要求。
app.config['LIVENESS_POLICY'] = os.environ.get(
    'LIVENESS_POLICY', 'passive_adaptive').strip().lower()
app.config['LIVENESS_WINDOW'] = float(os.environ.get('LIVENESS_WINDOW', '8.0'))
app.config['LIVENESS_EAR_CLOSED'] = float(os.environ.get('LIVENESS_EAR_CLOSED', '0.20'))
app.config['LIVENESS_EAR_OPEN'] = float(os.environ.get('LIVENESS_EAR_OPEN', '0.24'))
app.config['LIVENESS_HEAD_DELTA'] = float(os.environ.get('LIVENESS_HEAD_DELTA', '0.12'))
app.config['ANTI_SPOOF_ENABLED'] = os.environ.get(
    'ANTI_SPOOF_ENABLED', 'true').strip().lower() not in ('0', 'false', 'no', 'off')
app.config['LIVENESS_PASSIVE_MIN_FRAMES'] = int(os.environ.get(
    'LIVENESS_PASSIVE_MIN_FRAMES', '2'))
app.config['LIVENESS_PASSIVE_ACCEPT'] = float(os.environ.get(
    'LIVENESS_PASSIVE_ACCEPT', '0.80'))
app.config['LIVENESS_PASSIVE_REJECT'] = float(os.environ.get(
    'LIVENESS_PASSIVE_REJECT', '0.15'))
app.config['LIVENESS_IDENTITY_STRONG'] = float(os.environ.get(
    'LIVENESS_IDENTITY_STRONG', '0.55'))
app.config['LIVENESS_QUALITY_STRONG'] = float(os.environ.get(
    'LIVENESS_QUALITY_STRONG', '0.55'))
db = SQLAlchemy(app)
csrf = CSRFProtect(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)

# 注：Flask-Login 0.6 已移除 session_protection 属性（0.5 起弃用），
# 此处不再设置，避免无效代码误导。
login_manager.login_view = 'login'
login_manager.login_message = u"你需要登录才能访问这个页面."

'''
Models
'''


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    users = db.relationship('User', backref='role', lazy='dynamic')

    @staticmethod
    def insert_roles():
        roles = ('Student', 'Admin')
        for r in roles:
            role = Role.query.filter_by(name=r).first()
            if role is None:
                role = Role(name=r)
            db.session.add(role)
        db.session.commit()

    def __repr__(self):
        return '<Role %r>' % self.name


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    # 学号：B 开头 + 11 位数字（如 B20240304323），存为字符串
    number = db.Column(db.String(20), unique=True, index=True)
    username = db.Column(db.String(64), index=True)
    # 密码以哈希形式存储（默认空密码哈希）
    password = db.Column(db.String(256))
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    # 人脸特征编码：128 维浮点数数组的字符串表示，约 2300+ 字符
    face_encoding = db.Column(db.String(5000))
    # OpenCV SFace 的归一化 128 维特征；与 dlib 特征空间不同，迁移期并存。
    face_encoding_sface = db.Column(db.Text)
    # 演示学生与真实学生严格隔离；演示账号永远不写入生物特征。
    is_demo = db.Column(db.Boolean, nullable=False, default=False, index=True)
    sign_ins = db.relationship('SignIn', backref='user', lazy='dynamic',
                               cascade='all, delete-orphan')

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        # 新添加的用户，初始其角色为学生。
        if self.role is None:
            self.role = Role.query.filter_by(name='Student').first()

    def __repr__(self):
        return '<User %r>' % self.username

    # 初次运行程序时生成初始管理员的静态方法
    @staticmethod
    def generate_admin():
        admin = Role.query.filter_by(name='Admin').first()
        u = User.query.filter_by(role=admin).first()
        if u is None:
            initial_password = current_app.config.get('AdminPassword')
            if not initial_password:
                raise click.ClickException(
                    u'首次初始化前请在 .env 中设置 ADMIN_PASSWORD')
            u = User(number='0', username='Admin', \
                     password=generate_password_hash(initial_password), \
                     role=Role.query.filter_by(name='Admin').first())
            db.session.add(u)
            db.session.commit()

    def verify_password(self, password):
        """
        校验密码。兼容历史明文密码：比对通过后自动升级为哈希存储。
        """
        if self.password and self.password.startswith(('pbkdf2:', 'scrypt:', 'sha256:')):
            return check_password_hash(self.password, password)
        # 历史明文密码：直接比对，通过则惰性升级为哈希
        ok = (self.password == password)
        if ok:
            self.password = generate_password_hash(password)
            db.session.add(self)
            db.session.commit()
        return ok


class SignIn(db.Model):
    """
    签到记录：同一学生同一天最多一条记录（数据库唯一约束兜底，防止重复签到）。
    时间统一使用本地时间（datetime.now / date.today），保证 sign_time 与
    sign_date 的日期归属一致——若用 UTC，中国时区（UTC+8）在 0:00–7:59 的
    签到会被记到前一天。
    """
    __tablename__ = 'sign_ins'
    __table_args__ = (
        db.UniqueConstraint('user_id', 'sign_date', name='uq_sign_ins_user_date'),
    )
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    sign_time = db.Column(db.DateTime, nullable=False, default=datetime.now)
    sign_date = db.Column(db.Date, nullable=False, default=date.today, index=True)

    def __repr__(self):
        return '<SignIn user=%s date=%s>' % (self.user_id, self.sign_date)


class Course(db.Model):
    """课程/班级，是新版考勤业务的最上层实体。"""
    __tablename__ = 'courses'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(32), nullable=False, unique=True, index=True)
    name = db.Column(db.String(128), nullable=False)
    teacher = db.Column(db.String(64), nullable=False)
    classroom = db.Column(db.String(64))
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_demo = db.Column(db.Boolean, nullable=False, default=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    enrollments = db.relationship('CourseEnrollment', back_populates='course',
                                  cascade='all, delete-orphan')
    sessions = db.relationship('AttendanceSession', back_populates='course',
                               cascade='all, delete-orphan')


class CourseEnrollment(db.Model):
    """学生与课程的多对多选课关系。"""
    __tablename__ = 'course_enrollments'
    __table_args__ = (
        db.UniqueConstraint('course_id', 'user_id', name='uq_course_enrollment'),
    )
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False,
                          index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False,
                        index=True)
    enrolled_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    course = db.relationship('Course', back_populates='enrollments')
    user = db.relationship('User', backref=db.backref(
        'course_enrollments', cascade='all, delete-orphan'))


class AttendanceSession(db.Model):
    """一次有明确开始、迟到阈值和结束时间的考勤任务。"""
    __tablename__ = 'attendance_sessions'
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False,
                          index=True)
    title = db.Column(db.String(128), nullable=False)
    starts_at = db.Column(db.DateTime, nullable=False)
    late_after = db.Column(db.DateTime, nullable=False)
    ends_at = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(16), nullable=False, default='scheduled', index=True)
    is_demo = db.Column(db.Boolean, nullable=False, default=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    activated_at = db.Column(db.DateTime)
    closed_at = db.Column(db.DateTime)
    course = db.relationship('Course', back_populates='sessions')
    records = db.relationship('AttendanceRecord', back_populates='attendance_session',
                              cascade='all, delete-orphan')
    recognition_events = db.relationship('RecognitionEvent',
                                         back_populates='attendance_session')


class AttendanceRecord(db.Model):
    """课程场次级考勤结果；同一学生在同一场次只有一个最终状态。"""
    __tablename__ = 'attendance_records'
    __table_args__ = (
        db.UniqueConstraint('session_id', 'user_id', name='uq_attendance_record'),
    )
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_sessions.id'),
                           nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False,
                        index=True)
    checked_at = db.Column(db.DateTime)
    status = db.Column(db.String(16), nullable=False, index=True)
    method = db.Column(db.String(16), nullable=False, default='face')
    recognition_event_id = db.Column(db.Integer, db.ForeignKey(
        'recognition_events.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    attendance_session = db.relationship('AttendanceSession', back_populates='records')
    recognition_event = db.relationship('RecognitionEvent',
                                        foreign_keys=[recognition_event_id])
    user = db.relationship('User', backref=db.backref(
        'attendance_records', cascade='all, delete-orphan'))


class RecognitionEvent(db.Model):
    """一条轨迹的一次最终识别决策，而不是每一帧的原始日志。"""
    __tablename__ = 'recognition_events'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_sessions.id'),
                           nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True,
                        index=True)
    track_id = db.Column(db.String(32))
    outcome = db.Column(db.String(24), nullable=False, index=True)
    engine = db.Column(db.String(32), nullable=False)
    quality = db.Column(db.Float)
    similarity = db.Column(db.Float)
    margin = db.Column(db.Float)
    passive_score = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    risk_level = db.Column(db.String(16), nullable=False, default='low', index=True)
    latency_ms = db.Column(db.Float)
    liveness_mode = db.Column(db.String(24))
    is_demo = db.Column(db.Boolean, nullable=False, default=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now,
                           index=True)
    attendance_session = db.relationship('AttendanceSession',
                                         back_populates='recognition_events')
    user = db.relationship('User', backref='recognition_events')
    risk_events = db.relationship('RiskEvent', back_populates='recognition_event',
                                  cascade='all, delete-orphan')


class RiskEvent(db.Model):
    """由活体、身份歧义或系统异常产生的可审计风险事件。"""
    __tablename__ = 'risk_events'
    id = db.Column(db.Integer, primary_key=True)
    recognition_event_id = db.Column(db.Integer, db.ForeignKey(
        'recognition_events.id'), nullable=True, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_sessions.id'),
                           nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True,
                        index=True)
    event_type = db.Column(db.String(32), nullable=False, index=True)
    severity = db.Column(db.String(16), nullable=False, index=True)
    message = db.Column(db.String(255), nullable=False)
    is_demo = db.Column(db.Boolean, nullable=False, default=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now,
                           index=True)
    recognition_event = db.relationship('RecognitionEvent',
                                        back_populates='risk_events')
    attendance_session = db.relationship('AttendanceSession', backref='risk_events')
    user = db.relationship('User', backref='risk_events')


'''
Forms
'''


class LoginForm(FlaskForm):
    # 仅管理员登录，账号即用户名（如 Admin），无需学号
    username = StringField(u'账号', validators=[DataRequired(message=u'请输入账号')])
    password = PasswordField(u'密码', validators=[DataRequired()])
    remember_me = BooleanField(u'记住我')
    submit = SubmitField(u'登录')


class SearchForm(FlaskForm):
    number = StringField(u'学号', validators=[DataRequired(message=u'请输入学号')])
    submit = SubmitField(u'搜索')


class UserForm(FlaskForm):
    username = StringField(u'姓名', validators=[DataRequired()])
    number = StringField(u'学号', validators=[DataRequired(message=u'请输入学号'),
                                              Regexp(r'^B\d{11}$', message=u'学号格式应为 B 开头 + 11 位数字')])
    file = FileField('面部照片', validators=[FileAllowed(['jpg', 'jpeg', 'png', 'gif'], u'仅支持图片文件')])
    submit = SubmitField(u'添加')

    def validate_number(self, field):
        if User.query.filter_by(number=field.data).first():
            raise ValidationError(u'此学生已存在，请检查学号！')


class EditForm(FlaskForm):
    username = StringField(u'姓名', validators=[DataRequired()])
    number = StringField(u'学号', validators=[DataRequired(message=u'请输入学号'),
                                              Regexp(r'^B\d{11}$', message=u'学号格式应为 B 开头 + 11 位数字')])
    # 密码留空表示不修改
    password = PasswordField(u'新密码（留空则不修改）', validators=[Optional(), Length(1, 64), \
                                                              Regexp('^[a-zA-Z0-9_.]*$', 0, \
                                                                     u'密码由字母、数字和_.组成')])
    file = FileField(u'重新录入人脸（可选）',
                     validators=[FileAllowed(['jpg', 'jpeg', 'png', 'gif'],
                                             u'仅支持图片文件')])
    role = SelectField(u'身份', coerce=int)
    submit = SubmitField(u'修改')

    def __init__(self, user, *args, **kwargs):
        super(EditForm, self).__init__(*args, **kwargs)
        self.role.choices = [(role.id, role.name)
                             for role in Role.query.order_by(Role.name).all()]
        self.user = user

    def validate_number(self, field):
        if field.data != self.user.number and \
                User.query.filter_by(number=field.data).first():
            raise ValidationError(u'此学生已存在，请检查学号！')


class ManualSignInForm(FlaskForm):
    user_id = SelectField(u'学生', coerce=int, validators=[DataRequired()])
    sign_date = DateField(u'签到日期', default=date.today, validators=[DataRequired()])
    submit = SubmitField(u'补签')

    def __init__(self, *args, **kwargs):
        super(ManualSignInForm, self).__init__(*args, **kwargs)
        student_role = Role.query.filter_by(name='Student').first()
        self.user_id.choices = [(u.id, u'%s（%s）' % (u.username, u.number))
                                for u in User.query.filter_by(role=student_role)
                                .order_by(User.number).all()]


class CourseForm(FlaskForm):
    code = StringField(u'课程编号', validators=[DataRequired(), Length(2, 32)])
    name = StringField(u'课程名称', validators=[DataRequired(), Length(2, 128)])
    teacher = StringField(u'授课教师', validators=[DataRequired(), Length(2, 64)])
    classroom = StringField(u'教室', validators=[Optional(), Length(0, 64)])
    submit = SubmitField(u'创建课程')

    def validate_code(self, field):
        if Course.query.filter_by(code=field.data.strip()).first():
            raise ValidationError(u'课程编号已存在')


class AttendanceSessionForm(FlaskForm):
    course_id = SelectField(u'课程', coerce=int, validators=[DataRequired()])
    title = StringField(u'任务名称', validators=[DataRequired(), Length(2, 128)])
    starts_at = DateTimeLocalField(u'开始时间', format='%Y-%m-%dT%H:%M',
                                   validators=[DataRequired()])
    late_after = DateTimeLocalField(u'迟到时间', format='%Y-%m-%dT%H:%M',
                                    validators=[DataRequired()])
    ends_at = DateTimeLocalField(u'结束时间', format='%Y-%m-%dT%H:%M',
                                 validators=[DataRequired()])
    submit = SubmitField(u'创建考勤任务')

    def __init__(self, demo_mode=False, *args, **kwargs):
        super(AttendanceSessionForm, self).__init__(*args, **kwargs)
        self.course_id.choices = [
            (course.id, u'%s · %s' % (course.code, course.name))
            for course in Course.query.filter_by(is_demo=demo_mode, is_active=True)
            .order_by(Course.created_at.desc()).all()
        ]

    def validate(self, extra_validators=None):
        valid = super().validate(extra_validators)
        if not valid:
            return False
        if not self.starts_at.data < self.late_after.data < self.ends_at.data:
            self.ends_at.errors.append(u'时间应满足：开始 < 迟到阈值 < 结束')
            return False
        return True


'''
权限装饰器
'''


def admin_required(f):
    """要求管理员身份；未登录先走登录流程，普通学生无权执行"""
    @functools.wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        admin = Role.query.filter_by(name='Admin').first()
        if current_user.role != admin:
            flash(u'该操作仅管理员可用！', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


def _csv_metric(value, digits=3):
    """CSV 指标格式化；缺失值保持空白，避免被误读为 0。"""
    if value is None:
        return ''
    return ('%%.%df' % digits) % float(value)


'''
views
'''


@app.route('/', methods=['GET'])
@login_required
def index():
    from attendance_platform import build_dashboard
    dashboard = build_dashboard(bool(flask_session.get('demo_mode', False)))
    return render_template('dashboard.html', dashboard=dashboard)


@app.route('/students', methods=['GET', 'POST'])
@login_required
def students():
    form = SearchForm()
    demo_mode = bool(flask_session.get('demo_mode', False))
    student_role = Role.query.filter_by(name='Student').first()
    query = User.query.filter_by(role=student_role, is_demo=demo_mode)
    if form.validate_on_submit():
        # 获得学生列表，其学号包含form中的数字
        student_rows = query.filter(User.number.like(
            '%{}%'.format(form.number.data))).order_by(User.number).all()
    else:
        student_rows = query.order_by(User.number.asc()).all()
    enrolled_count = sum(1 for item in student_rows if item.face_encoding_sface)
    return render_template('index.html', form=form, students=student_rows,
                           enrolled_count=enrolled_count,
                           demo_mode=demo_mode)


# 增加新学生
@app.route('/add-user', methods=['GET', 'POST'])
@admin_required
def add_user():
    form = UserForm()
    if form.validate_on_submit():
        try:
            result = get_face_encoding_in_image(form.file.data)
        except OSError:
            flash(u'文件格式有误', 'danger')
            return render_template('add_user.html', form=form)
        if result["face_found_in_image"] == False:
            flash(u'未检测到人脸，添加失败！', 'danger')
            return render_template('add_user.html', form=form)
        elif result.get("face_count", 1) > 1:
            flash(u'照片中包含多张人脸，请上传单人照片！', 'danger')
            return render_template('add_user.html', form=form)
        else:
            user = User(username=form.username.data,
                        number=form.number.data,
                        face_encoding=result["unknown_face_encodings"],
                        face_encoding_sface=result.get("sface_encoding"),
                        password=generate_password_hash('123456'))
            db.session.add(user)
            db.session.commit()
            flash(u'成功添加学生（初始密码：123456）', 'success')
            return redirect(url_for('students'))
    return render_template('add_user.html', form=form)


# 删除学生（POST，带 CSRF 保护）
@app.route('/remove-user/<int:id>', methods=['POST'])
@admin_required
def remove_user(id):
    user = db.get_or_404(User, id)
    if user.role == Role.query.filter_by(name='Admin').first():
        flash(u'不能删除管理员', 'warning')
    else:
        db.session.delete(user)
        db.session.commit()
        flash(u'成功删除此学生', 'success')
    return redirect(url_for('students'))


# 修改学生资料
@app.route('/edit-user/<int:id>', methods=['GET', 'POST'])
@admin_required
def edit_user(id):
    user = db.get_or_404(User, id)
    form = EditForm(user=user)
    if form.validate_on_submit():
        user.username = form.username.data
        user.number = form.number.data
        if form.file.data:
            try:
                result = get_face_encoding_in_image(form.file.data)
            except OSError:
                flash(u'文件格式有误', 'danger')
                return render_template('edit_user.html', form=form, user=user)
            if not result['face_found_in_image']:
                flash(u'未检测到人脸，人脸信息未更新', 'danger')
                return render_template('edit_user.html', form=form, user=user)
            if result.get('face_count', 1) != 1:
                flash(u'照片中必须且只能包含一张人脸', 'danger')
                return render_template('edit_user.html', form=form, user=user)
            user.face_encoding = result['unknown_face_encodings']
            user.face_encoding_sface = result.get('sface_encoding')
        # 密码留空则不修改，避免覆盖已有哈希
        if form.password.data:
            user.password = generate_password_hash(form.password.data)
        user.role = db.session.get(Role, form.role.data)
        db.session.add(user)
        db.session.commit()
        flash(u'个人信息已更改', 'success')
        return redirect(url_for('students'))
    form.username.data = user.username
    form.number.data = user.number
    # 不回显密码
    form.role.data = user.role_id
    return render_template('edit_user.html', form=form, user=user)


# 登录，系统只允许管理员登录
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is not None and user.verify_password(form.password.data):
            if user.role != Role.query.filter_by(name='Admin').first():
                flash(u'系统只对管理员开放，请联系管理员获得权限！', 'warning')
            else:
                login_user(user, form.remember_me.data)
                return redirect(url_for('index'))
        flash(u'账号或密码错误！', 'danger')
    return render_template('login.html', form=form)


# 注销改为 POST：避免 GET 注销被第三方页面（如 <img src="/logout">）CSRF 强制登出
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    flash(u'成功注销！', 'success')
    return redirect(url_for('login'))


# 签到记录列表（支持按日期/学号筛选）
@app.route('/sign-ins', methods=['GET'])
@login_required
def sign_ins():
    date_str = request.args.get('date', '')
    number = request.args.get('number', '').strip()
    query = SignIn.query.join(User).order_by(SignIn.sign_time.desc())
    if date_str:
        try:
            date_parsed = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            date_parsed = None
        if date_parsed:
            query = query.filter(SignIn.sign_date == date_parsed)
    if number:
        query = query.filter(User.number == number)
    records = query.all()
    return render_template('sign_ins.html', records=records,
                           date_str=date_str, number=number)


# 考勤统计：查看某日全体学生的签到/未签到情况
@app.route('/sign-ins/statistics', methods=['GET'])
@login_required
def sign_ins_statistics():
    date_str = request.args.get('date', '')
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date() \
            if date_str else date.today()
    except ValueError:
        target_date = date.today()
    admin = Role.query.filter_by(name='Admin').first()
    students = User.query.filter(User.role != admin).order_by(User.number).all()
    signed_ids = {r.user_id for r in
                  SignIn.query.filter_by(sign_date=target_date).all()}
    for s in students:
        s.is_signed = s.id in signed_ids
    # 出勤率（百分比，保留 1 位小数）
    attendance_rate = round(len(signed_ids) / len(students) * 100, 1) if students else 0.0
    # 近 7 日出勤趋势（含今天，往前 6 天）
    trend = []
    for offset in range(6, -1, -1):
        d = date.today() - timedelta(days=offset)
        cnt = SignIn.query.filter_by(sign_date=d).count()
        trend.append({'date': d.strftime('%m-%d'),
                      'weekday': '周' + '一二三四五六日'[d.weekday()],
                      'count': cnt})
    # 补签表单的默认日期跟随当前查看的统计日期，避免误签其它日期
    return render_template('sign_in_statistics.html', students=students,
                           target_date=target_date, signed_count=len(signed_ids),
                           total=len(students),
                           attendance_rate=attendance_rate, attendance_trend=trend,
                           form=ManualSignInForm(sign_date=target_date))


# 手动补签
@app.route('/sign-ins/manual', methods=['POST'])
@admin_required
def sign_in_manual():
    from attendance import record_sign_in  # 延迟导入，避免与 app 循环依赖
    form = ManualSignInForm()
    if form.validate_on_submit():
        record = record_sign_in(form.user_id.data, sign_time=form.sign_date.data)
        if record:
            flash(u'补签成功！', 'success')
        else:
            flash(u'该学生当日已签到，不能重复补签！', 'warning')
        return redirect(url_for('sign_ins_statistics',
                                date=form.sign_date.data.strftime('%Y-%m-%d')))
    flash(u'补签失败，请检查输入！', 'danger')
    return redirect(url_for('sign_ins_statistics'))


@app.route('/courses', methods=['GET', 'POST'])
@admin_required
def courses():
    demo_mode = bool(flask_session.get('demo_mode', False))
    form = CourseForm()
    if form.validate_on_submit():
        course = Course(code=form.code.data.strip().upper(),
                        name=form.name.data.strip(),
                        teacher=form.teacher.data.strip(),
                        classroom=form.classroom.data.strip(),
                        is_demo=demo_mode)
        db.session.add(course)
        db.session.flush()
        student_role = Role.query.filter_by(name='Student').first()
        for student in User.query.filter_by(role=student_role,
                                            is_demo=demo_mode).all():
            db.session.add(CourseEnrollment(course_id=course.id,
                                            user_id=student.id))
        db.session.commit()
        flash(u'课程已创建，并自动加入当前模式下的学生', 'success')
        return redirect(url_for('courses'))
    course_rows = (Course.query.filter_by(is_demo=demo_mode)
                   .order_by(Course.created_at.desc()).all())
    return render_template('courses.html', form=form, courses=course_rows,
                           demo_mode=demo_mode)


@app.route('/attendance-sessions', methods=['GET', 'POST'])
@admin_required
def attendance_sessions():
    demo_mode = bool(flask_session.get('demo_mode', False))
    form = AttendanceSessionForm(demo_mode=demo_mode)
    if request.method == 'GET':
        start = datetime.now().replace(second=0, microsecond=0)
        form.starts_at.data = start
        form.late_after.data = start + timedelta(minutes=10)
        form.ends_at.data = start + timedelta(minutes=60)
    if form.validate_on_submit():
        course = db.get_or_404(Course, form.course_id.data)
        if bool(course.is_demo) != demo_mode:
            flash(u'课程与当前数据模式不一致', 'danger')
            return redirect(url_for('attendance_sessions'))
        item = AttendanceSession(
            course_id=course.id, title=form.title.data.strip(),
            starts_at=form.starts_at.data, late_after=form.late_after.data,
            ends_at=form.ends_at.data, status='scheduled', is_demo=demo_mode)
        db.session.add(item)
        db.session.commit()
        flash(u'考勤任务已创建，可在开始上课时激活', 'success')
        return redirect(url_for('attendance_sessions'))
    session_rows = (AttendanceSession.query.filter_by(is_demo=demo_mode)
                    .order_by(AttendanceSession.starts_at.desc()).all())
    return render_template('attendance_sessions.html', form=form,
                           sessions=session_rows, demo_mode=demo_mode)


@app.route('/attendance-sessions/<int:session_id>/activate', methods=['POST'])
@admin_required
def activate_attendance_session(session_id):
    from attendance_platform import close_attendance_session
    item = db.get_or_404(AttendanceSession, session_id)
    demo_mode = bool(flask_session.get('demo_mode', False))
    if bool(item.is_demo) != demo_mode:
        return jsonify({'error': 'data mode mismatch'}), 400
    active_items = AttendanceSession.query.filter_by(
        status='active', is_demo=demo_mode).all()
    for active in active_items:
        if active.id != item.id:
            close_attendance_session(active)
    item.status = 'active'
    item.activated_at = datetime.now()
    if item.ends_at <= datetime.now():
        item.ends_at = datetime.now() + timedelta(minutes=60)
    db.session.commit()
    flash(u'考勤任务已激活，签到台将自动写入本场次', 'success')
    return redirect(url_for('attendance_sessions'))


@app.route('/attendance-sessions/<int:session_id>')
@login_required
def attendance_session_detail(session_id):
    from attendance_platform import build_session_detail
    item = db.get_or_404(AttendanceSession, session_id)
    demo_mode = bool(flask_session.get('demo_mode', False))
    if bool(item.is_demo) != demo_mode:
        abort(404)
    detail = build_session_detail(item)
    status_filter = request.args.get('status', '').strip().lower()
    allowed = {'on_time', 'late', 'absent', 'manual', 'pending'}
    visible_rows = (detail['rows'] if status_filter not in allowed else
                    [row for row in detail['rows']
                     if row['status'] == status_filter])
    return render_template('attendance_session_detail.html', item=item,
                           detail=detail, rows=visible_rows,
                           status_filter=status_filter)


@app.route('/attendance-sessions/<int:session_id>/export.csv')
@login_required
def export_attendance_session(session_id):
    from attendance_platform import build_session_detail
    item = db.get_or_404(AttendanceSession, session_id)
    demo_mode = bool(flask_session.get('demo_mode', False))
    if bool(item.is_demo) != demo_mode:
        abort(404)
    detail = build_session_detail(item)
    output = io.StringIO(newline='')
    output.write('\ufeff')
    writer = csv.writer(output)
    writer.writerow([
        u'课程编号', u'课程名称', u'考勤任务', u'日期', u'学号', u'姓名',
        u'考勤状态', u'签到时间', u'签到方式', u'识别引擎', u'人脸质量',
        u'相似度', u'活体概率', u'风险分', u'端到端时延(ms)'])
    status_names = {'on_time': u'正常', 'late': u'迟到', 'absent': u'缺勤',
                    'manual': u'补签', 'pending': u'待签到'}
    method_names = {'face': u'人脸识别', 'manual': u'手动补签',
                    'system': u'系统判定'}
    for row in detail['rows']:
        writer.writerow([
            item.course.code, item.course.name, item.title,
            item.starts_at.strftime('%Y-%m-%d'), row['number'], row['name'],
            status_names.get(row['status'], row['status']),
            row['checked_at'].strftime('%Y-%m-%d %H:%M:%S')
            if row['checked_at'] else '',
            method_names.get(row['method'], row['method'] or ''),
            row['engine'] or '', _csv_metric(row['quality']),
            _csv_metric(row['similarity']), _csv_metric(row['passive_score']),
            _csv_metric(row['risk_score']), _csv_metric(row['latency_ms'], 1),
        ])
    filename = '%s-%s-attendance.csv' % (
        item.course.code, item.starts_at.strftime('%Y%m%d'))
    return Response(output.getvalue(), mimetype='text/csv; charset=utf-8',
                    headers={'Content-Disposition':
                             'attachment; filename="%s"' % filename})


@app.route('/attendance-sessions/<int:session_id>/close', methods=['POST'])
@admin_required
def close_session(session_id):
    from attendance_platform import close_attendance_session
    item = db.get_or_404(AttendanceSession, session_id)
    close_attendance_session(item)
    flash(u'考勤任务已结束，未到学生已标记为缺勤', 'success')
    return redirect(url_for('attendance_sessions'))


@app.route('/recognition-center')
@login_required
def recognition_center():
    from attendance_platform import build_dashboard
    dashboard = build_dashboard(bool(flask_session.get('demo_mode', False)))
    return render_template('recognition_center.html', dashboard=dashboard)


@app.route('/experiments')
@login_required
def experiments():
    from experiment_service import load_experiment_dashboard
    try:
        experiment = load_experiment_dashboard()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        current_app.logger.exception('unable to load experiment artifacts')
        flash(u'实验产物读取失败：%s' % exc, 'danger')
        experiment = None
    return render_template('experiments.html', experiment=experiment)


@app.route('/risk-events')
@login_required
def risk_events():
    demo_mode = bool(flask_session.get('demo_mode', False))
    severity = request.args.get('severity', '').strip().lower()
    query = RiskEvent.query.filter_by(is_demo=demo_mode)
    if severity in ('low', 'medium', 'high'):
        query = query.filter_by(severity=severity)
    rows = query.order_by(RiskEvent.created_at.desc()).limit(200).all()
    return render_template('risk_events.html', risks=rows,
                           severity=severity, demo_mode=demo_mode)


@app.route('/api/dashboard/summary')
@login_required
def dashboard_summary():
    from attendance_platform import build_dashboard
    data = build_dashboard(bool(flask_session.get('demo_mode', False)))
    return jsonify(data)


@app.route('/demo/seed', methods=['POST'])
@admin_required
def seed_demo():
    from attendance_platform import seed_demo_data
    result = seed_demo_data()
    flask_session['demo_mode'] = True
    if result['created']:
        flash(u'演示数据已生成并进入演示模式；所有记录都有「演示」标识', 'success')
    else:
        flash(u'演示数据已存在，已切换到演示模式', 'info')
    return redirect(request.referrer or url_for('index'))


@app.route('/demo/toggle', methods=['POST'])
@admin_required
def toggle_demo_mode():
    target = request.form.get('enabled') == '1'
    flask_session['demo_mode'] = target
    flash(u'已切换到%s数据' % (u'演示' if target else u'真实'), 'info')
    return redirect(request.referrer or url_for('index'))


# 公开 Web 人脸签到（无需登录）：浏览器只采集截图，识别与活体判定均在后端。
@app.route('/web-sign-in', methods=['GET'])
def web_sign_in():
    return render_template('web_sign_in.html')


# Web 签到比对接口：接收前端描述子，后端找最佳匹配 + 多帧确认 + 落库
@app.route('/api/sign-in', methods=['POST'])
def api_sign_in():
    from web_signin import handle_sign_in_request  # 延迟导入，避免循环依赖
    payload, status_code = handle_sign_in_request()
    return jsonify(payload), status_code


# 主签到接口：接收截图，后端完成多人跟踪、质量调度、识别与活体检测。
@app.route('/api/sign-in-image', methods=['POST'])
def api_sign_in_image():
    from web_signin import handle_sign_in_image  # 延迟导入，避免循环依赖
    payload, status_code = handle_sign_in_image()
    return jsonify(payload), status_code


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# 加载用户的回调函数
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


'''
初始化命令：python -m flask --app app init
创建角色与初始管理员账号
'''


@app.cli.command('init')
def init():
    Role.insert_roles()
    User.generate_admin()
    print(u'初始化完成：已创建初始管理员账号 Admin（密码来自 ADMIN_PASSWORD）')


@app.cli.command('enroll-sface')
@click.argument('number')
@click.argument('image_path', type=click.Path(exists=True, dir_okay=False))
def enroll_sface(number, image_path):
    """为历史学生生成 SFace 特征，并校验照片与原 dlib 身份一致。"""
    user = User.query.filter_by(number=number).first()
    if user is None:
        raise click.ClickException(u'未找到学号：%s' % number)
    with open(image_path, 'rb') as stream:
        result = get_face_encoding_in_image(stream)
    if result.get('face_count') != 1 or result.get('sface_face_count') != 1:
        raise click.ClickException(u'照片必须且只能被两套模型检测到一张人脸')
    old_feature = parse_face_encoding(user.face_encoding)
    new_dlib_feature = parse_face_encoding(result['unknown_face_encodings'])
    if old_feature is not None and new_dlib_feature is not None:
        distance = float(np.linalg.norm(old_feature - new_dlib_feature))
        if distance > app.config['FACE_MATCH_TOLERANCE']:
            raise click.ClickException(
                u'照片与原身份不一致（dlib 距离 %.4f），已拒绝写入' % distance)
    user.face_encoding_sface = result['sface_encoding']
    db.session.add(user)
    db.session.commit()
    click.echo(u'已为 %s（%s）生成 SFace 特征' %
               (user.username, user.number))


@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db, User=User, Role=Role, SignIn=SignIn)


if __name__ == '__main__':
    # 默认关闭 debug；需要调试时设置 FLASK_DEBUG=1
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
