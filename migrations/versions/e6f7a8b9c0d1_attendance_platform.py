"""add course attendance analytics and risk event model

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-08-20
"""
from alembic import op
import sqlalchemy as sa


revision = 'e6f7a8b9c0d1'
down_revision = 'd5e6f7a8b9c0'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('users') as batch_op:
        batch_op.add_column(sa.Column('is_demo', sa.Boolean(),
                                      nullable=False, server_default=sa.false()))
        batch_op.create_index('ix_users_is_demo', ['is_demo'], unique=False)

    op.create_table(
        'courses',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('code', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('teacher', sa.String(length=64), nullable=False),
        sa.Column('classroom', sa.String(length=64)),
        sa.Column('is_active', sa.Boolean(), nullable=False,
                  server_default=sa.true()),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('code'),
    )
    op.create_index('ix_courses_code', 'courses', ['code'], unique=True)
    op.create_index('ix_courses_is_demo', 'courses', ['is_demo'], unique=False)

    op.create_table(
        'course_enrollments',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('enrolled_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.UniqueConstraint('course_id', 'user_id', name='uq_course_enrollment'),
    )
    op.create_index('ix_course_enrollments_course_id', 'course_enrollments',
                    ['course_id'], unique=False)
    op.create_index('ix_course_enrollments_user_id', 'course_enrollments',
                    ['user_id'], unique=False)

    op.create_table(
        'attendance_sessions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=128), nullable=False),
        sa.Column('starts_at', sa.DateTime(), nullable=False),
        sa.Column('late_after', sa.DateTime(), nullable=False),
        sa.Column('ends_at', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False,
                  server_default='scheduled'),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('activated_at', sa.DateTime()),
        sa.Column('closed_at', sa.DateTime()),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id']),
    )
    op.create_index('ix_attendance_sessions_course_id', 'attendance_sessions',
                    ['course_id'], unique=False)
    op.create_index('ix_attendance_sessions_status', 'attendance_sessions',
                    ['status'], unique=False)
    op.create_index('ix_attendance_sessions_is_demo', 'attendance_sessions',
                    ['is_demo'], unique=False)

    op.create_table(
        'recognition_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('session_id', sa.Integer()),
        sa.Column('user_id', sa.Integer()),
        sa.Column('track_id', sa.String(length=32)),
        sa.Column('outcome', sa.String(length=24), nullable=False),
        sa.Column('engine', sa.String(length=32), nullable=False),
        sa.Column('quality', sa.Float()),
        sa.Column('similarity', sa.Float()),
        sa.Column('margin', sa.Float()),
        sa.Column('passive_score', sa.Float()),
        sa.Column('risk_score', sa.Float()),
        sa.Column('risk_level', sa.String(length=16), nullable=False,
                  server_default='low'),
        sa.Column('latency_ms', sa.Float()),
        sa.Column('liveness_mode', sa.String(length=24)),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['attendance_sessions.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
    )
    for column in ('session_id', 'user_id', 'outcome', 'risk_level',
                   'is_demo', 'created_at'):
        op.create_index('ix_recognition_events_' + column,
                        'recognition_events', [column], unique=False)

    op.create_table(
        'attendance_records',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('checked_at', sa.DateTime()),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('method', sa.String(length=16), nullable=False,
                  server_default='face'),
        sa.Column('recognition_event_id', sa.Integer()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['recognition_event_id'], ['recognition_events.id']),
        sa.ForeignKeyConstraint(['session_id'], ['attendance_sessions.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.UniqueConstraint('session_id', 'user_id', name='uq_attendance_record'),
    )
    op.create_index('ix_attendance_records_session_id', 'attendance_records',
                    ['session_id'], unique=False)
    op.create_index('ix_attendance_records_user_id', 'attendance_records',
                    ['user_id'], unique=False)
    op.create_index('ix_attendance_records_status', 'attendance_records',
                    ['status'], unique=False)

    op.create_table(
        'risk_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('recognition_event_id', sa.Integer()),
        sa.Column('session_id', sa.Integer()),
        sa.Column('user_id', sa.Integer()),
        sa.Column('event_type', sa.String(length=32), nullable=False),
        sa.Column('severity', sa.String(length=16), nullable=False),
        sa.Column('message', sa.String(length=255), nullable=False),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['recognition_event_id'], ['recognition_events.id']),
        sa.ForeignKeyConstraint(['session_id'], ['attendance_sessions.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
    )
    for column in ('recognition_event_id', 'session_id', 'user_id',
                   'event_type', 'severity', 'is_demo', 'created_at'):
        op.create_index('ix_risk_events_' + column, 'risk_events', [column],
                        unique=False)


def downgrade():
    op.drop_table('risk_events')
    op.drop_table('attendance_records')
    op.drop_table('recognition_events')
    op.drop_table('attendance_sessions')
    op.drop_table('course_enrollments')
    op.drop_table('courses')
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_index('ix_users_is_demo')
        batch_op.drop_column('is_demo')
