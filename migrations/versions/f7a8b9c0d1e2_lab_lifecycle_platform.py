"""add laboratory identity lifecycle and trusted access platform

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-08-26
"""
from alembic import op
import sqlalchemy as sa


revision = 'f7a8b9c0d1e2'
down_revision = 'e6f7a8b9c0d1'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('users') as batch_op:
        batch_op.add_column(sa.Column('person_type', sa.String(length=24),
                                      nullable=False, server_default='student'))
        batch_op.add_column(sa.Column('lifecycle_status', sa.String(length=24),
                                      nullable=False, server_default='active'))
        batch_op.add_column(sa.Column('joined_at', sa.Date()))
        batch_op.add_column(sa.Column('expected_leave_at', sa.Date()))
        batch_op.add_column(sa.Column('suspended_at', sa.DateTime()))
        batch_op.add_column(sa.Column('offboarded_at', sa.DateTime()))
        batch_op.add_column(sa.Column('privacy_consent_at', sa.DateTime()))
        batch_op.add_column(sa.Column('alternative_auth', sa.String(length=24),
                                      nullable=False, server_default='manual'))
        batch_op.create_index('ix_users_person_type', ['person_type'], unique=False)
        batch_op.create_index('ix_users_lifecycle_status', ['lifecycle_status'],
                              unique=False)
        batch_op.create_index('ix_users_expected_leave_at', ['expected_leave_at'],
                              unique=False)

    op.create_table(
        'organization_units',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('code', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('unit_type', sa.String(length=24), nullable=False,
                  server_default='laboratory'),
        sa.Column('location', sa.String(length=128)),
        sa.Column('manager', sa.String(length=64)),
        sa.Column('description', sa.String(length=500)),
        sa.Column('is_active', sa.Boolean(), nullable=False,
                  server_default=sa.true()),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('code'),
    )
    op.create_index('ix_organization_units_code', 'organization_units', ['code'],
                    unique=True)
    op.create_index('ix_organization_units_is_active', 'organization_units',
                    ['is_active'], unique=False)
    op.create_index('ix_organization_units_is_demo', 'organization_units',
                    ['is_demo'], unique=False)

    op.create_table(
        'access_points',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('code', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('location', sa.String(length=128)),
        sa.Column('is_active', sa.Boolean(), nullable=False,
                  server_default=sa.true()),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organization_units.id']),
        sa.UniqueConstraint('code'),
    )
    op.create_index('ix_access_points_organization_id', 'access_points',
                    ['organization_id'], unique=False)
    op.create_index('ix_access_points_code', 'access_points', ['code'], unique=True)
    op.create_index('ix_access_points_is_active', 'access_points', ['is_active'],
                    unique=False)
    op.create_index('ix_access_points_is_demo', 'access_points', ['is_demo'],
                    unique=False)

    op.create_table(
        'face_templates',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('dlib_encoding', sa.Text()),
        sa.Column('sface_encoding', sa.Text()),
        sa.Column('algorithm', sa.String(length=64), nullable=False,
                  server_default='YuNet+SFace'),
        sa.Column('model_version', sa.String(length=64), nullable=False,
                  server_default='opencv-zoo-2021dec'),
        sa.Column('status', sa.String(length=24), nullable=False,
                  server_default='active'),
        sa.Column('enrolled_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime()),
        sa.Column('destroyed_at', sa.DateTime()),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.UniqueConstraint('user_id'),
    )
    op.create_index('ix_face_templates_user_id', 'face_templates', ['user_id'],
                    unique=True)
    op.create_index('ix_face_templates_status', 'face_templates', ['status'],
                    unique=False)
    op.create_index('ix_face_templates_expires_at', 'face_templates', ['expires_at'],
                    unique=False)
    op.create_index('ix_face_templates_is_demo', 'face_templates', ['is_demo'],
                    unique=False)
    # 平滑迁移历史人脸库：复制特征到独立模板表，旧字段暂时保留以便回滚。
    op.execute(sa.text("""
        INSERT INTO face_templates
            (user_id, dlib_encoding, sface_encoding, algorithm, model_version,
             status, enrolled_at, updated_at, is_demo)
        SELECT id, face_encoding, face_encoding_sface, 'YuNet+SFace',
               'legacy-migrated', 'active', CURRENT_TIMESTAMP,
               CURRENT_TIMESTAMP, is_demo
        FROM users
        WHERE face_encoding IS NOT NULL OR face_encoding_sface IS NOT NULL
    """))

    op.create_table(
        'lab_memberships',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('role_title', sa.String(length=64), nullable=False,
                  server_default='实验室成员'),
        sa.Column('status', sa.String(length=24), nullable=False,
                  server_default='active'),
        sa.Column('joined_at', sa.Date(), nullable=False),
        sa.Column('ends_at', sa.Date()),
        sa.Column('ended_at', sa.DateTime()),
        sa.ForeignKeyConstraint(['organization_id'], ['organization_units.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.UniqueConstraint('organization_id', 'user_id',
                            name='uq_lab_membership'),
    )
    for column in ('organization_id', 'user_id', 'status', 'ends_at'):
        op.create_index('ix_lab_memberships_' + column, 'lab_memberships',
                        [column], unique=False)

    op.create_table(
        'access_grants',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('access_point_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False,
                  server_default='active'),
        sa.Column('starts_at', sa.DateTime(), nullable=False),
        sa.Column('ends_at', sa.DateTime()),
        sa.Column('source', sa.String(length=32), nullable=False,
                  server_default='membership'),
        sa.Column('granted_at', sa.DateTime(), nullable=False),
        sa.Column('revoked_at', sa.DateTime()),
        sa.ForeignKeyConstraint(['access_point_id'], ['access_points.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.UniqueConstraint('access_point_id', 'user_id', name='uq_access_grant'),
    )
    for column in ('access_point_id', 'user_id', 'status', 'ends_at'):
        op.create_index('ix_access_grants_' + column, 'access_grants', [column],
                        unique=False)

    op.create_table(
        'access_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer()),
        sa.Column('access_point_id', sa.Integer()),
        sa.Column('recognition_event_id', sa.Integer()),
        sa.Column('decision', sa.String(length=24), nullable=False),
        sa.Column('reason_code', sa.String(length=48), nullable=False),
        sa.Column('reason_text', sa.String(length=255), nullable=False),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['access_point_id'], ['access_points.id']),
        sa.ForeignKeyConstraint(['recognition_event_id'], ['recognition_events.id']),
    )
    for column in ('user_id', 'access_point_id', 'recognition_event_id',
                   'decision', 'reason_code', 'is_demo', 'occurred_at'):
        op.create_index('ix_access_events_' + column, 'access_events', [column],
                        unique=False)

    op.create_table(
        'lifecycle_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer()),
        sa.Column('event_type', sa.String(length=32), nullable=False),
        sa.Column('from_status', sa.String(length=24)),
        sa.Column('to_status', sa.String(length=24)),
        sa.Column('summary', sa.String(length=255), nullable=False),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
    )
    for column in ('user_id', 'event_type', 'is_demo', 'created_at'):
        op.create_index('ix_lifecycle_events_' + column, 'lifecycle_events',
                        [column], unique=False)

    op.create_table(
        'erasure_tasks',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('receipt_code', sa.String(length=48), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False,
                  server_default='pending'),
        sa.Column('trigger_type', sa.String(length=32), nullable=False,
                  server_default='offboarding'),
        sa.Column('templates_deleted', sa.Integer(), nullable=False,
                  server_default='0'),
        sa.Column('grants_revoked', sa.Integer(), nullable=False,
                  server_default='0'),
        sa.Column('records_deleted', sa.Integer(), nullable=False,
                  server_default='0'),
        sa.Column('profile_action', sa.String(length=32), nullable=False,
                  server_default='physical_delete'),
        sa.Column('message', sa.String(length=255), nullable=False),
        sa.Column('is_demo', sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column('requested_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime()),
        sa.UniqueConstraint('receipt_code'),
    )
    op.create_index('ix_erasure_tasks_receipt_code', 'erasure_tasks',
                    ['receipt_code'], unique=True)
    op.create_index('ix_erasure_tasks_status', 'erasure_tasks', ['status'],
                    unique=False)
    op.create_index('ix_erasure_tasks_is_demo', 'erasure_tasks', ['is_demo'],
                    unique=False)


def downgrade():
    op.drop_table('erasure_tasks')
    op.drop_table('lifecycle_events')
    op.drop_table('access_events')
    op.drop_table('access_grants')
    op.drop_table('lab_memberships')
    op.drop_table('face_templates')
    op.drop_table('access_points')
    op.drop_table('organization_units')
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_index('ix_users_expected_leave_at')
        batch_op.drop_index('ix_users_lifecycle_status')
        batch_op.drop_index('ix_users_person_type')
        batch_op.drop_column('alternative_auth')
        batch_op.drop_column('privacy_consent_at')
        batch_op.drop_column('offboarded_at')
        batch_op.drop_column('suspended_at')
        batch_op.drop_column('expected_leave_at')
        batch_op.drop_column('joined_at')
        batch_op.drop_column('lifecycle_status')
        batch_op.drop_column('person_type')
