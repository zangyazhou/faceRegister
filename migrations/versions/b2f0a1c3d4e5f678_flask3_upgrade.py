"""flask3 upgrade: 补齐 face_encoding 列、number 改 BigInteger、新增签到表

Revision ID: b2f0a1c3d4e5f678
Revises: a3ea5d4b480b
Create Date: 2026-08-18

说明：
- 老库 users.face_encoding 列是手工添加的（VARCHAR(5000)），迁移链里没有，
  因此本迁移做存在性检查，live 库为 no-op、干净库可正常创建。
- number 由 SmallInteger 改为 BigInteger（学号 8~10 位超出 16 位整数范围）。
  SQLite 原生 ALTER 不支持改列类型，必须用 batch 模式（重建表并拷贝数据）。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b2f0a1c3d4e5f678'
down_revision = 'a3ea5d4b480b'
branch_labels = None
depends_on = None


def _has_table(table):
    from sqlalchemy import inspect
    return inspect(op.get_bind()).has_table(table)


def _has_column(table, column):
    from sqlalchemy import inspect
    cols = [c['name'] for c in inspect(op.get_bind()).get_columns(table)]
    return column in cols


def upgrade():
    # 1. users.face_encoding：老库已有（手工添加），仅缺失时补齐
    if _has_table('users') and not _has_column('users', 'face_encoding'):
        op.add_column('users',
                      sa.Column('face_encoding', sa.String(length=5000), nullable=True))

    # 2. number: SmallInteger -> BigInteger（SQLite 需 batch 重建表）
    if _has_table('users'):
        with op.batch_alter_table('users') as batch_op:
            batch_op.alter_column('number',
                                  existing_type=sa.SmallInteger(),
                                  type_=sa.BigInteger(),
                                  existing_nullable=True,
                                  existing_unique=True)

    # 3. 新增签到表
    op.create_table('sign_ins',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.Column('sign_time', sa.DateTime(), nullable=False),
                    sa.Column('sign_date', sa.Date(), nullable=False),
                    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('user_id', 'sign_date',
                                        name='uq_sign_ins_user_date')
                    )
    op.create_index(op.f('ix_sign_ins_user_id'), 'sign_ins', ['user_id'], unique=False)
    op.create_index(op.f('ix_sign_ins_sign_date'), 'sign_ins', ['sign_date'], unique=False)

    # 4. 清理历史迁移残留表
    if _has_table('_users_old_20190425'):
        op.drop_table('_users_old_20190425')


def downgrade():
    op.drop_index(op.f('ix_sign_ins_sign_date'), table_name='sign_ins')
    op.drop_index(op.f('ix_sign_ins_user_id'), table_name='sign_ins')
    op.drop_table('sign_ins')

    if _has_table('users'):
        with op.batch_alter_table('users') as batch_op:
            batch_op.alter_column('number',
                                  existing_type=sa.BigInteger(),
                                  type_=sa.SmallInteger(),
                                  existing_nullable=True,
                                  existing_unique=True)
        if _has_column('users', 'face_encoding'):
            op.drop_column('users', 'face_encoding')
