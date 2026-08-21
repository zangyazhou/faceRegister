"""number BigInteger -> String (B + 11 位学号)

Revision ID: c4d5e6f7a8b9
Revises: b2f0a1c3d4e5f678
Create Date: 2026-08-19

学号格式从纯数字改为「B 开头 + 11 位数字」（如 B20240304323），
number 字段类型由 BigInteger 改为 String(20)。SQLite 需 batch 重建表。
"""
import sqlalchemy as sa
from alembic import op

revision = 'c4d5e6f7a8b9'
down_revision = 'b2f0a1c3d4e5f678'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('number',
                              existing_type=sa.BigInteger(),
                              type_=sa.String(length=20),
                              existing_nullable=True,
                              existing_unique=True)


def downgrade():
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('number',
                              existing_type=sa.String(length=20),
                              type_=sa.BigInteger(),
                              existing_nullable=True,
                              existing_unique=True)
