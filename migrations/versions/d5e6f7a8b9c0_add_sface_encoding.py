"""add independent SFace encoding for incremental model migration

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-08-20
"""
from alembic import op
import sqlalchemy as sa


revision = 'd5e6f7a8b9c0'
down_revision = 'c4d5e6f7a8b9'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('users', sa.Column('face_encoding_sface', sa.Text(),
                                     nullable=True))


def downgrade():
    op.drop_column('users', 'face_encoding_sface')
