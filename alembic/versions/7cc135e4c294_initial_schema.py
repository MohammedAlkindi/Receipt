"""initial_schema

Revision ID: 7cc135e4c294
Revises:
Create Date: 2026-05-15 16:33:00.660790

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect


# revision identifiers, used by Alembic.
revision: str = '7cc135e4c294'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa_inspect(bind).get_table_names()

    if 'analysis_runs' not in existing:
        op.create_table(
            'analysis_runs',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('run_id', sa.String(length=64), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
            sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
            sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
            sa.Column('source_file', sa.String(length=512), nullable=True),
            sa.Column('transaction_count', sa.Integer(), nullable=False),
            sa.Column('narrative_json', sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('analysis_runs', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_analysis_runs_run_id'), ['run_id'], unique=True)

    if 'merchants' not in existing:
        op.create_table(
            'merchants',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('name', sa.String(length=512), nullable=False),
            sa.Column('normalized_name', sa.String(length=512), nullable=False),
            sa.Column('category', sa.String(length=64), nullable=True),
            sa.Column('first_seen', sa.DateTime(timezone=True), nullable=True),
            sa.Column('last_seen', sa.DateTime(timezone=True), nullable=True),
            sa.Column('total_spent', sa.Float(), nullable=False),
            sa.Column('transaction_count', sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('merchants', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_merchants_name'), ['name'], unique=False)
            batch_op.create_index(batch_op.f('ix_merchants_normalized_name'), ['normalized_name'], unique=True)

    if 'transactions' not in existing:
        op.create_table(
            'transactions',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('transaction_id', sa.String(length=16), nullable=False),
            sa.Column('date', sa.DateTime(timezone=True), nullable=False),
            sa.Column('description', sa.String(length=512), nullable=False),
            sa.Column('raw_description', sa.String(length=512), nullable=False),
            sa.Column('amount', sa.Float(), nullable=False),
            sa.Column('source', sa.String(length=64), nullable=False),
            sa.Column('category', sa.String(length=64), nullable=True),
            sa.Column('category_confidence', sa.Float(), nullable=True),
            sa.Column('cluster_id', sa.Integer(), nullable=True),
            sa.Column('anomaly_score', sa.Float(), nullable=True),
            sa.Column('is_anomaly', sa.Boolean(), nullable=False),
            sa.Column('is_weekend', sa.Boolean(), nullable=True),
            sa.Column('day_of_week', sa.String(length=16), nullable=True),
            sa.Column('run_id', sa.String(length=64), nullable=True),
            sa.ForeignKeyConstraint(['run_id'], ['analysis_runs.run_id']),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('transactions', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_transactions_category'), ['category'], unique=False)
            batch_op.create_index(batch_op.f('ix_transactions_date'), ['date'], unique=False)
            batch_op.create_index(batch_op.f('ix_transactions_run_id'), ['run_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_transactions_transaction_id'), ['transaction_id'], unique=True)


def downgrade() -> None:
    with op.batch_alter_table('transactions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_transactions_transaction_id'))
        batch_op.drop_index(batch_op.f('ix_transactions_run_id'))
        batch_op.drop_index(batch_op.f('ix_transactions_date'))
        batch_op.drop_index(batch_op.f('ix_transactions_category'))

    op.drop_table('transactions')
    with op.batch_alter_table('merchants', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_merchants_normalized_name'))
        batch_op.drop_index(batch_op.f('ix_merchants_name'))

    op.drop_table('merchants')
    with op.batch_alter_table('analysis_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_analysis_runs_run_id'))

    op.drop_table('analysis_runs')
