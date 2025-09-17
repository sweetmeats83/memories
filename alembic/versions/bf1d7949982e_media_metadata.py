""" + media metadata

Revision ID: bf1d7949982e
Revises: 7520bb31a20a
Create Date: 2025-08-16 04:12:18.087708
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'bf1d7949982e'
down_revision: Union[str, Sequence[str], None] = '7520bb31a20a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # --- prompt_media ---
    op.add_column('prompt_media', sa.Column('thumbnail_url', sa.String(), nullable=True))
    op.add_column('prompt_media', sa.Column('mime_type', sa.String(), nullable=True))
    op.add_column('prompt_media', sa.Column('duration_sec', sa.Integer(), nullable=True))
    op.add_column('prompt_media', sa.Column('sample_rate', sa.Integer(), nullable=True))
    op.add_column('prompt_media', sa.Column('channels', sa.Integer(), nullable=True))
    op.add_column('prompt_media', sa.Column('width', sa.Integer(), nullable=True))
    op.add_column('prompt_media', sa.Column('height', sa.Integer(), nullable=True))
    op.add_column('prompt_media', sa.Column('size_bytes', sa.Integer(), nullable=True))
    op.add_column('prompt_media', sa.Column('codec_audio', sa.String(), nullable=True))
    op.add_column('prompt_media', sa.Column('codec_video', sa.String(), nullable=True))
    op.add_column('prompt_media', sa.Column('wav_path', sa.String(), nullable=True))

    # --- response (primary media fields) ---
    op.add_column('response', sa.Column('primary_thumbnail_path', sa.String(), nullable=True))
    op.add_column('response', sa.Column('primary_mime_type', sa.String(), nullable=True))
    op.add_column('response', sa.Column('primary_duration_sec', sa.Integer(), nullable=True))
    op.add_column('response', sa.Column('primary_sample_rate', sa.Integer(), nullable=True))
    op.add_column('response', sa.Column('primary_channels', sa.Integer(), nullable=True))
    op.add_column('response', sa.Column('primary_width', sa.Integer(), nullable=True))
    op.add_column('response', sa.Column('primary_height', sa.Integer(), nullable=True))
    op.add_column('response', sa.Column('primary_size_bytes', sa.Integer(), nullable=True))
    op.add_column('response', sa.Column('primary_codec_audio', sa.String(), nullable=True))
    op.add_column('response', sa.Column('primary_codec_video', sa.String(), nullable=True))
    op.add_column('response', sa.Column('primary_wav_path', sa.String(), nullable=True))

    # --- supporting_media ---
    op.add_column('supporting_media', sa.Column('thumbnail_url', sa.String(), nullable=True))
    op.add_column('supporting_media', sa.Column('mime_type', sa.String(), nullable=True))
    op.add_column('supporting_media', sa.Column('duration_sec', sa.Integer(), nullable=True))
    op.add_column('supporting_media', sa.Column('sample_rate', sa.Integer(), nullable=True))
    op.add_column('supporting_media', sa.Column('channels', sa.Integer(), nullable=True))
    op.add_column('supporting_media', sa.Column('width', sa.Integer(), nullable=True))
    op.add_column('supporting_media', sa.Column('height', sa.Integer(), nullable=True))
    op.add_column('supporting_media', sa.Column('size_bytes', sa.Integer(), nullable=True))
    op.add_column('supporting_media', sa.Column('codec_audio', sa.String(), nullable=True))
    op.add_column('supporting_media', sa.Column('codec_video', sa.String(), nullable=True))
    op.add_column('supporting_media', sa.Column('wav_path', sa.String(), nullable=True))
    # NOTE: no unique constraints created here — they already exist


def downgrade() -> None:
    """Downgrade schema (remove added columns)."""
    # --- supporting_media ---
    op.drop_column('supporting_media', 'wav_path')
    op.drop_column('supporting_media', 'codec_video')
    op.drop_column('supporting_media', 'codec_audio')
    op.drop_column('supporting_media', 'size_bytes')
    op.drop_column('supporting_media', 'height')
    op.drop_column('supporting_media', 'width')
    op.drop_column('supporting_media', 'channels')
    op.drop_column('supporting_media', 'sample_rate')
    op.drop_column('supporting_media', 'duration_sec')
    op.drop_column('supporting_media', 'mime_type')
    op.drop_column('supporting_media', 'thumbnail_url')

    # --- response ---
    op.drop_column('response', 'primary_wav_path')
    op.drop_column('response', 'primary_codec_video')
    op.drop_column('response', 'primary_codec_audio')
    op.drop_column('response', 'primary_size_bytes')
    op.drop_column('response', 'primary_height')
    op.drop_column('response', 'primary_width')
    op.drop_column('response', 'primary_channels')
    op.drop_column('response', 'primary_sample_rate')
    op.drop_column('response', 'primary_duration_sec')
    op.drop_column('response', 'primary_mime_type')
    op.drop_column('response', 'primary_thumbnail_path')

    # --- prompt_media ---
    op.drop_column('prompt_media', 'wav_path')
    op.drop_column('prompt_media', 'codec_video')
    op.drop_column('prompt_media', 'codec_audio')
    op.drop_column('prompt_media', 'size_bytes')
    op.drop_column('prompt_media', 'height')
    op.drop_column('prompt_media', 'width')
    op.drop_column('prompt_media', 'channels')
    op.drop_column('prompt_media', 'sample_rate')
    op.drop_column('prompt_media', 'duration_sec')
    op.drop_column('prompt_media', 'mime_type')
    op.drop_column('prompt_media', 'thumbnail_url')
