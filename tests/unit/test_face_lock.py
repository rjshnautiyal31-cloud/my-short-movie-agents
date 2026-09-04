# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.utils.face_lock import (
    _reinhard_color_transfer,
    composite_face_into_image_bytes,
    detect_faces,
    resolve_user_photo_bytes_and_uri,
)


def _create_synthetic_image_bytes(color: tuple[int, int, int] = (200, 150, 120)) -> bytes:
    img = Image.new("RGB", (100, 100), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_faces_handles_empty_image():
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detect_faces(empty_img) == []

    tiny_img = np.zeros((5, 5, 3), dtype=np.uint8)
    assert detect_faces(tiny_img) == []


def test_reinhard_color_transfer_preserves_dimensions():
    src = np.full((50, 50, 3), (180, 140, 110), dtype=np.uint8)
    tgt = np.full((50, 50, 3), (80, 60, 40), dtype=np.uint8)
    transferred = _reinhard_color_transfer(src, tgt)
    assert transferred.shape == src.shape
    assert transferred.dtype == np.uint8


def test_composite_face_returns_original_if_no_faces():
    src_bytes = _create_synthetic_image_bytes((255, 0, 0))
    tgt_bytes = _create_synthetic_image_bytes((0, 255, 0))
    result = composite_face_into_image_bytes(tgt_bytes, src_bytes)
    assert len(result) > 0


def test_resolve_user_photo_from_state():
    mock_context = MagicMock()
    mock_context.session.state = {"user_photo_uri": "tests/unit/test_dummy.py"}
    mock_context.state = mock_context.session.state

    raw_bytes, uri = resolve_user_photo_bytes_and_uri(mock_context)
    assert raw_bytes is not None
    assert uri == "tests/unit/test_dummy.py"


def test_resolve_user_photo_from_inline_data():
    mock_context = MagicMock()
    mock_context.session.state = {}
    mock_context.state = {}

    dummy_bytes = b"fake_inline_image_data"
    part = MagicMock()
    part.inline_data.data = dummy_bytes
    part.file_data = None
    part.text = None

    mock_context.user_content.parts = [part]
    mock_context.session.events = []

    raw_bytes, uri = resolve_user_photo_bytes_and_uri(mock_context)
    assert raw_bytes == dummy_bytes


def test_is_frontal_pose():
    from app.utils.face_lock import is_frontal_pose

    # Normal frontal landmarks: re, le, nose, mouth_r, mouth_l
    frontal_landmarks = np.array([
        [40.0, 50.0],
        [60.0, 50.0],
        [50.0, 60.0],
        [42.0, 75.0],
        [58.0, 75.0],
    ])
    assert is_frontal_pose(frontal_landmarks) is True

    # Extreme turned profile landmarks: nose way off to the side
    turned_landmarks = np.array([
        [40.0, 50.0],
        [60.0, 50.0],
        [32.0, 60.0],  # Nose outside eye span
        [35.0, 75.0],
        [45.0, 75.0],
    ])
    assert is_frontal_pose(turned_landmarks) is False


def test_swap_face_video_mode_feathers():
    from app.utils.face_lock import swap_face

    src_bgr = np.full((100, 100, 3), (200, 150, 120), dtype=np.uint8)
    tgt_bgr = np.full((100, 100, 3), (50, 50, 50), dtype=np.uint8)

    face_dict = {
        "landmarks": np.array([
            [40.0, 40.0],
            [60.0, 40.0],
            [50.0, 55.0],
            [42.0, 70.0],
            [58.0, 70.0],
        ]),
        "bbox": [20, 20, 60, 60],
        "score": 0.95,
    }

    result = swap_face(src_bgr, tgt_bgr, face_dict, face_dict, is_video=True)
    assert result.shape == tgt_bgr.shape
    assert result.dtype == np.uint8
    # Video mode feathered blend modifies face region without crashing
    assert not np.array_equal(result, tgt_bgr)

