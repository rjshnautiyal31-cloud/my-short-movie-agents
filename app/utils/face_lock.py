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
import logging
import os
import re
import subprocess
import tempfile
import urllib.request
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
from google.cloud import storage
from PIL import Image

logger = logging.getLogger(__name__)

# Path to the YuNet ONNX face detection model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def _ensure_model_exists() -> str:
    """Ensure the YuNet ONNX face detector model is present locally."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"Downloading YuNet face detector model to {MODEL_PATH}...")
        try:
            ctx = urllib.request.ssl._create_unverified_context()
            with urllib.request.urlopen(MODEL_URL, context=ctx) as response, open(
                MODEL_PATH, "wb"
            ) as out_file:
                out_file.write(response.read())
            logger.info("YuNet model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download YuNet model via urllib: {e}")
            try:
                subprocess.run(
                    ["curl", "-k", "-L", "-o", MODEL_PATH, MODEL_URL],
                    check=True,
                    capture_output=True,
                )
            except Exception as curl_err:
                logger.error(f"Failed to download YuNet model via curl: {curl_err}")
    return MODEL_PATH


def _get_detector(
    width: int, height: int, score_threshold: float = 0.3
) -> cv2.FaceDetectorYN | None:
    """Initialize or configure YuNet face detector with adaptive sensitivity."""
    try:
        model_path = _ensure_model_exists()
        if not os.path.exists(model_path):
            return None
        detector = cv2.FaceDetectorYN_create(
            model_path,
            "",
            (width, height),
            score_threshold,  # Adaptive score threshold
            0.3,              # NMS threshold
            5000,             # Top K
        )
        detector.setInputSize((width, height))
        return detector
    except Exception as e:
        logger.warning(f"Could not initialize YuNet detector: {e}")
        return None


def detect_faces(
    image_bgr: np.ndarray, score_threshold: float = 0.3
) -> list[dict[str, Any]]:
    """Detect faces and 5 key landmarks in a BGR image.

    Uses adaptive multi-scale (1x, 1.5x, 2x) and multi-threshold detection
    to detect faces even when characters are rendered at various camera distances,
    angles, or stylizations.
    """
    h, w = image_bgr.shape[:2]
    if h <= 10 or w <= 10:
        return []

    # Adaptive thresholds: start with requested, then try sensitive thresholds
    thresholds = [score_threshold]
    if score_threshold > 0.2:
        thresholds.append(0.2)
    if score_threshold > 0.15:
        thresholds.append(0.15)

    scales = [1.0, 1.5, 2.0]

    for scale in scales:
        if scale == 1.0:
            proc_img = image_bgr
            pw, ph = w, h
        else:
            pw, ph = int(w * scale), int(h * scale)
            proc_img = cv2.resize(image_bgr, (pw, ph), interpolation=cv2.INTER_LINEAR)

        for thresh in thresholds:
            detector = _get_detector(pw, ph, score_threshold=thresh)
            if detector is None:
                continue

            try:
                detector.setInputSize((pw, ph))
                _, faces = detector.detect(proc_img)
                if faces is not None and len(faces) > 0:
                    results = []
                    inv_scale = 1.0 / scale
                    for face in faces:
                        bbox = [int(v * inv_scale) for v in face[0:4]]
                        landmarks = np.array(
                            face[4:14].reshape((5, 2)) * inv_scale, dtype=np.float32
                        )
                        score = float(face[14])
                        results.append(
                            {
                                "bbox": bbox,
                                "landmarks": landmarks,
                                "score": score,
                            }
                        )
                    return results
            except Exception as e:
                logger.debug(f"Face detection failed (scale={scale}, thresh={thresh}): {e}")

    return []


def _reinhard_color_transfer(
    source_bgr: np.ndarray, target_bgr: np.ndarray
) -> np.ndarray:
    """Match lighting and ambient warmth while strictly preserving user facial identity."""
    if source_bgr.size == 0 or target_bgr.size == 0:
        return source_bgr

    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean = np.mean(src_lab, axis=(0, 1))
    src_std = np.std(src_lab, axis=(0, 1))
    tgt_mean = np.mean(tgt_lab, axis=(0, 1))
    tgt_std = np.std(tgt_lab, axis=(0, 1))

    src_std = np.where(src_std < 1e-5, 1.0, src_std)

    result_lab = src_lab.copy()
    # Match luminance L to target scene lighting
    result_lab[:, :, 0] = (src_lab[:, :, 0] - src_mean[0]) * (tgt_std[0] / src_std[0]) + tgt_mean[0]
    # Retain 80% authentic facial chromaticity (A, B) and blend 20% ambient scene color
    result_lab[:, :, 1:] = 0.80 * src_lab[:, :, 1:] + 0.20 * (
        (src_lab[:, :, 1:] - src_mean[1:]) * (tgt_std[1:] / src_std[1:]) + tgt_mean[1:]
    )
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def swap_face(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_face: dict[str, Any],
    target_face: dict[str, Any],
) -> np.ndarray:
    """Seamlessly swap and lock source face onto target face using 5-point landmark similarity."""
    try:
        src_landmarks = source_face["landmarks"]
        tgt_landmarks = target_face["landmarks"]

        # Compute optimal 2D affine / similarity transform using all 5 landmarks
        warp_mat, _ = cv2.estimateAffinePartial2D(src_landmarks, tgt_landmarks)
        if warp_mat is None:
            # Fallback to 3 points if partial estimate fails
            src_pts = np.float32([src_landmarks[0], src_landmarks[1], src_landmarks[2]])
            tgt_pts = np.float32([tgt_landmarks[0], tgt_landmarks[1], tgt_landmarks[2]])
            warp_mat = cv2.getAffineTransform(src_pts, tgt_pts)

        h, w = target_bgr.shape[:2]
        warped_source = cv2.warpAffine(
            source_bgr,
            warp_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Target bounding box
        tx, ty, tw, th = target_face["bbox"]
        tx, ty = max(0, tx), max(0, ty)
        tw, th = min(w - tx, tw), min(h - ty, th)

        if tw <= 10 or th <= 10:
            return target_bgr

        # Crop target face region for lighting matching
        tgt_crop = target_bgr[ty : ty + th, tx : tx + tw]
        warped_crop = warped_source[ty : ty + th, tx : tx + tw]

        # Apply lighting and color transfer
        matched_source = warped_source.copy()
        matched_source[ty : ty + th, tx : tx + tw] = _reinhard_color_transfer(
            warped_crop, tgt_crop
        )

        # Create smooth elliptical boundary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(tx + tw / 2), int(ty + th / 2))
        axes = (int(tw * 0.44), int(th * 0.54))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 10)

        # Seamless Poisson cloning
        try:
            blended = cv2.seamlessClone(
                matched_source, target_bgr, mask, center, cv2.NORMAL_CLONE
            )
            return blended
        except Exception:
            # Alpha blend fallback
            alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            blended = (
                matched_source.astype(np.float32) * alpha
                + target_bgr.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)
            return blended

    except Exception as e:
        logger.warning(f"Face swap failed: {e}")
        return target_bgr


def composite_face_into_image_bytes(
    scene_image_bytes: bytes, user_photo_bytes: bytes
) -> bytes:
    """Composite the user's authentic face into a generated storyboard scene.

    If faces are detected in both the user photo and storyboard, seamlessly swaps
    the user's face onto the storyboard character.
    """
    if not user_photo_bytes or not scene_image_bytes:
        return scene_image_bytes

    try:
        src_pil = Image.open(io.BytesIO(user_photo_bytes)).convert("RGB")
        tgt_pil = Image.open(io.BytesIO(scene_image_bytes)).convert("RGB")

        src_bgr = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2BGR)
        tgt_bgr = cv2.cvtColor(np.array(tgt_pil), cv2.COLOR_RGB2BGR)

        src_faces = detect_faces(src_bgr, score_threshold=0.3)
        tgt_faces = detect_faces(tgt_bgr, score_threshold=0.2)

        if not src_faces or not tgt_faces:
            logger.info(
                f"Face detection count: src={len(src_faces)}, tgt={len(tgt_faces)}; keeping original."
            )
            return scene_image_bytes

        src_face = max(src_faces, key=lambda f: f["score"])
        # Main protagonist in the scene is the largest face by bounding box area
        tgt_face = max(tgt_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])

        logger.info(
            f"Locking exact face onto storyboard character (conf: src={src_face['score']:.2f}, tgt={tgt_face['score']:.2f})"
        )
        blended_bgr = swap_face(src_bgr, tgt_bgr, src_face, tgt_face)

        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        out_pil = Image.fromarray(blended_rgb)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Error in composite_face_into_image_bytes: {e}")
        return scene_image_bytes


def apply_face_lock_to_video(
    input_video_path: str, user_photo_bytes: bytes, output_video_path: str
) -> bool:
    """Process a generated video frame-by-frame to lock the user's exact face."""
    if not user_photo_bytes or not os.path.exists(input_video_path):
        return False

    try:
        src_pil = Image.open(io.BytesIO(user_photo_bytes)).convert("RGB")
        src_bgr = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2BGR)
        src_faces = detect_faces(src_bgr, score_threshold=0.3)

        if not src_faces:
            logger.warning("No face detected in user photo for video face locking.")
            return False

        src_face = max(src_faces, key=lambda f: f["score"])

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open input video {input_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Applying Face-Lock to video ({total_frames} frames @ {fps} fps, {width}x{height})"
        )

        temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

        # Temporal smoothing buffer for target face landmarks
        prev_landmarks = None
        smoothing_alpha = 0.75  # 75% current frame, 25% previous frame for stability

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tgt_faces = detect_faces(frame, score_threshold=0.2)
            if tgt_faces:
                # Main character in video is the largest face by area
                tgt_face = max(tgt_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
                if prev_landmarks is not None:
                    # Smooth landmarks across frames
                    tgt_face["landmarks"] = (
                        smoothing_alpha * tgt_face["landmarks"]
                        + (1.0 - smoothing_alpha) * prev_landmarks
                    )
                prev_landmarks = tgt_face["landmarks"].copy()

                frame = swap_face(src_bgr, frame, src_face, tgt_face)
            else:
                prev_landmarks = None

            writer.write(frame)

        cap.release()
        writer.release()

        # Re-encode with ffmpeg to h264 for web/GCS streaming compatibility
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            temp_out,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if os.path.exists(temp_out):
            os.remove(temp_out)

        logger.info(f"Successfully locked exact face in video -> {output_video_path}")
        return True

    except Exception as e:
        logger.error(f"Error in apply_face_lock_to_video: {e}")
        return False


def _load_image_bytes(
    path_or_uri: str, project_id: str = "", bucket_name: str = ""
) -> bytes | None:
    """Helper to load image bytes from GCS, local path, or URL."""
    if not path_or_uri:
        return None

    try:
        normalized = path_or_uri.strip()
        authorized_uri = "https://storage.mtls.cloud.google.com/"
        if normalized.startswith(authorized_uri):
            normalized = normalized.replace(authorized_uri, "gs://")
        elif normalized.startswith("https://storage.googleapis.com/"):
            normalized = normalized.replace("https://storage.googleapis.com/", "gs://")

        # Case 1: GCS URI
        if normalized.startswith("gs://"):
            storage_client = storage.Client(project=project_id or os.getenv("GOOGLE_CLOUD_PROJECT"))
            blob = storage.Blob.from_string(normalized, client=storage_client)
            return blob.download_as_bytes()

        # Case 2: Local file path
        if os.path.exists(normalized):
            with open(normalized, "rb") as f:
                return f.read()

        # Case 3: Public HTTP/HTTPS URL
        if normalized.startswith("http://") or normalized.startswith("https://"):
            ctx = urllib.request.ssl._create_unverified_context()
            req = urllib.request.Request(
                normalized, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, context=ctx) as resp:
                return resp.read()

    except Exception as e:
        logger.warning(f"Could not load image bytes from '{path_or_uri}': {e}")

    return None


def resolve_user_photo_bytes_and_uri(
    context: Any, photo_path_or_url: str = ""
) -> tuple[bytes | None, str]:
    """Resolves and loads user photo bytes and GCS URI across all input modalities.

    Works seamlessly with both CallbackContext (from director_agent) and ToolContext
    (from sub-agents). Checks state, incoming user content, session history, and
    raw byte uploads, persisting the photo to GCS so it is never dropped.
    """
    try:
        session = getattr(context, "session", None)
        if session is None and hasattr(context, "_invocation_context"):
            session = context._invocation_context.session

        state = getattr(context, "state", None)
        if state is None and session is not None:
            state = session.state

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME", "")
        session_id = session.id if session else "default"

        # 1. If explicit path/url was passed
        if photo_path_or_url:
            raw_bytes = _load_image_bytes(photo_path_or_url, project_id, bucket_name)
            if raw_bytes:
                return raw_bytes, photo_path_or_url

        # 2. If already saved in GCS
        if state and state.get("user_photo_gcs_uri"):
            gcs_uri = str(state["user_photo_gcs_uri"])
            raw_bytes = _load_image_bytes(gcs_uri, project_id, bucket_name)
            if raw_bytes:
                return raw_bytes, gcs_uri

        # 3. If local or web URI is in state
        if state and state.get("user_photo_uri"):
            uri = str(state["user_photo_uri"])
            raw_bytes = _load_image_bytes(uri, project_id, bucket_name)
            if raw_bytes:
                return raw_bytes, uri

        # 4. Check incoming user_content if on CallbackContext
        user_content = getattr(context, "user_content", None)
        if user_content and getattr(user_content, "parts", None):
            for part in user_content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    raw_bytes = part.inline_data.data
                    if project_id and bucket_name and session_id:
                        try:
                            storage_client = storage.Client(project=project_id)
                            bucket = storage_client.bucket(bucket_name)
                            blob = bucket.blob(f"{session_id}/user_photo.png")
                            blob.upload_from_string(raw_bytes, content_type="image/png")
                            gcs_uri = f"gs://{bucket_name}/{session_id}/user_photo.png"
                            if state:
                                state["user_photo_gcs_uri"] = gcs_uri
                                state["user_photo_uri"] = gcs_uri
                            return raw_bytes, gcs_uri
                        except Exception as e:
                            logger.warning(f"Failed to upload user photo to GCS: {e}")
                    return raw_bytes, "inline_upload"
                elif getattr(part, "file_data", None) and part.file_data.file_uri:
                    uri = part.file_data.file_uri
                    raw_bytes = _load_image_bytes(uri, project_id, bucket_name)
                    if raw_bytes:
                        if state:
                            state["user_photo_uri"] = uri
                        return raw_bytes, uri
                elif getattr(part, "text", None):
                    match = re.search(
                        r"(https?://\S+\.(?:png|jpe?g|webp)|gs://\S+\.(?:png|jpe?g|webp)|/[^\s'\"<>]+\.(?:png|jpe?g|webp)|[a-zA-Z0-9_\-./]+\.(?:png|jpe?g|webp))",
                        part.text,
                        re.IGNORECASE,
                    )
                    if match:
                        uri = match.group(1).strip("'\")")
                        raw_bytes = _load_image_bytes(uri, project_id, bucket_name)
                        if raw_bytes:
                            if state:
                                state["user_photo_uri"] = uri
                            return raw_bytes, uri

        # 5. Scan session events (checking inline_data, file_data, and text)
        if session and getattr(session, "events", None):
            for event in reversed(session.events):
                if event.content and getattr(event.content, "parts", None):
                    for part in event.content.parts:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            raw_bytes = part.inline_data.data
                            if project_id and bucket_name and session_id:
                                try:
                                    storage_client = storage.Client(project=project_id)
                                    bucket = storage_client.bucket(bucket_name)
                                    blob = bucket.blob(f"{session_id}/user_photo.png")
                                    blob.upload_from_string(raw_bytes, content_type="image/png")
                                    gcs_uri = f"gs://{bucket_name}/{session_id}/user_photo.png"
                                    if state:
                                        state["user_photo_gcs_uri"] = gcs_uri
                                        state["user_photo_uri"] = gcs_uri
                                    return raw_bytes, gcs_uri
                                except Exception as e:
                                    logger.warning(f"Failed to upload user photo to GCS: {e}")
                            return raw_bytes, "inline_upload"
                        elif getattr(part, "file_data", None) and part.file_data.file_uri:
                            uri = part.file_data.file_uri
                            raw_bytes = _load_image_bytes(uri, project_id, bucket_name)
                            if raw_bytes:
                                if state:
                                    state["user_photo_uri"] = uri
                                return raw_bytes, uri
                        elif getattr(part, "text", None):
                            match = re.search(
                                r"(https?://\S+\.(?:png|jpe?g|webp)|gs://\S+\.(?:png|jpe?g|webp)|/[^\s'\"<>]+\.(?:png|jpe?g|webp)|[a-zA-Z0-9_\-./]+\.(?:png|jpe?g|webp))",
                                part.text,
                                re.IGNORECASE,
                            )
                            if match:
                                uri = match.group(1).strip("'\")")
                                raw_bytes = _load_image_bytes(uri, project_id, bucket_name)
                                if raw_bytes:
                                    if state:
                                        state["user_photo_uri"] = uri
                                    return raw_bytes, uri
    except Exception as ex:
        logger.warning(f"Error resolving user photo bytes: {ex}")

    return None, ""

