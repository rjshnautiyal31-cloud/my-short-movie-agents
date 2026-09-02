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

import argparse
import logging
import os
import re
import subprocess
import tempfile
from dotenv import load_dotenv
import imageio_ffmpeg
from google.cloud import storage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge_session(session_id: str, bucket_name: str | None = None, project_id: str | None = None) -> str:
    """Consolidates all scene video clips for a given session ID into a single movie."""
    bucket_name = bucket_name or os.getenv("GOOGLE_CLOUD_BUCKET_NAME", "my-short-movies")
    project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    authorized_uri = "https://storage.mtls.cloud.google.com/"

    print(f"Connecting to bucket gs://{bucket_name} for session {session_id}...")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=f"{session_id}/scene_"))
    scene_blobs = [b for b in blobs if b.name.endswith(".mp4") and not b.name.endswith("final_movie.mp4")]

    if not scene_blobs:
        raise ValueError(f"No scene MP4 files found under prefix: {session_id}/scene_ in bucket {bucket_name}")

    def get_scene_num(blob):
        match = re.search(r"scene_(\d+)", blob.name)
        return int(match.group(1)) if match else 999

    scene_blobs.sort(key=get_scene_num)
    print(f"Found {len(scene_blobs)} scene clips to consolidate:")
    for b in scene_blobs:
        print(f"  - {b.name}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded_files = []
        for i, b in enumerate(scene_blobs):
            local_path = os.path.join(tmpdir, f"scene_{i + 1}.mp4")
            b.download_to_filename(local_path)
            downloaded_files.append(local_path)
            print(f"Downloaded {b.name} -> {local_path} ({os.path.getsize(local_path)} bytes)")

        concat_file = os.path.join(tmpdir, "concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in downloaded_files:
                f.write(f"file '{p}'\n")

        merged_output = os.path.join(tmpdir, "final_movie.mp4")
        cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", merged_output]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            print(f"Fast concat failed, re-encoding: {res.stderr}")
            cmd_reencode = [
                ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264",
                "-c:a", "aac",
                merged_output,
            ]
            subprocess.run(cmd_reencode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f"Consolidated video created ({os.path.getsize(merged_output)} bytes)")
        final_blob_path = f"{session_id}/final_movie.mp4"
        final_blob = bucket.blob(final_blob_path)
        final_blob.upload_from_filename(merged_output, content_type="video/mp4")

        final_url = f"gs://{bucket_name}/{final_blob_path}".replace("gs://", authorized_uri)
        print(f"✅ Successfully uploaded final movie to: {final_url}")
        return final_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge scene clips from a session into a single final movie.")
    parser.add_argument("session_id", help="The session ID containing the scene clips (e.g. a0ea7f1f-b17d-4cb3-b3a8-aab6e8c91e58)")
    parser.add_argument("--bucket", default=None, help="Google Cloud Storage bucket name")
    parser.add_argument("--project", default=None, help="Google Cloud project ID")
    args = parser.parse_args()

    merge_session(args.session_id, args.bucket, args.project)
