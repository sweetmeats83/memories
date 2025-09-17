import json, os, shutil, subprocess, uuid, mimetypes, shlex, asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
from PIL import Image
from datetime import datetime
from tempfile import NamedTemporaryFile

BOOMERANG_MAX_SEC = 5.0  # under 5s => boomerang
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
# ---- Strategy for bucketed paths ----
class UserBucketsStrategy:
    """
    Places files under:
      admin/prompts/<prompt_id>/<media_id>/
      users/<user_slug-or-id>/responses/<response_id>/(primary|supporting)/<id>/
    """
    def __init__(self, users_root="users", admin_root="admin/prompts"):
        self.users_root = users_root
        self.admin_root = admin_root

    def prompt_dir(self, prompt_id: int, media_id: int) -> Path:
        return Path(self.admin_root) / str(prompt_id) / str(media_id)

    def response_primary_dir(self, user_slug_or_id: str, response_id: int) -> Path:
        return Path(self.users_root) / str(user_slug_or_id) / "responses" / str(response_id) / "primary"

    def response_supporting_dir(self, user_slug_or_id: str, response_id: int, media_id: int) -> Path:
        return Path(self.users_root) / str(user_slug_or_id) / "responses" / str(response_id) / "supporting" / str(media_id)

# ---- Artifact & result payload ----
@dataclass
class MediaArtifact:
    playable_rel: str
    wav_rel: Optional[str]
    thumb_rel: Optional[str]
    mime_type: str
    duration_sec: Optional[float]
    codec_a: Optional[str]
    codec_v: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    width: Optional[int]
    height: Optional[int]
    size_bytes: int

# ---- Pipeline ----
class MediaPipeline:
    def __init__(self, static_root: Path, path_strategy: UserBucketsStrategy):
        self.static_root = static_root
        self.upload_root = static_root / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.strategy = path_strategy

    # ffprobe helpers
    def _ffprobe(self, abspath: Path) -> dict:
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(abspath)
        ]
        out = subprocess.run(cmd, check=True, capture_output=True)
        return json.loads(out.stdout.decode("utf-8", "ignore"))

    def _probe_fields(self, abspath: Path) -> dict:
        meta = self._ffprobe(abspath)
        fmt = meta.get("format", {}) or {}
        streams = meta.get("streams", []) or []
        a = next((s for s in streams if s.get("codec_type")=="audio"), None)
        v = next((s for s in streams if s.get("codec_type")=="video"), None)
        return {
            "duration_sec": float(fmt.get("duration")) if fmt.get("duration") else None,
            "codec_a": a.get("codec_name") if a else None,
            "codec_v": v.get("codec_name") if v else None,
            "sample_rate": int(a.get("sample_rate")) if a and a.get("sample_rate") else None,
            "channels": int(a.get("channels")) if a and a.get("channels") else None,
            "width": int(v.get("width")) if v and v.get("width") else None,
            "height": int(v.get("height")) if v and v.get("height") else None,
        }

    # transcodes


    def _ffprobe_duration_seconds(src: Path) -> float | None:
        """
        Return duration in seconds (float) via ffprobe, or None if unknown.
        """
        try:
            cmd = f'{FFPROBE} -v error -select_streams v:0 -show_entries stream=duration ' \
                f'-of default=nokey=1:noprint_wrappers=1 {shlex.quote(str(src))}'
            out = subprocess.check_output(cmd, shell=True, text=True).strip()
            if not out or out == "N/A":
                # fallback to container duration
                cmd2 = f'{FFPROBE} -v error -show_entries format=duration ' \
                    f'-of default=nokey=1:noprint_wrappers=1 {shlex.quote(str(src))}'
                out = subprocess.check_output(cmd2, shell=True, text=True).strip()
            return float(out) if out and out != "N/A" else None
        except Exception:
            return None

    def _to_boomerang_mp4(
        self,
        src: Path,
        dst: Path,
        *,
        crf: int = 23,
        preset: str = "veryfast",
        scale_filter: str | None = None,
        mute_audio: bool = True,
    ):
        """
        Create a 'boomerang' (forward + reverse) H.264 MP4.
        By default mutes audio to avoid reversed-audio artifacts.
        """
        # Video: forward + reverse, concat
        # If scaling is desired, apply after concat
        if scale_filter:
            vf = (
                "[0:v]split[vf][vr];"
                "[vr]reverse[vrr];"
                "[vf][vrr]concat=n=2:v=1:a=0[vcat];"
                f"[vcat]{scale_filter}[v]"
            )
        else:
            vf = "[0:v]split[vf][vr];[vr]reverse[vrr];[vf][vrr]concat=n=2:v=1:a=0[v]"

        base = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-filter_complex", vf,
            "-map", "[v]",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            str(dst),
        ]

        if mute_audio:
            subprocess.check_call(base)
            return

        # Optional palindromic audio; try and fall back to mute if it fails
        full = (
            vf + ";"
            "[0:a]asplit[af][ar];[ar]areverse[arr];[af][arr]concat=n=2:v=0:a=1[a]"
        )
        with_audio = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-filter_complex", full,
            "-map", "[v]", "-map", "[a]",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", preset, "-crf", str(crf),
            str(dst),
        ]
        try:
            subprocess.check_call(with_audio)
        except subprocess.CalledProcessError:
            subprocess.check_call(base)
    
    def _to_m4a(self, src: Path, dst: Path):
        """
        Normalize ALL audio we ingest to a consistent format so concat never fails:
        AAC, 128k, 44.1kHz, stereo.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vn",
            "-c:a", "aac", "-b:a", "128k",
            "-ar", "44100", "-ac", "2",
            str(dst),
        ]
        subprocess.run(cmd, check=True)

    def _to_mp4(self, src: Path, dst: Path):
        """
        Normalize video too (so the *audio track* is uniform when later concatenated).
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-ar", "44100", "-ac", "2",
            str(dst),
        ]
        subprocess.run(cmd, check=True)

    def _to_wav(self, src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            # Bubble up details to the log; much easier to debug
            raise RuntimeError(f"ffmpeg to_wav failed: {e.stderr or e}") from e


    def _thumbnail(self, src_video: Path, dst_thumb: Path):
        dst_thumb.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg","-y","-ss","0.5","-i",str(src_video),
            "-frames:v","1","-vf","scale=640:-1",str(dst_thumb)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[pipeline] thumbnail failed for {src_video} -> {dst_thumb}\n{e.stderr}")
            try:
                if dst_thumb.exists() and dst_thumb.stat().st_size == 0:
                    dst_thumb.unlink()
            except Exception:
                pass

    def _image_thumb(self, src: Path, dst: Path, size: tuple[int, int] = (320, 320)):
        """
        Generate a thumbnail for an image file using Pillow.
        src: original uploaded image
        dst: output path for thumbnail
        size: max width/height of thumbnail
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img.thumbnail(size)
            img = img.convert("RGB")   # ensure JPEG-compatible
            img.save(dst, format="JPEG", quality=85)

        # main
    def process_upload(
        self,
        *,
        temp_path: Path,
        logical: Literal["prompt","response"],
        role:    Literal["prompt","primary","supporting"],
        user_slug_or_id: Optional[str],
        prompt_id: Optional[int],
        response_id: Optional[int],
        media_id: Optional[int],
        original_filename: str,
        content_type: Optional[str]
    ) -> MediaArtifact:
        """
        Store uploaded media in the correct bucket, normalize format for playback,
        and return paths/metadata. IMPORTANT: no WAV is produced here — WAV is
        created by transcription.py from the uploaded temp file.
        """
        # Per-artifact random suffix to reduce guessability of URLs
        rand = uuid.uuid4().hex[:8]

        # ---- 1) Decide target folder (based on your existing strategy) ----
        if logical == "prompt":
            if prompt_id is None or media_id is None:
                raise ValueError("prompt_id and media_id are required for logical='prompt'")
            target_dir = (self.upload_root / self.strategy.prompt_dir(prompt_id, media_id)).resolve()

        elif logical == "response":
            if user_slug_or_id is None or response_id is None:
                raise ValueError("user_slug_or_id and response_id are required for logical='response'")
            safe_user = (user_slug_or_id or "").strip().replace("\\", "_").replace("/", "_")
            if role == "primary":
                target_dir = (self.upload_root / self.strategy.response_primary_dir(safe_user, response_id)).resolve()
            elif role == "supporting":
                if media_id is None:
                    raise ValueError("media_id is required for supporting media")
                target_dir = (self.upload_root / self.strategy.response_supporting_dir(safe_user, response_id, media_id)).resolve()
            else:
                # 'role' can also be 'prompt' here, but in your app 'prompt' role is used only with logical='prompt'
                raise ValueError(f"Unsupported role for logical='response': {role}")
        else:
            raise ValueError(f"Unknown logical bucket: {logical}")
        # Prevent directory escape: ensure under uploads root
        uploads_root = self.upload_root.resolve()
        if uploads_root not in target_dir.parents and target_dir != uploads_root:
            raise ValueError("calculated target directory escapes uploads root")
        target_dir.mkdir(parents=True, exist_ok=True)

        # ---- 2) Robust media-type choice (images via Pillow; A/V by probing) ----
        ctype = (content_type or "").lower()
        ext = temp_path.suffix.lower()

        looks_like_image = (
            ctype.startswith("image/")
            or ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
        )
        looks_like_video = (
            ctype.startswith("video/")
            or ext in {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
        )
        looks_like_audio = (
            ctype.startswith("audio/")
            or ext in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aiff", ".aif", ".alac", ".mka"}
        )

        # We will fill these and return
        thumb_abs = None
        wav_abs = None  # always None — WAV is owned by transcription.py

        if looks_like_image:
            # ---- 2a) Image: normalize to JPEG and make a Pillow thumbnail ----
            playable_name = f"playback-{rand}.jpg"
            thumb_name = f"thumb-{rand}.jpg"
            playable_abs = target_dir / playable_name
            thumb_abs = target_dir / thumb_name

            with Image.open(temp_path) as img:
                img = img.convert("RGB")
                img.save(playable_abs, format="JPEG", quality=90)

            self._image_thumb(playable_abs, thumb_abs)
            mime_type = "image/jpeg"

            with Image.open(playable_abs) as i2:
                width, height = i2.size

            fields = {
                "duration_sec": None, "codec_a": None, "codec_v": None,
                "sample_rate": None, "channels": None, "width": width, "height": height,
            }

        else:
            # ---- 2b) Non-image: probe to distinguish true video vs audio-only ----
            src_fields = self._probe_fields(temp_path)
            has_audio = bool(src_fields.get("codec_a"))
            has_video = bool(src_fields.get("codec_v"))

            if has_video:
                # Treat as video → normalize to MP4 (H.264/AAC) + thumbnail
                playable_name = f"playback-{rand}.mp4"
                thumb_name = f"thumb-{rand}.jpg"
                playable_abs = target_dir / playable_name
                thumb_abs = target_dir / thumb_name

                # Decide boomerang based on source duration
                duration_src = src_fields.get("duration_sec")
                make_boomerang = (duration_src is not None) and (float(duration_src) < 5.0)

                if make_boomerang:
                    # 👇 mark with a distinct filename
                    playable_name = f"playback_boomerang-{rand}.mp4"
                else:
                    playable_name = f"playback-{rand}.mp4"

                thumb_name = f"thumb-{rand}.jpg"
                playable_abs = target_dir / playable_name
                thumb_abs = target_dir / thumb_name

                if make_boomerang:
                    scale_filter = None  # or your normal scale rule
                    self._to_boomerang_mp4(
                        temp_path, playable_abs,
                        crf=23, preset="veryfast",
                        scale_filter=scale_filter,
                        mute_audio=True,
                    )
                else:
                    self._to_mp4(temp_path, playable_abs)

                fields = self._probe_fields(playable_abs)
                mime_type = "video/mp4"

                # Only video gets a thumbnail
                try:
                    self._thumbnail(playable_abs, thumb_abs)
                except subprocess.CalledProcessError:
                    # Don't fail the request just for a thumb
                    thumb_abs = None

            elif has_audio or looks_like_audio or looks_like_video:
                # Audio-only (or “video” container with no video stream) → normalize to M4A (AAC)
                playable_name = f"playback-{rand}.m4a"
                playable_abs = target_dir / playable_name

                self._to_m4a(temp_path, playable_abs)
                fields = self._probe_fields(playable_abs)
                mime_type = "audio/mp4"  # m4a

            else:
                # Fallback: store as-is
                playable_name = f"file-{rand}{temp_path.suffix.lower()}"
                playable_abs = target_dir / playable_name
                shutil.copy2(temp_path, playable_abs)

                fields = {
                    "duration_sec": None, "codec_a": None, "codec_v": None,
                    "sample_rate": None, "channels": None, "width": None, "height": None,
                }
                mime_type = mimetypes.guess_type(playable_abs.name)[0] or "application/octet-stream"

        # ---- 3) Return artifact (note: wav_rel is always None here) ----
        size_bytes = playable_abs.stat().st_size
        try:
            return MediaArtifact(
                playable_rel=str(playable_abs.relative_to(self.static_root)),
                wav_rel=None,  # transcription.py handles temp WAV creation
                thumb_rel=(str(thumb_abs.relative_to(self.static_root)) if thumb_abs else None),
                mime_type=mime_type,
                duration_sec=fields["duration_sec"],
                codec_a=fields["codec_a"],
                codec_v=fields["codec_v"],
                sample_rate=fields["sample_rate"],
                channels=fields["channels"],
                width=fields["width"],
                height=fields["height"],
                size_bytes=size_bytes,
            )
        finally:
            # Always delete temp upload to reduce disk usage
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


        
    def delete_artifacts(self, *rel_paths: Optional[str]):
        uploads_root = (self.static_root / "uploads").resolve()
        for rp in rel_paths:
            if not rp:
                continue
            # accept either "uploads/..." or "static/uploads/..."
            rp_norm = rp.lstrip("/").replace("\\", "/")
            if rp_norm.startswith("static/"):
                rp_norm = rp_norm[len("static/"):]
            abspath = (self.static_root / rp_norm).resolve()
            try:
                if abspath.exists():
                    abspath.unlink()
                    # prune empty folders up to uploads/
                    cur = abspath.parent
                    while cur != uploads_root and uploads_root in cur.parents:
                        try:
                            cur.rmdir()
                        except OSError:
                            break
                        cur = cur.parent
            except Exception:
                pass



    async def concat_audio_async(self, *, sources_rel: list[str], response_id: int) -> dict:
        if not sources_rel:
            return {"ok": False, "error": "no sources"}

        uploads_root = self.static_root / "uploads"
        out_dir = uploads_root / "responses" / str(response_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Normalize each input (any container) -> .m4a (AAC, 44.1k, 2ch)
        norm_dir = out_dir / "_norm"
        norm_dir.mkdir(parents=True, exist_ok=True)
        norm_paths = []

        for i, rel in enumerate(sources_rel):
            src_abs = (uploads_root / rel).resolve()
            if not src_abs.exists() or src_abs.stat().st_size == 0:
                return {"ok": False, "error": f"missing_or_empty: {rel}"}
            tgt_abs = norm_dir / f"part-{i:03d}.m4a"
            p = await asyncio.create_subprocess_exec(
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(src_abs),
                "-map", "a:0?", "-vn",
                "-c:a", "aac", "-b:a", "128k",
                "-ar", "44100", "-ac", "2",
                "-movflags", "+faststart",
                "-y", str(tgt_abs),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, se = await p.communicate()
            if p.returncode != 0:
                return {"ok": False, "error": f"normalize failed for {rel}: {(se or b'').decode().strip()}"}
            if not tgt_abs.exists() or tgt_abs.stat().st_size == 0:
                return {"ok": False, "error": f"normalized empty: {tgt_abs.name}"}
            norm_paths.append(tgt_abs)

        # 2) Concat normalized parts via concat demuxer
        from datetime import datetime
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        final_rel = f"responses/{response_id}/composite-{stamp}.m4a"
        final_abs = uploads_root / final_rel
        list_file = out_dir / f"concat-{stamp}.txt"
        try:
            with list_file.open("w", encoding="utf-8") as f:
                for pth in norm_paths:
                    s = str(pth).replace("'", "'\\''")
                    f.write(f"file '{s}'\n")

            p2 = await asyncio.create_subprocess_exec(
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-vn",
                "-c:a", "aac", "-b:a", "128k",
                "-ar", "44100", "-ac", "2",
                "-movflags", "+faststart",
                "-y", str(final_abs),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, se2 = await p2.communicate()
            if p2.returncode != 0:
                return {"ok": False, "error": f"ffmpeg concat failed: {(se2 or b'').decode().strip()}"}

            return {"ok": True, "dest_rel": final_rel}
        finally:
            try:
                list_file.unlink(missing_ok=True)
                for pth in norm_paths:
                    pth.unlink(missing_ok=True)
                if norm_dir.exists() and not any(norm_dir.iterdir()):
                    norm_dir.rmdir()
            except Exception:
                pass


