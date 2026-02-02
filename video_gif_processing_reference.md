# Video/GIF Processing Reference

This document contains the extracted video and GIF processing code from Super Browser for future re-implementation.

## Overview

The video processing pipeline:
1. Downloads video/GIF from source URL
2. Extracts frames using FFmpeg
3. Runs NudeNet detection on each frame
4. Applies censoring (black boxes) to detected regions
5. Reassembles frames back into video

## Dependencies

```bash
# FFmpeg must be installed on the system
apt-get install ffmpeg

# Or on Railway, add to nixpacks.toml:
# [phases.setup]
# nixPkgs = ["ffmpeg"]
```

### Python Requirements

```
# Already in requirements.txt for image processing:
onnxruntime>=1.18.0
numpy>=1.24.0
Pillow>=10.4.0
httpx>=0.26.0
```

## Configuration (Environment Variables)

```python
# Video processing configuration
VIDEO_CACHE_DIR = Path(os.getenv("VIDEO_CACHE_DIR", "video_cache"))
VIDEO_KEYFRAME_INTERVAL_SECONDS = float(os.getenv("VIDEO_KEYFRAME_INTERVAL_SECONDS", "1.0"))
VIDEO_SCENE_CHANGE_THRESHOLD = float(os.getenv("VIDEO_SCENE_CHANGE_THRESHOLD", "0.15"))
VIDEO_MAX_DURATION = int(os.getenv("VIDEO_MAX_DURATION", "60"))  # Max 60 seconds
VIDEO_TARGET_FPS = int(os.getenv("VIDEO_TARGET_FPS", "15"))  # Process at 15fps
VIDEO_USE_DUAL_MODELS = os.getenv("VIDEO_USE_DUAL_MODELS", "false").lower() == "true"
VIDEO_PRIMARY_MODEL = os.getenv("VIDEO_PRIMARY_MODEL", "320n")  # or "640m"
VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_BATCH_SIZE", "20"))  # Frames per batch
VIDEO_VERBOSE_LOGGING = os.getenv("VIDEO_VERBOSE_LOGGING", "true").lower() == "true"

# Ensure directory exists
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

## VideoCensorProcessor Class

```python
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

class VideoCensorProcessor:
    """Process and censor videos/GIFs using FFmpeg and NudeNet."""

    def __init__(self, censor: NudeNetCensor):
        self.censor = censor
        self._http_client: httpx.AsyncClient | None = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        return self._http_client

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def download_video(self, url: str) -> bytes:
        """Download video from URL."""
        response = await self.http_client.get(url)
        response.raise_for_status()
        return response.content

    async def process_video(
        self,
        video_bytes: bytes,
        output_format: str = "mp4",
        max_duration: int | None = None
    ) -> tuple[bytes, dict]:
        """
        Process and censor a video.

        Returns:
            Tuple of (censored_video_bytes, processing_stats)
        """
        max_duration = max_duration or VIDEO_MAX_DURATION

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.mp4"
            frames_dir = temp_path / "frames"
            censored_dir = temp_path / "censored"
            output_path = temp_path / f"output.{output_format}"

            frames_dir.mkdir()
            censored_dir.mkdir()

            # Write input video
            input_path.write_bytes(video_bytes)

            # Get video info using ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-of", "json",
                str(input_path)
            ]

            probe_result = await asyncio.to_thread(
                subprocess.run, probe_cmd, capture_output=True, text=True
            )

            video_info = {"width": 0, "height": 0, "fps": 30, "duration": 0}
            try:
                probe_data = json.loads(probe_result.stdout)
                stream = probe_data.get("streams", [{}])[0]
                video_info["width"] = int(stream.get("width", 0))
                video_info["height"] = int(stream.get("height", 0))

                fps_str = stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = map(int, fps_str.split("/"))
                    video_info["fps"] = num / den if den else 30
                else:
                    video_info["fps"] = float(fps_str)

                video_info["duration"] = float(stream.get("duration", 0))
            except Exception as e:
                print(f"[Video] Probe error: {e}")

            # Limit duration
            duration_limit = min(video_info["duration"], max_duration) if video_info["duration"] > 0 else max_duration

            # Extract frames at target FPS
            target_fps = min(VIDEO_TARGET_FPS, video_info["fps"])
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-t", str(duration_limit),
                "-vf", f"fps={target_fps}",
                "-q:v", "2",
                str(frames_dir / "frame_%05d.jpg")
            ]

            await asyncio.to_thread(
                subprocess.run, extract_cmd, capture_output=True
            )

            # Get frame files
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            total_frames = len(frame_files)

            if total_frames == 0:
                raise RuntimeError("No frames extracted from video")

            print(f"[Video] Processing {total_frames} frames...")

            # Process frames in batches
            stats = {
                "total_frames": total_frames,
                "censored_frames": 0,
                "detections": 0
            }

            for i in range(0, total_frames, VIDEO_BATCH_SIZE):
                batch = frame_files[i:i + VIDEO_BATCH_SIZE]

                async def process_frame(frame_path: Path) -> tuple[Path, int]:
                    frame_bytes = frame_path.read_bytes()
                    censored_bytes, detections = await self.censor.censor_image(
                        frame_bytes,
                        use_dual_models=VIDEO_USE_DUAL_MODELS,
                        primary_model=VIDEO_PRIMARY_MODEL,
                        threshold=CENSOR_THRESHOLD
                    )

                    output_frame = censored_dir / frame_path.name
                    output_frame.write_bytes(censored_bytes)

                    return output_frame, len([d for d in detections if d["class"] in CENSOR_CLASSES])

                results = await asyncio.gather(*[process_frame(f) for f in batch])

                for _, detection_count in results:
                    if detection_count > 0:
                        stats["censored_frames"] += 1
                        stats["detections"] += detection_count

            # Reassemble video with FFmpeg
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(target_fps),
                "-i", str(censored_dir / "frame_%05d.jpg"),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]

            await asyncio.to_thread(
                subprocess.run, encode_cmd, capture_output=True
            )

            if not output_path.exists():
                raise RuntimeError("Failed to encode output video")

            output_bytes = output_path.read_bytes()
            stats["output_size"] = len(output_bytes)

            print(f"[Video] Complete: {stats}")
            return output_bytes, stats
```

## Usage in MediaProcessor

```python
class MediaProcessor:
    """Service for processing and serving censored media."""

    def __init__(self, censor: NudeNetCensor, rule34: Rule34Client):
        self.censor = censor
        self.rule34 = rule34
        self.video_processor = VideoCensorProcessor(censor)
        # ... rest of init

    async def get_or_process_media(self, post_id: int) -> tuple[str, str]:
        """Get or process censored media for a post."""
        # ... check cache ...

        file_type = post["file_type"]

        # Download original
        response = await self.http_client.get(file_url)
        original_bytes = response.content

        # Process based on type
        if file_type == "video":
            # Process video
            processed_bytes, stats = await self.video_processor.process_video(original_bytes)
            extension = "mp4"
            media_type = "video"
        elif file_type == "gif":
            # Process GIF as video (converts to mp4)
            processed_bytes, stats = await self.video_processor.process_video(
                original_bytes, output_format="mp4"
            )
            extension = "mp4"
            media_type = "gif"
        else:
            # Process image
            processed_bytes, detections = await self.censor.censor_image(
                original_bytes,
                threshold=CENSOR_THRESHOLD
            )
            extension = "jpg"
            media_type = "image"

        # Save and cache...
```

## Filtering Configuration (for re-enabling GIFs)

To re-enable GIF support, update `ALLOWED_FILE_TYPES`:

```python
# In configuration section:
ALLOWED_FILE_TYPES = ["image", "gif"]  # Add "video" for full video support
```

## FFmpeg Commands Reference

### Extract frames from video
```bash
ffmpeg -y -i input.mp4 -t 60 -vf "fps=15" -q:v 2 frames/frame_%05d.jpg
```
- `-t 60`: Limit to 60 seconds
- `-vf "fps=15"`: Extract at 15 FPS
- `-q:v 2`: High quality JPEG

### Get video info
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of json input.mp4
```

### Reassemble frames to video
```bash
ffmpeg -y -framerate 15 -i frames/frame_%05d.jpg -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart output.mp4
```
- `-c:v libx264`: H.264 codec
- `-preset fast`: Fast encoding
- `-crf 23`: Quality (lower = better, 18-28 typical)
- `-pix_fmt yuv420p`: Compatibility
- `-movflags +faststart`: Web streaming optimization

## Notes

1. **GIF Duration**: Rule34 API doesn't provide GIF duration, so we can't filter by length without downloading
2. **Memory**: Processing many frames in parallel can use significant memory - adjust `VIDEO_BATCH_SIZE`
3. **CPU**: ONNX inference is CPU-bound - consider `VIDEO_USE_DUAL_MODELS=false` for faster processing
4. **Temp Storage**: Frame extraction uses temp directory - ensure adequate disk space
