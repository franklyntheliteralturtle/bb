"""
Super Browser - Poe Server Bot
A Poe-protocol-compatible server for browsing and processing RedGifs content.
Features automated nudity detection and censoring using direct ONNX inference.
"""

import os
import io
import json
import hashlib
import asyncio
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import AsyncIterable
from contextlib import asynccontextmanager

import httpx
import aiosqlite
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from dotenv import load_dotenv

import fastapi_poe as fp
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

import redgifs

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", "cache.db")
MEDIA_CACHE_DIR = Path(os.getenv("MEDIA_CACHE_DIR", "media_cache"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "models"))
POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY", "")
BOT_NAME = os.getenv("BOT_NAME", "SuperBrowser")
SERVER_URL = os.getenv("SERVER_URL", "")

# NudeNet model URLs - using dual models for better coverage
# 320n: Fast, lightweight model (~6MB)
# 640m: Larger, more accurate model (~25MB)
# Using Hugging Face mirrors for reliable downloads
NUDENET_320N_URLS = [
    "https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx",
    "https://huggingface.co/vladmandic/nudenet/resolve/main/nudenet.onnx",
]
NUDENET_640M_URLS = [
    "https://huggingface.co/spaces/xxparthparekhxx/NudeNet-FastAPI/resolve/794a185a301917f1a3505ab3b8d55b268ea81f0e/640m.onnx",
]

NUDENET_320N_PATH = MODEL_CACHE_DIR / "320n.onnx"
NUDENET_640M_PATH = MODEL_CACHE_DIR / "640m.onnx"
NUDENET_MODEL_MIN_SIZE_320 = 5 * 1024 * 1024   # Expect at least 5MB for 320n
NUDENET_MODEL_MIN_SIZE_640 = 20 * 1024 * 1024  # Expect at least 20MB for 640m

# Nudity censoring configuration
CENSOR_THRESHOLD = float(os.getenv("CENSOR_THRESHOLD", "0.4"))

# NudeNet class labels - supports both vladmandic and official 320n models
# The vladmandic model uses slightly different naming
NUDENET_LABELS_VLADMANDIC = [
    "female-private-area",   # 0 - covered
    "female-face",           # 1
    "buttocks-bare",         # 2 - NSFW
    "female-breast-bare",    # 3 - NSFW
    "female-vagina",         # 4 - NSFW
    "male-breast-bare",      # 5
    "anus-bare",             # 6 - NSFW
    "feet-bare",             # 7
    "belly",                 # 8 - covered
    "feet",                  # 9 - covered
    "armpits",               # 10 - covered
    "armpits-bare",          # 11
    "male-face",             # 12
    "belly-bare",            # 13
    "male-penis",            # 14 - NSFW
    "anus-area",             # 15 - covered
    "female-breast",         # 16 - covered
    "buttocks",              # 17 - covered
]

# Official NudeNet 320n model labels
NUDENET_LABELS_320N = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# Unified set of NSFW classes to censor (covers both naming conventions)
# Includes both exposed AND covered versions of sensitive areas
DEFAULT_CENSOR_CLASSES = [
    # vladmandic naming - exposed
    "buttocks-bare",
    "female-breast-bare",
    "female-vagina",
    "anus-bare",
    "male-penis",
    "male-breast-bare",
    # vladmandic naming - covered
    "female-private-area",
    "female-breast",
    "buttocks",
    "anus-area",
    # Official 320n naming - exposed
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    # Official 320n naming - covered
    "BUTTOCKS_COVERED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "ANUS_COVERED",
]

CENSOR_CLASSES = os.getenv("CENSOR_CLASSES", "").split(",") if os.getenv("CENSOR_CLASSES") else DEFAULT_CENSOR_CLASSES
CENSOR_CLASSES = [c.strip() for c in CENSOR_CLASSES if c.strip()]

# Video processing configuration
VIDEO_CACHE_DIR = Path(os.getenv("VIDEO_CACHE_DIR", "video_cache"))
VIDEO_KEYFRAME_INTERVAL = int(os.getenv("VIDEO_KEYFRAME_INTERVAL", "30"))  # frames (1 sec at 30fps)
VIDEO_SCENE_CHANGE_THRESHOLD = float(os.getenv("VIDEO_SCENE_CHANGE_THRESHOLD", "0.15"))  # 15% pixel diff
VIDEO_MAX_DURATION = int(os.getenv("VIDEO_MAX_DURATION", "60"))  # Max video duration in seconds
VIDEO_TARGET_FPS = int(os.getenv("VIDEO_TARGET_FPS", "30"))  # Target FPS for processing

# Ensure directories exist
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Direct ONNX NudeNet Detector (No OpenCV!)
# =============================================================================

class NudeDetectorONNX:
    """
    NudeNet detector using direct ONNX inference.
    No OpenCV dependency - uses only Pillow and numpy.
    """

    def __init__(
        self,
        model_path: Path,
        model_urls: list[str],
        min_size: int,
        input_size: int = 320,
        name: str = "detector"
    ):
        self.model_path = model_path
        self.model_urls = model_urls
        self.min_size = min_size
        self.input_size = input_size
        self.name = name
        self.session: ort.InferenceSession | None = None
        self.labels: list[str] = NUDENET_LABELS_VLADMANDIC  # Default, updated on load
        self._lock = asyncio.Lock()

    async def ensure_model_downloaded(self):
        """Download the ONNX model if not present or corrupted."""
        # Check if model exists and is valid size
        if self.model_path.exists():
            file_size = self.model_path.stat().st_size
            if file_size >= self.min_size:
                print(f"[{self.name}] Model already exists: {file_size / 1024 / 1024:.1f} MB")
                return
            else:
                print(f"[{self.name}] Model file too small ({file_size} bytes), re-downloading...")
                self.model_path.unlink()

        # Try each URL until one works
        last_error = None
        for url in self.model_urls:
            try:
                print(f"[{self.name}] Downloading model from {url}...")
                await self._download_model(url)
                print(f"[{self.name}] Model downloaded successfully: {self.model_path.stat().st_size / 1024 / 1024:.1f} MB")
                return
            except Exception as e:
                print(f"[{self.name}] Failed to download from {url}: {e}")
                last_error = e
                # Clean up partial download
                if self.model_path.exists():
                    self.model_path.unlink()

        raise RuntimeError(f"[{self.name}] Failed to download model from any source. Last error: {last_error}")

    async def _download_model(self, url: str):
        """Download model from a specific URL."""
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.content
            content_size = len(content)

            # Verify we got actual model data, not an HTML error page
            if content_size < self.min_size:
                raise RuntimeError(
                    f"Downloaded file too small ({content_size} bytes). "
                    f"Expected at least {self.min_size} bytes."
                )

            # Check it's not HTML
            if content[:100].lower().find(b'<!doctype') >= 0 or content[:100].lower().find(b'<html') >= 0:
                raise RuntimeError("Downloaded HTML instead of ONNX model file")

            self.model_path.write_bytes(content)

    async def load(self):
        """Load the ONNX model."""
        async with self._lock:
            if self.session is not None:
                return

            await self.ensure_model_downloaded()

            print(f"[{self.name}] Loading ONNX model...")
            # Run in thread to avoid blocking
            self.session = await asyncio.to_thread(
                ort.InferenceSession,
                str(self.model_path),
                providers=["CPUExecutionProvider"]
            )

            # Detect model type based on output shape
            output_shape = self.session.get_outputs()[0].shape
            print(f"[{self.name}] Model output shape: {output_shape}")

            # Determine input size from model
            input_shape = self.session.get_inputs()[0].shape
            print(f"[{self.name}] Model input shape: {input_shape}")

            # Update input_size based on model (if detectable from shape)
            # Shape is typically [batch, channels, height, width] = [1, 3, 320, 320] or [1, 3, 640, 640]
            if input_shape and len(input_shape) >= 4 and isinstance(input_shape[2], int):
                self.input_size = input_shape[2]
                print(f"[{self.name}] Using input size: {self.input_size}")

            # Both models should have 18 classes, but file size differs
            # vladmandic is ~12MB, 320n/640m are from official repo
            file_size = self.model_path.stat().st_size
            if file_size > 10 * 1024 * 1024 and file_size < 20 * 1024 * 1024:
                # 10-20MB = vladmandic model
                self.labels = NUDENET_LABELS_VLADMANDIC
                print(f"[{self.name}] Detected vladmandic/nudenet model")
            else:
                # Official models (320n ~6MB, 640m ~25MB)
                self.labels = NUDENET_LABELS_320N
                print(f"[{self.name}] Detected official NudeNet model")

            print(f"[{self.name}] Model loaded successfully with {len(self.labels)} classes")

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
        """
        Preprocess image for NudeNet model.

        Returns:
            - input_tensor: Preprocessed image tensor
            - original_size: (width, height) of original image
            - scale: (scale_x, scale_y) to map boxes back to original size
        """
        original_size = image.size  # (width, height)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size (320x320)
        resized = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)

        # Convert to numpy array and normalize to 0-1
        img_array = np.array(resized, dtype=np.float32) / 255.0

        # Transpose from HWC to CHW format (height, width, channels -> channels, height, width)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension: (1, 3, 320, 320) - NCHW format
        input_tensor = np.expand_dims(img_array, axis=0)

        # Calculate scale factors to map detections back to original size
        scale_x = original_size[0] / self.input_size
        scale_y = original_size[1] / self.input_size

        return input_tensor, original_size, (scale_x, scale_y)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale: tuple[float, float],
        threshold: float = 0.25
    ) -> list[dict]:
        """
        Postprocess model outputs to get detections.

        NudeNet 320n output format:
        - outputs[0]: shape (1, 18, 2100) - predictions
          - 18 = number of classes
          - 2100 = number of anchor boxes
          - Each column: [x_center, y_center, width, height, class_scores...]

        Returns list of detections with class, score, and box.
        """
        predictions = outputs[0]  # Shape: (1, num_features, num_boxes)

        # Squeeze batch dimension and transpose to (num_boxes, num_features)
        predictions = predictions[0].T  # Shape: (2100, 18+4) or similar

        # The model outputs: first 4 values are box coords, rest are class scores
        # Actually for YOLOv8 format: (batch, 4+num_classes, num_boxes)
        # After transpose: (num_boxes, 4+num_classes)

        scale_x, scale_y = scale
        detections = []

        for pred in predictions:
            # First 4 values: x_center, y_center, width, height (in input scale 0-320)
            x_center, y_center, w, h = pred[:4]

            # Remaining values are class scores
            class_scores = pred[4:]

            # Get best class
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            if score < threshold:
                continue

            # Convert from center format to corner format and scale to original size
            x1 = (x_center - w / 2) * scale_x
            y1 = (y_center - h / 2) * scale_y
            box_w = w * scale_x
            box_h = h * scale_y

            # Clamp to positive values
            x1 = max(0, x1)
            y1 = max(0, y1)

            detections.append({
                "class": self.labels[class_id] if class_id < len(self.labels) else f"CLASS_{class_id}",
                "score": round(score, 4),
                "box": [int(x1), int(y1), int(box_w), int(box_h)]
            })

        # Apply Non-Maximum Suppression
        detections = self._nms(detections, iou_threshold=0.45)

        return detections

    def _nms(self, detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if not detections:
            return []

        # Sort by score descending
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best["box"], d["box"]) < iou_threshold
            ]

        return keep

    def _iou(self, box1: list[int], box2: list[int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to corner format
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        # Calculate intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    async def detect(self, image_bytes: bytes, threshold: float = 0.25) -> list[dict]:
        """
        Detect nudity in an image.

        Args:
            image_bytes: Raw image bytes
            threshold: Minimum confidence score

        Returns:
            List of detections with class, score, and box
        """
        if self.session is None:
            await self.load()

        # Load image
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        input_tensor, original_size, scale = self._preprocess(image)

        # Get input name from model
        input_name = self.session.get_inputs()[0].name

        # Run inference
        outputs = await asyncio.to_thread(
            self.session.run,
            None,
            {input_name: input_tensor}
        )

        # Postprocess
        detections = self._postprocess(outputs, scale, threshold)

        return detections


# =============================================================================
# NudeNet Censor (using our ONNX detector)
# =============================================================================

class NudeNetCensor:
    """
    Wrapper that combines detection and censoring.
    Uses dual models (320n and 640m) for better coverage.
    """

    def __init__(self):
        # Initialize both detectors
        self.detector_320n = NudeDetectorONNX(
            model_path=NUDENET_320N_PATH,
            model_urls=NUDENET_320N_URLS,
            min_size=NUDENET_MODEL_MIN_SIZE_320,
            input_size=320,
            name="320n"
        )
        self.detector_640m = NudeDetectorONNX(
            model_path=NUDENET_640M_PATH,
            model_urls=NUDENET_640M_URLS,
            min_size=NUDENET_MODEL_MIN_SIZE_640,
            input_size=640,
            name="640m"
        )

    async def load(self):
        """Pre-load both models in parallel."""
        print("Loading dual NudeNet models...")
        await asyncio.gather(
            self.detector_320n.load(),
            self.detector_640m.load()
        )
        print("Both models loaded successfully!")

    async def censor_image(
        self,
        image_bytes: bytes,
        classes: list[str] | None = None,
        threshold: float = CENSOR_THRESHOLD,
    ) -> tuple[bytes, list[dict]]:
        """
        Detect and censor nudity in an image using dual models.
        Merges results from both 320n and 640m models for maximum coverage.

        Args:
            image_bytes: Raw image bytes
            classes: List of body part classes to censor
            threshold: Minimum confidence score for detection

        Returns:
            Tuple of (censored_image_bytes, all_detections)
        """
        classes_to_censor = classes or CENSOR_CLASSES

        # Run both detectors in parallel
        detections_320n, detections_640m = await asyncio.gather(
            self.detector_320n.detect(image_bytes, threshold=threshold),
            self.detector_640m.detect(image_bytes, threshold=threshold)
        )

        # Tag detections with source model for debugging
        for d in detections_320n:
            d["model"] = "320n"
        for d in detections_640m:
            d["model"] = "640m"

        # Merge all detections
        all_detections = detections_320n + detections_640m

        # Apply NMS across merged detections to remove duplicates
        merged_detections = self._merge_detections(all_detections)

        # Filter detections to censor
        filtered_detections = [
            d for d in merged_detections
            if d["class"] in classes_to_censor
        ]

        if not filtered_detections:
            # No nudity detected that needs censoring
            return image_bytes, merged_detections

        # Apply black boxes
        censored_bytes = await asyncio.to_thread(
            self._apply_black_boxes,
            image_bytes,
            filtered_detections
        )

        return censored_bytes, merged_detections

    def _merge_detections(self, detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """
        Merge detections from multiple models using NMS.
        Keeps the detection with highest score when boxes overlap.
        """
        if not detections:
            return []

        # Sort by score descending
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections of the same class
            detections = [
                d for d in detections
                if d["class"] != best["class"] or self._iou(best["box"], d["box"]) < iou_threshold
            ]

        return keep

    def _iou(self, box1: list[int], box2: list[int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def _apply_black_boxes(self, image_bytes: bytes, detections: list[dict]) -> bytes:
        """Apply black boxes over detected regions."""
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box
            # Draw black rectangle
            draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))

        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()


# Global NudeNet censor instance
nudenet_censor = NudeNetCensor()


# =============================================================================
# Video Processor - Frame-by-frame with Smart Keyframes & Interpolation
# =============================================================================

class VideoProcessor:
    """
    Processes videos by detecting nudity on keyframes and interpolating
    censor boxes smoothly between them.

    Features:
    - Smart keyframe detection (scene changes)
    - Time-distributed keyframes as fallback
    - Smooth box interpolation between keyframes
    """

    def __init__(self, censor: NudeNetCensor):
        self.censor = censor
        self.keyframe_interval = VIDEO_KEYFRAME_INTERVAL
        self.scene_change_threshold = VIDEO_SCENE_CHANGE_THRESHOLD
        self.target_fps = VIDEO_TARGET_FPS
        self.max_duration = VIDEO_MAX_DURATION

    async def process_video(self, video_bytes: bytes, quality: str = "sd") -> tuple[bytes, dict]:
        """
        Process a video with nudity censoring.

        Args:
            video_bytes: Raw video bytes
            quality: Output quality ("sd" or "hd")

        Returns:
            Tuple of (processed_video_bytes, processing_stats)
        """
        stats = {
            "total_frames": 0,
            "keyframes_detected": 0,
            "detections_total": 0,
            "processing_time_seconds": 0,
        }

        import time
        start_time = time.time()

        # Create temporary directory for frame processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_video = temp_path / "input.mp4"
            output_video = temp_path / "output.mp4"
            frames_dir = temp_path / "frames"
            processed_dir = temp_path / "processed"
            audio_file = temp_path / "audio.aac"

            frames_dir.mkdir()
            processed_dir.mkdir()

            # Write input video
            await asyncio.to_thread(input_video.write_bytes, video_bytes)

            # Get video info
            video_info = await self._get_video_info(input_video)
            fps = min(video_info.get("fps", 30), self.target_fps)
            duration = min(video_info.get("duration", 10), self.max_duration)
            has_audio = video_info.get("has_audio", False)

            # Extract frames
            await self._extract_frames(input_video, frames_dir, fps)

            # Extract audio if present
            if has_audio:
                await self._extract_audio(input_video, audio_file)

            # Get list of frames
            frame_files = sorted(frames_dir.glob("*.png"))
            stats["total_frames"] = len(frame_files)

            if not frame_files:
                raise ValueError("No frames extracted from video")

            # Identify keyframes and run detection
            keyframe_data = await self._process_keyframes(frame_files)
            stats["keyframes_detected"] = len(keyframe_data)
            stats["detections_total"] = sum(len(kf["detections"]) for kf in keyframe_data.values())

            # Process all frames with interpolation
            await self._apply_censorship_with_interpolation(
                frame_files,
                keyframe_data,
                processed_dir
            )

            # Encode output video
            await self._encode_video(
                processed_dir,
                output_video,
                fps,
                audio_file if has_audio and audio_file.exists() else None
            )

            # Read output
            output_bytes = await asyncio.to_thread(output_video.read_bytes)

            stats["processing_time_seconds"] = round(time.time() - start_time, 2)

            return output_bytes, stats

    async def _get_video_info(self, video_path: Path) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"ffprobe error: {result.stderr}")
            return {"fps": 30, "duration": 10, "has_audio": False}

        try:
            data = json.loads(result.stdout)

            # Find video stream
            fps = 30.0
            has_audio = False

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    # Parse frame rate (could be "30/1" or "29.97")
                    fps_str = stream.get("r_frame_rate", "30/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fps = float(num) / float(den) if float(den) != 0 else 30
                    else:
                        fps = float(fps_str)
                elif stream.get("codec_type") == "audio":
                    has_audio = True

            duration = float(data.get("format", {}).get("duration", 10))

            return {"fps": fps, "duration": duration, "has_audio": has_audio}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing video info: {e}")
            return {"fps": 30, "duration": 10, "has_audio": False}

    async def _extract_frames(self, video_path: Path, output_dir: Path, fps: float):
        """Extract frames from video using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality PNG
            str(output_dir / "frame_%06d.png"),
            "-y",
            "-loglevel", "error"
        ]

        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")

    async def _extract_audio(self, video_path: Path, audio_path: Path):
        """Extract audio track from video."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "aac",
            "-b:a", "128k",
            str(audio_path),
            "-y",
            "-loglevel", "error"
        ]

        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )

        # Audio extraction can fail if no audio - that's OK
        if result.returncode != 0:
            print(f"Audio extraction skipped: {result.stderr}")

    async def _process_keyframes(self, frame_files: list[Path]) -> dict[int, dict]:
        """
        Identify keyframes and run detection on them.

        Returns dict mapping frame index to detection data.
        """
        keyframe_data = {}
        prev_frame_array = None
        last_keyframe_idx = -self.keyframe_interval  # Force first frame to be keyframe

        for idx, frame_path in enumerate(frame_files):
            # Load frame for comparison
            frame_bytes = await asyncio.to_thread(frame_path.read_bytes)
            frame_img = Image.open(io.BytesIO(frame_bytes))
            frame_array = self._get_comparison_array(frame_img)

            # Determine if this is a keyframe
            is_keyframe = False

            # Always make first frame a keyframe
            if idx == 0:
                is_keyframe = True
            # Time-distributed keyframe (fallback interval)
            elif idx - last_keyframe_idx >= self.keyframe_interval:
                is_keyframe = True
            # Scene change detection
            elif prev_frame_array is not None:
                diff = self._frame_difference(prev_frame_array, frame_array)
                if diff > self.scene_change_threshold:
                    is_keyframe = True
                    print(f"Scene change detected at frame {idx}, diff={diff:.3f}")

            if is_keyframe:
                # Run detection on this keyframe
                detections = await self._detect_frame(frame_bytes)
                keyframe_data[idx] = {
                    "detections": detections,
                    "boxes": self._extract_censor_boxes(detections)
                }
                last_keyframe_idx = idx

            prev_frame_array = frame_array

        return keyframe_data

    def _get_comparison_array(self, image: Image.Image) -> np.ndarray:
        """Convert image to small grayscale array for fast comparison."""
        # Downsample to 64x64 grayscale for fast comparison
        small = image.resize((64, 64), Image.Resampling.BILINEAR).convert('L')
        return np.array(small, dtype=np.float32)

    def _frame_difference(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate normalized difference between two frame arrays."""
        diff = np.abs(arr1 - arr2)
        return float(np.mean(diff) / 255.0)

    async def _detect_frame(self, frame_bytes: bytes) -> list[dict]:
        """Run nudity detection on a single frame."""
        # Use the dual-model detection
        detections_320n, detections_640m = await asyncio.gather(
            self.censor.detector_320n.detect(frame_bytes, threshold=CENSOR_THRESHOLD),
            self.censor.detector_640m.detect(frame_bytes, threshold=CENSOR_THRESHOLD)
        )

        # Tag and merge
        for d in detections_320n:
            d["model"] = "320n"
        for d in detections_640m:
            d["model"] = "640m"

        all_detections = detections_320n + detections_640m
        return self.censor._merge_detections(all_detections)

    def _extract_censor_boxes(self, detections: list[dict]) -> list[dict]:
        """Extract boxes that need censoring."""
        return [
            {
                "class": d["class"],
                "box": d["box"],
                "score": d["score"]
            }
            for d in detections
            if d["class"] in CENSOR_CLASSES
        ]

    async def _apply_censorship_with_interpolation(
        self,
        frame_files: list[Path],
        keyframe_data: dict[int, dict],
        output_dir: Path
    ):
        """Apply censorship to all frames with smooth box interpolation."""

        if not keyframe_data:
            # No keyframes = no detections, just copy frames
            for idx, frame_path in enumerate(frame_files):
                shutil.copy(frame_path, output_dir / f"frame_{idx:06d}.png")
            return

        # Get sorted keyframe indices
        keyframe_indices = sorted(keyframe_data.keys())

        # Process frames in batches for efficiency
        batch_size = 10
        frame_batches = [
            frame_files[i:i + batch_size]
            for i in range(0, len(frame_files), batch_size)
        ]

        for batch_start_idx, batch in enumerate(frame_batches):
            tasks = []
            for local_idx, frame_path in enumerate(batch):
                frame_idx = batch_start_idx * batch_size + local_idx
                output_path = output_dir / f"frame_{frame_idx:06d}.png"

                # Get interpolated boxes for this frame
                boxes = self._get_interpolated_boxes(
                    frame_idx,
                    keyframe_indices,
                    keyframe_data
                )

                tasks.append(
                    self._process_single_frame(frame_path, output_path, boxes)
                )

            await asyncio.gather(*tasks)

    def _get_interpolated_boxes(
        self,
        frame_idx: int,
        keyframe_indices: list[int],
        keyframe_data: dict[int, dict]
    ) -> list[list[int]]:
        """
        Get interpolated censor boxes for a specific frame.
        Smoothly interpolates between keyframes.
        """
        # Find surrounding keyframes
        prev_kf_idx = None
        next_kf_idx = None

        for kf_idx in keyframe_indices:
            if kf_idx <= frame_idx:
                prev_kf_idx = kf_idx
            if kf_idx >= frame_idx and next_kf_idx is None:
                next_kf_idx = kf_idx
                break

        # If we're exactly on a keyframe, use its boxes directly
        if frame_idx in keyframe_data:
            return [b["box"] for b in keyframe_data[frame_idx]["boxes"]]

        # If no previous keyframe, use next (shouldn't happen with frame 0 as keyframe)
        if prev_kf_idx is None:
            if next_kf_idx is not None:
                return [b["box"] for b in keyframe_data[next_kf_idx]["boxes"]]
            return []

        # If no next keyframe, use previous
        if next_kf_idx is None:
            return [b["box"] for b in keyframe_data[prev_kf_idx]["boxes"]]

        # Interpolate between prev and next keyframes
        prev_boxes = keyframe_data[prev_kf_idx]["boxes"]
        next_boxes = keyframe_data[next_kf_idx]["boxes"]

        # Calculate interpolation factor (0 = prev, 1 = next)
        if next_kf_idx == prev_kf_idx:
            t = 0
        else:
            t = (frame_idx - prev_kf_idx) / (next_kf_idx - prev_kf_idx)

        # Match boxes between keyframes and interpolate
        return self._interpolate_box_sets(prev_boxes, next_boxes, t)

    def _interpolate_box_sets(
        self,
        prev_boxes: list[dict],
        next_boxes: list[dict],
        t: float
    ) -> list[list[int]]:
        """
        Interpolate between two sets of boxes.
        Matches boxes by class and IoU, then interpolates positions.
        """
        result = []
        used_next = set()

        # For each box in prev, find best match in next
        for prev_box in prev_boxes:
            best_match = None
            best_iou = 0.3  # Minimum IoU to consider a match

            for idx, next_box in enumerate(next_boxes):
                if idx in used_next:
                    continue
                if prev_box["class"] != next_box["class"]:
                    continue

                iou = self._calculate_iou(prev_box["box"], next_box["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = idx

            if best_match is not None:
                # Interpolate between matched boxes
                used_next.add(best_match)
                interpolated = self._interpolate_single_box(
                    prev_box["box"],
                    next_boxes[best_match]["box"],
                    t
                )
                result.append(interpolated)
            else:
                # No match - box is disappearing, keep it until halfway
                if t < 0.5:
                    result.append(prev_box["box"])

        # Add unmatched next boxes (appearing boxes) after halfway
        if t >= 0.5:
            for idx, next_box in enumerate(next_boxes):
                if idx not in used_next:
                    result.append(next_box["box"])

        return result

    def _interpolate_single_box(
        self,
        box1: list[int],
        box2: list[int],
        t: float
    ) -> list[int]:
        """Linearly interpolate between two boxes."""
        x1_a, y1_a, w_a, h_a = box1
        x1_b, y1_b, w_b, h_b = box2

        return [
            int(x1_a + (x1_b - x1_a) * t),
            int(y1_a + (y1_b - y1_a) * t),
            int(w_a + (w_b - w_a) * t),
            int(h_a + (h_b - h_a) * t),
        ]

    def _calculate_iou(self, box1: list[int], box2: list[int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    async def _process_single_frame(
        self,
        input_path: Path,
        output_path: Path,
        boxes: list[list[int]]
    ):
        """Apply censor boxes to a single frame and save."""
        frame_bytes = await asyncio.to_thread(input_path.read_bytes)

        if not boxes:
            # No censoring needed, just copy
            await asyncio.to_thread(shutil.copy, input_path, output_path)
            return

        # Apply black boxes
        img = Image.open(io.BytesIO(frame_bytes))

        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)

        for box in boxes:
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))

        # Save processed frame
        await asyncio.to_thread(img.save, output_path, format='PNG')

    async def _encode_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float,
        audio_path: Path | None = None
    ):
        """Encode processed frames back into a video."""
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
        ]

        # Add audio if available
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path)])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",  # Good quality
            "-pix_fmt", "yuv420p",  # Compatibility
        ])

        # Add audio codec if we have audio
        if audio_path and audio_path.exists():
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])

        cmd.extend([
            "-movflags", "+faststart",  # Web optimization
            str(output_path),
            "-y",
            "-loglevel", "error"
        ])

        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Video encoding failed: {result.stderr}")


# Global video processor instance
video_processor: VideoProcessor | None = None


# =============================================================================
# Database Setup
# =============================================================================

async def init_db():
    """Initialize SQLite database with required tables."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Main GIF cache table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS gif_cache (
                id TEXT PRIMARY KEY,
                thumbnail_url TEXT,
                hd_url TEXT,
                sd_url TEXT,
                web_url TEXT,
                width INTEGER,
                height INTEGER,
                duration REAL,
                tags TEXT,
                username TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tag index table - tracks which tags exist and their counts
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tag_index (
                tag TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # GIF-to-tag mapping for efficient tag-based queries
        await db.execute("""
            CREATE TABLE IF NOT EXISTS gif_tags (
                gif_id TEXT,
                tag TEXT,
                PRIMARY KEY (gif_id, tag),
                FOREIGN KEY (gif_id) REFERENCES gif_cache(id)
            )
        """)

        # Create index for faster tag lookups
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_gif_tags_tag ON gif_tags(tag)
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS captions (
                gif_id TEXT PRIMARY KEY,
                caption TEXT,
                persona TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS processed_media (
                gif_id TEXT PRIMARY KEY,
                processed_path TEXT,
                original_type TEXT,
                detections TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS processed_videos (
                gif_id TEXT PRIMARY KEY,
                processed_path TEXT,
                quality TEXT,
                stats TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
    print("Database initialized")


# =============================================================================
# RedGifs Client
# =============================================================================

class RedGifsClient:
    """Wrapper around the redgifs library."""

    def __init__(self):
        self._api: redgifs.API | None = None
        self._lock = asyncio.Lock()

    async def get_api(self) -> redgifs.API:
        async with self._lock:
            if self._api is None:
                self._api = await asyncio.to_thread(self._create_api)
            return self._api

    def _create_api(self) -> redgifs.API:
        api = redgifs.API()
        api.login()
        return api

    async def get_trending(self, page: int = 1, count: int = 40) -> dict:
        """
        Get trending GIFs from RedGifs using get_top_this_week().
        This is our primary content source - we cache everything and filter by tags locally.
        """
        api = await self.get_api()
        result = await asyncio.to_thread(api.get_top_this_week)
        return self._parse_search_result(result)

    async def get_gif(self, gif_id: str) -> dict:
        api = await self.get_api()
        result = await asyncio.to_thread(api.get_gif, gif_id)
        return self._parse_gif(result)

    async def download_media(self, gif_id: str) -> tuple[bytes, str]:
        """Download media for a GIF. Returns (bytes, content_type)."""
        api = await self.get_api()
        gif = await asyncio.to_thread(api.get_gif, gif_id)
        url = gif.urls.thumbnail or gif.urls.poster
        if not url:
            raise Exception("No thumbnail available")

        content_type = "image/jpeg"
        suffix = ".jpg"
        if url.endswith(".png"):
            content_type = "image/png"
            suffix = ".png"
        elif url.endswith(".gif"):
            content_type = "image/gif"
            suffix = ".gif"
        elif url.endswith(".webp"):
            content_type = "image/webp"
            suffix = ".webp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        try:
            await asyncio.to_thread(api.download, url, tmp_path)
            media_bytes = await asyncio.to_thread(Path(tmp_path).read_bytes)
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

        return media_bytes, content_type

    async def download_video(self, gif_id: str, quality: str = "sd") -> tuple[bytes, str]:
        """
        Download video for a GIF. Returns (bytes, content_type).

        Args:
            gif_id: The GIF ID to download
            quality: "sd" or "hd"
        """
        api = await self.get_api()
        gif = await asyncio.to_thread(api.get_gif, gif_id)

        # Choose URL based on quality preference
        if quality == "hd":
            url = gif.urls.hd or gif.urls.sd
        else:
            url = gif.urls.sd or gif.urls.hd

        if not url:
            raise Exception(f"No video URL available for {gif_id}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name

        try:
            await asyncio.to_thread(api.download, url, tmp_path)
            video_bytes = await asyncio.to_thread(Path(tmp_path).read_bytes)
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

        return video_bytes, "video/mp4"

    def _parse_search_result(self, result) -> dict:
        return {
            "page": result.page,
            "pages": result.pages,
            "total": result.total,
            "gifs": [self._parse_gif(gif) for gif in (result.gifs or [])]
        }

    def _parse_gif(self, gif) -> dict:
        return {
            "id": gif.id,
            "thumbnail_url": gif.urls.thumbnail,
            "hd_url": gif.urls.hd,
            "sd_url": gif.urls.sd,
            "web_url": gif.urls.web_url,
            "width": getattr(gif, 'width', None),
            "height": getattr(gif, 'height', None),
            "duration": getattr(gif, 'duration', None),
            "tags": getattr(gif, 'tags', []),
            "username": getattr(gif, 'username', None),
        }

    async def close(self):
        if self._api:
            await asyncio.to_thread(self._api.close)
            self._api = None


# Global client
redgifs_client = RedGifsClient()


# =============================================================================
# Image Processing
# =============================================================================

async def process_image(image_bytes: bytes) -> tuple[bytes, list[dict]]:
    """Process an image by detecting and censoring nudity."""
    return await nudenet_censor.censor_image(image_bytes)


# =============================================================================
# Caching Functions
# =============================================================================

async def get_cached_gif(gif_id: str) -> dict | None:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM gif_cache WHERE id = ?", (gif_id,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
    return None


async def cache_gif(gif_data: dict):
    """Cache a GIF and update tag indexes."""
    gif_id = gif_data["id"]
    tags = gif_data.get("tags", []) or []

    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Insert/update the main GIF cache
        await db.execute("""
            INSERT OR REPLACE INTO gif_cache
            (id, thumbnail_url, hd_url, sd_url, web_url, width, height, duration, tags, username)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            gif_id,
            gif_data.get("thumbnail_url"),
            gif_data.get("hd_url"),
            gif_data.get("sd_url"),
            gif_data.get("web_url"),
            gif_data.get("width"),
            gif_data.get("height"),
            gif_data.get("duration"),
            ",".join(tags),
            gif_data.get("username"),
        ))

        # Update gif_tags mapping and tag_index counts
        for tag in tags:
            tag_lower = tag.lower().strip()
            if not tag_lower:
                continue

            # Check if this gif-tag relationship already exists
            cursor = await db.execute(
                "SELECT 1 FROM gif_tags WHERE gif_id = ? AND tag = ?",
                (gif_id, tag_lower)
            )
            exists = await cursor.fetchone()

            if not exists:
                # Insert new gif-tag relationship
                await db.execute(
                    "INSERT OR IGNORE INTO gif_tags (gif_id, tag) VALUES (?, ?)",
                    (gif_id, tag_lower)
                )

                # Update tag count (insert or increment)
                await db.execute("""
                    INSERT INTO tag_index (tag, count, last_updated)
                    VALUES (?, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(tag) DO UPDATE SET
                        count = count + 1,
                        last_updated = CURRENT_TIMESTAMP
                """, (tag_lower,))

        await db.commit()


async def get_available_tags(limit: int = 100) -> list[dict]:
    """Get all available tags with their counts, sorted by count descending."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT tag, count FROM tag_index
            WHERE count > 0
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [{"tag": row["tag"], "count": row["count"]} for row in rows]


async def get_gifs_by_tag(tag: str, page: int = 1, count: int = 20) -> dict:
    """Get cached GIFs that have a specific tag."""
    tag_lower = tag.lower().strip()
    offset = (page - 1) * count

    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Get total count for this tag
        cursor = await db.execute(
            "SELECT count FROM tag_index WHERE tag = ?",
            (tag_lower,)
        )
        row = await cursor.fetchone()
        total = row["count"] if row else 0

        if total == 0:
            return {
                "page": page,
                "pages": 0,
                "total": 0,
                "gifs": []
            }

        # Get GIF IDs for this tag
        cursor = await db.execute("""
            SELECT gc.* FROM gif_cache gc
            INNER JOIN gif_tags gt ON gc.id = gt.gif_id
            WHERE gt.tag = ?
            ORDER BY gc.created_at DESC
            LIMIT ? OFFSET ?
        """, (tag_lower, count, offset))

        rows = await cursor.fetchall()
        gifs = []
        for row in rows:
            tags_str = row["tags"] or ""
            gifs.append({
                "id": row["id"],
                "thumbnail_url": row["thumbnail_url"],
                "hd_url": row["hd_url"],
                "sd_url": row["sd_url"],
                "web_url": row["web_url"],
                "width": row["width"],
                "height": row["height"],
                "duration": row["duration"],
                "tags": tags_str.split(",") if tags_str else [],
                "username": row["username"],
            })

        pages = (total + count - 1) // count  # Ceiling division

        return {
            "page": page,
            "pages": pages,
            "total": total,
            "gifs": gifs
        }


async def get_cached_caption(gif_id: str) -> str | None:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute("SELECT caption FROM captions WHERE gif_id = ?", (gif_id,))
        row = await cursor.fetchone()
        if row:
            return row[0]
    return None


async def cache_caption(gif_id: str, caption: str, persona: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO captions (gif_id, caption, persona)
            VALUES (?, ?, ?)
        """, (gif_id, caption, persona))
        await db.commit()


async def get_processed_media_path(gif_id: str) -> str | None:
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT processed_path FROM processed_media WHERE gif_id = ?", (gif_id,)
        )
        row = await cursor.fetchone()
        if row and Path(row[0]).exists():
            return row[0]
    return None


async def cache_processed_media(gif_id: str, processed_bytes: bytes, detections: list[dict]) -> str:
    filename = f"{hashlib.md5(gif_id.encode()).hexdigest()}.jpg"
    filepath = MEDIA_CACHE_DIR / filename
    await asyncio.to_thread(filepath.write_bytes, processed_bytes)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO processed_media (gif_id, processed_path, original_type, detections)
            VALUES (?, ?, ?, ?)
        """, (gif_id, str(filepath), "image/jpeg", json.dumps(detections)))
        await db.commit()
    return str(filepath)


async def get_processed_video_path(gif_id: str, quality: str = "sd") -> str | None:
    """Get path to cached processed video if it exists."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT processed_path FROM processed_videos WHERE gif_id = ? AND quality = ?",
            (gif_id, quality)
        )
        row = await cursor.fetchone()
        if row and Path(row[0]).exists():
            return row[0]
    return None


async def cache_processed_video(
    gif_id: str,
    processed_bytes: bytes,
    quality: str,
    stats: dict
) -> str:
    """Cache a processed video and return its path."""
    filename = f"{hashlib.md5(f'{gif_id}_{quality}'.encode()).hexdigest()}.mp4"
    filepath = VIDEO_CACHE_DIR / filename
    await asyncio.to_thread(filepath.write_bytes, processed_bytes)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO processed_videos (gif_id, processed_path, quality, stats)
            VALUES (?, ?, ?, ?)
        """, (gif_id, str(filepath), quality, json.dumps(stats)))
        await db.commit()
    return str(filepath)


async def get_video_processing_stats(gif_id: str, quality: str = "sd") -> dict | None:
    """Get processing stats for a cached video."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT stats FROM processed_videos WHERE gif_id = ? AND quality = ?",
            (gif_id, quality)
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
    return None


# =============================================================================
# Poe Bot Implementation
# =============================================================================

class SuperBrowserBot(fp.PoeBot):
    """Poe bot that handles browsing and processing RedGifs content."""

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        """Handle incoming messages from Poe."""
        last_message = request.query[-1].content.strip()

        parts = last_message.split()
        command = parts[0].lower() if parts else "help"
        args = parts[1:] if len(parts) > 1 else []

        try:
            if command == "browse":
                async for response in self._handle_browse(args):
                    yield response
            elif command == "trending":
                # Alias for browse with no tag
                async for response in self._handle_browse(args):
                    yield response
            elif command == "tags":
                async for response in self._handle_tags(args):
                    yield response
            elif command == "item":
                async for response in self._handle_item(args):
                    yield response
            elif command == "help":
                yield fp.PartialResponse(text=self._get_help_text())
            else:
                # Treat unknown command as a tag to browse
                async for response in self._handle_browse([command] + args):
                    yield response

        except Exception as e:
            yield fp.PartialResponse(text=f"Error: {str(e)}")

    async def _handle_browse(self, args: list[str]) -> AsyncIterable[fp.PartialResponse]:
        """
        Handle browse command.

        - No args: Get trending content from RedGifs (cached for future tag filtering)
        - With tag: Filter locally cached content by that tag

        Usage: browse [tag] [page]
        """
        page = 1
        count = 20
        tag = None

        # Parse arguments
        for arg in args:
            if arg.isdigit():
                page = int(arg)
            else:
                tag = arg

        if tag:
            # Filter by tag from local cache
            yield fp.PartialResponse(text=f"🏷️ Filtering cached content for **{tag}** (page {page})...\n\n")

            result = await get_gifs_by_tag(tag, page=page, count=count)

            if result["total"] == 0:
                # Get available tags to suggest alternatives
                available_tags = await get_available_tags(20)
                tag_list = ", ".join([f"`{t['tag']}` ({t['count']})" for t in available_tags[:10]])
                yield fp.PartialResponse(
                    text=f"No cached content found for tag **{tag}**.\n\n"
                    f"Available tags: {tag_list if tag_list else 'None yet - browse trending first!'}\n\n"
                    "Use `browse` (no tag) to fetch trending content and build up the cache."
                )
                return

            response_data = {
                "type": "browse_result",
                "source": "cache",
                "tag": tag,
                "page": result["page"],
                "pages": result["pages"],
                "total": result["total"],
                "items": []
            }

            for gif in result["gifs"]:
                media_url = f"{SERVER_URL}/media/{gif['id']}" if SERVER_URL else None
                response_data["items"].append({
                    "id": gif["id"],
                    "thumbnail_url": gif["thumbnail_url"],
                    "hd_url": gif["hd_url"],
                    "sd_url": gif["sd_url"],
                    "web_url": gif["web_url"],
                    "media_url": media_url,
                    "video_url": f"{SERVER_URL}/video/{gif['id']}" if SERVER_URL else None,
                    "username": gif.get("username"),
                    "tags": gif.get("tags", []),
                })

        else:
            # Fetch trending from RedGifs and cache it
            yield fp.PartialResponse(text=f"🔥 Fetching trending content (page {page})...\n\n")

            try:
                result = await redgifs_client.get_trending(page=page, count=count)
            except Exception as e:
                yield fp.PartialResponse(text=f"Error fetching from RedGifs: {str(e)}")
                return

            # Cache all GIFs and their tags
            for gif in result["gifs"]:
                await cache_gif(gif)

            response_data = {
                "type": "browse_result",
                "source": "trending",
                "page": result["page"],
                "pages": result["pages"],
                "total": result["total"],
                "items": []
            }

            for gif in result["gifs"]:
                media_url = f"{SERVER_URL}/media/{gif['id']}" if SERVER_URL else None
                response_data["items"].append({
                    "id": gif["id"],
                    "thumbnail_url": gif["thumbnail_url"],
                    "hd_url": gif["hd_url"],
                    "sd_url": gif["sd_url"],
                    "web_url": gif["web_url"],
                    "media_url": media_url,
                    "video_url": f"{SERVER_URL}/video/{gif['id']}" if SERVER_URL else None,
                    "username": gif.get("username"),
                    "tags": gif.get("tags", []),
                })

        yield fp.PartialResponse(text=f"```json\n{json.dumps(response_data, indent=2)}\n```")

    async def _handle_tags(self, args: list[str]) -> AsyncIterable[fp.PartialResponse]:
        """
        Handle tags command - list available tags from the cache.

        Usage: tags [limit]
        """
        limit = 50  # Default limit
        for arg in args:
            if arg.isdigit():
                limit = min(int(arg), 200)  # Cap at 200

        yield fp.PartialResponse(text="🏷️ Fetching available tags from cache...\n\n")

        tags = await get_available_tags(limit)

        if not tags:
            yield fp.PartialResponse(
                text="No tags cached yet.\n\n"
                "Use `browse` or `trending` to fetch content from RedGifs and build up the tag cache."
            )
            return

        response_data = {
            "type": "tags_result",
            "total": len(tags),
            "tags": tags
        }

        yield fp.PartialResponse(text=f"```json\n{json.dumps(response_data, indent=2)}\n```")

    async def _handle_item(self, args: list[str]) -> AsyncIterable[fp.PartialResponse]:
        """Handle item command."""
        if not args:
            yield fp.PartialResponse(text="Please specify a GIF ID. Example: `item abcxyz123`")
            return

        gif_id = args[0]

        yield fp.PartialResponse(text=f"📦 Loading item **{gif_id}**...\n\n")

        gif_data = await get_cached_gif(gif_id)
        if not gif_data:
            try:
                gif_data = await redgifs_client.get_gif(gif_id)
                await cache_gif(gif_data)
            except Exception as e:
                yield fp.PartialResponse(text=f"Error: Could not find GIF {gif_id}: {str(e)}")
                return

        caption = await get_cached_caption(gif_id)
        if not caption:
            caption = "A captivating moment captured in pixels! ✨"
            await cache_caption(gif_id, caption, "default")

        media_url = f"{SERVER_URL}/media/{gif_id}" if SERVER_URL else None

        response_data = {
            "type": "item_result",
            "id": gif_id,
            "thumbnail_url": gif_data.get("thumbnail_url"),
            "hd_url": gif_data.get("hd_url"),
            "sd_url": gif_data.get("sd_url"),
            "web_url": gif_data.get("web_url"),
            "media_url": media_url,
            "video_url": f"{SERVER_URL}/video/{gif_id}" if SERVER_URL else None,
            "caption": caption,
            "username": gif_data.get("username"),
            "tags": gif_data.get("tags", "").split(",") if isinstance(gif_data.get("tags"), str) else gif_data.get("tags", []),
        }

        yield fp.PartialResponse(text=f"```json\n{json.dumps(response_data, indent=2)}\n```")

    def _get_help_text(self) -> str:
        return """
# Super Browser Bot 🌐

**Commands:**

- `browse` or `trending` - Get trending content from RedGifs
  - Automatically caches all content and their tags
  - Use `browse 2`, `browse 3`, etc. for more pages
  - Examples:
    - `browse` - Get page 1 of trending
    - `trending 5` - Get page 5 of trending

- `browse <tag> [page]` or just `<tag> [page]` - Filter cached content by tag
  - Only searches content already cached from trending
  - Examples:
    - `browse blonde` - Cached content tagged "blonde"
    - `amateur 2` - Page 2 of cached "amateur" content
    - `milf` - Cached content tagged "milf"

- `tags [limit]` - List available tags from cache with counts
  - Shows which tags have cached content
  - Example: `tags 50` - Top 50 tags by count

- `item <gif_id>` - Get a specific item with caption
  - Example: `item abcxyz123`

- `help` - Show this help message

**How It Works:**
1. Use `browse` (no tag) to fetch trending content
2. Content is automatically cached with all its tags
3. Use `tags` to see what's available
4. Use `browse <tag>` to filter cached content by tag

**Features:**
- 🔒 Dual-model AI nudity censoring (320n + 640m)
- 📦 Processed images served from `/media/{id}`
- 🎬 **NEW: Video censoring** served from `/video/{id}`
  - Smart keyframe detection (scene changes)
  - Smooth censor box interpolation between keyframes
  - Audio preserved
  - First request takes 10-60s to process, then cached
- 🏷️ Local tag filtering (more reliable than API search)
- 💾 All content cached for instant tag-based browsing

**For Canvas Apps:**
Responses are returned as JSON in code blocks for easy parsing.
Each item includes `media_url` (censored thumbnail) and `video_url` (censored video).
"""

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Return bot settings."""
        return fp.SettingsResponse(
            server_bot_dependencies={},
            introduction_message="Welcome to Super Browser! 🌐\n\n"
            "**Quick Start:**\n"
            "• `browse` - Get trending content (builds cache)\n"
            "• `tags` - See available tags with counts\n"
            "• `browse <tag>` - Filter cached content by tag\n\n"
            "🔒 All images AND videos are processed with automated nudity censoring.\n"
            "🎬 Video censoring uses smart keyframe detection with smooth interpolation.\n\n"
            "Type `help` for more commands.",
        )


# =============================================================================
# FastAPI Application
# =============================================================================

bot = SuperBrowserBot()

app = fp.make_app(
    bot,
    access_key=POE_ACCESS_KEY if POE_ACCESS_KEY else None,
    bot_name=BOT_NAME,
    allow_without_key=not POE_ACCESS_KEY,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    global video_processor
    await init_db()
    print("Pre-loading NudeNet ONNX model...")
    await nudenet_censor.load()
    print("NudeNet model ready!")
    # Initialize video processor
    video_processor = VideoProcessor(nudenet_censor)
    print("Video processor initialized!")
    async with original_lifespan(app):
        yield
    await redgifs_client.close()


app.router.lifespan_context = combined_lifespan


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    # Get tag stats
    tags = await get_available_tags(limit=5)
    total_tags_result = await get_available_tags(limit=10000)
    total_tags = len(total_tags_result)

    # Check ffmpeg availability
    ffmpeg_available = False
    ffmpeg_version = None
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            ffmpeg_available = True
            # Extract version from first line
            first_line = result.stdout.split('\n')[0] if result.stdout else ""
            ffmpeg_version = first_line
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "Super Browser Bot",
        "features": {
            "nudity_censoring": True,
            "dual_model": True,
            "models": ["NudeNet 320n", "NudeNet 640m"],
            "censor_threshold": CENSOR_THRESHOLD,
            "local_tag_filtering": True,
            "video_processing": ffmpeg_available,
            "ffmpeg_available": ffmpeg_available,
            "ffmpeg_version": ffmpeg_version,
            "video_features": {
                "smart_keyframes": True,
                "scene_change_detection": True,
                "smooth_box_interpolation": True,
                "keyframe_interval_frames": VIDEO_KEYFRAME_INTERVAL,
                "scene_change_threshold": VIDEO_SCENE_CHANGE_THRESHOLD,
                "max_duration_seconds": VIDEO_MAX_DURATION,
            } if ffmpeg_available else None,
        },
        "cache_stats": {
            "total_tags": total_tags,
            "top_tags": tags[:5] if tags else [],
        },
        "endpoints": {
            "/tags": "GET - List all cached tags with counts",
            "/browse/tag/{tag}": "GET - Get cached GIFs by tag",
            "/media/{gif_id}": "GET - Get processed (censored) thumbnail image",
            "/video/{gif_id}": "GET - Get processed (censored) video (query: quality=sd|hd)" if ffmpeg_available else "UNAVAILABLE - ffmpeg not installed",
            "/video/{gif_id}/status": "GET - Check video processing status",
        }
    }


@app.get("/media/{gif_id}")
async def get_processed_media(gif_id: str):
    """Serve processed media with nudity censored."""
    cached_path = await get_processed_media_path(gif_id)
    if cached_path:
        return FileResponse(cached_path, media_type="image/jpeg")

    try:
        original_bytes, _ = await redgifs_client.download_media(gif_id)
    except Exception as e:
        return Response(content=f"Error downloading: {e}", status_code=404)

    try:
        processed_bytes, detections = await process_image(original_bytes)
    except Exception as e:
        return Response(content=f"Error processing: {e}", status_code=500)

    filepath = await cache_processed_media(gif_id, processed_bytes, detections)

    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/media/{gif_id}/detections")
async def get_media_detections(gif_id: str):
    """Get detection results for a processed media item."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT detections FROM processed_media WHERE gif_id = ?", (gif_id,)
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return {"gif_id": gif_id, "detections": json.loads(row[0])}

    return {"gif_id": gif_id, "detections": None, "message": "Not processed yet"}


@app.get("/tags")
async def list_tags(limit: int = 100):
    """Get available tags from cache with counts."""
    tags = await get_available_tags(limit=min(limit, 500))
    return {
        "total": len(tags),
        "tags": tags
    }


@app.get("/browse/tag/{tag}")
async def browse_by_tag(tag: str, page: int = 1, count: int = 20):
    """Get cached GIFs by tag."""
    result = await get_gifs_by_tag(tag, page=page, count=count)

    # Add media_url and video_url to each gif
    for gif in result["gifs"]:
        gif["media_url"] = f"{SERVER_URL}/media/{gif['id']}" if SERVER_URL else None
        gif["video_url"] = f"{SERVER_URL}/video/{gif['id']}" if SERVER_URL else None

    return {
        "source": "cache",
        "tag": tag,
        **result
    }


def check_ffmpeg_available() -> tuple[bool, str]:
    """Check if ffmpeg and ffprobe are available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, "ffmpeg not working"

        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, "ffprobe not working"

        return True, "ffmpeg available"
    except FileNotFoundError:
        return False, "ffmpeg/ffprobe not installed. Video processing requires ffmpeg."
    except Exception as e:
        return False, f"ffmpeg check failed: {e}"


@app.get("/video/{gif_id}")
async def get_processed_video(gif_id: str, quality: str = "sd"):
    """
    Serve processed video with nudity censored.

    This endpoint downloads the video, processes it frame-by-frame with
    smart keyframe detection and smooth box interpolation, then serves
    the censored result.

    Query params:
        quality: "sd" (default) or "hd"

    Note: First request for a video will take 10-60 seconds to process.
    Subsequent requests are served from cache.
    """
    global video_processor

    # Check ffmpeg availability
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if not ffmpeg_ok:
        return Response(
            content=f"Video processing unavailable: {ffmpeg_msg}",
            status_code=503
        )

    if video_processor is None:
        return Response(
            content="Video processor not initialized",
            status_code=503
        )

    # Validate quality param
    if quality not in ("sd", "hd"):
        quality = "sd"

    # Check cache first
    cached_path = await get_processed_video_path(gif_id, quality)
    if cached_path:
        return FileResponse(
            cached_path,
            media_type="video/mp4",
            headers={
                "X-Cache": "HIT",
                "X-Processing-Time": "0"
            }
        )

    # Download the video
    try:
        print(f"Downloading video {gif_id} ({quality})...")
        video_bytes, _ = await redgifs_client.download_video(gif_id, quality)
        print(f"Downloaded {len(video_bytes) / 1024 / 1024:.1f} MB")
    except Exception as e:
        return Response(
            content=f"Error downloading video: {e}",
            status_code=404
        )

    # Process the video
    try:
        print(f"Processing video {gif_id}...")
        processed_bytes, stats = await video_processor.process_video(video_bytes, quality)
        print(f"Video processed: {stats}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Error processing video: {e}",
            status_code=500
        )

    # Cache the result
    filepath = await cache_processed_video(gif_id, processed_bytes, quality, stats)

    return FileResponse(
        filepath,
        media_type="video/mp4",
        headers={
            "X-Cache": "MISS",
            "X-Processing-Time": str(stats.get("processing_time_seconds", 0)),
            "X-Keyframes": str(stats.get("keyframes_detected", 0)),
            "X-Total-Frames": str(stats.get("total_frames", 0)),
        }
    )


@app.get("/video/{gif_id}/status")
async def get_video_status(gif_id: str, quality: str = "sd"):
    """
    Check if a processed video exists in cache and get its stats.
    Useful for checking status before requesting a potentially long processing job.
    """
    if quality not in ("sd", "hd"):
        quality = "sd"

    cached_path = await get_processed_video_path(gif_id, quality)

    if cached_path:
        stats = await get_video_processing_stats(gif_id, quality)
        return {
            "gif_id": gif_id,
            "quality": quality,
            "status": "ready",
            "cached": True,
            "stats": stats,
            "video_url": f"{SERVER_URL}/video/{gif_id}?quality={quality}" if SERVER_URL else None
        }

    return {
        "gif_id": gif_id,
        "quality": quality,
        "status": "not_processed",
        "cached": False,
        "message": "Video has not been processed yet. Request /video/{gif_id} to trigger processing."
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
