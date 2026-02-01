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
from redgifs import Order, Tags

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

# Ensure directories exist
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
# Database Setup
# =============================================================================

async def init_db():
    """Initialize SQLite database with required tables."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
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

    async def search(self, query: str, page: int = 1, count: int = 20, order: str = "latest") -> dict:
        """
        Search for GIFs by tag/query.

        Args:
            query: Tag or search term
            page: Page number (1-indexed)
            count: Number of results per page
            order: Sort order - "latest", "trending", "top", "top7", "top28"
        """
        api = await self.get_api()

        # Map order string to Order enum
        # Map order string to Order enum - only use values that exist in the library
        # Available: TRENDING, LATEST, TOP (and aliases like BEST, NEW, RECENT)
        order_map = {
            "latest": Order.TRENDING,  # Use TRENDING as fallback since LATEST may not exist
            "new": Order.TRENDING,
            "trending": Order.TRENDING,
            "top": Order.TRENDING,
            "best": Order.TRENDING,
        }

        # Try to use the actual Order values if they exist
        try:
            order_map["latest"] = Order.LATEST
            order_map["new"] = Order.LATEST
        except AttributeError:
            pass

        try:
            order_map["top"] = Order.TOP
        except AttributeError:
            pass

        try:
            order_map["best"] = Order.BEST
        except AttributeError:
            pass

        order_enum = order_map.get(order.lower(), Order.TRENDING)

        # Try using Tags object for proper tag-based search
        # According to docs: search_text can be "a string or an instance of Tags"
        try:
            # Create a Tags instance with the query as a tag
            tags_obj = Tags(query)
            result = await asyncio.to_thread(
                api.search,
                tags_obj,
                order=order_enum,
                count=count,
                page=page
            )
        except (TypeError, ValueError) as e:
            # Fallback to regular string search if Tags doesn't work
            print(f"Tags search failed ({e}), falling back to text search")
            result = await asyncio.to_thread(
                api.search,
                query,
                order=order_enum,
                count=count,
                page=page
            )

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
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO gif_cache
            (id, thumbnail_url, hd_url, sd_url, web_url, width, height, duration, tags, username)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            gif_data["id"],
            gif_data.get("thumbnail_url"),
            gif_data.get("hd_url"),
            gif_data.get("sd_url"),
            gif_data.get("web_url"),
            gif_data.get("width"),
            gif_data.get("height"),
            gif_data.get("duration"),
            ",".join(gif_data.get("tags", []) if gif_data.get("tags") else []),
            gif_data.get("username"),
        ))
        await db.commit()


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
            elif command == "item":
                async for response in self._handle_item(args):
                    yield response
            elif command == "help":
                yield fp.PartialResponse(text=self._get_help_text())
            else:
                async for response in self._handle_browse([command] + args):
                    yield response

        except Exception as e:
            yield fp.PartialResponse(text=f"Error: {str(e)}")

    async def _handle_browse(self, args: list[str]) -> AsyncIterable[fp.PartialResponse]:
        """
        Handle browse command.

        Usage: browse <tag> [page] [order]
        - tag: The tag to search for
        - page: Page number (default: 1)
        - order: Sort order - latest, trending, top, top7, top28 (default: latest)
        """
        if not args:
            yield fp.PartialResponse(text="Please specify a tag to browse. Example: `browse cats`")
            return

        tag = args[0]
        page = 1
        order = "latest"  # Default to latest for fresh results
        count = 20  # More results per page for infinite scroll

        # Parse optional arguments
        for arg in args[1:]:
            if arg.isdigit():
                page = int(arg)
            elif arg.lower() in ["latest", "trending", "top", "new", "best"]:
                order = arg.lower()

        yield fp.PartialResponse(text=f"🔍 Searching for **{tag}** (page {page}, {order})...\n\n")

        try:
            # Always fetch fresh results - no caching of browse listings
            result = await redgifs_client.search(tag, page=page, count=count, order=order)
        except Exception as e:
            yield fp.PartialResponse(text=f"Error fetching from RedGifs: {str(e)}")
            return

        # Note: We don't cache browse results - only individual GIF metadata
        # when they are accessed via /media/{id} endpoint

        response_data = {
            "type": "browse_result",
            "tag": tag,
            "page": result["page"],
            "pages": result["pages"],
            "total": result["total"],
            "order": order,
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
                "username": gif.get("username"),
                "tags": gif.get("tags", []),
            })

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
            "caption": caption,
            "username": gif_data.get("username"),
            "tags": gif_data.get("tags", "").split(",") if isinstance(gif_data.get("tags"), str) else gif_data.get("tags", []),
        }

        yield fp.PartialResponse(text=f"```json\n{json.dumps(response_data, indent=2)}\n```")

    def _get_help_text(self) -> str:
        return """
# Super Browser Bot 🌐

**Commands:**

- `browse <tag> [page] [order]` - Browse GIFs by tag
  - `tag`: The tag to search for
  - `page`: Page number (default: 1)
  - `order`: Sort order (default: latest)
    - `latest` - Most recent uploads
    - `trending` - Currently popular
    - `top` - Top rated
  - Examples:
    - `browse blonde` - Latest blonde content
    - `browse amateur 2` - Page 2 of amateur
    - `browse milf trending` - Trending milf content

- `item <gif_id>` - Get a specific item with caption
  - Example: `item abcxyz123`

- `help` - Show this help message

**Features:**
- 🔒 Dual-model AI nudity censoring (320n + 640m)
- 📦 Processed images served from `/media/{id}`
- 🔄 Fresh results on every browse (no stale cache)

**For Canvas Apps:**
Responses are returned as JSON in code blocks for easy parsing.
"""

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Return bot settings."""
        return fp.SettingsResponse(
            server_bot_dependencies={},
            introduction_message="Welcome to Super Browser! 🌐\n\nUse `browse <tag>` to search for content, or type `help` for more commands.\n\n🔒 All images are processed with automated nudity censoring.",
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
    await init_db()
    print("Pre-loading NudeNet ONNX model...")
    await nudenet_censor.load()
    print("NudeNet model ready!")
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
    return {
        "status": "ok",
        "service": "Super Browser Bot",
        "features": {
            "nudity_censoring": True,
            "dual_model": True,
            "models": ["NudeNet 320n", "NudeNet 640m"],
            "censor_threshold": CENSOR_THRESHOLD,
            "censor_classes": CENSOR_CLASSES,
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


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
