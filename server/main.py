"""
Super Browser - Poe Server Bot
A Poe-protocol-compatible server for browsing and processing Rule34.xxx content.
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
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import AsyncIterable
from contextlib import asynccontextmanager
from urllib.parse import quote, urlencode

import httpx
import aiosqlite
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from dotenv import load_dotenv

import fastapi_poe as fp
from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

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

# Rule34 API configuration
RULE34_API_BASE = "https://api.rule34.xxx/index.php"
RULE34_AUTOCOMPLETE_BASE = "https://autocomplete.rule34.xxx/autocomplete.php"
RULE34_TAG_CACHE_TTL = int(os.getenv("RULE34_TAG_CACHE_TTL", "300"))  # 5 minutes default
RULE34_API_KEY = os.getenv("RULE34_API_KEY", "")
RULE34_USER_ID = os.getenv("RULE34_USER_ID", "")

# Hardcoded popular tags (Rule34's tag API ordering is unreliable)
POPULAR_TAGS = [
    "breasts",
    "femdom",
    "feet",
    "genshin_impact",
]

# NudeNet model URLs - using dual models for better coverage
NUDENET_320N_URLS = [
    "https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx",
    "https://huggingface.co/vladmandic/nudenet/resolve/main/nudenet.onnx",
]
NUDENET_640M_URLS = [
    "https://huggingface.co/spaces/xxparthparekhxx/NudeNet-FastAPI/resolve/794a185a301917f1a3505ab3b8d55b268ea81f0e/640m.onnx",
]

NUDENET_320N_PATH = MODEL_CACHE_DIR / "320n.onnx"
NUDENET_640M_PATH = MODEL_CACHE_DIR / "640m.onnx"
NUDENET_MODEL_MIN_SIZE_320 = 5 * 1024 * 1024
NUDENET_MODEL_MIN_SIZE_640 = 20 * 1024 * 1024

# Nudity censoring configuration
CENSOR_THRESHOLD = float(os.getenv("CENSOR_THRESHOLD", "0.4"))

# NudeNet class labels
NUDENET_LABELS_VLADMANDIC = [
    "female-private-area", "female-face", "buttocks-bare", "female-breast-bare",
    "female-vagina", "male-breast-bare", "anus-bare", "feet-bare", "belly",
    "feet", "armpits", "armpits-bare", "male-face", "belly-bare", "male-penis",
    "anus-area", "female-breast", "buttocks",
]

NUDENET_LABELS_320N = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

DEFAULT_CENSOR_CLASSES = [
    "buttocks-bare", "female-breast-bare", "female-vagina", "anus-bare",
    "male-penis", "male-breast-bare", "female-private-area", "female-breast",
    "buttocks", "anus-area", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED", "BUTTOCKS_COVERED", "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED", "ANUS_COVERED",
]

CENSOR_CLASSES = os.getenv("CENSOR_CLASSES", "").split(",") if os.getenv("CENSOR_CLASSES") else DEFAULT_CENSOR_CLASSES
CENSOR_CLASSES = [c.strip() for c in CENSOR_CLASSES if c.strip()]

# Video processing configuration
VIDEO_CACHE_DIR = Path(os.getenv("VIDEO_CACHE_DIR", "video_cache"))
VIDEO_KEYFRAME_INTERVAL_SECONDS = float(os.getenv("VIDEO_KEYFRAME_INTERVAL_SECONDS", "1.0"))
VIDEO_SCENE_CHANGE_THRESHOLD = float(os.getenv("VIDEO_SCENE_CHANGE_THRESHOLD", "0.15"))
VIDEO_MAX_DURATION = int(os.getenv("VIDEO_MAX_DURATION", "60"))
VIDEO_TARGET_FPS = int(os.getenv("VIDEO_TARGET_FPS", "15"))
VIDEO_USE_DUAL_MODELS = os.getenv("VIDEO_USE_DUAL_MODELS", "false").lower() == "true"
VIDEO_PRIMARY_MODEL = os.getenv("VIDEO_PRIMARY_MODEL", "320n")
VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_BATCH_SIZE", "20"))
VIDEO_VERBOSE_LOGGING = os.getenv("VIDEO_VERBOSE_LOGGING", "true").lower() == "true"

# Ensure directories exist
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Rule34 API Client
# =============================================================================

class Rule34Client:
    """Client for interacting with Rule34.xxx API."""

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None
        # In-memory cache for tags with TTL
        self._tag_cache: dict[str, tuple[float, list]] = {}
        self._tag_cache_lock = asyncio.Lock()

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "SuperBrowser/1.0 (Poe Bot)"
                }
            )
        return self._http_client

    def _add_auth_params(self, params: dict) -> dict:
        """Add authentication parameters to API request."""
        if RULE34_API_KEY and RULE34_USER_ID:
            params["api_key"] = RULE34_API_KEY
            params["user_id"] = RULE34_USER_ID
        return params

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get_popular_tags(self, limit: int = 50, order_by: str = "updated") -> list[dict]:
        """
        Get popular/trending tags from Rule34.

        Args:
            limit: Number of tags to return (max 100)
            order_by: "count" for most popular, "updated" for recently active

        Returns:
            List of tag dicts with 'name', 'count', 'type' fields
        """
        cache_key = f"popular_tags:{limit}:{order_by}"

        async with self._tag_cache_lock:
            if cache_key in self._tag_cache:
                cached_time, cached_data = self._tag_cache[cache_key]
                if time.time() - cached_time < RULE34_TAG_CACHE_TTL:
                    print(f"[Rule34] Returning cached tags for {cache_key}")
                    return cached_data

        params = self._add_auth_params({
            "page": "dapi",
            "s": "tag",
            "q": "index",
            "json": "1",
            "limit": min(limit, 100),
            "orderby": order_by,
        })

        try:
            response = await self.http_client.get(RULE34_API_BASE, params=params)
            response.raise_for_status()

            raw_text = response.text
            print(f"[Rule34] Tags response length: {len(raw_text)}")

            # Handle empty response
            if not raw_text or raw_text.strip() == "":
                print("[Rule34] Empty response from tags API")
                return []

            # Try to parse response - could be JSON or XML
            result = []

            # Check if it's XML (starts with <?xml or <tags)
            if raw_text.strip().startswith("<?xml") or raw_text.strip().startswith("<tags"):
                print("[Rule34] Parsing XML response for tags")
                result = self._parse_tags_xml(raw_text)
            else:
                # Try JSON parsing
                try:
                    tags = response.json()
                    if isinstance(tags, list):
                        for tag in tags:
                            result.append({
                                "name": tag.get("name", ""),
                                "count": int(tag.get("count", 0)),
                                "type": self._tag_type_name(int(tag.get("type", 0))),
                            })
                    else:
                        print(f"[Rule34] Unexpected JSON format: {type(tags)}")
                except Exception as json_err:
                    print(f"[Rule34] JSON parse error: {json_err}, trying XML")
                    result = self._parse_tags_xml(raw_text)

            # Cache the result
            if result:
                async with self._tag_cache_lock:
                    self._tag_cache[cache_key] = (time.time(), result)

            print(f"[Rule34] Fetched {len(result)} tags")
            return result

        except Exception as e:
            print(f"[Rule34] Error fetching tags: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _tag_type_name(self, type_id: int) -> str:
        """Convert Rule34 tag type ID to name."""
        types = {
            0: "general",
            1: "artist",
            3: "copyright",
            4: "character",
            5: "metadata",
        }
        return types.get(type_id, "general")

    def _parse_tags_xml(self, xml_text: str) -> list[dict]:
        """Parse XML response for tags."""
        result = []
        try:
            root = ET.fromstring(xml_text)
            for tag_elem in root.findall(".//tag"):
                name = tag_elem.get("name", "")
                count = int(tag_elem.get("count", 0))
                type_id = int(tag_elem.get("type", 0))
                if name:
                    result.append({
                        "name": name,
                        "count": count,
                        "type": self._tag_type_name(type_id),
                    })
        except ET.ParseError as e:
            print(f"[Rule34] XML parse error for tags: {e}")
        return result

    def _parse_posts_xml(self, xml_text: str) -> list[dict]:
        """Parse XML response for posts."""
        result = []
        try:
            root = ET.fromstring(xml_text)
            for post_elem in root.findall(".//post"):
                file_url = post_elem.get("file_url", "")
                result.append({
                    "id": int(post_elem.get("id", 0)),
                    "tags": post_elem.get("tags", ""),
                    "file_url": file_url,
                    "sample_url": post_elem.get("sample_url", "") or post_elem.get("preview_url", ""),
                    "preview_url": post_elem.get("preview_url", ""),
                    "width": int(post_elem.get("width", 0)),
                    "height": int(post_elem.get("height", 0)),
                    "score": int(post_elem.get("score", 0)),
                    "file_type": self._determine_file_type(file_url),
                    "source": post_elem.get("source", ""),
                })
        except ET.ParseError as e:
            print(f"[Rule34] XML parse error for posts: {e}")
        return result

    async def autocomplete_tags(self, query: str, limit: int = 20) -> list[dict]:
        """
        Get tag autocomplete suggestions.

        Args:
            query: Partial tag name
            limit: Max results to return

        Returns:
            List of tag suggestions with 'label' and 'value' fields
        """
        if not query or len(query) < 2:
            return []

        cache_key = f"autocomplete:{query.lower()}"

        async with self._tag_cache_lock:
            if cache_key in self._tag_cache:
                cached_time, cached_data = self._tag_cache[cache_key]
                if time.time() - cached_time < RULE34_TAG_CACHE_TTL:
                    return cached_data[:limit]

        try:
            # Autocomplete endpoint may also need auth
            params = self._add_auth_params({"q": query})
            response = await self.http_client.get(
                RULE34_AUTOCOMPLETE_BASE,
                params=params
            )
            response.raise_for_status()

            suggestions = response.json()

            if not isinstance(suggestions, list):
                return []

            # Normalize format
            result = []
            for s in suggestions[:limit]:
                if isinstance(s, dict):
                    result.append({
                        "label": s.get("label", s.get("value", "")),
                        "value": s.get("value", s.get("label", "")),
                    })
                elif isinstance(s, str):
                    result.append({"label": s, "value": s})

            # Cache results
            async with self._tag_cache_lock:
                self._tag_cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            print(f"[Rule34] Autocomplete error: {e}")
            return []

    async def get_posts(
        self,
        tags: str,
        limit: int = 50,
        page: int = 0,
        sort: str = "score"
    ) -> list[dict]:
        """
        Get posts matching given tags.

        Args:
            tags: Space-separated tags (use sort:score for highest rated)
            limit: Number of posts (max 1000)
            page: Page number (0-indexed)
            sort: "score" for highest rated, "id" for newest

        Returns:
            List of post dicts with id, file_url, sample_url, tags, etc.
        """
        # Add sort tag if specified
        search_tags = tags
        if sort == "score":
            search_tags = f"sort:score {tags}"
        elif sort == "id":
            search_tags = f"sort:id:desc {tags}"

        params = self._add_auth_params({
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "limit": min(limit, 1000),
            "pid": page,
            "tags": search_tags,
        })

        try:
            response = await self.http_client.get(RULE34_API_BASE, params=params)
            response.raise_for_status()

            raw_text = response.text

            # Handle empty response
            if not raw_text or raw_text.strip() == "":
                print("[Rule34] Empty response from posts API")
                return []

            posts = []

            # Check if it's XML
            if raw_text.strip().startswith("<?xml") or raw_text.strip().startswith("<posts"):
                print("[Rule34] Parsing XML response for posts")
                posts = self._parse_posts_xml(raw_text)
            else:
                # Try JSON parsing
                try:
                    data = response.json()
                    if isinstance(data, list):
                        for post in data:
                            file_url = post.get("file_url", "")
                            posts.append({
                                "id": int(post.get("id", 0)),
                                "tags": post.get("tags", ""),
                                "file_url": file_url,
                                "sample_url": post.get("sample_url", "") or post.get("preview_url", ""),
                                "preview_url": post.get("preview_url", ""),
                                "width": int(post.get("width", 0)),
                                "height": int(post.get("height", 0)),
                                "score": int(post.get("score", 0)),
                                "file_type": self._determine_file_type(file_url),
                                "source": post.get("source", ""),
                            })
                    else:
                        print(f"[Rule34] Unexpected JSON format for posts: {type(data)}")
                except Exception as json_err:
                    print(f"[Rule34] JSON parse error for posts: {json_err}, trying XML")
                    posts = self._parse_posts_xml(raw_text)

            print(f"[Rule34] Fetched {len(posts)} posts for tags: {tags}")
            return posts

        except Exception as e:
            print(f"[Rule34] Error fetching posts: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _determine_file_type(self, url: str) -> str:
        """Determine file type from URL."""
        url_lower = url.lower()
        if url_lower.endswith(('.mp4', '.webm')):
            return "video"
        elif url_lower.endswith('.gif'):
            return "gif"
        else:
            return "image"

    async def get_post_by_id(self, post_id: int) -> dict | None:
        """Get a single post by ID."""
        params = self._add_auth_params({
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "id": post_id,
        })

        try:
            response = await self.http_client.get(RULE34_API_BASE, params=params)
            response.raise_for_status()

            raw_text = response.text

            if not raw_text or raw_text.strip() == "":
                return None

            posts = []

            # Check if it's XML
            if raw_text.strip().startswith("<?xml") or raw_text.strip().startswith("<posts"):
                posts = self._parse_posts_xml(raw_text)
            else:
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        post = data[0]
                        file_url = post.get("file_url", "")
                        return {
                            "id": int(post.get("id", 0)),
                            "tags": post.get("tags", ""),
                            "file_url": file_url,
                            "sample_url": post.get("sample_url", "") or post.get("preview_url", ""),
                            "preview_url": post.get("preview_url", ""),
                            "width": int(post.get("width", 0)),
                            "height": int(post.get("height", 0)),
                            "score": int(post.get("score", 0)),
                            "file_type": self._determine_file_type(file_url),
                            "source": post.get("source", ""),
                        }
                except Exception:
                    posts = self._parse_posts_xml(raw_text)

            if posts:
                return posts[0]

            return None

        except Exception as e:
            print(f"[Rule34] Error fetching post {post_id}: {e}")
            return None


# =============================================================================
# Direct ONNX NudeNet Detector (No OpenCV!)
# =============================================================================

class NudeDetectorONNX:
    """NudeNet detector using direct ONNX inference."""

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
        self.labels: list[str] = NUDENET_LABELS_VLADMANDIC
        self._lock = asyncio.Lock()

    async def ensure_model_downloaded(self):
        """Download the ONNX model if not present or corrupted."""
        if self.model_path.exists():
            file_size = self.model_path.stat().st_size
            if file_size >= self.min_size:
                print(f"[{self.name}] Model already exists: {file_size / 1024 / 1024:.1f} MB")
                return
            else:
                print(f"[{self.name}] Model file too small ({file_size} bytes), re-downloading...")
                self.model_path.unlink()

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

            if content_size < self.min_size:
                raise RuntimeError(f"Downloaded file too small ({content_size} bytes).")

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
            self.session = await asyncio.to_thread(
                ort.InferenceSession,
                str(self.model_path),
                providers=["CPUExecutionProvider"]
            )

            output_shape = self.session.get_outputs()[0].shape
            print(f"[{self.name}] Model output shape: {output_shape}")

            input_shape = self.session.get_inputs()[0].shape
            print(f"[{self.name}] Model input shape: {input_shape}")

            if input_shape and len(input_shape) >= 4 and isinstance(input_shape[2], int):
                self.input_size = input_shape[2]
                print(f"[{self.name}] Using input size: {self.input_size}")

            file_size = self.model_path.stat().st_size
            if file_size > 10 * 1024 * 1024 and file_size < 20 * 1024 * 1024:
                self.labels = NUDENET_LABELS_VLADMANDIC
                print(f"[{self.name}] Detected vladmandic/nudenet model")
            else:
                self.labels = NUDENET_LABELS_320N
                print(f"[{self.name}] Detected official NudeNet model")

            print(f"[{self.name}] Model loaded successfully with {len(self.labels)} classes")

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
        """Preprocess image for NudeNet model."""
        original_size = image.size

        if image.mode != "RGB":
            image = image.convert("RGB")

        resized = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        img_array = np.array(resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        input_tensor = np.expand_dims(img_array, axis=0)

        scale_x = original_size[0] / self.input_size
        scale_y = original_size[1] / self.input_size

        return input_tensor, original_size, (scale_x, scale_y)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale: tuple[float, float],
        threshold: float = 0.25
    ) -> list[dict]:
        """Postprocess model outputs to get detections."""
        predictions = outputs[0]
        predictions = predictions[0].T

        scale_x, scale_y = scale
        detections = []

        for pred in predictions:
            x_center, y_center, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            if score < threshold:
                continue

            x1 = (x_center - w / 2) * scale_x
            y1 = (y_center - h / 2) * scale_y
            box_w = w * scale_x
            box_h = h * scale_y

            x1 = max(0, x1)
            y1 = max(0, y1)

            detections.append({
                "class": self.labels[class_id] if class_id < len(self.labels) else f"CLASS_{class_id}",
                "score": round(score, 4),
                "box": [int(x1), int(y1), int(box_w), int(box_h)]
            })

        detections = self._nms(detections, iou_threshold=0.45)
        return detections

    def _nms(self, detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self._iou(best["box"], d["box"]) < iou_threshold]

        return keep

    def _iou(self, box1: list[int], box2: list[int]) -> float:
        """Calculate Intersection over Union."""
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

    async def detect(self, image_bytes: bytes, threshold: float = 0.25) -> list[dict]:
        """Detect nudity in an image."""
        if self.session is None:
            await self.load()

        image = Image.open(io.BytesIO(image_bytes))
        input_tensor, original_size, scale = self._preprocess(image)
        input_name = self.session.get_inputs()[0].name

        outputs = await asyncio.to_thread(
            self.session.run,
            None,
            {input_name: input_tensor}
        )

        detections = self._postprocess(outputs, scale, threshold)
        return detections


# =============================================================================
# NudeNet Censor
# =============================================================================

class NudeNetCensor:
    """Wrapper that combines detection and censoring."""

    def __init__(self):
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
        censor_classes: list[str] | None = None,
        threshold: float = 0.25,
        use_dual_models: bool = True,
        primary_model: str = "320n"
    ) -> tuple[bytes, list[dict]]:
        """
        Detect and censor NSFW content in an image.

        Returns:
            Tuple of (censored_image_bytes, list_of_detections)
        """
        if censor_classes is None:
            censor_classes = CENSOR_CLASSES

        # Run detection with selected model(s)
        if use_dual_models:
            results_320n, results_640m = await asyncio.gather(
                self.detector_320n.detect(image_bytes, threshold),
                self.detector_640m.detect(image_bytes, threshold)
            )
            all_detections = results_320n + results_640m
        else:
            if primary_model == "640m":
                all_detections = await self.detector_640m.detect(image_bytes, threshold)
            else:
                all_detections = await self.detector_320n.detect(image_bytes, threshold)

        # Filter to censor classes
        to_censor = [d for d in all_detections if d["class"] in censor_classes]

        if not to_censor:
            return image_bytes, all_detections

        # Apply censoring
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image)

        for detection in to_censor:
            x, y, w, h = detection["box"]
            # Apply expanded box with rounded corners effect via solid fill
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.width, x + w + padding)
            y2 = min(image.height, y + h + padding)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        # Save to bytes
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue(), all_detections


# =============================================================================
# Video/GIF Processing
# =============================================================================

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

            # Get video info
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

            # Extract frames
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

            # Reassemble video
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


# =============================================================================
# Database Layer
# =============================================================================

async def init_db():
    """Initialize SQLite database with required tables."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Posts cache - stores Rule34 post metadata
        await db.execute("""
            CREATE TABLE IF NOT EXISTS post_cache (
                post_id INTEGER PRIMARY KEY,
                tags TEXT,
                file_url TEXT,
                sample_url TEXT,
                preview_url TEXT,
                width INTEGER,
                height INTEGER,
                file_type TEXT,
                score INTEGER,
                source TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Processed media cache - stores censored versions
        await db.execute("""
            CREATE TABLE IF NOT EXISTS processed_media (
                id TEXT PRIMARY KEY,
                post_id INTEGER,
                media_type TEXT,
                file_path TEXT,
                detections TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES post_cache(post_id)
            )
        """)

        # Captions cache
        await db.execute("""
            CREATE TABLE IF NOT EXISTS captions (
                id TEXT PRIMARY KEY,
                post_id INTEGER,
                caption TEXT,
                style TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES post_cache(post_id)
            )
        """)

        await db.commit()
        print("[DB] Database initialized")


async def cache_post(post: dict):
    """Cache a Rule34 post to the database."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO post_cache
            (post_id, tags, file_url, sample_url, preview_url, width, height, file_type, score, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post["id"],
            post["tags"],
            post["file_url"],
            post["sample_url"],
            post["preview_url"],
            post["width"],
            post["height"],
            post["file_type"],
            post["score"],
            post.get("source", ""),
        ))
        await db.commit()


async def get_cached_post(post_id: int) -> dict | None:
    """Get a cached post by ID."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM post_cache WHERE post_id = ?",
            (post_id,)
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None


async def cache_processed_media(
    media_id: str,
    post_id: int,
    media_type: str,
    file_path: str,
    detections: list[dict]
):
    """Cache processed (censored) media."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO processed_media
            (id, post_id, media_type, file_path, detections)
            VALUES (?, ?, ?, ?, ?)
        """, (
            media_id,
            post_id,
            media_type,
            file_path,
            json.dumps(detections)
        ))
        await db.commit()


async def get_cached_processed_media(media_id: str) -> dict | None:
    """Get cached processed media."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM processed_media WHERE id = ?",
            (media_id,)
        )
        row = await cursor.fetchone()
        if row:
            result = dict(row)
            result["detections"] = json.loads(result["detections"])
            return result
        return None


# =============================================================================
# Media Processing Service
# =============================================================================

class MediaProcessor:
    """Service for processing and serving censored media."""

    def __init__(self, censor: NudeNetCensor, rule34: Rule34Client):
        self.censor = censor
        self.rule34 = rule34
        self.video_processor = VideoCensorProcessor(censor)
        self._http_client: httpx.AsyncClient | None = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                headers={"User-Agent": "SuperBrowser/1.0"}
            )
        return self._http_client

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        await self.video_processor.close()

    def _generate_media_id(self, post_id: int) -> str:
        """Generate a unique ID for processed media."""
        return f"r34_{post_id}"

    async def get_or_process_media(self, post_id: int) -> tuple[str, str]:
        """
        Get or process censored media for a post.

        Returns:
            Tuple of (file_path, media_type)
        """
        media_id = self._generate_media_id(post_id)

        # Check cache first
        cached = await get_cached_processed_media(media_id)
        if cached and Path(cached["file_path"]).exists():
            print(f"[Media] Cache hit for post {post_id}")
            return cached["file_path"], cached["media_type"]

        # Get post info
        post = await get_cached_post(post_id)
        if not post:
            post = await self.rule34.get_post_by_id(post_id)
            if not post:
                raise ValueError(f"Post {post_id} not found")
            await cache_post(post)

        file_url = post["file_url"]
        file_type = post["file_type"]

        print(f"[Media] Processing post {post_id} ({file_type}): {file_url}")

        # Download original
        response = await self.http_client.get(file_url)
        response.raise_for_status()
        original_bytes = response.content

        # Process based on type
        if file_type == "video":
            # Process video
            processed_bytes, stats = await self.video_processor.process_video(original_bytes)
            extension = "mp4"
            media_type = "video"
            detections = [{"stats": stats}]
        elif file_type == "gif":
            # Process GIF as video
            processed_bytes, stats = await self.video_processor.process_video(
                original_bytes, output_format="mp4"
            )
            extension = "mp4"
            media_type = "gif"
            detections = [{"stats": stats}]
        else:
            # Process image
            processed_bytes, detections = await self.censor.censor_image(
                original_bytes,
                threshold=CENSOR_THRESHOLD
            )
            extension = "jpg"
            media_type = "image"

        # Save to cache directory
        file_path = MEDIA_CACHE_DIR / f"{media_id}.{extension}"
        file_path.write_bytes(processed_bytes)

        # Cache metadata
        await cache_processed_media(
            media_id=media_id,
            post_id=post_id,
            media_type=media_type,
            file_path=str(file_path),
            detections=detections
        )

        print(f"[Media] Saved processed media: {file_path}")
        return str(file_path), media_type


# =============================================================================
# Global Instances
# =============================================================================

rule34_client: Rule34Client | None = None
censor: NudeNetCensor | None = None
media_processor: MediaProcessor | None = None


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global rule34_client, censor, media_processor

    print("=" * 60)
    print("Super Browser - Starting up...")
    print("=" * 60)

    # Initialize database
    await init_db()

    # Initialize Rule34 client
    rule34_client = Rule34Client()

    # Initialize NudeNet censor
    censor = NudeNetCensor()
    await censor.load()

    # Initialize media processor
    media_processor = MediaProcessor(censor, rule34_client)

    print("=" * 60)
    print("Super Browser - Ready!")
    print("=" * 60)

    yield

    # Cleanup
    print("Shutting down...")
    if rule34_client:
        await rule34_client.close()
    if media_processor:
        await media_processor.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "super-browser"}


@app.get("/tags/popular")
async def get_popular_tags_endpoint():
    """Get hardcoded popular tags list."""
    return {"tags": [{"name": tag, "type": "general"} for tag in POPULAR_TAGS]}


@app.get("/tags/autocomplete")
async def autocomplete_tags(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50)
):
    """Autocomplete tag search."""
    if not rule34_client:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    suggestions = await rule34_client.autocomplete_tags(query=q, limit=limit)
    return {"suggestions": suggestions}


@app.get("/posts")
async def get_posts(
    tags: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100),
    page: int = Query(0, ge=0),
    sort: str = Query("score", regex="^(score|id)$")
):
    """Get posts for given tags."""
    if not rule34_client:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    posts = await rule34_client.get_posts(
        tags=tags,
        limit=limit,
        page=page,
        sort=sort
    )

    # Cache posts
    for post in posts:
        await cache_post(post)

    return {"posts": posts, "count": len(posts)}


@app.get("/posts/{post_id}")
async def get_post(post_id: int):
    """Get a single post by ID."""
    if not rule34_client:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    # Check cache first
    post = await get_cached_post(post_id)
    if not post:
        post = await rule34_client.get_post_by_id(post_id)
        if not post:
            return JSONResponse({"error": "Post not found"}, status_code=404)
        await cache_post(post)

    return {"post": post}


@app.get("/media/{post_id}")
async def get_media(post_id: int):
    """Get censored media for a post."""
    if not media_processor:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    try:
        file_path, media_type = await media_processor.get_or_process_media(post_id)

        # Determine content type
        if media_type == "video" or media_type == "gif":
            content_type = "video/mp4"
        else:
            content_type = "image/jpeg"

        return FileResponse(
            file_path,
            media_type=content_type,
            filename=f"r34_{post_id}.{file_path.split('.')[-1]}"
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        print(f"[Media] Error processing {post_id}: {e}")
        return JSONResponse({"error": "Processing failed"}, status_code=500)


@app.get("/media/{post_id}/status")
async def get_media_status(post_id: int):
    """Check if media has been processed."""
    media_id = f"r34_{post_id}"
    cached = await get_cached_processed_media(media_id)

    if cached and Path(cached["file_path"]).exists():
        return {
            "processed": True,
            "media_type": cached["media_type"],
            "url": f"{SERVER_URL}/media/{post_id}" if SERVER_URL else f"/media/{post_id}"
        }

    return {"processed": False}


# =============================================================================
# Poe Bot Handler
# =============================================================================

class SuperBrowserBot(fp.PoeBot):
    """Poe bot for browsing Rule34 content."""

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        """Handle incoming bot requests."""
        query = request.query[-1].content.strip().lower()

        # Simple command routing
        if query.startswith("/tags"):
            # Use hardcoded popular tags list
            tag_list = "\n".join([f"- **{tag}**" for tag in POPULAR_TAGS])
            yield fp.PartialResponse(text=f"## Popular Tags\n\n{tag_list}\n\nUse `/search <tag>` to browse posts.")

        elif query.startswith("/search "):
            # Parse: /search <tag> [page]
            parts = query[8:].strip().split()
            if not parts:
                yield fp.PartialResponse(text="Usage: `/search <tag> [page]`")
                return

            # Check if last part is a page number
            page = 0
            if len(parts) > 1 and parts[-1].isdigit():
                page = int(parts[-1])
                search_term = " ".join(parts[:-1])
            else:
                search_term = " ".join(parts)

            posts_per_page = 20
            posts = await rule34_client.get_posts(tags=search_term, limit=posts_per_page, page=page)

            if not posts:
                yield fp.PartialResponse(text=f"No posts found for: {search_term} (page {page})")
                return

            base_url = SERVER_URL if SERVER_URL else ""
            response = f"## Results for: {search_term} (page {page})\n\n"
            response += "| Post | Score | Type | Censored |\n"
            response += "|------|-------|------|----------|\n"
            for post in posts:
                censored_url = f"{base_url}/media/{post['id']}"
                response += f"| #{post['id']} | {post['score']} | {post['file_type']} | [View]({censored_url}) |\n"

            response += f"\n**Thumbnails** (uncensored):\n"
            for post in posts:
                if post.get('sample_url'):
                    response += f"- #{post['id']}: {post['sample_url']}\n"

            # Navigation hint
            if len(posts) == posts_per_page:
                response += f"\n*More results: `/search {search_term} {page + 1}`*"

            yield fp.PartialResponse(text=response)

        else:
            yield fp.PartialResponse(text="""
## Super Browser

Commands:
- `/tags` - View popular tags
- `/search <tag>` - Search posts by tag (page 0)
- `/search <tag> <page>` - Search with pagination

**REST API Endpoints:**
- `GET /tags/popular` - Get popular tags
- `GET /posts?tags=<tag>&page=0&limit=50` - Get posts (paginated)
- `GET /media/<post_id>` - Get censored image/video

Use the canvas app for the full browsing experience!
            """)

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Return bot settings."""
        return fp.SettingsResponse(
            server_bot_dependencies={},
            introduction_message="Welcome to Super Browser! Use /tags to see trending content.",
        )


poe_bot = SuperBrowserBot()


# Mount Poe bot endpoint
# Poe sends POST to / so we need to handle it there
# But we also have REST endpoints, so we use add_api_route for POST only
poe_app = fp.make_app(poe_bot, access_key=POE_ACCESS_KEY)

# Get the actual endpoint handler from the Poe app
for route in poe_app.routes:
    if hasattr(route, 'methods') and 'POST' in route.methods:
        app.add_api_route("/", route.endpoint, methods=["POST"])


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
