"""
Super Browser - Poe Server Bot
A Poe-protocol-compatible server for browsing and processing RedGifs content.
Features automated nudity detection and censoring using NudeNet.
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
from PIL import Image, ImageDraw
from dotenv import load_dotenv

import fastapi_poe as fp
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

import redgifs
from redgifs import Order

# NudeNet for nudity detection and censoring
from nudenet import NudeDetector

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", "cache.db")
MEDIA_CACHE_DIR = Path(os.getenv("MEDIA_CACHE_DIR", "media_cache"))
POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY", "")
BOT_NAME = os.getenv("BOT_NAME", "SuperBrowser")
SERVER_URL = os.getenv("SERVER_URL", "")  # Your Railway URL, e.g., https://xxx.up.railway.app

# Nudity censoring configuration
# Minimum confidence score for detection (0.0 to 1.0)
CENSOR_THRESHOLD = float(os.getenv("CENSOR_THRESHOLD", "0.4"))

# Body parts to censor (exposed parts only by default)
DEFAULT_CENSOR_CLASSES = [
    "FEMALE_BREAST_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]

# Parse custom censor classes from env if provided
CENSOR_CLASSES = os.getenv("CENSOR_CLASSES", "").split(",") if os.getenv("CENSOR_CLASSES") else DEFAULT_CENSOR_CLASSES
CENSOR_CLASSES = [c.strip() for c in CENSOR_CLASSES if c.strip()]

# Ensure media cache directory exists
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# NudeNet Detector (Singleton)
# =============================================================================

class NudeNetCensor:
    """
    Singleton wrapper for NudeNet detector.
    Handles thread-safe initialization and provides censoring functionality.
    """

    def __init__(self):
        self._detector: NudeDetector | None = None
        self._lock = asyncio.Lock()

    async def get_detector(self) -> NudeDetector:
        """Get or initialize the NudeNet detector."""
        async with self._lock:
            if self._detector is None:
                # Initialize detector in a thread to avoid blocking
                self._detector = await asyncio.to_thread(self._create_detector)
            return self._detector

    def _create_detector(self) -> NudeDetector:
        """Create and return a NudeDetector instance."""
        print("Initializing NudeNet detector...")
        detector = NudeDetector()
        print("NudeNet detector initialized successfully")
        return detector

    async def detect(self, image_path: str) -> list[dict]:
        """
        Detect nudity in an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of detections, each containing:
            - class: The body part class (e.g., "FEMALE_BREAST_EXPOSED")
            - score: Confidence score (0.0 to 1.0)
            - box: Bounding box [x, y, width, height]
        """
        detector = await self.get_detector()
        return await asyncio.to_thread(detector.detect, image_path)

    async def censor_image(
        self,
        image_bytes: bytes,
        classes: list[str] | None = None,
        threshold: float = CENSOR_THRESHOLD,
    ) -> tuple[bytes, list[dict]]:
        """
        Detect and censor nudity in an image.

        Args:
            image_bytes: Raw image bytes
            classes: List of body part classes to censor (uses CENSOR_CLASSES if None)
            threshold: Minimum confidence score for detection

        Returns:
            Tuple of (censored_image_bytes, detections_list)
        """
        detector = await self.get_detector()
        classes_to_censor = classes or CENSOR_CLASSES

        # NudeNet requires a file path, so we use temp files
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in:
            tmp_in.write(image_bytes)
            tmp_in_path = tmp_in.name

        try:
            # Detect nudity
            detections = await asyncio.to_thread(detector.detect, tmp_in_path)

            # Filter detections by class and threshold
            filtered_detections = [
                d for d in detections
                if d["class"] in classes_to_censor and d["score"] >= threshold
            ]

            if not filtered_detections:
                # No nudity detected that needs censoring
                return image_bytes, detections

            # Apply black boxes using Pillow for more control
            censored_bytes = await asyncio.to_thread(
                self._apply_black_boxes,
                image_bytes,
                filtered_detections
            )

            return censored_bytes, detections

        finally:
            # Clean up temp file
            try:
                Path(tmp_in_path).unlink()
            except Exception:
                pass

    def _apply_black_boxes(self, image_bytes: bytes, detections: list[dict]) -> bytes:
        """
        Apply black boxes over detected regions.

        Args:
            image_bytes: Raw image bytes
            detections: List of detections with bounding boxes

        Returns:
            Censored image bytes
        """
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
            # NudeNet returns [x, y, width, height]
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

    async def search(self, query: str, page: int = 1, count: int = 20) -> dict:
        api = await self.get_api()
        result = await asyncio.to_thread(
            api.search, query, order=Order.TRENDING, count=count, page=page
        )
        return self._parse_search_result(result)

    async def get_gif(self, gif_id: str) -> dict:
        api = await self.get_api()
        result = await asyncio.to_thread(api.get_gif, gif_id)
        return self._parse_gif(result)

    async def download_media(self, gif_id: str) -> tuple[bytes, str]:
        """
        Download media for a GIF. Returns (bytes, content_type).
        Uses the library's download method which requires a file path.
        """
        api = await self.get_api()
        gif = await asyncio.to_thread(api.get_gif, gif_id)
        url = gif.urls.thumbnail or gif.urls.poster
        if not url:
            raise Exception("No thumbnail available")

        # Determine content type from URL
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

        # Download to temp file (redgifs library requires file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        try:
            await asyncio.to_thread(api.download, url, tmp_path)
            media_bytes = await asyncio.to_thread(Path(tmp_path).read_bytes)
        finally:
            # Clean up temp file
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
# Image Processing (with NudeNet censoring)
# =============================================================================

async def process_image(image_bytes: bytes) -> tuple[bytes, list[dict]]:
    """
    Process an image by detecting and censoring nudity.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of (processed_image_bytes, detections)
    """
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
    """
    Poe bot that handles browsing and processing RedGifs content.

    Commands:
        browse <tag> [page]  - Browse GIFs by tag
        item <gif_id>        - Get processed item with caption
        help                 - Show available commands
    """

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        """Handle incoming messages from Poe."""

        # Get the latest user message
        last_message = request.query[-1].content.strip()

        # Parse command
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
                # Default to browse if it looks like a tag
                async for response in self._handle_browse([command] + args):
                    yield response

        except Exception as e:
            yield fp.PartialResponse(text=f"Error: {str(e)}")

    async def _handle_browse(self, args: list[str]) -> AsyncIterable[fp.PartialResponse]:
        """Handle browse command: browse <tag> [page]"""
        if not args:
            yield fp.PartialResponse(text="Please specify a tag to browse. Example: `browse cats`")
            return

        tag = args[0]
        page = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1
        count = 10  # Items per page

        yield fp.PartialResponse(text=f"🔍 Searching for **{tag}** (page {page})...\n\n")

        try:
            result = await redgifs_client.search(tag, page=page, count=count)
        except Exception as e:
            yield fp.PartialResponse(text=f"Error fetching from RedGifs: {str(e)}")
            return

        # Cache all GIFs
        for gif in result["gifs"]:
            await cache_gif(gif)

        # Build response as JSON for Canvas app to parse
        response_data = {
            "type": "browse_result",
            "tag": tag,
            "page": result["page"],
            "pages": result["pages"],
            "total": result["total"],
            "items": []
        }

        for gif in result["gifs"]:
            # Build the media URL that points back to our server
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

        # Return as JSON for the Canvas app
        yield fp.PartialResponse(text=f"```json\n{json.dumps(response_data, indent=2)}\n```")

    async def _handle_item(self, args: list[str]) -> AsyncIterable[fp.PartialResponse]:
        """Handle item command: item <gif_id>"""
        if not args:
            yield fp.PartialResponse(text="Please specify a GIF ID. Example: `item abcxyz123`")
            return

        gif_id = args[0]

        yield fp.PartialResponse(text=f"📦 Loading item **{gif_id}**...\n\n")

        # Get GIF metadata
        gif_data = await get_cached_gif(gif_id)
        if not gif_data:
            try:
                gif_data = await redgifs_client.get_gif(gif_id)
                await cache_gif(gif_data)
            except Exception as e:
                yield fp.PartialResponse(text=f"Error: Could not find GIF {gif_id}: {str(e)}")
                return

        # Get or generate caption
        caption = await get_cached_caption(gif_id)
        if not caption:
            # For now, use a placeholder caption
            # In the future, this would call a Poe bot for caption generation
            caption = "A captivating moment captured in pixels! ✨"
            await cache_caption(gif_id, caption, "default")

        # Build response
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

- `browse <tag>` - Browse GIFs by tag
  - Example: `browse cats`
  - Example: `browse funny 2` (page 2)

- `item <gif_id>` - Get a specific item with caption
  - Example: `item abcxyz123`

- `help` - Show this help message

**Features:**
- 🔒 Automated nudity censoring using AI detection
- 📦 Cached and processed images served from `/media/{id}`

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Pre-initialize NudeNet detector on startup
    print("Pre-loading NudeNet detector...")
    await nudenet_censor.get_detector()
    print("NudeNet detector ready!")
    yield
    await redgifs_client.close()


# Create the bot instance
bot = SuperBrowserBot()

# Create FastAPI app with Poe bot handlers
app = fp.make_app(
    bot,
    access_key=POE_ACCESS_KEY if POE_ACCESS_KEY else None,
    bot_name=BOT_NAME,
    allow_without_key=not POE_ACCESS_KEY,  # Allow without key if not configured
)

# Add CORS for direct API access (media endpoints)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add lifespan handler
original_lifespan = app.router.lifespan_context

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    await init_db()
    # Pre-initialize NudeNet detector on startup
    print("Pre-loading NudeNet detector...")
    await nudenet_censor.get_detector()
    print("NudeNet detector ready!")
    async with original_lifespan(app):
        yield
    await redgifs_client.close()

app.router.lifespan_context = combined_lifespan


# =============================================================================
# Additional REST Endpoints (for serving processed media)
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Super Browser Bot",
        "features": {
            "nudity_censoring": True,
            "censor_threshold": CENSOR_THRESHOLD,
            "censor_classes": CENSOR_CLASSES,
        }
    }


@app.get("/media/{gif_id}")
async def get_processed_media(gif_id: str):
    """
    Serve processed media with nudity censored.
    This endpoint is called directly by the Canvas app to load images.
    """
    # Check cache
    cached_path = await get_processed_media_path(gif_id)
    if cached_path:
        return FileResponse(cached_path, media_type="image/jpeg")

    # Download original
    try:
        original_bytes, _ = await redgifs_client.download_media(gif_id)
    except Exception as e:
        return Response(content=f"Error downloading: {e}", status_code=404)

    # Process with nudity censoring
    try:
        processed_bytes, detections = await process_image(original_bytes)
    except Exception as e:
        return Response(content=f"Error processing: {e}", status_code=500)

    # Cache the result
    filepath = await cache_processed_media(gif_id, processed_bytes, detections)

    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/media/{gif_id}/detections")
async def get_media_detections(gif_id: str):
    """
    Get detection results for a processed media item.
    Returns the list of detected body parts and their bounding boxes.
    """
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
