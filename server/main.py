"""
Super Browser - Poe Server Bot
A Poe-protocol-compatible server for browsing and processing RedGifs content.
"""

import os
import io
import json
import hashlib
import asyncio
from pathlib import Path
from typing import AsyncIterable
from contextlib import asynccontextmanager

import httpx
import aiosqlite
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

import fastapi_poe as fp
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

import redgifs
from redgifs import Order

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", "cache.db")
MEDIA_CACHE_DIR = Path(os.getenv("MEDIA_CACHE_DIR", "media_cache"))
POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY", "")
BOT_NAME = os.getenv("BOT_NAME", "SuperBrowser")
SERVER_URL = os.getenv("SERVER_URL", "")  # Your Railway URL, e.g., https://xxx.up.railway.app

# Ensure media cache directory exists
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
        api = await self.get_api()
        gif = await asyncio.to_thread(api.get_gif, gif_id)
        url = gif.urls.thumbnail or gif.urls.poster
        if not url:
            raise Exception("No thumbnail available")
        media_bytes = await asyncio.to_thread(api.download, url)
        content_type = "image/jpeg"
        if url.endswith(".png"):
            content_type = "image/png"
        elif url.endswith(".gif"):
            content_type = "image/gif"
        elif url.endswith(".webp"):
            content_type = "image/webp"
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

def process_image(image_bytes: bytes, text: str = "Hello") -> bytes:
    """Process an image by adding text overlay."""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode in ('RGBA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)
    font_size = max(20, img.width // 15)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img.width - text_width) // 2
    y = img.height - text_height - 20

    # Shadow
    for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85)
    return output.getvalue()


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


async def cache_processed_media(gif_id: str, processed_bytes: bytes) -> str:
    filename = f"{hashlib.md5(gif_id.encode()).hexdigest()}.jpg"
    filepath = MEDIA_CACHE_DIR / filename
    await asyncio.to_thread(filepath.write_bytes, processed_bytes)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO processed_media (gif_id, processed_path, original_type)
            VALUES (?, ?, ?)
        """, (gif_id, str(filepath), "image/jpeg"))
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

**For Canvas Apps:**
Responses are returned as JSON in code blocks for easy parsing.
"""

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Return bot settings."""
        return fp.SettingsResponse(
            server_bot_dependencies={},
            introduction_message="Welcome to Super Browser! 🌐\n\nUse `browse <tag>` to search for content, or type `help` for more commands.",
        )


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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
    return {"status": "ok", "service": "Super Browser Bot"}


@app.get("/media/{gif_id}")
async def get_processed_media(gif_id: str, overlay_text: str = "Hello"):
    """
    Serve processed media with overlay.
    This endpoint is called directly by the Canvas app to load images.
    """
    # Check cache
    cached_path = await get_processed_media_path(gif_id)
    if cached_path:
        return FileResponse(cached_path, media_type="image/jpeg")

    # Download and process
    try:
        original_bytes, _ = await redgifs_client.download_media(gif_id)
    except Exception as e:
        return Response(content=f"Error downloading: {e}", status_code=404)

    try:
        processed_bytes = await asyncio.to_thread(process_image, original_bytes, overlay_text)
    except Exception as e:
        return Response(content=f"Error processing: {e}", status_code=500)

    # Cache
    filepath = await cache_processed_media(gif_id, processed_bytes)

    return FileResponse(filepath, media_type="image/jpeg")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)