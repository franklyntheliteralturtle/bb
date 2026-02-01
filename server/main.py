"""
Super Browser Server
FastAPI application for browsing, caching, and processing RedGifs content.
"""

import os
import io
import hashlib
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import aiosqlite
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import redgifs
from redgifs import Order

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", "cache.db")
MEDIA_CACHE_DIR = Path(os.getenv("MEDIA_CACHE_DIR", "media_cache"))
POE_API_KEY = os.getenv("POE_API_KEY", "")
POE_BOT_NAME = os.getenv("POE_BOT_NAME", "GPT-4o")  # Bot to use for captions

# Ensure media cache directory exists
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Database Setup
# =============================================================================

async def init_db():
    """Initialize SQLite database with required tables."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Table for cached GIF metadata
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

        # Table for generated captions
        await db.execute("""
            CREATE TABLE IF NOT EXISTS captions (
                gif_id TEXT PRIMARY KEY,
                caption TEXT,
                persona TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gif_id) REFERENCES gif_cache(id)
            )
        """)

        # Table for processed media paths
        await db.execute("""
            CREATE TABLE IF NOT EXISTS processed_media (
                gif_id TEXT PRIMARY KEY,
                processed_path TEXT,
                original_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gif_id) REFERENCES gif_cache(id)
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
        self._api: Optional[redgifs.API] = None
        self._lock = asyncio.Lock()

    async def get_api(self) -> redgifs.API:
        """Get or create an authenticated API instance."""
        async with self._lock:
            if self._api is None:
                # Using sync API in async context - run in thread pool
                self._api = await asyncio.to_thread(self._create_api)
            return self._api

    def _create_api(self) -> redgifs.API:
        """Create and authenticate a new API instance."""
        api = redgifs.API()
        api.login()  # Gets temporary token
        return api

    async def search(self, query: str, page: int = 1, count: int = 20) -> dict:
        """Search for GIFs by query."""
        api = await self.get_api()
        result = await asyncio.to_thread(
            api.search, query, order=Order.TRENDING, count=count, page=page
        )
        return self._parse_search_result(result)

    async def get_trending(self) -> dict:
        """Get trending GIFs."""
        api = await self.get_api()
        result = await asyncio.to_thread(api.fetch_trending_gifs)
        return self._parse_gif_list(result)

    async def get_gif(self, gif_id: str) -> dict:
        """Get a specific GIF by ID."""
        api = await self.get_api()
        result = await asyncio.to_thread(api.get_gif, gif_id)
        return self._parse_gif(result)

    async def download_media(self, gif_id: str) -> tuple[bytes, str]:
        """
        Download media for a GIF. Returns (bytes, content_type).
        Uses the library's download method to handle auth properly.
        """
        api = await self.get_api()
        gif = await asyncio.to_thread(api.get_gif, gif_id)

        # Get the thumbnail/poster for image processing
        # (We'll use thumbnail for now since it's an image)
        url = gif.urls.thumbnail or gif.urls.poster

        if not url:
            raise HTTPException(status_code=404, detail="No thumbnail available")

        # Download using the library's method
        media_bytes = await asyncio.to_thread(api.download, url)

        # Determine content type from URL
        content_type = "image/jpeg"
        if url.endswith(".png"):
            content_type = "image/png"
        elif url.endswith(".gif"):
            content_type = "image/gif"
        elif url.endswith(".webp"):
            content_type = "image/webp"

        return media_bytes, content_type

    def _parse_search_result(self, result) -> dict:
        """Parse a SearchResult into a dict."""
        return {
            "page": result.page,
            "pages": result.pages,
            "total": result.total,
            "gifs": [self._parse_gif(gif) for gif in (result.gifs or [])]
        }

    def _parse_gif_list(self, gifs) -> dict:
        """Parse a list of GIFs into a dict."""
        return {
            "gifs": [self._parse_gif(gif) for gif in gifs]
        }

    def _parse_gif(self, gif) -> dict:
        """Parse a GIF object into a dict."""
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
        """Close the API connection."""
        if self._api:
            await asyncio.to_thread(self._api.close)
            self._api = None


# Global client instance
redgifs_client = RedGifsClient()


# =============================================================================
# Image Processing
# =============================================================================

def process_image(image_bytes: bytes, text: str = "Hello") -> bytes:
    """
    Process an image by adding text overlay.
    Returns processed image as bytes.
    """
    # Open image
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (for JPEG output)
    if img.mode in ('RGBA', 'P'):
        # Create white background for transparency
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    font_size = max(20, img.width // 15)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Calculate text position (bottom center with padding)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (img.width - text_width) // 2
    y = img.height - text_height - 20  # 20px padding from bottom

    # Draw text shadow/outline for visibility
    shadow_offset = 2
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((x - shadow_offset, y - shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((x + shadow_offset, y - shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((x - shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))

    # Draw main text
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # Save to bytes
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85)
    return output.getvalue()


# =============================================================================
# Poe API Client (for caption generation)
# =============================================================================

async def generate_caption_with_poe(image_url: str, persona: str = "witty") -> str:
    """
    Generate a caption for an image using Poe API.
    Returns the generated caption.
    """
    if not POE_API_KEY:
        # Return placeholder if no API key configured
        return f"[{persona}] A captivating moment captured in pixels."

    # Poe API endpoint
    url = "https://api.poe.com/bot/message"

    headers = {
        "Authorization": f"Bearer {POE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Craft prompt based on persona
    persona_prompts = {
        "witty": "You're a witty comedian. Give a short, funny one-liner caption for this image. Keep it under 15 words.",
        "poetic": "You're a romantic poet. Give a beautiful, evocative one-line caption for this image. Keep it under 15 words.",
        "sarcastic": "You're hilariously sarcastic. Give a sarcastic but friendly caption for this image. Keep it under 15 words.",
        "wholesome": "You're a wholesome, supportive friend. Give an encouraging caption for this image. Keep it under 15 words.",
    }

    prompt = persona_prompts.get(persona, persona_prompts["witty"])

    payload = {
        "bot": POE_BOT_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "attachments": [{"type": "image", "url": image_url}]
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data.get("text", "No caption generated")
    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"[{persona}] Couldn't generate caption, but this image speaks for itself!"


# =============================================================================
# Caching Functions
# =============================================================================

async def get_cached_gif(gif_id: str) -> Optional[dict]:
    """Get cached GIF metadata."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM gif_cache WHERE id = ?", (gif_id,)
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
    return None


async def cache_gif(gif_data: dict):
    """Cache GIF metadata."""
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
            ",".join(gif_data.get("tags", [])),
            gif_data.get("username"),
        ))
        await db.commit()


async def get_cached_caption(gif_id: str) -> Optional[str]:
    """Get cached caption for a GIF."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT caption FROM captions WHERE gif_id = ?", (gif_id,)
        )
        row = await cursor.fetchone()
        if row:
            return row[0]
    return None


async def cache_caption(gif_id: str, caption: str, persona: str):
    """Cache a generated caption."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO captions (gif_id, caption, persona)
            VALUES (?, ?, ?)
        """, (gif_id, caption, persona))
        await db.commit()


async def get_processed_media_path(gif_id: str) -> Optional[str]:
    """Get path to processed media file."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT processed_path FROM processed_media WHERE gif_id = ?", (gif_id,)
        )
        row = await cursor.fetchone()
        if row and Path(row[0]).exists():
            return row[0]
    return None


async def cache_processed_media(gif_id: str, processed_bytes: bytes) -> str:
    """Save processed media to disk and record in database."""
    # Generate filename from gif_id hash
    filename = f"{hashlib.md5(gif_id.encode()).hexdigest()}.jpg"
    filepath = MEDIA_CACHE_DIR / filename

    # Write to disk
    await asyncio.to_thread(filepath.write_bytes, processed_bytes)

    # Record in database
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO processed_media (gif_id, processed_path, original_type)
            VALUES (?, ?, ?)
        """, (gif_id, str(filepath), "image/jpeg"))
        await db.commit()

    return str(filepath)


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    await init_db()
    yield
    await redgifs_client.close()


app = FastAPI(
    title="Super Browser API",
    description="API for browsing, caching, and processing RedGifs content",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Canvas app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Canvas apps come from various Poe domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Super Browser API"}


@app.get("/browse")
async def browse(
    tag: str = Query(..., description="Tag/category to browse"),
    page: int = Query(1, ge=1, description="Page number"),
    count: int = Query(20, ge=1, le=50, description="Items per page"),
):
    """
    Browse GIFs by tag/category.
    Returns paginated list of GIFs with metadata.
    """
    try:
        result = await redgifs_client.search(tag, page=page, count=count)

        # Cache all GIF metadata
        for gif in result["gifs"]:
            await cache_gif(gif)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trending")
async def trending():
    """Get trending GIFs."""
    try:
        result = await redgifs_client.get_trending()

        # Cache all GIF metadata
        for gif in result["gifs"]:
            await cache_gif(gif)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gif/{gif_id}")
async def get_gif(gif_id: str):
    """Get metadata for a specific GIF."""
    # Check cache first
    cached = await get_cached_gif(gif_id)
    if cached:
        return cached

    # Fetch from API
    try:
        gif_data = await redgifs_client.get_gif(gif_id)
        await cache_gif(gif_data)
        return gif_data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"GIF not found: {e}")


@app.get("/media/{gif_id}")
async def get_processed_media(
    gif_id: str,
    overlay_text: str = Query("Hello", description="Text to overlay on image"),
):
    """
    Get processed media for a GIF.
    Downloads, processes (adds text overlay), caches, and returns the image.
    """
    # Check for cached processed version
    cached_path = await get_processed_media_path(gif_id)
    if cached_path:
        image_bytes = await asyncio.to_thread(Path(cached_path).read_bytes)
        return Response(content=image_bytes, media_type="image/jpeg")

    # Download original
    try:
        original_bytes, _ = await redgifs_client.download_media(gif_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not download media: {e}")

    # Process image
    try:
        processed_bytes = await asyncio.to_thread(process_image, original_bytes, overlay_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process image: {e}")

    # Cache processed version
    await cache_processed_media(gif_id, processed_bytes)

    return Response(content=processed_bytes, media_type="image/jpeg")


@app.get("/caption/{gif_id}")
async def get_caption(
    gif_id: str,
    persona: str = Query("witty", description="Caption persona: witty, poetic, sarcastic, wholesome"),
):
    """
    Get or generate a caption for a GIF.
    Uses cached caption if available, otherwise generates via Poe API.
    """
    # Check cache first
    cached_caption = await get_cached_caption(gif_id)
    if cached_caption:
        return {"gif_id": gif_id, "caption": cached_caption, "cached": True}

    # Get GIF metadata for image URL
    gif_data = await get_cached_gif(gif_id)
    if not gif_data:
        try:
            gif_data = await redgifs_client.get_gif(gif_id)
            await cache_gif(gif_data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"GIF not found: {e}")

    # Generate caption using Poe API
    image_url = gif_data.get("thumbnail_url") or gif_data.get("sd_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL available for captioning")

    caption = await generate_caption_with_poe(image_url, persona)

    # Cache the caption
    await cache_caption(gif_id, caption, persona)

    return {"gif_id": gif_id, "caption": caption, "cached": False}


@app.get("/item/{gif_id}")
async def get_full_item(
    gif_id: str,
    overlay_text: str = Query("Hello", description="Text to overlay on image"),
    persona: str = Query("witty", description="Caption persona"),
):
    """
    Get fully processed item: metadata, processed media URL, and caption.
    This is the main endpoint for the Canvas app to use.
    """
    # Get metadata
    gif_data = await get_cached_gif(gif_id)
    if not gif_data:
        try:
            gif_data = await redgifs_client.get_gif(gif_id)
            await cache_gif(gif_data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"GIF not found: {e}")

    # Get or generate caption
    cached_caption = await get_cached_caption(gif_id)
    if not cached_caption:
        image_url = gif_data.get("thumbnail_url") or gif_data.get("sd_url")
        if image_url:
            cached_caption = await generate_caption_with_poe(image_url, persona)
            await cache_caption(gif_id, cached_caption, persona)
        else:
            cached_caption = "No caption available"

    return {
        "id": gif_id,
        "metadata": gif_data,
        "caption": cached_caption,
        "media_url": f"/media/{gif_id}?overlay_text={overlay_text}",
        "web_url": gif_data.get("web_url"),
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)