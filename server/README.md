# Super Browser Server

FastAPI server for browsing, processing, and caching RedGifs content.

## Features

- **RedGifs Integration**: Fetch trending GIFs and search by tags
- **Image Processing**: Add text overlays to thumbnails (extensible for more)
- **Caption Generation**: Generate captions via Poe API with different personas
- **Caching**: SQLite for metadata/captions, filesystem for processed images

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /browse?tag=X&page=1&count=20` | Browse GIFs by tag |
| `GET /trending` | Get trending GIFs |
| `GET /gif/{gif_id}` | Get metadata for a specific GIF |
| `GET /media/{gif_id}?overlay_text=Hello` | Get processed image with overlay |
| `GET /caption/{gif_id}?persona=witty` | Get/generate caption |
| `GET /item/{gif_id}` | Get full item (metadata + caption + media URL) |

## Deploy to Railway

### Option 1: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create new project
railway init

# Deploy
railway up
```

### Option 2: GitHub Integration

1. Push this `server` folder to a GitHub repo
2. Go to [railway.app](https://railway.app)
3. New Project → Deploy from GitHub repo
4. Railway auto-detects the Python app

### Configure Environment Variables

In Railway dashboard, add these variables:

| Variable | Description |
|----------|-------------|
| `POE_API_KEY` | Your Poe API key (get from poe.com/api) |
| `POE_BOT_NAME` | Bot for captions (default: `GPT-4o`) |

Railway automatically provides `PORT`.

### Add Persistent Storage (Optional)

For cache persistence across deploys:

1. In Railway dashboard, go to your service
2. Add a Volume
3. Mount path: `/app/media_cache`
4. Set env var: `MEDIA_CACHE_DIR=/app/media_cache`

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your values

# Run server
python main.py
```

Server runs at `http://localhost:8000`

## Extending Image Processing

The `process_image()` function in `main.py` is where you add image detection/processing:

```python
def process_image(image_bytes: bytes, text: str = "Hello") -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    
    # TODO: Add your image detection here
    # Example: detected_objects = your_model.detect(img)
    
    # Add overlays, filters, etc.
    draw = ImageDraw.Draw(img)
    # ...
    
    return output_bytes
```

## Architecture

```
Canvas App  →  Poe Server Bot  →  This Server  →  RedGifs API
                                       ↓
                                  SQLite Cache
                                  File Storage
```