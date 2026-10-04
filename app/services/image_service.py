"""Image downloading, validation, optimization, and base64 encoding service."""

import base64
import io
import logging
from typing import Tuple
import httpx
from PIL import Image, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}

# Standard browser headers to prevent CDN 403 blocks from retail sites (e.g. Myntra, Ajio)
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}


class ImageProcessingError(Exception):
    """Custom exception raised when image download or validation fails."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def fetch_and_prepare_image(image_url: str) -> str:
    """Fetches an image from a URL, validates it, optimizes if necessary,

    and returns a base64 Data URI string ready for the Groq Vision API.

    Returns:
        Data URI string (e.g. "data:image/jpeg;base64,...")

    Raises:
        ImageProcessingError: on network, timeout, format, or parsing errors.
    """
    logger.info(f"Fetching image from URL: {image_url}")

    try:
        async with httpx.AsyncClient(
            headers=HTTP_HEADERS,
            follow_redirects=True,
            timeout=settings.IMAGE_DOWNLOAD_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(image_url)

    except httpx.TimeoutException:
        logger.error(f"Image download timed out for URL: {image_url}")
        raise ImageProcessingError(
            f"Image download timed out after {settings.IMAGE_DOWNLOAD_TIMEOUT_SECONDS}s. Please check the image host.",
            status_code=504
        )
    except httpx.ConnectError:
        logger.error(f"Could not connect to image server: {image_url}")
        raise ImageProcessingError(
            "Could not connect to the image host. The URL may be inaccessible or invalid.",
            status_code=400
        )
    except httpx.RequestError as exc:
        logger.error(f"Network error while fetching image: {exc}")
        raise ImageProcessingError(
            f"Failed to fetch image from URL: {str(exc)}",
            status_code=400
        )

    if response.status_code != 200:
        logger.error(f"Image download failed with status {response.status_code} for URL: {image_url}")
        raise ImageProcessingError(
            f"Failed to download image. Server returned HTTP {response.status_code}.",
            status_code=400
        )

    image_bytes = response.content
    if not image_bytes:
        raise ImageProcessingError(
            "The retrieved response is empty.",
            status_code=400
        )

    # Validate image format and integrity using Pillow
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError) as exc:
        logger.error(f"Invalid image content: {exc}")
        raise ImageProcessingError(
            "The downloaded content is not a valid or recognizable image file.",
            status_code=422
        )

    if image_format not in SUPPORTED_FORMATS:
        raise ImageProcessingError(
            f"Unsupported image format: '{image_format}'. Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}.",
            status_code=422
        )

    # Optimize and convert to standard JPEG Data URI
    # This ensures it is under the 4MB Groq base64 limit and standardized for LLM vision
    optimized_bytes, mime_type = _optimize_image(image)
    b64_encoded = base64.b64encode(optimized_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{b64_encoded}"


def _optimize_image(image: Image.Image) -> Tuple[bytes, str]:
    """Resizes large images down to max dimensions and converts to JPEG.

    Ensures the payload is compact, reliable, and high-fidelity for vision analysis.
    """
    max_dim = settings.MAX_IMAGE_DIMENSION

    # Convert to RGB mode if necessary (handles RGBA, P, etc.)
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[3] if image.mode == "RGBA" else None)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    if width > max_dim or height > max_dim:
        scale = min(max_dim / width, max_dim / height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    output_buffer = io.BytesIO()
    # Save as high-quality JPEG
    image.save(output_buffer, format="JPEG", quality=85, optimize=True)
    return output_buffer.getvalue(), "image/jpeg"
