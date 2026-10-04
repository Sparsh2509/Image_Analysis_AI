"""FestFit AI Product Enrichment API.

Analyzes fashion product images using Groq Vision to determine suitable skin tones,
body types, and occasions strictly adhering to predefined platform categories.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    ProductEnrichmentRequest,
    ProductEnrichmentResponse,
    HealthResponse,
)
from app.services.image_service import (
    fetch_and_prepare_image,
    ImageProcessingError,
)
from app.services.groq_service import (
    groq_service,
    GroqServiceError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("festfit-enrichment-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup check
    keys = settings.groq_keys
    logger.info("Starting FestFit AI Product Enrichment API")
    logger.info(f"Configured Groq Vision Model: {settings.GROQ_VISION_MODEL}")
    logger.info(f"Available Groq API Keys: {len(keys)}")
    if not keys:
        logger.warning(
            "⚠️ No Groq API keys found! Set GROQ_API_KEY_1 and/or GROQ_API_KEY_2 in your environment or .env file."
        )
    yield
    logger.info("Shutting down FestFit AI Product Enrichment API")


app = FastAPI(
    title="FestFit AI Product Enrichment API",
    description="Vision-based AI API to enrich fashion clothing products with suitable skin tones, body types, and occasions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin teammate services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"], tags=["General"])
async def root():
    """Root info endpoint."""
    return {
        "service": "FestFit AI Product Enrichment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "enrich_product": "POST /api/enrich-product"
        }
    }


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Health check endpoint to inspect API status and configured credentials."""
    keys = settings.groq_keys
    return HealthResponse(
        status="healthy" if keys else "degraded (no Groq keys configured)",
        vision_model=settings.GROQ_VISION_MODEL,
        available_api_keys=len(keys),
    )


@app.post(
    "/api/enrich-product",
    response_model=ProductEnrichmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Product Enrichment"],
    summary="Enrich clothing product with AI fashion suitability attributes",
    responses={
        200: {
            "description": "Product successfully analyzed and enriched.",
            "content": {
                "application/json": {
                    "example": {
                        "suitable_skin_tones": ["Wheatish", "Dusky"],
                        "suitable_body_types": ["Rectangle", "Pear"],
                        "suitable_occasions": ["Casual", "College", "Travel"]
                    }
                }
            }
        },
        400: {"description": "Invalid image URL or download failed."},
        422: {"description": "Unsupported image format or corrupt image file."},
        502: {"description": "Groq Vision provider error or invalid model response."},
        503: {"description": "All Groq API keys exhausted or rate-limited."},
        504: {"description": "Image download or Groq request timed out."}
    }
)
async def enrich_product(request: ProductEnrichmentRequest) -> ProductEnrichmentResponse:
    """Analyze a clothing product image from `imageUrl` and determine:

    - `suitable_skin_tones` (from fixed list: Fair, Light, Wheatish, Dusky, Deep)
    - `suitable_body_types` (from fixed list: Rectangle, Hourglass, Pear, Apple, Inverted Triangle)
    - `suitable_occasions` (from fixed list of 27 platform occasions)

    Accepts either just `{"imageUrl": "..."}` or a full product JSON payload.
    Existing product fields are untouched; only the three suitability fields are returned.
    """
    image_url = request.imageUrl

    # Step 1: Download, validate, and optimize image to base64 Data URI
    try:
        image_data_uri = await fetch_and_prepare_image(image_url)
    except ImageProcessingError as exc:
        logger.warning(f"Image processing failed for {image_url}: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.error(f"Unexpected error processing image {image_url}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while downloading and processing product image."
        )

    # Step 2: Pass optional metadata context if present
    context = {}
    if request.name:
        context["name"] = request.name
    if request.category:
        context["category"] = request.category
    if request.brand:
        context["brand"] = request.brand
    if request.description:
        context["description"] = request.description

    # Step 3: Analyze with Groq Vision (with dual-key fallback & strict filtering)
    try:
        enrichment_result = await groq_service.analyze_product(
            image_data_uri=image_data_uri,
            context=context if context else None
        )
        return enrichment_result

    except GroqServiceError as exc:
        logger.error(f"Groq vision analysis error: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.error(f"Unexpected error during enrichment: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during product enrichment analysis."
        )


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

