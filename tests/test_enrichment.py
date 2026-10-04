"""Unit and integration tests for the FestFit AI Product Enrichment API."""

import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
from groq import RateLimitError

from app.main import app
from app.constants import (
    VALID_SKIN_TONES,
    VALID_BODY_TYPES,
    VALID_OCCASIONS,
    filter_skin_tones,
    filter_body_types,
    filter_occasions,
)
from app.services.image_service import (
    fetch_and_prepare_image,
    ImageProcessingError,
    _optimize_image,
)
from app.services.groq_service import GroqVisionService, GroqServiceError
from app.schemas import ProductEnrichmentRequest


client = TestClient(app)


# ---------------------------------------------------------
# 1. Category Filtering & Normalization Tests
# ---------------------------------------------------------

def test_skin_tone_filtering():
    raw_input = ["Fair", "wheatish", "AlienBlue", "Deep", "RandomTone"]
    filtered = filter_skin_tones(raw_input)
    assert filtered == ["Fair", "Wheatish", "Deep"]
    # Check that no non-standard values exist
    for item in filtered:
        assert item in VALID_SKIN_TONES


def test_body_type_filtering():
    raw_input = ["Hourglass", "rectangle", "Muscular", "Pear", "UnknownShape"]
    filtered = filter_body_types(raw_input)
    assert filtered == ["Hourglass", "Rectangle", "Pear"]
    for item in filtered:
        assert item in VALID_BODY_TYPES


def test_occasion_filtering():
    raw_input = ["Casual", "party", "SpaceTravel", "Wedding", "Diwali", "FakeHoliday"]
    filtered = filter_occasions(raw_input)
    assert filtered == ["Casual", "Party", "Wedding", "Diwali"]
    for item in filtered:
        assert item in VALID_OCCASIONS


# ---------------------------------------------------------
# 2. Schema Validation Tests
# ---------------------------------------------------------

def test_request_schema_with_image_url_only():
    req = ProductEnrichmentRequest(imageUrl="https://example.com/item.jpg")
    assert req.imageUrl == "https://example.com/item.jpg"


def test_request_schema_with_alias_image_url():
    req = ProductEnrichmentRequest(image_url="https://example.com/item.jpg")
    assert req.imageUrl == "https://example.com/item.jpg"


def test_request_schema_with_full_product_payload():
    full_product = {
        "brand": "See Designs",
        "category": "Pyjamas",
        "currency": "INR",
        "description": "See Designs Men Mid Rise Pure Cotton Pyjama",
        "imageUrl": "https://assets.myntassets.com/sample.jpg",
        "name": "See Designs Men Mid Rise Pure Cotton Pyjama",
        "price": 454,
        "productUrl": "https://www.myntra.com/sample",
        "source": "MYNTRA"
    }
    req = ProductEnrichmentRequest(**full_product)
    assert req.imageUrl == "https://assets.myntassets.com/sample.jpg"
    assert req.name == "See Designs Men Mid Rise Pure Cotton Pyjama"
    assert req.category == "Pyjamas"


def test_request_schema_invalid_url():
    with pytest.raises(ValueError):
        ProductEnrichmentRequest(imageUrl="not-a-valid-url")

    with pytest.raises(ValueError):
        ProductEnrichmentRequest(imageUrl="")


# ---------------------------------------------------------
# 3. Image Processing Service Tests
# ---------------------------------------------------------

def test_optimize_image_resizes_large_images():
    # Create large 3000x2000 test image in memory
    large_img = Image.new("RGB", (3000, 2000), color=(200, 100, 50))
    optimized_bytes, mime_type = _optimize_image(large_img)
    assert mime_type == "image/jpeg"

    # Verify resized dimension is within max limit
    result_img = Image.open(io.BytesIO(optimized_bytes))
    assert max(result_img.size) <= 1280


@pytest.mark.asyncio
async def test_fetch_and_prepare_image_invalid_content():
    # Mock httpx returning non-image bytes (e.g. HTML or text)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"<html><body>Not an image</body></html>"

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        with pytest.raises(ImageProcessingError) as exc_info:
            await fetch_and_prepare_image("https://example.com/not-image.jpg")
        assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_fetch_and_prepare_image_404_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.content = b"Not Found"

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        with pytest.raises(ImageProcessingError) as exc_info:
            await fetch_and_prepare_image("https://example.com/missing.jpg")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------
# 4. Groq Service Dual-Key Fallback Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_fallback_when_key_1_rate_limited():
    service = GroqVisionService()

    # Mock _get_api_keys to return two keys
    with patch.object(service, "_get_api_keys", return_value=["key_1_mock", "key_2_mock"]):
        # Mock completions create
        mock_choice = MagicMock()
        mock_choice.message.content = (
            '{"suitable_skin_tones": ["Wheatish", "Dusky"], '
            '"suitable_body_types": ["Rectangle", "Pear"], '
            '"suitable_occasions": ["Casual", "College"]}'
        )
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        # Key 1 raises RateLimitError, Key 2 succeeds
        mock_create = AsyncMock(
            side_effect=[
                RateLimitError(message="Rate limit reached", response=MagicMock(status_code=429), body={}),
                mock_completion
            ]
        )

        with patch("groq.AsyncGroq.chat") as mock_chat:
            mock_chat.completions.create = mock_create

            res = await service.analyze_product("data:image/jpeg;base64,abc123mock")
            assert res.suitable_skin_tones == ["Wheatish", "Dusky"]
            assert res.suitable_body_types == ["Rectangle", "Pear"]
            assert res.suitable_occasions == ["Casual", "College"]
            # Verify it attempted Key 1, then fell back to Key 2
            assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_groq_both_keys_fail_raises_503():
    service = GroqVisionService()

    with patch.object(service, "_get_api_keys", return_value=["key_1_mock", "key_2_mock"]):
        mock_create = AsyncMock(
            side_effect=[
                RateLimitError(message="Key 1 rate limited", response=MagicMock(status_code=429), body={}),
                RateLimitError(message="Key 2 rate limited", response=MagicMock(status_code=429), body={}),
            ]
        )

        with patch("groq.AsyncGroq.chat") as mock_chat:
            mock_chat.completions.create = mock_create

            with pytest.raises(GroqServiceError) as exc_info:
                await service.analyze_product("data:image/jpeg;base64,abc123mock")
            assert exc_info.value.status_code == 503


# ---------------------------------------------------------
# 5. API Endpoint Tests (Integration)
# ---------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vision_model" in data
    assert "available_api_keys" in data


def test_enrich_product_endpoint_success():
    fake_data_uri = "data:image/jpeg;base64,dGVzdA=="

    mock_enrichment = {
        "suitable_skin_tones": ["Fair", "Wheatish"],
        "suitable_body_types": ["Hourglass"],
        "suitable_occasions": ["Party", "Wedding"]
    }

    with patch("app.main.fetch_and_prepare_image", new_callable=AsyncMock) as mock_fetch, \
         patch("app.main.groq_service.analyze_product", new_callable=AsyncMock) as mock_analyze:

        mock_fetch.return_value = fake_data_uri
        mock_analyze.return_value = mock_enrichment

        # Send full product JSON
        payload = {
            "brand": "FabIndia",
            "category": "Kurtas",
            "imageUrl": "https://example.com/kurta.jpg",
            "name": "Men Cotton Kurta",
            "price": 1299
        }

        response = client.post("/api/enrich-product", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["suitable_skin_tones"] == ["Fair", "Wheatish"]
        assert data["suitable_body_types"] == ["Hourglass"]
        assert data["suitable_occasions"] == ["Party", "Wedding"]


def test_enrich_product_endpoint_image_fetch_failure():
    with patch("app.main.fetch_and_prepare_image", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = ImageProcessingError("Image download timed out", status_code=504)

        payload = {"imageUrl": "https://example.com/timeout.jpg"}
        response = client.post("/api/enrich-product", json=payload)
        assert response.status_code == 504
        assert "Image download timed out" in response.json()["detail"]
