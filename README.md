# FestFit - AI Product Enrichment API

A clean, production-friendly FastAPI microservice that enriches fashion clothing products using Groq Vision. 

When a product image URL is submitted, the API downloads the image, analyzes the clothing via a vision-capable LLM, and maps visual suitability across **skin tones**, **body types**, and **occasions** strictly to FestFit's predefined platform categories.

---

## 🌟 Features

- **Strict Category Adherence**: Restricts outputs to fixed platform categories (5 skin tones, 5 body types, 27 occasions). Discards hallucinations or invalid values automatically.
- **Direct Image Processing**: Fetches actual image bytes from the URL, validates format/integrity, resizes/optimizes large images, and base64-encodes them for the vision model (does not ask the LLM to inspect URLs as plain text).
- **Dual Groq API Key Fallback**: Supports `GROQ_API_KEY_1` (primary) and `GROQ_API_KEY_2` (backup). Automatically fails over on rate limits (HTTP 429), quota limits, or temporary provider errors with exponential backoff.
- **Efficient API Usage**: Exactly **1 vision LLM call** per product submission under normal conditions. No redundant calls or agent loops.
- **Teammate-Friendly**: Accepts either a minimal payload `{"imageUrl": "..."}` or a full product object (e.g. from Myntra/scrapers) without validation errors. Existing product fields are never modified.
- **Production-Ready**: Comprehensive error handling for timeouts, invalid URLs, unsupported formats, and provider downtime.

---

## 📋 Fixed Category Reference

The API guarantees responses will **only** contain values from these exact lists:

### 1. Suitable Skin Tones (`suitable_skin_tones`)
- `Fair`
- `Light`
- `Wheatish`
- `Dusky`
- `Deep`

### 2. Suitable Body Types (`suitable_body_types`)
- `Rectangle`
- `Hourglass`
- `Pear`
- `Apple`
- `Inverted Triangle`

### 3. Suitable Occasions (`suitable_occasions`)
- `Casual`, `Formal`, `Office`, `College`, `Party`, `Wedding`, `Date`, `Travel`, `Sports`, `Beach`, `Brunch`, `Dinner`, `Interview`
- **Festivals & Cultural**: `Diwali`, `Holi`, `Eid`, `Navratri`, `Dussehra`, `Durga Puja`, `Ganesh Chaturthi`, `Raksha Bandhan`, `Janmashtami`, `Pongal`, `Onam`, `Baisakhi`, `Christmas`, `Karwa Chauth`

> **Note**: These fields indicate which user body types, skin tones, and occasions the *garment* is visually flattering and suitable for. They do **not** describe the model photographed wearing the clothing.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ installed
- One or two Groq API keys from [Groq Console](https://console.groq.com/)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Groq API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Primary key used normally
GROQ_API_KEY_1=gsk_your_primary_key_here

# Backup key used automatically if Key 1 hits rate limits or quota
GROQ_API_KEY_2=gsk_your_backup_key_here

# Vision model (default: qwen/qwen3.8-27b)
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

### 4. Run the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at:
👉 **`http://localhost:8000/docs`**

---

## 📡 API Usage

### Endpoint: `POST /api/enrich-product`

#### Option A: Minimal Request
```bash
curl -X POST "http://localhost:8000/api/enrich-product" \
  -H "Content-Type: application/json" \
  -d '{
    "imageUrl": "https://assets.myntassets.com/h_1440,q_90,w_1080/v1/assets/images/2024/product.jpg"
  }'
```

#### Option B: Full Product Payload (Backend Teammate)
The API accepts full product objects without schema rejection. Optional fields (`name`, `category`, `brand`, `description`) are used as contextual guidance for the vision model:
```bash
curl -X POST "http://localhost:8000/api/enrich-product" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "See Designs",
    "category": "Pyjamas",
    "currency": "INR",
    "description": "See Designs Men Mid Rise Pure Cotton Pyjama",
    "imageUrl": "https://assets.myntassets.com/h_1440,q_90,w_1080/v1/assets/images/sample.jpg",
    "name": "See Designs Men Mid Rise Pure Cotton Pyjama",
    "price": 454,
    "productUrl": "https://www.myntra.com/...",
    "source": "MYNTRA"
  }'
```

#### Success Response (`200 OK`)
```json
{
  "suitable_skin_tones": [
    "Wheatish",
    "Dusky"
  ],
  "suitable_body_types": [
    "Rectangle",
    "Pear"
  ],
  "suitable_occasions": [
    "Casual",
    "College",
    "Travel"
  ]
}
```

The backend teammate can directly assign these three fields to the product document in their database.

---

### Endpoint: `GET /health`
Inspects server readiness and number of active Groq keys configured:
```bash
curl "http://localhost:8000/health"
```
```json
{
  "status": "healthy",
  "vision_model": "qwen/qwen3.8-27b",
  "available_api_keys": 2
}
```

---

## 🛡️ Error Handling & Status Codes

| Status Code | Reason | Cause / Action |
|:---|:---|:---|
| `400 Bad Request` | Invalid URL / Download Failed | The provided `imageUrl` is not a valid HTTP/HTTPS URL, cannot be reached, or returned an HTTP error (e.g. 404). |
| `422 Unprocessable Entity` | Invalid / Corrupt Image | The file at the URL is not a valid image format (e.g. HTML error page or unsupported format). |
| `502 Bad Gateway` | Groq Provider Error | Groq Vision model returned an unexpected or malformed response. |
| `503 Service Unavailable` | Keys Exhausted | Both Groq keys were rate-limited or temporarily unavailable after retries. |
| `504 Gateway Timeout` | Download Timeout | Fetching the image exceeded the timeout limit (`IMAGE_DOWNLOAD_TIMEOUT_SECONDS`). |

---

## 🧪 Running Automated Tests

Run the test suite with pytest:
```bash
pytest
```
Output:
```text
tests\test_enrichment.py ...............                                 [100%]
======================= 15 passed in 4.93s ========================
```

The tests cover:
- Strict whitelist filtering and case-insensitive normalization
- Image optimization & resizing
- Corrupt and 404 image handling
- Dual-key Groq fallback upon rate limit (HTTP 429)
- Full API endpoint integration with mocked external calls

---

## 📂 Project Structure

```
Image_Analysis_AI/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application & /api/enrich-product endpoint
│   ├── config.py                # Environment configuration & settings
│   ├── constants.py             # Predefined skin tones, body types, occasions & validation
│   ├── schemas.py               # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── image_service.py     # Image downloading, validation, optimization & base64 encoding
│       └── groq_service.py      # Groq Vision service with dual-key fallback & strict filtering
├── tests/
│   ├── __init__.py
│   └── test_enrichment.py       # Comprehensive pytest suite
├── .env.example                 # Example configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```
