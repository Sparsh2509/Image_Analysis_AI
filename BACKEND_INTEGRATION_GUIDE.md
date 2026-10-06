# 📘 FestFit — Backend Integration Guide for AI Product Enrichment

> **API Base URL**: `https://image-analysis-ai.onrender.com`  
> **Interactive Swagger Docs**: `https://image-analysis-ai.onrender.com/docs`

---

## 1. Quick Overview

When you submit a clothing product's `imageUrl`, the AI API downloads the image, analyzes the garment (cut, color palette, style, formality, cultural appearance), and returns **three enriched fields**:

1. `suitable_skin_tones`
2. `suitable_body_types`
3. `suitable_occasions`

Your backend simply stores these three returned arrays in the product document in your database. Existing product attributes are not modified.

---

## 2. API Endpoint

### `POST /api/enrich-product`
- **Method**: `POST`
- **URL**: `https://image-analysis-ai.onrender.com/api/enrich-product`
- **Headers**: `Content-Type: application/json`

---

## 3. Request Format

You can send either **only the image URL** or forward the **entire product object**. Extra fields are safely ignored.

### Option A: Minimal (Recommended)
```json
{
  "imageUrl": "https://assets.myntassets.com/h_1440,q_90,w_1080/v1/assets/images/sample.jpg"
}
```

### Option B: Full Product Payload (Direct Forward)
```json
{
  "brand": "See Designs",
  "category": "Pyjamas",
  "currency": "INR",
  "description": "See Designs Men Mid Rise Pure Cotton Pyjama",
  "imageUrl": "https://assets.myntassets.com/h_1440,q_90,w_1080/v1/assets/images/sample.jpg",
  "name": "See Designs Men Mid Rise Pure Cotton Pyjama",
  "price": 454,
  "productUrl": "https://www.myntra.com/...",
  "source": "MYNTRA"
}
```
*(Both `imageUrl` and `image_url` are accepted).*

---

## 4. Response Format (`200 OK`)

The response is always valid JSON containing strictly the three enriched fields:

```json
{
  "suitable_skin_tones": [
    "Fair",
    "Light",
    "Wheatish"
  ],
  "suitable_body_types": [
    "Rectangle",
    "Inverted Triangle",
    "Apple"
  ],
  "suitable_occasions": [
    "Office",
    "College",
    "Casual",
    "Interview"
  ]
}
```

---

## 5. Fixed Categories Reference (Enum Whitelist)

The AI strictly outputs values from these fixed sets. No external or hallucinated values will ever be returned.

### 1. `suitable_skin_tones` (5 allowed values)
```json
["Fair", "Light", "Wheatish", "Dusky", "Deep"]
```

### 2. `suitable_body_types` (5 allowed values)
```json
["Rectangle", "Hourglass", "Pear", "Apple", "Inverted Triangle"]
```

### 3. `suitable_occasions` (27 allowed values)
- **Work / Social / Casual**:
  `"Casual"`, `"Formal"`, `"Office"`, `"College"`, `"Party"`, `"Wedding"`, `"Date"`, `"Travel"`, `"Sports"`, `"Beach"`, `"Brunch"`, `"Dinner"`, `"Interview"`
- **Indian Festivals & Cultural**:
  `"Diwali"`, `"Holi"`, `"Eid"`, `"Navratri"`, `"Dussehra"`, `"Durga Puja"`, `"Ganesh Chaturthi"`, `"Raksha Bandhan"`, `"Janmashtami"`, `"Pongal"`, `"Onam"`, `"Baisakhi"`, `"Christmas"`, `"Karwa Chauth"`

---

## 6. Integration Examples

### Node.js / Express (Axios)
```javascript
const axios = require('axios');

async function enrichProduct(product) {
  try {
    const response = await axios.post(
      'https://image-analysis-ai.onrender.com/api/enrich-product',
      { imageUrl: product.imageUrl },
      { timeout: 35000 } // Set 35s timeout for vision processing
    );

    const { suitable_skin_tones, suitable_body_types, suitable_occasions } = response.data;

    // Attach to product and save in DB
    product.suitable_skin_tones = suitable_skin_tones;
    product.suitable_body_types = suitable_body_types;
    product.suitable_occasions = suitable_occasions;

    await product.save();
    return product;
  } catch (error) {
    console.error('AI enrichment failed:', error.response?.data || error.message);
    throw error;
  }
}
```

### Python (Requests / HTTPX)
```python
import requests

def enrich_product(product_data: dict) -> dict:
    url = "https://image-analysis-ai.onrender.com/api/enrich-product"
    payload = {"imageUrl": product_data["imageUrl"]}
    
    response = requests.post(url, json=payload, timeout=35)
    response.raise_for_status()
    
    ai_data = response.json()
    product_data["suitable_skin_tones"] = ai_data["suitable_skin_tones"]
    product_data["suitable_body_types"] = ai_data["suitable_body_types"]
    product_data["suitable_occasions"] = ai_data["suitable_occasions"]
    
    return product_data
```

---

## 7. HTTP Error Codes

| Status Code | Description | Recommended Backend Action |
| :--- | :--- | :--- |
| `200 OK` | Enrichment successful | Save fields to DB. |
| `400 Bad Request` | Missing `imageUrl` or URL returned 404/Connection error | Verify image URL exists and is publicly accessible. |
| `422 Unprocessable` | Corrupt or non-image content | Check that the URL points to a standard image (JPEG, PNG, WebP). |
| `502 / 503` | Vision LLM temporarily unavailable | Retry request after a short delay (dual API key failover is handled automatically). |
| `504 Gateway Timeout`| Image download timed out (>12s) | Retry or check image host CDN latency. |
