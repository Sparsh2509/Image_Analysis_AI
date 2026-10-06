"""Groq Vision service with dual API-key fallback, exponential backoff, and strict output validation."""

import json
import logging
import re
import asyncio
from typing import Optional, Dict, Any, List
import groq
from groq import AsyncGroq, RateLimitError, APIStatusError, APIConnectionError

from app.config import settings
from app.constants import (
    VALID_SKIN_TONES,
    VALID_BODY_TYPES,
    VALID_OCCASIONS,
    filter_skin_tones,
    filter_body_types,
    filter_occasions,
)
from app.schemas import ProductEnrichmentResponse

logger = logging.getLogger(__name__)


class GroqServiceError(Exception):
    """Custom exception raised when Groq vision analysis fails."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


SYSTEM_PROMPT = f"""You are a professional high-fashion stylist and AI enrichment expert for the FestFit fashion platform.
Your task is to analyze the clothing product shown in the provided product image and recommend suitability across three dimensions:

1. suitable_skin_tones
2. suitable_body_types
3. suitable_occasions

STRICT CATEGORY CONSTRAINTS:
You MUST select values ONLY from these predefined lists. NEVER invent or hallucinate new categories.

ALLOWED SKIN TONES:
{json.dumps(VALID_SKIN_TONES)}

ALLOWED BODY TYPES:
{json.dumps(VALID_BODY_TYPES)}

ALLOWED OCCASIONS:
{json.dumps(VALID_OCCASIONS)}

SELECTIVITY RULES BY FIELD:
1. suitable_skin_tones: Select top 2 to 3 most flattering skin tones based on color harmony, undertones, and contrast. Do NOT return all 5 unless truly universal.
2. suitable_body_types: Select top 2 to 3 body types whose silhouette is enhanced by the garment's cut and proportions. Do NOT return all 5.
3. suitable_occasions: Be culturally accurate and comprehensive! Assign all appropriate occasions from the allowed list (typically 4 to 8 for versatile garments).

INDIAN ETHNIC & WEDDING/KURTA RULES (CRITICAL):
- Carefully detect Indian ethnic garments: Kurtas (mandarin/nehru collar, long tunic length to thighs/knees, side slits, front plackets), Kurti, Saree, Lehenga, Sherwani, Nehru jacket, Anarkali, ethnic sets.
- OCCASION ACCURACY:
  * HEAVY / BRIDAL / GROOM WEAR (e.g. Royal Sherwanis with turbans/shawls, bridal Lehengas, heavy Silk Sarees):
    - Must include: "Wedding", "Party" (sangeet/reception).
    - Can include major festivals: "Diwali", "Karwa Chauth", "Eid".
    - NEVER assign "Dinner", "Brunch", "Casual", "College", or "Office" to heavy bridal/groom wedding wear like a Sherwani! Nobody wears a groom sherwani to a casual dinner.
  * VERSATILE / SEMI-FORMAL ETHNIC WEAR (Festive Kurtas, printed kurtas, Nehru jacket sets):
    - Include: "Diwali", "Eid", "Navratri", "Durga Puja", "Ganesh Chaturthi", "Raksha Bandhan", "Party", "Wedding".
  * DAILY COTTON KURTAS / CASUAL ETHNIC:
    - Include Indian festivals ("Diwali", "Raksha Bandhan", "Ganesh Chaturthi", "Durga Puja", "Eid") AND daily daytime wear ("Casual", "College", "Office").

OCCASION GUIDE FOR "Dinner" & "Brunch":
- "Dinner": Assign ONLY to evening date night outfits, chic cocktail/party dresses, smart casual dinner shirts, or evening blazers. NEVER assign "Dinner" to heavy traditional wedding attire or sleepwear.
- "Brunch": Assign ONLY to relaxed daytime casuals, sundresses, linen shirts, or light daytime loungewear.

SKIN TONE STYLING GUIDE (SELECT TOP 2 TO 3 MOST COMPLEMENTARY):
- "Fair" and "Light":
  * Flattered by soft pastels (blush pink, powder blue, peach, mint, lilac, lavender).
  * Flattered by light neutrals and regal metallics: Ivory, cream, beige, champagne, off-white, silver, antique gold (e.g. an ivory/cream groom sherwani or pastel kurta looks stunning on Fair and Light skin).
  * High-contrast bold hues: Deep maroon, navy blue, ruby red, and emerald green provide a striking, gorgeous contrast on Fair and Light complexions.
- "Wheatish" and "Dusky":
  * Flattered by warm earth tones: Mustard, rust, terracotta, olive green, warm ochre, marigold.
  * Flattered by rich jewel tones: Deep maroon, wine, teal, royal blue, peacock green, warm gold.
  * Flattered by warm neutrals: Warm beige, tan, cream.
- "Deep":
  * Flattered by vibrant high-contrast colors: Crisp white, bright yellow, cobalt blue, fuchsia, orange.
  * Flattered by rich metallic gold, bronze, emerald, and deep jewel tones.

DETAILED BODY TYPE CRITERIA (Select 2 to 3 best matches based on silhouette):
- Hourglass: Fitted waists, wrap silhouettes, bodycon, tailored kurtas/dresses with defined waist.
- Pear: A-line skirts/kurtas, boat necks, statement collars, structured shoulders, flared bottoms.
- Apple: Empire waists, vertical paneling, fluid straight kurtas, V-necks, flowy silhouettes.
- Rectangle: Belts, peplum, defined waistbands, flared anarkalis, straight sherwanis with structured shoulders.
- Inverted Triangle: Wide-leg bottoms, flared silhouettes, scoop or deep V-necks, structured jackets/sherwanis balancing broad shoulders.

REMINDER:
- These fields represent which users and occasions the CLOTHING PRODUCT is visually flattering and suitable for.
- Do NOT describe the model or mannequin in the picture.

OUTPUT FORMAT:
Return ONLY a valid JSON object with no additional text or explanation:
{{
  "suitable_skin_tones": ["Fair", "Light", "Wheatish"],
  "suitable_body_types": ["Rectangle", "Inverted Triangle"],
  "suitable_occasions": ["Wedding", "Party"]
}}
"""


class GroqVisionService:
    def __init__(self):
        self.model = settings.GROQ_VISION_MODEL

    def _get_api_keys(self) -> List[str]:
        keys = []
        if settings.GROQ_API_KEY_1 and settings.GROQ_API_KEY_1.strip():
            keys.append(settings.GROQ_API_KEY_1.strip())
        if settings.GROQ_API_KEY_2 and settings.GROQ_API_KEY_2.strip():
            k2 = settings.GROQ_API_KEY_2.strip()
            if k2 not in keys:
                keys.append(k2)
        if not keys and settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
            keys.append(settings.GROQ_API_KEY.strip())

        if not keys:
            logger.warning("No Groq API keys configured in environment (GROQ_API_KEY_1, GROQ_API_KEY_2).")
        return keys

    async def analyze_product(
        self,
        image_data_uri: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ProductEnrichmentResponse:
        """Analyzes an image using Groq Vision with dual-key fallback and exponential backoff."""
        api_keys = self._get_api_keys()
        if not api_keys:
            raise GroqServiceError(
                "Groq API keys are not configured. Please set GROQ_API_KEY_1 or GROQ_API_KEY_2 in .env.",
                status_code=500
            )

        # Build user prompt with optional product metadata context
        user_prompt_text = (
            "Analyze this clothing product image and return the JSON with suitable_skin_tones, suitable_body_types, and suitable_occasions. "
            "Pay close attention to whether the garment is an Indian ethnic item (such as a Kurta, Kurti, Saree, Lehenga, Sherwani, or Nehru Jacket), "
            "and ensure you include appropriate Indian festival and celebratory occasions (e.g., Diwali, Eid, Navratri, Durga Puja, Ganesh Chaturthi, Raksha Bandhan, Wedding, Party) as instructed."
        )
        if context:
            context_bits = []
            if context.get("name"):
                context_bits.append(f"Name: {context['name']}")
            if context.get("category"):
                context_bits.append(f"Category: {context['category']}")
            if context.get("brand"):
                context_bits.append(f"Brand: {context['brand']}")
            if context.get("description"):
                context_bits.append(f"Description: {context['description']}")
            if context_bits:
                user_prompt_text += f"\n\nProduct Context:\n" + "\n".join(context_bits)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    }
                ]
            }
        ]

        last_error = None

        # Iterate through primary key (Key 1) and backup key (Key 2)
        for key_index, api_key in enumerate(api_keys, start=1):
            client = AsyncGroq(
                api_key=api_key,
                timeout=settings.GROQ_REQUEST_TIMEOUT_SECONDS
            )

            max_retries = settings.MAX_RETRIES_PER_KEY
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        f"Sending vision request to Groq using Key #{key_index} (attempt {attempt}/{max_retries}, model={self.model})..."
                    )
                    completion = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=500,
                        response_format={"type": "json_object"}
                    )

                    response_content = completion.choices[0].message.content
                    if not response_content:
                        raise GroqServiceError("Received empty response from vision model.", status_code=502)

                    logger.info(f"Successfully received analysis from Groq using Key #{key_index}")
                    return self._parse_and_validate_response(response_content)

                except RateLimitError as exc:
                    logger.warning(
                        f"Rate limit exceeded on Groq Key #{key_index}: {exc}. Triggering fallback."
                    )
                    last_error = exc
                    # Break out of inner loop to switch immediately to the next API key
                    break

                except APIStatusError as exc:
                    logger.warning(
                        f"Groq API error on Key #{key_index} (HTTP {exc.status_code}): {exc}"
                    )
                    last_error = exc
                    # If quota/auth or server 5xx error, switch to next key immediately
                    if exc.status_code in (401, 403, 429, 500, 502, 503, 504):
                        break
                    # For other status codes, back off and retry
                    if attempt < max_retries:
                        backoff = settings.RETRY_BACKOFF_FACTOR ** attempt
                        await asyncio.sleep(backoff)

                except (APIConnectionError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        f"Connection or timeout error with Groq on Key #{key_index}: {exc}"
                    )
                    last_error = exc
                    if attempt < max_retries:
                        backoff = settings.RETRY_BACKOFF_FACTOR ** attempt
                        await asyncio.sleep(backoff)

                except Exception as exc:
                    logger.error(f"Unexpected error calling Groq on Key #{key_index}: {exc}")
                    last_error = exc
                    break

        # If we exhausted all keys
        error_msg = f"All Groq API keys failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise GroqServiceError(error_msg, status_code=503)

    def _parse_and_validate_response(self, raw_content: str) -> ProductEnrichmentResponse:
        """Parses model response JSON and strictly filters values against fixed category lists."""
        logger.debug(f"Raw Groq response: {raw_content}")

        # Extract JSON block if wrapped in markdown code fence
        cleaned_json = raw_content.strip()
        if "```" in cleaned_json:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_json, re.DOTALL)
            if match:
                cleaned_json = match.group(1)

        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to decode JSON from Groq output: {cleaned_json}. Error: {exc}")
            raise GroqServiceError("Vision model returned an invalid JSON response format.", status_code=502)

        # Extract raw arrays
        raw_skin = parsed.get("suitable_skin_tones", [])
        raw_body = parsed.get("suitable_body_types", [])
        raw_occasions = parsed.get("suitable_occasions", [])

        if not isinstance(raw_skin, list):
            raw_skin = [str(raw_skin)] if raw_skin else []
        if not isinstance(raw_body, list):
            raw_body = [str(raw_body)] if raw_body else []
        if not isinstance(raw_occasions, list):
            raw_occasions = [str(raw_occasions)] if raw_occasions else []

        # Strictly filter against fixed whitelist
        valid_skin = filter_skin_tones(raw_skin)
        valid_body = filter_body_types(raw_body)
        valid_occasions = filter_occasions(raw_occasions)

        # Fallbacks for extreme edge cases where model filtered down to empty
        if not valid_skin:
            valid_skin = ["Wheatish", "Light"]
        if not valid_body:
            valid_body = ["Rectangle", "Hourglass"]
        if not valid_occasions:
            valid_occasions = ["Casual"]

        return ProductEnrichmentResponse(
            suitable_skin_tones=valid_skin,
            suitable_body_types=valid_body,
            suitable_occasions=valid_occasions,
        )


groq_service = GroqVisionService()
