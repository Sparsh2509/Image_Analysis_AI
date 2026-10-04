"""Pydantic schemas for request and response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ProductEnrichmentRequest(BaseModel):
    """Product enrichment request.

    Accepts either just imageUrl or a full product object containing imageUrl.
    Supports both camelCase `imageUrl` and snake_case `image_url`.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    imageUrl: str = Field(
        ...,
        alias="image_url",
        description="Publicly accessible HTTP/HTTPS URL of the product image.",
        examples=["https://assets.myntassets.com/h_1440,q_90,w_1080/v1/assets/images/product.jpg"]
    )
    # Optional context fields that may be passed along with the product
    name: Optional[str] = Field(default=None, description="Optional product name")
    category: Optional[str] = Field(default=None, description="Optional product category")
    description: Optional[str] = Field(default=None, description="Optional product description")
    brand: Optional[str] = Field(default=None, description="Optional product brand")

    @field_validator("imageUrl", mode="before")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("imageUrl must be a non-empty string")
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("imageUrl must be a valid HTTP or HTTPS URL")
        return v


class ProductEnrichmentResponse(BaseModel):
    """Enriched fashion product attributes."""
    model_config = ConfigDict(extra="forbid")

    suitable_skin_tones: List[str] = Field(
        ...,
        description="Skin tones suitable for this clothing product.",
        examples=[["Wheatish", "Dusky"]]
    )
    suitable_body_types: List[str] = Field(
        ...,
        description="Body types suitable for this clothing product.",
        examples=[["Rectangle", "Pear"]]
    )
    suitable_occasions: List[str] = Field(
        ...,
        description="Occasions suitable for this clothing product.",
        examples=[["Casual", "College", "Travel"]]
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vision_model: str
    available_api_keys: int
