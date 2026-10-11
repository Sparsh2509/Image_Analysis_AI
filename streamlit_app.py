"""Streamlit UI for FestFit AI Product Enrichment.

A clean and simple interface to input a fashion product image URL,
preview the image, and view the AI-predicted skin tones, body types, and occasions.
"""

import asyncio
import os
import streamlit as st
import httpx
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

from app.services.image_service import fetch_and_prepare_image, ImageProcessingError
from app.services.groq_service import groq_service, GroqServiceError
from app.config import settings

# Page Configuration
st.set_page_config(
    page_title="FestFit — AI Product Enrichment",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom minimal CSS for clean styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 6px;
        margin-bottom: 16px;
    }
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .tag-skin {
        background-color: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFE0B2;
    }
    .tag-body {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #C8E6C9;
    }
    .tag-occasion {
        background-color: #E3F2FD;
        color: #1565C0;
        border: 1px solid #BBDEFB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">👗 FestFit AI Product Enrichment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analyze fashion clothing images to determine suitable <b>skin tones</b>, <b>body types</b>, and <b>occasions</b>.</div>',
    unsafe_allow_html=True,
)

# Input Field
image_url = st.text_input(
    "Product Image URL:",
    value="",
    placeholder="https://example.com/clothing-image.jpg",
    help="Enter a direct publicly accessible image URL (JPG, PNG, WebP).",
)

# Image Preview
if image_url and image_url.strip():
    clean_url = image_url.strip()
    st.markdown("#### 🖼️ Image Preview")
    try:
        st.image(clean_url, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render image preview directly in browser: {e}. You can still click Analyze below.")

# Analyze Action
async def run_analysis(url: str):
    # Step 1: Fetch and prepare image (validation, resize, base64)
    image_data_uri = await fetch_and_prepare_image(url)
    # Step 2: Groq Vision analysis
    result = await groq_service.analyze_product(image_data_uri)
    return result

if st.button("✨ Analyze Clothing Product", type="primary", use_container_width=True):
    if not image_url or not image_url.strip():
        st.error("Please enter a valid image URL first.")
    else:
        with st.spinner("Analyzing garment cut, color, and occasions with Groq Vision..."):
            try:
                # Run the async analysis
                result = asyncio.run(run_analysis(image_url.strip()))

                st.success("Analysis Complete!")
                st.divider()

                st.markdown("### 🎯 Predicted Fashion Attributes")

                # Field 1: Suitable Skin Tones
                st.markdown("##### 1. Suitable Skin Tones")
                skin_tags_html = "".join([f'<span class="tag tag-skin">{tone}</span>' for tone in result.suitable_skin_tones])
                st.markdown(f'<div class="tag-container">{skin_tags_html}</div>', unsafe_allow_html=True)

                # Field 2: Suitable Body Types
                st.markdown("##### 2. Suitable Body Types")
                body_tags_html = "".join([f'<span class="tag tag-body">{body}</span>' for body in result.suitable_body_types])
                st.markdown(f'<div class="tag-container">{body_tags_html}</div>', unsafe_allow_html=True)

                # Field 3: Suitable Occasions
                st.markdown("##### 3. Suitable Occasions")
                occ_tags_html = "".join([f'<span class="tag tag-occasion">{occ}</span>' for occ in result.suitable_occasions])
                st.markdown(f'<div class="tag-container">{occ_tags_html}</div>', unsafe_allow_html=True)

                # Raw JSON Output
                with st.expander("📋 View Raw JSON (for Backend Teammate)"):
                    st.json(result.model_dump())

            except ImageProcessingError as exc:
                st.error(f"Image Processing Error: {exc.message}")
            except GroqServiceError as exc:
                st.error(f"AI Service Error: {exc.message}")
            except Exception as exc:
                st.error(f"Unexpected Error: {str(exc)}")

st.markdown("---")
st.caption("FestFit AI Product Enrichment • Powered by Groq Vision")
