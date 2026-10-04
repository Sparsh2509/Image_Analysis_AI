"""Fixed category definitions and validation helpers for FestFit product enrichment."""

from typing import List

# Fixed Skin Tone Categories
VALID_SKIN_TONES: List[str] = [
    "Fair",
    "Light",
    "Wheatish",
    "Dusky",
    "Deep",
]

# Fixed Body Type Categories
VALID_BODY_TYPES: List[str] = [
    "Rectangle",
    "Hourglass",
    "Pear",
    "Apple",
    "Inverted Triangle",
]

# Fixed Occasion Categories
VALID_OCCASIONS: List[str] = [
    "Casual",
    "Formal",
    "Office",
    "College",
    "Party",
    "Wedding",
    "Date",
    "Travel",
    "Sports",
    "Beach",
    "Brunch",
    "Dinner",
    "Interview",
    "Diwali",
    "Holi",
    "Eid",
    "Navratri",
    "Dussehra",
    "Durga Puja",
    "Ganesh Chaturthi",
    "Raksha Bandhan",
    "Janmashtami",
    "Pongal",
    "Onam",
    "Baisakhi",
    "Christmas",
    "Karwa Chauth",
]

# Lookup dictionaries for case-insensitive normalization to canonical casing
_SKIN_TONE_LOOKUP = {item.strip().lower(): item for item in VALID_SKIN_TONES}
_BODY_TYPE_LOOKUP = {item.strip().lower(): item for item in VALID_BODY_TYPES}
_OCCASION_LOOKUP = {item.strip().lower(): item for item in VALID_OCCASIONS}


def normalize_categories(items: List[str], lookup: dict) -> List[str]:
    """Filters and maps a raw list of categories to canonical allowed values.

    Preserves order and discards any values not in the allowed set.
    """
    seen = set()
    normalized = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if cleaned in lookup:
            canonical = lookup[cleaned]
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
    return normalized


def filter_skin_tones(items: List[str]) -> List[str]:
    return normalize_categories(items, _SKIN_TONE_LOOKUP)


def filter_body_types(items: List[str]) -> List[str]:
    return normalize_categories(items, _BODY_TYPE_LOOKUP)


def filter_occasions(items: List[str]) -> List[str]:
    return normalize_categories(items, _OCCASION_LOOKUP)
