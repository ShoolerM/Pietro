"""Model context window database and detection utilities."""

import re
from typing import Tuple, Optional


# Comprehensive model context window database
MODEL_CONTEXT_DATABASE = {
    "open_source_llms": {
        "gpt_oss": {"gpt-oss-20b": 131072, "gpt-oss-120b": 131072},
        "llama": {
            "llama2-7b": 4096,
            "llama2-13b": 4096,
            "llama3-8b": 128000,
            "llama3-70b": 128000,
            "llama4-scout": 10000000,
            "llama4-maverick": 1000000,
        },
        "gemma3": {
            "gemma3-1b": 32768,
            "gemma3-4b": 128000,
            "gemma3-12b": 128000,
            "gemma3-27b": 128000,
        },
        "qwen": {
            "qwen2.5-7b": 128000,
            "qwen2.5-14b": 128000,
            "qwen2.5-1m": 1000000,
            "qwen3-next-80b": 262144,
            "qwen3-omni-vl": 262144,
        },
        "mistral": {
            "mistral-7b-v0.2": 32768,
            "mixtral-8x7b": 32768,
            "codestral-22b": 32768,
        },
        "devstral": {"devstral-24b": 128000, "devstral2-123b": 256000},
        "deepseek": {"deepseek-v3": 128000, "deepseek-coder-base": 16384},
        "yi_models": {"yi-6b-200k": 200000, "yi-34b-200k": 200000},
        "falcon": {"falcon-7b": 2048, "falcon-40b": 2048},
        "stablelm": {"stablelm-base-alpha-7b": 4096},
        "tiny_models": {"tinyllama-1.1b": 4096},
    },
    "multimodal_and_vision_llms": {
        "llava": {
            "llava-7b-32k": 32768,
            "llava-13b-32k": 32768,
            "llava-34b-32k": 32768,
        },
        "qwen_vl": {
            "qwen2.5-vl-3b": 128000,
            "qwen2.5-vl-7b": 128000,
            "qwen3-vl": 262144,
        },
        "glm_vision": {"glm-4.6v": 128000},
        "phi_vision": {"phi-3.5-vision": 128000},
    },
    "specialized_code_models": {
        "codestral": {"codestral-22b": 32768},
        "devstral": {"devstral-coding-24b": 128000, "devstral2-coding-123b": 256000},
    },
    "proprietary_and_cloud_models": {
        "openai_cloud": {"gpt-4-turbo": 128000, "gpt-4o": 128000, "gpt-5.2": 400000},
        "anthropic_claude": {"claude-3.5-sonnet": 200000, "claude-opus-4.6": 1000000},
        "google_gemini": {"gemini-1.5-pro": 2000000},
    },
}


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for matching.

    Removes common suffixes like -chat, -instruct, -int4, -int8, etc.
    Converts to lowercase and removes extra spaces/slashes.

    Args:
        model_name: Raw model name from user

    Returns:
        Normalized model name
    """
    if not model_name:
        return ""

    # Convert to lowercase
    normalized = model_name.lower().strip()

    # Remove common path prefixes (e.g., "openai/", "qwen/")
    if "/" in normalized:
        normalized = normalized.split("/")[-1]

    # Remove common suffixes that don't affect context window
    suffixes_to_remove = [
        "-chat",
        "-instruct",
        "-base",
        "-int4",
        "-int8",
        "-gptq",
        "-awq",
        "-gguf",
        "-ggml",
        "-q4",
        "-q8",
        "-uncensored",
        "-unfiltered",
    ]

    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    return normalized


def extract_context_from_name(model_name: str) -> Optional[int]:
    """
    Try to extract context window size from model name patterns.

    Looks for patterns like: 32k, 200k, 1m, etc.

    Args:
        model_name: Model name to analyze

    Returns:
        Context window size if found, None otherwise
    """
    # Pattern for context hints in model name (e.g., "32k", "200k", "1m")
    patterns = [
        (r"(\d+)m\b", 1000000),  # e.g., "1m" -> 1,000,000
        (r"(\d+)k\b", 1000),  # e.g., "32k" -> 32,000
    ]

    model_lower = model_name.lower()

    for pattern, multiplier in patterns:
        match = re.search(pattern, model_lower)
        if match:
            value = int(match.group(1))
            return value * multiplier

    return None


def detect_context_window(model_name: str) -> Tuple[int, str, str]:
    """
    Detect context window size for a given model name.

    Uses multiple strategies:
    1. Exact match in database (100% confidence)
    2. Fuzzy/substring match in database (80% confidence)
    3. Pattern extraction from name (50% confidence)
    4. Safe default fallback (0% confidence)

    Args:
        model_name: The model name to detect context for

    Returns:
        Tuple of (context_length, confidence_level, detection_source)
        - context_length: Detected context window size in tokens
        - confidence_level: "high", "medium", "low", or "default"
        - detection_source: Description of how it was detected
    """
    if not model_name:
        return 4096, "default", "No model specified - using safe default"

    normalized = normalize_model_name(model_name)

    # Build flat lookup dictionary from nested structure
    flat_db = {}
    for category, subcategories in MODEL_CONTEXT_DATABASE.items():
        for subcategory, models in subcategories.items():
            flat_db.update(models)

    # Strategy 1: Exact match (highest confidence)
    if normalized in flat_db:
        context = flat_db[normalized]
        return context, "high", f"Exact match: {normalized}"

    # Strategy 2: Fuzzy match - check if normalized name is substring of any database key
    best_match = None
    best_match_key = None
    best_match_score = 0

    for db_key, context in flat_db.items():
        # Check if either is substring of the other
        if normalized in db_key or db_key in normalized:
            # Calculate match score based on length similarity
            score = min(len(normalized), len(db_key)) / max(
                len(normalized), len(db_key)
            )
            if score > best_match_score:
                best_match = context
                best_match_key = db_key
                best_match_score = score

    if best_match and best_match_score > 0.5:
        return best_match, "medium", f"Fuzzy match: {normalized} → {best_match_key}"

    # Strategy 3: Extract from name pattern (e.g., "32k", "200k")
    extracted_context = extract_context_from_name(model_name)
    if extracted_context:
        return extracted_context, "low", f"Extracted from name pattern: {model_name}"

    # Strategy 4: Safe default fallback
    return 4096, "default", "No match found - using safe default of 4096"


def get_all_known_models() -> dict:
    """
    Get all known models from the database.

    Returns:
        Flattened dictionary of all models and their context windows
    """
    flat_db = {}
    for category, subcategories in MODEL_CONTEXT_DATABASE.items():
        for subcategory, models in subcategories.items():
            flat_db.update(models)
    return flat_db


def is_vision_model(model_name: str) -> bool:
    """Return True if the model is known to support image inputs."""
    if not model_name:
        return False

    normalized = normalize_model_name(model_name)

    vision_models = set()
    for subcategory, models in MODEL_CONTEXT_DATABASE.get(
        "multimodal_and_vision_llms", {}
    ).items():
        vision_models.update(models.keys())

    vision_models.update(
        {
            "gpt-4o",
            "gpt-4-turbo",
            "gemini-1.5-pro",
            "claude-3.5-sonnet",
            "claude-opus-4.6",
        }
    )

    if normalized in vision_models:
        return True

    # Heuristic for common vision naming
    if "vision" in normalized or "-vl" in normalized or "omni-vl" in normalized:
        return True

    return False
