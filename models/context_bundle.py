"""Context bundle and shared constants for context assembly.

This module provides:
- ``ContextBundle``: typed dataclass returned by every ``assemble_for_*``
  method on ``ContextManager``.
- Named constants that replace all magic numbers in token-budget arithmetic,
  giving every call site a single authoritative source of truth.
"""

from dataclasses import dataclass

# ── Output / overhead reserves ────────────────────────────────────────────────

# Tokens reserved for the model's output on every call.
OUTPUT_RESERVE_TOKENS: int = 2000

# Conservative estimate for static system-prompt text in planning-conversation
# mode (used before the actual system prompt length is known).
SYSTEM_PROMPT_ESTIMATE_TOKENS: int = 800

# Fixed framing text in planning-chunk queries: section labels, separators,
# backtick fences, upcoming-tasks boilerplate, and the closing instruction.
QUERY_OVERHEAD_TOKENS: int = 500

# Conservative system-prompt estimate used in inline edit-mode budget math.
EDIT_SYSTEM_ESTIMATE_TOKENS: int = 500

# ── RAG allocation ratios ─────────────────────────────────────────────────────
# Fraction of available (post-fixed-cost) tokens to allocate to RAG per mode.

RAG_RATIO_PLANNING: float = 0.20  # Planning-conversation mode
RAG_RATIO_GENERATION: float = 0.30  # Normal write / generation mode
RAG_RATIO_EDIT: float = 0.25  # Inline text-edit (Update Selected Text)
RAG_RATIO_AUTO_BUILD: float = 0.25  # Auto-build (Story Mode) chunks
RAG_RATIO_CHUNK: float = 0.25  # Outline-driven planning chunks

# ── RAG hard limits (min / max tokens per call) ───────────────────────────────

MIN_RAG_TOKENS_PLANNING: int = 500
MAX_RAG_TOKENS_PLANNING: int = 2000

MIN_RAG_TOKENS_GENERATION: int = 500
MAX_RAG_TOKENS_GENERATION: int = 4000

MIN_RAG_TOKENS_EDIT: int = 300
MAX_RAG_TOKENS_EDIT: int = 1500

MIN_RAG_TOKENS_AUTO_BUILD: int = 500
MAX_RAG_TOKENS_AUTO_BUILD: int = 3000

# No minimum for chunk mode — skip RAG entirely when there is no headroom.
MAX_RAG_TOKENS_CHUNK: int = 3000

# ── Story-context extraction settings ────────────────────────────────────────

# Maximum tokens of recent story text passed to ``get_context_for_llm`` in
# planning-conversation and auto-build modes.
STORY_MAX_RECENT_TOKENS_PLANNING: int = 2000

# ── Inline-edit context windows ──────────────────────────────────────────────

# Characters of story text extracted before / after the selected region.
EDIT_CONTEXT_CHARS_BEFORE: int = 2000
EDIT_CONTEXT_CHARS_AFTER: int = 2000

# Portions of the surrounding context fed into the RAG query string.
EDIT_RAG_QUERY_BEFORE_CHARS: int = 200
EDIT_RAG_QUERY_SELECTION_CHARS: int = 500
EDIT_RAG_QUERY_AFTER_CHARS: int = 200

# ── Auto-build RAG query settings ────────────────────────────────────────────

# Characters of recent story text appended to the initial prompt when
# constructing the RAG query for auto-build (Story Mode) chunks.
AUTO_BUILD_RAG_RECENT_CHARS: int = 500

# ── Chunk-mode enriched RAG query excerpt lengths ────────────────────────────
# Characters from notes, outline, and supplemental text prepended to the
# current plot-point task when building the RAG query for outline-driven chunk
# generation.  This ensures that character names defined only in notes/outline
# are visible to the filename-boost tokenizer.

RAG_CHUNK_OUTLINE_EXCERPT_CHARS: int = 800
RAG_CHUNK_NOTES_EXCERPT_CHARS: int = 400
RAG_CHUNK_SUPP_EXCERPT_CHARS: int = 400


@dataclass
class ContextBundle:
    """Typed container for all context components assembled for an LLM call.

    All ``assemble_for_*`` methods on ``ContextManager`` return one of these.
    The caller uses ``bundle.query`` as the ready-to-send LLM prompt and may
    inspect the component fields (``rag_context``, ``notes``, etc.) for
    logging or further processing.

    Fields
    ------
    story_context : str
        Story text included in the prompt (raw or post-summarisation).
    rag_context : str
        Retrieved RAG chunks joined by ``\\n\\n---\\n\\n``.
    notes : str
        Author's notes included in the prompt.
    system_prompt : str
        System prompt text for the LLM call.
    supp_text : str
        Supplemental prompt text.
    query : str
        Fully assembled LLM query string.  Empty for ``assemble_for_chunk``
        where the planning controller builds the final query itself.
    total_tokens : int
        Estimated total tokens across all included context components.
    max_rag_tokens : int
        Token budget allocated for the RAG query (0 = RAG was skipped).
    rag_tokens : int
        Actual tokens retrieved from RAG (0 = nothing returned).
    """

    # Context components
    story_context: str = ""
    rag_context: str = ""
    notes: str = ""
    system_prompt: str = ""
    supp_text: str = ""

    # Assembled prompt
    query: str = ""

    # Budget / diagnostic info
    total_tokens: int = 0
    max_rag_tokens: int = 0
    rag_tokens: int = 0
