"""Context Manager

Centralises all context gathering and assembly for every LLM call site.

Provides typed ``assemble_for_*`` methods that encapsulate:
- Dynamic token-budget calculation using shared named constants
- RAG database queries with a correctly sized budget
- Story-context extraction via the summary model
- Final query-string assembly

All methods return a ``ContextBundle`` — a typed dataclass whose ``query``
field is the ready-to-send LLM prompt, with component fields available for
logging and diagnostics.
"""

from typing import Optional

from models.context_bundle import (
    ContextBundle,
    OUTPUT_RESERVE_TOKENS,
    SYSTEM_PROMPT_ESTIMATE_TOKENS,
    QUERY_OVERHEAD_TOKENS,
    EDIT_SYSTEM_ESTIMATE_TOKENS,
    RAG_RATIO_PLANNING,
    RAG_RATIO_GENERATION,
    RAG_RATIO_EDIT,
    RAG_RATIO_AUTO_BUILD,
    RAG_RATIO_CHUNK,
    MIN_RAG_TOKENS_PLANNING,
    MAX_RAG_TOKENS_PLANNING,
    MIN_RAG_TOKENS_GENERATION,
    MAX_RAG_TOKENS_GENERATION,
    MIN_RAG_TOKENS_EDIT,
    MAX_RAG_TOKENS_EDIT,
    MIN_RAG_TOKENS_AUTO_BUILD,
    MAX_RAG_TOKENS_AUTO_BUILD,
    MAX_RAG_TOKENS_CHUNK,
    STORY_MAX_RECENT_TOKENS_PLANNING,
    EDIT_CONTEXT_CHARS_BEFORE,
    EDIT_CONTEXT_CHARS_AFTER,
    EDIT_RAG_QUERY_BEFORE_CHARS,
    EDIT_RAG_QUERY_SELECTION_CHARS,
    EDIT_RAG_QUERY_AFTER_CHARS,
    AUTO_BUILD_RAG_RECENT_CHARS,
    RAG_CHUNK_OUTLINE_EXCERPT_CHARS,
    RAG_CHUNK_NOTES_EXCERPT_CHARS,
    RAG_CHUNK_SUPP_EXCERPT_CHARS,
)


class ContextManager:
    """Centralised context assembly for all LLM call sites.

    One instance is shared across MainController and PlanningController.

    Responsibilities
    ----------------
    - Compute token budgets using shared constants (no magic numbers).
    - Query RAG databases with a correctly sized budget.
    - Extract story context via the summary model (rolling summary + recent tail).
    - Assemble fully formed query strings for each operation mode.
    """

    def __init__(
        self,
        story_model,
        settings_model,
        summary_model,
        rag_controller,
        rag_model,
    ):
        """Initialise the context manager.

        Args:
            story_model: Story data model (content, notes, token helpers).
            settings_model: App settings (context_limit, etc.).
            summary_model: Rolling-summary model for story compression.
            rag_controller: RAG controller used to query knowledge bases.
            rag_model: RAG configuration (summary_chunk_size, etc.).
        """
        self.story_model = story_model
        self.settings_model = settings_model
        self.summary_model = summary_model
        self.rag_controller = rag_controller
        self.rag_model = rag_model

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_rag_budget(
        self,
        available_tokens: int,
        ratio: float,
        min_tokens: int,
        max_tokens: int,
    ) -> int:
        """Calculate the RAG token budget as a clamped fraction of available space.

        Args:
            available_tokens: Tokens remaining after all fixed context costs.
            ratio: Fraction of available_tokens to allocate to RAG.
            min_tokens: Hard minimum (applied even when context is tight).
            max_tokens: Hard upper cap.

        Returns:
            Token budget clamped to [min_tokens, max_tokens].
            Returns min_tokens when available_tokens <= 0.
        """
        if available_tokens <= 0:
            return min_tokens
        budget: int = int(available_tokens * ratio)
        return max(min_tokens, min(budget, max_tokens))

    def _extract_story_context_for_planning(self) -> tuple[str, int]:
        """Extract story context for planning and auto-build modes.

        Uses ``summary_model.get_context_for_llm`` to prepend a rolling
        summary when one exists, keeping the text within
        ``STORY_MAX_RECENT_TOKENS_PLANNING`` raw tokens.

        Returns:
            Tuple of (story_context_text, story_context_token_count).
        """
        existing_story: str = self.story_model.content
        if not existing_story:
            return "", 0

        # Extract the recent tail of the story up to the token limit.
        raw_recent, _ = self.story_model.extract_recent_content(
            existing_story, STORY_MAX_RECENT_TOKENS_PLANNING
        )
        raw_tokens: int = self.story_model.estimate_token_count(raw_recent)

        # Prepend any cached rolling summary.
        story_context, story_tokens = self.summary_model.get_context_for_llm(raw_recent, raw_tokens)
        return story_context, story_tokens

    # ── Public assembly methods ───────────────────────────────────────────────

    def assemble_for_generation(
        self,
        user_input: str,
        story_for_llm: str,
        notes: str,
        supp_text: str,
        system_prompt: str,
        planning_outline: Optional[str] = None,
        attachments_text: str = "",
    ) -> ContextBundle:
        """Assemble context for normal write / generation mode.

        The caller is responsible for ensuring ``story_for_llm`` is already
        in its final form — either raw story text that fits in the context
        window, or the output of
        ``llm_controller.process_story_with_summarization``.

        Args:
            user_input: The user's current prompt.
            story_for_llm: Story text to include (possibly summarised).
            notes: Author's notes.
            supp_text: Supplemental prompt text.
            system_prompt: System prompt text.
            planning_outline: Active planning outline markdown, if any.
            attachments_text: File attachments rendered as plain text.

        Returns:
            ``ContextBundle`` with ``query`` fully assembled.
        """
        context_limit: int = self.settings_model.context_limit

        # Estimate fixed costs to determine how much headroom is left for RAG.
        user_tokens: int = self.story_model.estimate_token_count(user_input)
        supp_tokens: int = self.story_model.estimate_token_count(supp_text)
        notes_tokens: int = self.story_model.estimate_token_count(notes)
        system_tokens: int = self.story_model.estimate_token_count(system_prompt)
        fixed_costs: int = (
            user_tokens + supp_tokens + notes_tokens + system_tokens + OUTPUT_RESERVE_TOKENS
        )
        available: int = context_limit - fixed_costs

        # Allocate 30% of available tokens to RAG.
        max_rag_tokens: int = self._compute_rag_budget(
            available,
            RAG_RATIO_GENERATION,
            MIN_RAG_TOKENS_GENERATION,
            MAX_RAG_TOKENS_GENERATION,
        )

        # Query RAG databases with the calculated budget.
        rag_context: str = self.rag_controller.query_databases(
            user_input, max_tokens=max_rag_tokens
        )
        rag_tokens: int = self.story_model.estimate_token_count(rag_context) if rag_context else 0

        # Build the final query string.
        if story_for_llm:
            final_query: str = (
                f"Based on this story so far:\n```\n{story_for_llm}\n```\n"
                f"the following should happen next (user input):\n{user_input}"
            )
        else:
            final_query = user_input

        # Prepend the planning outline when one is active.
        if planning_outline:
            final_query = (
                f"STORY OUTLINE (must address all tasks):\n{planning_outline}\n\n" + final_query
            )

        if rag_context:
            final_query += f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}"
        if supp_text:
            final_query += f"\n\n{supp_text}"
        if notes:
            final_query += f"\n\nAUTHOR'S NOTES (for context):\n{notes}"
        if attachments_text:
            final_query += f"\n\n{attachments_text}"

        total_tokens: int = (
            self.story_model.estimate_token_count(story_for_llm) + fixed_costs + rag_tokens
        )

        return ContextBundle(
            story_context=story_for_llm,
            rag_context=rag_context,
            notes=notes,
            system_prompt=system_prompt,
            supp_text=supp_text,
            query=final_query,
            total_tokens=total_tokens,
            max_rag_tokens=max_rag_tokens,
            rag_tokens=rag_tokens,
        )

    def assemble_for_planning(
        self,
        user_text: str,
        current_outline: Optional[str],
        attachments_text: str,
    ) -> ContextBundle:
        """Assemble context for a planning-conversation message.

        Reads story content and notes directly from the story model, which is
        kept in sync by ``MainController._on_send`` before any planning message
        is dispatched.

        Args:
            user_text: The user's current planning message.
            current_outline: The outline text currently displayed, if any.
            attachments_text: File attachments rendered as plain text.

        Returns:
            ``ContextBundle`` where ``query`` is the dynamic system-context
            string appended to the BAML planning prompt template.
        """
        # Notes and story are read from the model (synced before this is called).
        notes: str = self.story_model.notes
        story_context, story_tokens = self._extract_story_context_for_planning()

        context_limit: int = self.settings_model.context_limit
        notes_tokens: int = self.story_model.estimate_token_count(notes)
        outline_tokens: int = (
            self.story_model.estimate_token_count(current_outline) if current_outline else 0
        )
        user_tokens: int = self.story_model.estimate_token_count(user_text)

        # Budget subtracts a static system-prompt estimate, story, notes, outline,
        # user text, and the output reserve from the context limit.
        fixed_costs: int = (
            SYSTEM_PROMPT_ESTIMATE_TOKENS
            + story_tokens
            + notes_tokens
            + outline_tokens
            + user_tokens
            + OUTPUT_RESERVE_TOKENS
        )
        available: int = context_limit - fixed_costs

        max_rag_tokens: int = self._compute_rag_budget(
            available,
            RAG_RATIO_PLANNING,
            MIN_RAG_TOKENS_PLANNING,
            MAX_RAG_TOKENS_PLANNING,
        )

        # Query RAG with the calculated budget.
        rag_context: str = self.rag_controller.query_databases(user_text, max_tokens=max_rag_tokens)
        rag_tokens: int = self.story_model.estimate_token_count(rag_context) if rag_context else 0

        # Build the dynamic system-content block for the BAML template.
        # Static planning instructions live in models/baml_src/outline.baml;
        # only runtime-specific data goes here.
        system_content: str = ""

        if story_context:
            system_content += "\n\n=== EXISTING STORY CONTENT (WHAT HAS BEEN WRITTEN) ==="
            system_content += f"\n{story_context}"
            system_content += (
                "\n\nCRITICAL: Mark plot points as completed [x] ONLY if they describe "
                "events that are clearly written in the story text above. "
                "If you're adding new plot points to continue the story, those MUST be "
                "unchecked [ ] because they haven't been written yet."
            )
        else:
            system_content += "\n\n=== NO STORY CONTENT YET ==="
            system_content += (
                "\nThe story has not been started yet. "
                "ALL plot points in your outline must be marked as [ ] (unchecked)."
            )

        if notes:
            system_content += f"\n\nEXISTING NOTES:\n{notes}"

        if current_outline:
            system_content += (
                f"\n\nCURRENT OUTLINE:\n{current_outline}\n\n"
                "The user may want to refine this outline or discuss changes."
            )

        if rag_context:
            system_content += f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}"

        if attachments_text:
            system_content += f"\n\nATTACHMENTS:\n{attachments_text}"

        total_tokens: int = story_tokens + notes_tokens + outline_tokens + rag_tokens + user_tokens

        return ContextBundle(
            story_context=story_context,
            rag_context=rag_context,
            notes=notes,
            query=system_content,
            total_tokens=total_tokens,
            max_rag_tokens=max_rag_tokens,
            rag_tokens=rag_tokens,
        )

    def assemble_for_edit(
        self,
        selected_text: str,
        start_pos: int,
        end_pos: int,
        prompt: str,
        system_prompt: str,
    ) -> ContextBundle:
        """Assemble context for inline text editing (Update Selected Text).

        Reads the full story from ``story_model.content`` to extract the
        surrounding context window, and reads notes from ``story_model.notes``.

        Args:
            selected_text: The highlighted text to be rewritten.
            start_pos: Character start offset of the selection in the story.
            end_pos: Character end offset of the selection in the story.
            prompt: The user's rewrite instruction.
            system_prompt: System prompt for this LLM call.

        Returns:
            ``ContextBundle`` with the full edit query in ``query``.
        """
        full_story: str = self.story_model.content
        notes: str = self.story_model.notes

        # Extract fixed-size context windows around the selection.
        context_before: str = full_story[max(0, start_pos - EDIT_CONTEXT_CHARS_BEFORE) : start_pos]
        context_after: str = full_story[end_pos : end_pos + EDIT_CONTEXT_CHARS_AFTER]

        context_limit: int = self.settings_model.context_limit
        selected_tokens: int = self.story_model.estimate_token_count(selected_text)
        before_tokens: int = self.story_model.estimate_token_count(context_before)
        after_tokens: int = self.story_model.estimate_token_count(context_after)
        notes_tokens: int = self.story_model.estimate_token_count(notes) if notes else 0
        prompt_tokens: int = self.story_model.estimate_token_count(prompt)

        fixed_costs: int = (
            EDIT_SYSTEM_ESTIMATE_TOKENS
            + selected_tokens
            + before_tokens
            + after_tokens
            + notes_tokens
            + prompt_tokens
            + OUTPUT_RESERVE_TOKENS
        )
        available: int = context_limit - fixed_costs

        max_rag_tokens: int = self._compute_rag_budget(
            available, RAG_RATIO_EDIT, MIN_RAG_TOKENS_EDIT, MAX_RAG_TOKENS_EDIT
        )

        # Build an enriched RAG query from the selection's immediate neighbourhood.
        rag_query_parts: list[str] = []
        if context_before:
            rag_query_parts.append(context_before[-EDIT_RAG_QUERY_BEFORE_CHARS:])
        rag_query_parts.append(selected_text[:EDIT_RAG_QUERY_SELECTION_CHARS])
        rag_query_parts.append(prompt)
        if context_after:
            rag_query_parts.append(context_after[:EDIT_RAG_QUERY_AFTER_CHARS])
        rag_query: str = " ".join(rag_query_parts)

        rag_context: str = self.rag_controller.query_databases(rag_query, max_tokens=max_rag_tokens)
        rag_tokens: int = self.story_model.estimate_token_count(rag_context) if rag_context else 0

        # Assemble the full edit query.
        query: str = "Rewrite the following text according to the instruction."

        if context_before:
            query += f"\n\nCONTEXT BEFORE (do not modify this):\n{context_before}"

        query += f"\n\nTEXT TO REWRITE:\n{selected_text}"

        if context_after:
            query += f"\n\nCONTEXT AFTER (do not modify this):\n{context_after}"

        query += f"\n\nINSTRUCTION:\n{prompt}"

        if rag_context:
            query += f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}"

        if notes:
            query += f"\n\nADDITIONAL CONTEXT (author's notes):\n{notes}"

        query += "\n\nREWRITTEN VERSION (output only the rewritten text, nothing else):"

        total_tokens: int = (
            selected_tokens
            + before_tokens
            + after_tokens
            + notes_tokens
            + rag_tokens
            + prompt_tokens
        )

        return ContextBundle(
            story_context=selected_text,
            rag_context=rag_context,
            notes=notes,
            system_prompt=system_prompt,
            query=query,
            total_tokens=total_tokens,
            max_rag_tokens=max_rag_tokens,
            rag_tokens=rag_tokens,
        )

    def assemble_for_auto_build(
        self,
        initial_prompt: str,
        story_for_llm: str,
        notes: str,
        supp_text: str,
        system_prompt: str,
        paragraphs_per_chunk: int,
        attachments_text: str = "",
    ) -> ContextBundle:
        """Assemble context for one auto-build (Story Mode) chunk.

        The caller passes the story text that should appear in the prompt.
        If the story was too large to fit directly, that text is the output
        of ``llm_controller.process_story_with_summarization``; otherwise it
        is the raw story content from the view.

        Args:
            initial_prompt: The user's original story kick-off prompt.
            story_for_llm: Story text to include (possibly summarised).
            notes: Author's notes.
            supp_text: Supplemental prompt text.
            system_prompt: System prompt text.
            paragraphs_per_chunk: Number of paragraphs to request per chunk.
            attachments_text: File attachments rendered as plain text.

        Returns:
            ``ContextBundle`` with ``query`` fully assembled.
        """
        context_limit: int = self.settings_model.context_limit
        supp_tokens: int = self.story_model.estimate_token_count(supp_text)
        notes_tokens: int = self.story_model.estimate_token_count(notes)
        system_tokens: int = self.story_model.estimate_token_count(system_prompt)

        # Fixed costs exclude story tokens because the story budget is handled
        # separately by the summarisation logic before this method is called.
        fixed_costs: int = supp_tokens + notes_tokens + system_tokens + OUTPUT_RESERVE_TOKENS
        available: int = context_limit - fixed_costs

        max_rag_tokens: int = self._compute_rag_budget(
            available,
            RAG_RATIO_AUTO_BUILD,
            MIN_RAG_TOKENS_AUTO_BUILD,
            MAX_RAG_TOKENS_AUTO_BUILD,
        )

        # Build the RAG query using the initial prompt plus recent story tail.
        rag_query: str = initial_prompt
        if story_for_llm:
            recent_tail: str = (
                story_for_llm[-AUTO_BUILD_RAG_RECENT_CHARS:]
                if len(story_for_llm) > AUTO_BUILD_RAG_RECENT_CHARS
                else story_for_llm
            )
            rag_query = f"{initial_prompt}\n\nRecent story content:\n{recent_tail}"

        rag_context: str = self.rag_controller.query_databases(rag_query, max_tokens=max_rag_tokens)
        rag_tokens: int = self.story_model.estimate_token_count(rag_context) if rag_context else 0

        # Assemble the final chunk query.
        query_parts: list[str] = []

        if story_for_llm:
            query_parts.append(f"Story so far:\n```\n{story_for_llm}\n```\n\n")
        if rag_context:
            query_parts.append(f"Relevant context:\n{rag_context}\n\n")
        if notes:
            query_parts.append(f"Author notes:\n{notes}\n\n")
        if supp_text:
            query_parts.append(f"Additional instructions:\n{supp_text}\n\n")
        if attachments_text:
            query_parts.append(f"{attachments_text}\n\n")

        query_parts.append(f"Initial prompt: {initial_prompt}\n\n")
        query_parts.append(
            f"Continue the story. Write EXACTLY {paragraphs_per_chunk} paragraphs. "
            "Maintain narrative flow and character consistency."
        )

        final_query: str = "".join(query_parts)
        total_tokens: int = (
            self.story_model.estimate_token_count(story_for_llm) + fixed_costs + rag_tokens
        )

        return ContextBundle(
            story_context=story_for_llm,
            rag_context=rag_context,
            notes=notes,
            system_prompt=system_prompt,
            supp_text=supp_text,
            query=final_query,
            total_tokens=total_tokens,
            max_rag_tokens=max_rag_tokens,
            rag_tokens=rag_tokens,
        )

    def assemble_for_ask(
        self,
        user_input: str,
        attachments_text: str,
        selected_dbs_override: Optional[list] = None,
    ) -> ContextBundle:
        """Assemble context for Ask mode.

        Ask mode is a lightweight RAG-backed chat; no story, notes, or
        supplemental prompts are included.  The caller is responsible for
        preparing ``selected_dbs_override`` (including the hidden
        ``__ask_readme__`` database when appropriate).

        Args:
            user_input: The user's question.
            attachments_text: File attachments rendered as plain text.
            selected_dbs_override: Override the active RAG database list.

        Returns:
            ``ContextBundle`` with ``query`` fully assembled.
        """
        # Ask mode uses the RAG controller's default token budget (quiet mode).
        rag_context: str = self.rag_controller.query_databases(
            user_input,
            selected_dbs_override=selected_dbs_override,
            quiet=True,
        )
        rag_tokens: int = self.story_model.estimate_token_count(rag_context) if rag_context else 0

        # Assemble the final query.
        final_query: str = user_input
        if rag_context:
            final_query += f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}"
        if attachments_text:
            final_query += f"\n\n{attachments_text}"

        return ContextBundle(
            rag_context=rag_context,
            query=final_query,
            rag_tokens=rag_tokens,
        )

    def assemble_for_chunk(
        self,
        story_for_llm: str,
        current_task: str,
        outline: str,
        notes: str,
        supp_text: str,
        system_prompt: str,
    ) -> ContextBundle:
        """Assemble context for one outline-driven planning chunk.

        Only the RAG context and budget metadata are populated here.  The
        final query string is built by ``PlanningController._build_chunk_query``
        because it requires task-sequencing state (upcoming tasks, is_last_chunk)
        that is not available to the context manager.

        Args:
            story_for_llm: Story text (possibly summarised) to include.
            current_task: The plot point currently being written.
            outline: Full outline markdown text.
            notes: Author's notes.
            supp_text: Supplemental prompt text.
            system_prompt: System prompt text.

        Returns:
            ``ContextBundle`` with ``rag_context`` populated.
            ``query`` is intentionally empty — built by the planning controller.
        """
        context_limit: int = self.settings_model.context_limit
        story_tokens: int = self.story_model.estimate_token_count(story_for_llm)
        outline_tokens: int = self.story_model.estimate_token_count(outline)
        system_tokens: int = self.story_model.estimate_token_count(system_prompt)
        notes_tokens: int = self.story_model.estimate_token_count(notes)
        supp_tokens: int = self.story_model.estimate_token_count(supp_text)

        # Account for every component that appears in the final prompt so the
        # RAG budget reflects the true available headroom.
        fixed_costs: int = (
            system_tokens
            + outline_tokens
            + story_tokens
            + notes_tokens
            + supp_tokens
            + QUERY_OVERHEAD_TOKENS
            + OUTPUT_RESERVE_TOKENS
        )
        available: int = context_limit - fixed_costs

        # Only allocate RAG tokens when genuine headroom exists.
        max_rag_tokens: int = (
            min(int(available * RAG_RATIO_CHUNK), MAX_RAG_TOKENS_CHUNK) if available > 0 else 0
        )

        rag_context: str = ""
        rag_tokens: int = 0

        if max_rag_tokens > 0:
            # Enrich the RAG query with names from notes/outline so the
            # filename-boost tokenizer can match character-specific files.
            rag_query_parts: list[str] = [current_task]
            if outline:
                rag_query_parts.append(outline[:RAG_CHUNK_OUTLINE_EXCERPT_CHARS])
            if notes:
                rag_query_parts.append(notes[:RAG_CHUNK_NOTES_EXCERPT_CHARS])
            if supp_text:
                rag_query_parts.append(supp_text[:RAG_CHUNK_SUPP_EXCERPT_CHARS])
            rag_query: str = " ".join(rag_query_parts)

            rag_context = self.rag_controller.query_databases(rag_query, max_tokens=max_rag_tokens)
            rag_tokens = self.story_model.estimate_token_count(rag_context) if rag_context else 0

        total_tokens: int = story_tokens + outline_tokens + notes_tokens + supp_tokens + rag_tokens

        return ContextBundle(
            story_context=story_for_llm,
            rag_context=rag_context,
            notes=notes,
            system_prompt=system_prompt,
            supp_text=supp_text,
            query="",  # Built by PlanningController._build_chunk_query
            total_tokens=total_tokens,
            max_rag_tokens=max_rag_tokens,
            rag_tokens=rag_tokens,
        )
