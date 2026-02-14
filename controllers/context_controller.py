"""Context Controller

Centralizes all context gathering and management for different LLM operations.
Handles context budgeting, RAG queries, story summarization, and context assembly
for various use cases (planning, editing, generation, etc.).
"""

from typing import Optional, Tuple, Dict, Any


class ContextController:
    """Controller for managing context assembly and token budgeting.

    Centralizes logic for:
    - RAG context queries with dynamic token budgets
    - Story context extraction (recent + summarized)
    - Context assembly for different operations (planning, editing, generation)
    - Token budget calculation and allocation
    """

    def __init__(
        self,
        story_model,
        settings_model,
        summary_model,
        rag_controller,
        rag_model,
    ):
        """Initialize the context controller.

        Args:
            story_model: Story data model
            settings_model: Settings model (for context limits)
            summary_model: Summary model (for rolling summarization)
            rag_controller: RAG controller (for database queries)
            rag_model: Smart Model (for configuration)
        """
        self.story_model = story_model
        self.settings_model = settings_model
        self.summary_model = summary_model
        self.rag_controller = rag_controller
        self.rag_model = rag_model

    def gather_context_for_planning(
        self,
        user_text: str,
        current_outline: Optional[str] = None,
    ) -> Tuple[str, str, int]:
        """Gather context for planning/outline mode.

        Args:
            user_text: User's current message/query
            current_outline: Current outline if it exists

        Returns:
            Tuple of (rag_context, story_context, rag_tokens)
        """
        # Get existing story content
        existing_story = self.story_model.content
        existing_notes = self.story_model.notes

        # Extract story context using summarization
        story_context = ""
        story_tokens = 0
        if existing_story:
            max_recent_tokens = 2000
            raw_recent, split_pos = self.story_model.extract_recent_content(
                existing_story, max_recent_tokens
            )
            raw_tokens = self.story_model.estimate_token_count(raw_recent)
            story_context, story_tokens = self.summary_model.get_context_for_llm(
                raw_recent, raw_tokens
            )

        # Calculate dynamic RAG token budget for planning
        context_limit = self.settings_model.context_limit
        output_reserve = 2000
        system_prompt_estimate = 800
        notes_tokens = (
            self.story_model.estimate_token_count(existing_notes)
            if existing_notes
            else 0
        )
        outline_tokens = (
            self.story_model.estimate_token_count(current_outline)
            if current_outline
            else 0
        )
        user_tokens = self.story_model.estimate_token_count(user_text)

        available_for_rag = (
            context_limit
            - system_prompt_estimate
            - story_tokens
            - notes_tokens
            - outline_tokens
            - user_tokens
            - output_reserve
        )
        # Use 20% for RAG in planning mode (moderate to not overwhelm outline generation)
        max_rag_tokens = int(available_for_rag * 0.20)
        max_rag_tokens = max(500, min(max_rag_tokens, 2000))

        # Query RAG databases with calculated budget
        rag_context = self.rag_controller.query_databases(
            user_text, max_tokens=max_rag_tokens
        )
        rag_tokens = (
            self.story_model.estimate_token_count(rag_context) if rag_context else 0
        )

        return rag_context, story_context, rag_tokens

    def gather_context_for_edit(
        self,
        selected_text: str,
        start_pos: int,
        end_pos: int,
        prompt: str,
        context_chars_before: int = 2000,
        context_chars_after: int = 2000,
    ) -> Dict[str, Any]:
        """Gather context for inline text editing.

        Args:
            selected_text: The text that was selected
            start_pos: Start position of selection in document
            end_pos: End position of selection in document
            prompt: The edit instruction
            context_chars_before: Characters of context to include before selection
            context_chars_after: Characters of context to include after selection

        Returns:
            Dictionary containing:
                - selected_text: The selected text
                - context_before: Text before selection
                - context_after: Text after selection
                - rag_context: Relevant RAG context
                - notes: Author's notes
                - total_context_tokens: Estimated token count for all context
        """
        # Get full story content
        full_story = self.story_model.content

        # Extract context before and after selection
        context_before = full_story[
            max(0, start_pos - context_chars_before) : start_pos
        ]
        context_after = full_story[end_pos : end_pos + context_chars_after]

        # Get notes
        notes = self.story_model.notes

        # Calculate RAG budget for editing
        context_limit = self.settings_model.context_limit
        output_reserve = 2000
        system_prompt_estimate = 500

        # Estimate current context tokens
        selected_tokens = self.story_model.estimate_token_count(selected_text)
        before_tokens = self.story_model.estimate_token_count(context_before)
        after_tokens = self.story_model.estimate_token_count(context_after)
        notes_tokens = self.story_model.estimate_token_count(notes) if notes else 0
        prompt_tokens = self.story_model.estimate_token_count(prompt)

        available_for_rag = (
            context_limit
            - system_prompt_estimate
            - selected_tokens
            - before_tokens
            - after_tokens
            - notes_tokens
            - prompt_tokens
            - output_reserve
        )

        # Use 25% for RAG in edit mode
        max_rag_tokens = int(available_for_rag * 0.25)
        max_rag_tokens = max(300, min(max_rag_tokens, 1500))

        # Build RAG query from selected text + prompt + surrounding context
        rag_query_parts = []
        if context_before:
            # Take last 200 chars of before context
            rag_query_parts.append(context_before[-200:])
        rag_query_parts.append(selected_text[:500])  # First 500 chars of selection
        rag_query_parts.append(prompt)
        if context_after:
            # Take first 200 chars of after context
            rag_query_parts.append(context_after[:200])

        rag_query = " ".join(rag_query_parts)

        # Query RAG
        rag_context = self.rag_controller.query_databases(
            rag_query, max_tokens=max_rag_tokens
        )
        rag_tokens = (
            self.story_model.estimate_token_count(rag_context) if rag_context else 0
        )

        total_context_tokens = (
            selected_tokens
            + before_tokens
            + after_tokens
            + notes_tokens
            + rag_tokens
            + prompt_tokens
        )

        return {
            "selected_text": selected_text,
            "context_before": context_before,
            "context_after": context_after,
            "rag_context": rag_context,
            "notes": notes,
            "total_context_tokens": total_context_tokens,
            "rag_tokens": rag_tokens,
        }

    def gather_context_for_generation(
        self,
        user_input: str,
        notes: str,
        supp_text: str,
        system_prompt: str,
        planning_outline: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Gather context for story generation (normal send operation).

        Args:
            user_input: User's query input
            notes: Author's notes
            supp_text: Supplemental prompts text
            system_prompt: System prompt
            planning_outline: Planning outline if active

        Returns:
            Dictionary containing:
                - story_context: Summarized story context
                - rag_context: RAG context
                - story_tokens: Token count for story context
                - rag_tokens: Token count for RAG context
                - max_rag_tokens: Maximum RAG budget allocated
        """
        context_limit = self.settings_model.context_limit
        output_reserve = 2000

        # Estimate token counts
        supp_tokens = self.story_model.estimate_token_count(supp_text)
        notes_tokens = self.story_model.estimate_token_count(notes)
        user_tokens = self.story_model.estimate_token_count(user_input)
        system_tokens = self.story_model.estimate_token_count(system_prompt)
        outline_tokens = (
            self.story_model.estimate_token_count(planning_outline)
            if planning_outline
            else 0
        )

        # Get story context
        current_story = self.story_model.content
        story_tokens = self.story_model.estimate_token_count(current_story)

        # Calculate available space
        fixed_costs = (
            supp_tokens
            + notes_tokens
            + user_tokens
            + system_tokens
            + outline_tokens
            + output_reserve
        )
        available_for_rag_and_story = context_limit - fixed_costs

        # Allocate 30% to RAG, 70% to story
        max_rag_tokens = int(available_for_rag_and_story * 0.30)
        max_rag_tokens = max(500, min(max_rag_tokens, 4000))

        max_story_tokens = int(available_for_rag_and_story * 0.70)

        # Get story context (summarized if needed)
        if story_tokens > max_story_tokens:
            max_recent_tokens = min(self.rag_model.summary_chunk_size, max_story_tokens)
            raw_recent, split_pos = self.story_model.extract_recent_content(
                current_story, max_recent_tokens
            )
            raw_tokens = self.story_model.estimate_token_count(raw_recent)
            story_context, story_context_tokens = (
                self.summary_model.get_context_for_llm(raw_recent, raw_tokens)
            )
        else:
            story_context = current_story
            story_context_tokens = story_tokens

        # Query RAG
        rag_context = self.rag_controller.query_databases(
            user_input, max_tokens=max_rag_tokens
        )
        rag_tokens = (
            self.story_model.estimate_token_count(rag_context) if rag_context else 0
        )

        return {
            "story_context": story_context,
            "rag_context": rag_context,
            "story_tokens": story_context_tokens,
            "rag_tokens": rag_tokens,
            "max_rag_tokens": max_rag_tokens,
        }

    def gather_context_for_auto_build(
        self,
        initial_prompt: str,
        notes: str,
        supp_text: str,
        system_prompt: str,
        current_story: str,
    ) -> Dict[str, Any]:
        """Gather context for auto-build mode chunk generation.

        Args:
            initial_prompt: Initial prompt for auto-build
            notes: Author's notes
            supp_text: Supplemental prompts
            system_prompt: System prompt
            current_story: Current story content (possibly summarized)

        Returns:
            Dictionary containing RAG context and token information
        """
        context_limit = self.settings_model.context_limit
        output_reserve = 2000

        # Calculate token counts
        supp_tokens = self.story_model.estimate_token_count(supp_text)
        notes_tokens = self.story_model.estimate_token_count(notes)
        system_tokens = self.story_model.estimate_token_count(system_prompt)

        available_for_rag_and_story = (
            context_limit - supp_tokens - notes_tokens - system_tokens - output_reserve
        )

        # Use 25% for RAG in auto-build mode
        max_rag_tokens = int(available_for_rag_and_story * 0.25)
        max_rag_tokens = max(500, min(max_rag_tokens, 3000))

        # Build RAG query from initial prompt + recent story
        rag_query_parts = [initial_prompt]
        if current_story:
            # Use last 1000 chars of story for context
            rag_query_parts.append(current_story[-1000:])
        rag_query = " ".join(rag_query_parts)

        # Query RAG
        rag_context = self.rag_controller.query_databases(
            rag_query, max_tokens=max_rag_tokens
        )
        rag_tokens = (
            self.story_model.estimate_token_count(rag_context) if rag_context else 0
        )

        return {
            "rag_context": rag_context,
            "rag_tokens": rag_tokens,
            "max_rag_tokens": max_rag_tokens,
        }

    def gather_context_for_outline_build(
        self,
        current_task: str,
        outline: str,
        system_prompt: str,
        story_for_llm: str,
    ) -> Dict[str, Any]:
        """Gather context for outline-driven story generation.

        Args:
            current_task: Current plot point being written
            outline: Full story outline
            system_prompt: System prompt
            story_for_llm: Story content (possibly summarized)

        Returns:
            Dictionary containing RAG context and token information
        """
        context_limit = self.settings_model.context_limit
        output_reserve = 2000

        story_tokens = self.story_model.estimate_token_count(story_for_llm)
        outline_tokens = self.story_model.estimate_token_count(outline)
        system_tokens = self.story_model.estimate_token_count(system_prompt)

        available_for_rag = (
            context_limit
            - story_tokens
            - outline_tokens
            - system_tokens
            - output_reserve
        )

        # Use 25% for RAG in outline-driven mode
        max_rag_tokens = int(available_for_rag * 0.25)
        max_rag_tokens = max(500, min(max_rag_tokens, 3000))

        # Query RAG with current task
        rag_context = self.rag_controller.query_databases(
            current_task, max_tokens=max_rag_tokens
        )
        rag_tokens = (
            self.story_model.estimate_token_count(rag_context) if rag_context else 0
        )

        return {
            "rag_context": rag_context,
            "rag_tokens": rag_tokens,
            "max_rag_tokens": max_rag_tokens,
        }
