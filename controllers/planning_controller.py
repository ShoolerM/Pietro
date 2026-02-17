"""Planning Mode Controller

Handles all planning mode logic including:
- Interactive LLM conversation for outline creation
- Structured output generation
- Outline-driven story generation
"""

import threading
import traceback
from typing import Optional, Dict, Any, Callable
from PyQt5 import QtCore

from models import base_prompts
from models.planning_model import PlanningModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class PlanningController(QtCore.QObject):
    """Controller for planning mode functionality."""

    def __init__(
        self,
        planning_model: PlanningModel,
        story_model,
        settings_model,
        llm_controller,
        rag_controller,
        rag_model,
        summary_model,
        notes_controller,
        view,
    ):
        super().__init__()
        self.planning_model = planning_model
        self.story_model = story_model
        self.settings_model = settings_model
        self.llm_controller = llm_controller
        self.rag_controller = rag_controller
        self.rag_model = rag_model
        self.summary_model = summary_model
        self.notes_controller = notes_controller
        self.view = view

        # Callbacks for delegation back to main controller
        self._on_notes_ready_callback: Optional[Callable] = None
        self._on_chunk_complete_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        on_notes_ready: Optional[Callable] = None,
        on_chunk_complete: Optional[Callable] = None,
    ):
        """Set callback functions for main controller integration.

        Args:
            on_notes_ready: Called when notes generation completes
            on_chunk_complete: Called when chunk generation completes
        """
        self._on_notes_ready_callback = on_notes_ready
        self._on_chunk_complete_callback = on_chunk_complete

    def handle_planning_message(
        self, user_input, conversation_history, current_outline, attachments_text
    ):
        """Handle a planning conversation message without dialog.

        Args:
            user_input: Users message
            conversation_history: Full conversation history
            current_outline: Current outline state
        """
        # Build context for planning
        context = self._build_planning_context(
            user_input, current_outline, attachments_text
        )

        # Set waiting state
        self.view.set_waiting(True)

        # Start background thread for LLM streaming to panel
        thread = threading.Thread(
            target=self._planning_thread_for_panel,
            args=(user_input, context, conversation_history),
            daemon=True,
        )
        thread.start()

    def _planning_thread_for_panel(self, user_input, context, conversation_history):
        """Background thread for planning LLM interaction (panel version).

        Args:
            user_input: User's message
            context: Built context string (system message)
            conversation_history: Full conversation history
        """
        try:
            # Build messages for LLM
            messages = [
                SystemMessage(content=context),
            ]

            # Add conversation history (excluding the just-added user message)
            for msg in conversation_history[:-1]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

            # Add current user message
            messages.append(HumanMessage(content=user_input))

            # Detect if user is asking for an outline
            is_outline_request = self._is_outline_request(user_input)

            # Use structured output for outline requests (strict, no parsing fallback)
            if is_outline_request:
                outline_result = self._request_structured_outline(messages)
                if outline_result is None:
                    self._append_planning_error(
                        "\n\n❌ Failed to generate a valid outline. Please try again.\n\n"
                    )
                    return

                response_text, outline_text = outline_result

                # Display in panel
                QtCore.QMetaObject.invokeMethod(
                    self.view.llm_panel,
                    "append_llm_panel_text",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "\n**Assistant:** "),
                )
                QtCore.QMetaObject.invokeMethod(
                    self.view.llm_panel,
                    "append_llm_panel_text",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, response_text),
                )
                QtCore.QMetaObject.invokeMethod(
                    self.view.llm_panel,
                    "append_llm_panel_text",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "\n\n"),
                )

                # Add to conversation
                QtCore.QMetaObject.invokeMethod(
                    self.view.llm_panel,
                    "add_planning_message",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "assistant"),
                    QtCore.Q_ARG(str, response_text),
                )

                # Store only the validated outline checklist
                QtCore.QMetaObject.invokeMethod(
                    self.view.llm_panel,
                    "set_current_outline",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, outline_text),
                )

                # Save conversation with stored outline
                self.settings_model.save_planning_conversation(
                    self.view.llm_panel.get_planning_conversation(),
                    outline_text,
                )

                return

            # Fallback: Stream response to panel (for non-outline or if structured failed)
            response_buffer = []

            # Add assistant prefix to panel
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "append_llm_panel_text",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "\n**Assistant:** "),
            )

            # Invoke LLM with streaming
            for chunk in self.llm_controller.llm.stream(messages):
                token = chunk.content
                response_buffer.append(token)
                # Stream to LLM panel
                QtCore.QMetaObject.invokeMethod(
                    self.view.llm_panel,
                    "append_llm_panel_text",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, token),
                )

            # Add newline after response
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "append_llm_panel_text",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "\n\n"),
            )

            # Complete response text
            response_text = "".join(response_buffer)

            # Add to panel's conversation
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "add_planning_message",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "assistant"),
                QtCore.Q_ARG(str, response_text),
            )

            # Save conversation via settings model (thread-safe file I/O)
            self.settings_model.save_planning_conversation(
                self.view.llm_panel.get_planning_conversation(),
                self.view.llm_panel.get_current_outline(),
            )

        except Exception as e:
            error_msg = f"\n\n❌ Error: {str(e)}\n\n"
            self._append_planning_error(error_msg)
            print(f"Planning thread error: {e}")
            traceback.print_exc()
        finally:
            # Hide waiting indicator
            QtCore.QMetaObject.invokeMethod(
                self.view,
                "set_waiting",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False),
            )

    def _is_outline_request(self, user_input: str) -> bool:
        """Check if user request is asking for an outline."""
        outline_keywords = [
            "outline",
            "plan",
            "structure",
            "plot points",
            "story beats",
        ]
        lower_text = user_input.lower()
        return any(keyword in lower_text for keyword in outline_keywords)

    def _request_structured_outline(self, messages):
        """Request a structured outline with strict retries for invalid plot points.

        Returns:
            Tuple[str, str]: (response_text, outline_text) or None on failure
        """
        from models.planning_model import StoryOutline

        # Tool-based structured output can trigger unsupported tool_choice errors.
        # Use strict JSON-only outlines for reliability.
        return self._request_outline_via_json(messages)

    def _build_structured_outline_llm(self, schema, fallback: bool = False):
        """Build a structured-output LLM for outlines with a safe method.

        Args:
            schema: Pydantic schema to enforce
            fallback: If True, use alternate method

        Returns:
            Structured LLM or None if unsupported
        """
        try:
            import inspect

            with_structured = self.llm_controller.llm.with_structured_output
            supports_method = "method" in inspect.signature(with_structured).parameters

            if supports_method:
                # Prefer function_calling when available; fallback disables tool use
                if not fallback:
                    return with_structured(schema, method="function_calling")
                return None

            # If method isn't supported, avoid tool-based structured output
            return None
        except Exception as e:
            self._append_planning_log(f"❌ Structured output init failed: {e}\n")
            return None

    def _request_outline_via_json(self, messages):
        """Request outline via strict JSON-only response and validate with schema.

        Returns:
            Tuple[str, str]: (response_text, outline_text) or None on failure
        """
        from models.planning_model import StoryOutline
        import json

        json_instructions = (
            "Return ONLY valid JSON that matches this schema: "
            "{discussion: string|null, plot_points: [{description: string, completed: boolean}], "
            "suggestions: [string]}. No extra text."
        )

        json_messages = messages + [HumanMessage(content=json_instructions)]
        response = self.llm_controller.llm.invoke(json_messages)

        raw_text = response.content if hasattr(response, "content") else str(response)
        try:
            data = json.loads(raw_text)
            result = StoryOutline(**data)
        except Exception as e:
            self._append_planning_log(
                f"❌ JSON outline parse failed: {e}\n↪ Please try again.\n"
            )
            return None

        discussion = result.discussion
        suggestions = result.suggestions or []
        error = self._validate_plot_points(result.plot_points)
        if error:
            retry_result = self._retry_outline_plot_points_json(
                messages,
                error,
                discussion,
                suggestions,
            )
            if retry_result is None:
                return None
            result, discussion, suggestions = retry_result

        outline_text = self._format_outline_checklist(result.plot_points)
        response_text = self._format_outline_response(
            discussion, outline_text, suggestions
        )
        return response_text, outline_text

    def _retry_outline_plot_points_json(
        self,
        messages,
        error: str,
        discussion: Optional[str],
        suggestions,
        max_retries: int = 2,
    ):
        """Retry only plot_points using strict JSON responses."""
        from models.planning_model import StoryOutline
        import json

        retry_instruction = (
            "The plot_points were invalid: "
            f"{error} Return ONLY JSON with plot_points corrected. "
            "You may omit discussion and suggestions."
        )

        for _ in range(max_retries):
            retry_messages = messages + [HumanMessage(content=retry_instruction)]
            response = self.llm_controller.llm.invoke(retry_messages)
            raw_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            try:
                data = json.loads(raw_text)
                retry_result = StoryOutline(**data)
            except Exception:
                continue

            retry_error = self._validate_plot_points(retry_result.plot_points)
            if not retry_error:
                merged_discussion = retry_result.discussion or discussion
                merged_suggestions = (
                    retry_result.suggestions
                    if retry_result.suggestions
                    else suggestions
                )
                return retry_result, merged_discussion, merged_suggestions
        return None

    def _validate_plot_points(self, plot_points) -> Optional[str]:
        """Validate plot points and return error string if invalid."""
        if not plot_points:
            return "No plot points were provided."

        for idx, point in enumerate(plot_points, 1):
            if not getattr(point, "description", "").strip():
                return f"Plot point {idx} is missing a description."
        return None

    def _retry_outline_plot_points(
        self,
        messages,
        error: str,
        discussion: Optional[str],
        suggestions,
        structured_llm,
        max_retries: int = 2,
    ):
        """Retry only the outline portion when plot points are invalid."""
        retry_instruction = (
            "The outline plot points were invalid: "
            f"{error} Please return ONLY corrected plot_points following the schema. "
            "Discussion and suggestions may be omitted."
        )

        for _ in range(max_retries):
            retry_messages = messages + [HumanMessage(content=retry_instruction)]
            retry_result = structured_llm.invoke(retry_messages)
            retry_error = self._validate_plot_points(retry_result.plot_points)
            if not retry_error:
                merged_discussion = retry_result.discussion or discussion
                merged_suggestions = (
                    retry_result.suggestions
                    if retry_result.suggestions
                    else suggestions
                )
                return retry_result, merged_discussion, merged_suggestions
        return None

    def _format_outline_checklist(self, plot_points) -> str:
        """Format plot points as a markdown checklist."""
        lines = []
        for point in plot_points:
            checkbox = "[x]" if point.completed else "[ ]"
            lines.append(f"- {checkbox} {point.description}")
        return "\n".join(lines)

    def _format_outline_response(
        self,
        discussion: Optional[str],
        outline_text: str,
        suggestions,
    ) -> str:
        """Format discussion + outline + suggestions response text."""
        parts = []
        if discussion:
            parts.append("**Discussion**")
            parts.append(discussion.strip())
            parts.append("")

        parts.append("**Outline**")
        parts.append(outline_text)

        if suggestions:
            parts.append("")
            parts.append("**Suggestions**")
            for suggestion in suggestions:
                if suggestion.strip():
                    parts.append(f"- {suggestion.strip()}")

        return "\n".join(parts)

    def _append_planning_error(self, error_msg: str):
        """Append a planning error to the LLM panel."""
        QtCore.QMetaObject.invokeMethod(
            self.view.llm_panel,
            "append_llm_panel_text",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, error_msg),
        )

    def _append_planning_log(self, log_msg: str):
        """Append a log message from a background thread safely."""
        QtCore.QMetaObject.invokeMethod(
            self.view.llm_panel,
            "append_logs",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, log_msg),
        )

    def _build_planning_context(
        self, user_text: str, current_outline: Optional[str], attachments_text: str
    ) -> str:
        """Build system context for planning conversation.

        Args:
            user_text: User's current message
            current_outline: Current outline if it exists

        Returns:
            System message content with full context
        """
        # Get existing story content and notes
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

        # Calculate dynamic RAG token budget
        context_limit = self.settings_model.context_limit
        output_reserve = 2000
        # Estimate system prompt tokens (rough estimate)
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
        # Use 20% for RAG in planning mode (keep it moderate to not overwhelm outline generation)
        max_rag_tokens = int(available_for_rag * 0.20)
        max_rag_tokens = max(500, min(max_rag_tokens, 2000))

        # Query RAG databases with calculated budget
        rag_context = self.rag_controller.query_databases(
            user_text, max_tokens=max_rag_tokens
        )

        # Build system message
        system_content = base_prompts.PLANNING_PROMPT
        # Add story context
        if story_context:
            system_content += (
                "\n\n=== EXISTING STORY CONTENT (WHAT HAS BEEN WRITTEN) ==="
            )
            system_content += f"\n{story_context}"
            system_content += (
                "\n\nCRITICAL: Mark plot points as completed [x] ONLY if they describe events that are clearly written in the story text above. "
                "If you're adding new plot points to continue the story, those MUST be unchecked [ ] because they haven't been written yet."
            )
        else:
            system_content += "\n\n=== NO STORY CONTENT YET ==="
            system_content += "\nThe story has not been started yet. ALL plot points in your outline must be marked as [ ] (unchecked)."

        # Add notes
        if existing_notes:
            system_content += f"\n\nEXISTING NOTES:\n{existing_notes}"

        # Add current outline
        if current_outline:
            system_content += f"\n\nCURRENT OUTLINE:\n{current_outline}\n\nThe user may want to refine this outline or discuss changes."

        # Add RAG context
        if rag_context:
            system_content += (
                f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}"
            )

        if attachments_text:
            system_content += f"\n\nATTACHMENTS:\n{attachments_text}"

        return system_content

    def start_outline_build(
        self, outline: str, notes: str, supp_text: str, system_prompt: str
    ):
        """Start outline-driven story generation.

        Args:
            outline: Markdown checklist outline
            notes: Author notes
            supp_text: Supplemental prompts
            system_prompt: System prompt
        """
        # Parse outline into tasks
        tasks = self.planning_model.parse_outline_tasks(outline)

        if not tasks:
            self.view.show_warning(
                "Invalid Outline",
                "No tasks found in outline. Please ensure outline uses markdown checklist format:\n- [ ] Task 1\n- [ ] Task 2",
            )
            return

        # Initialize build state
        build_state = {
            "outline": outline,
            "original_tasks": tasks.copy(),
            "remaining_tasks": tasks.copy(),
            "notes": notes,
            "supp_text": supp_text,
            "system_prompt": system_prompt,
            "current_task_index": 0,
            "chunk_count": 0,
            "task_chunk_count": 0,
            "max_chunks_per_task": 2,
            "paragraphs_per_chunk": 3,
            "chunks_before_summary": 3,
            "last_rag_context": None,
        }

        self.planning_model.build_state = build_state
        self.planning_model.is_active = True

        # Reset stop flag
        self.llm_controller.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)

        # Show status in LLM panel (logs)
        self.view.append_logs(f"\n{'=' * 60}\n")
        self.view.append_logs("📋 PLANNING MODE: OUTLINE-DRIVEN GENERATION\n")
        self.view.append_logs(f"{'=' * 60}\n\n")
        self.view.append_logs(f"Outline contains {len(tasks)} plot points:\n")
        for i, task in enumerate(tasks, 1):
            task_preview = task[:80] + "..." if len(task) > 80 else task
            self.view.append_logs(f"  {i}. {task_preview}\n")
        self.view.append_logs(f"\n{'=' * 60}\n\n")

        # Save history
        self.story_model.save_to_history()

        # Start first chunk
        self.generate_next_chunk()

    def generate_next_chunk(self):
        """Generate next chunk in outline-driven mode."""
        state = self.planning_model.build_state
        if not state:
            return

        # Check completion
        if state["current_task_index"] >= len(state["original_tasks"]):
            self._finish_build()
            return

        # Check stop flag
        if self.llm_controller.llm_model.stop_generation:
            self._stop_build()
            return

        # Update counters
        current_task = state["original_tasks"][state["current_task_index"]]
        state["chunk_count"] += 1
        state["task_chunk_count"] += 1

        # Log progress
        self.view.append_logs(f"\n{'─' * 60}\n")
        self.view.append_logs(
            f"📝 PLOT POINT {state['current_task_index'] + 1}/{len(state['original_tasks'])} "
            f"(Chunk {state['task_chunk_count']}/{state['max_chunks_per_task']})\n"
        )
        task_preview = (
            current_task[:80] + "..." if len(current_task) > 80 else current_task
        )
        self.view.append_logs(f"Task: {task_preview}\n")
        self.view.append_logs(f"{'─' * 60}\n\n")

        # Get current story
        current_story = self.view.get_story_content()

        # Check if summarization needed
        if self._should_summarize(state, current_story):
            self._run_summarization(state, current_story)
            return

        # Execute chunk generation
        self._execute_chunk_generation(state, current_story)

    def _should_summarize(self, state: Dict[str, Any], current_story: str) -> bool:
        """Check if summarization is needed.

        Args:
            state: Build state
            current_story: Current story content

        Returns:
            True if summarization should run
        """
        if (
            state["chunk_count"] <= 1
            or state["chunk_count"] % state["chunks_before_summary"] != 0
            or not self.settings_model.summarize_prompts
        ):
            return False

        story_tokens = self.story_model.estimate_token_count(current_story)
        context_limit = self.settings_model.context_limit

        # Calculate budgets
        supp_tokens = self.story_model.estimate_token_count(state["supp_text"])
        notes_tokens = self.story_model.estimate_token_count(state["notes"])
        system_tokens = self.story_model.estimate_token_count(state["system_prompt"])
        safety_buffer = 500

        fixed_costs = supp_tokens + notes_tokens + system_tokens + safety_buffer
        available_for_story = context_limit - fixed_costs
        max_raw_tokens = min(
            self.rag_model.summary_chunk_size,
            int(available_for_story * 0.6),
        )

        return story_tokens > max_raw_tokens

    def _run_summarization(self, state: Dict[str, Any], current_story: str):
        """Run summarization on story content.

        Args:
            state: Build state
            current_story: Current story content
        """
        story_tokens = self.story_model.estimate_token_count(current_story)
        self.view.append_logs(f"\n📊 Story getting large ({story_tokens} tokens)\n")
        self.view.append_logs(
            "🔄 Running summarization to compress older content...\n\n"
        )

        # Calculate token limits
        context_limit = self.settings_model.context_limit
        supp_tokens = self.story_model.estimate_token_count(state["supp_text"])
        notes_tokens = self.story_model.estimate_token_count(state["notes"])
        system_tokens = self.story_model.estimate_token_count(state["system_prompt"])
        safety_buffer = 500

        fixed_costs = supp_tokens + notes_tokens + system_tokens + safety_buffer
        available_for_story = context_limit - fixed_costs
        max_raw_tokens = min(
            self.rag_model.summary_chunk_size,
            int(available_for_story * 0.6),
        )
        max_rolling_summary_tokens = min(1000, int(available_for_story * 0.4))

        # Start summarization
        self.llm_controller.process_story_with_summarization(
            current_story,
            max_raw_tokens,
            max_rolling_summary_tokens,
            self.summary_model,
            self.view.append_logs,
            lambda story, tokens: self._on_summarization_complete(story, tokens),
            lambda error: self._on_build_error(error),
            self.view.set_waiting,
        )

    def _on_summarization_complete(self, story_for_llm: str, tokens: int):
        """Handle summarization completion.

        Args:
            story_for_llm: Summarized story content
            tokens: Token count
        """
        state = self.planning_model.build_state
        if not state:
            return

        self.view.append_logs(f"\n✅ Summarization complete ({tokens} tokens)\n")
        self.view.append_logs("Continuing with next chunk...\n\n")

        # Continue generation
        self._execute_chunk_generation(state, story_for_llm)

    def _execute_chunk_generation(self, state: Dict[str, Any], story_for_llm: str):
        """Execute chunk generation for current task.

        Args:
            state: Build state
            story_for_llm: Story content (possibly summarized)
        """
        current_task = state["original_tasks"][state["current_task_index"]]

        # Calculate dynamic RAG token budget
        context_limit = self.settings_model.context_limit
        output_reserve = 2000
        story_tokens = self.story_model.estimate_token_count(story_for_llm)
        outline_tokens = self.story_model.estimate_token_count(state["outline"])
        system_tokens = self.story_model.estimate_token_count(state["system_prompt"])

        available_for_rag_and_story = (
            context_limit
            - story_tokens
            - outline_tokens
            - system_tokens
            - output_reserve
        )
        # Use 25% for RAG in outline-driven mode (similar to auto-build)
        max_rag_tokens = int(available_for_rag_and_story * 0.25)
        max_rag_tokens = max(500, min(max_rag_tokens, 3000))

        # Query RAG
        self.view.append_logs(
            f"🔍 Querying knowledge bases (budget: {max_rag_tokens:,} tokens)...\n"
        )
        rag_context = self.rag_controller.query_databases(
            current_task, max_tokens=max_rag_tokens
        )
        state["last_rag_context"] = rag_context

        if rag_context:
            rag_tokens = self.story_model.estimate_token_count(rag_context)
            self.view.append_logs(f"  ✓ Retrieved {rag_tokens} tokens of context\n")
        else:
            self.view.append_logs("  • No additional context found\n")

        # Build query
        query = self._build_chunk_query(state, story_for_llm, current_task, rag_context)

        # Generate
        self.view.append_logs(
            f"✍️ Generating {state['paragraphs_per_chunk']} paragraphs...\n\n"
        )
        self.view.set_waiting(True)

        self.llm_controller.generate_story_chunk(
            query,
            state["system_prompt"],
            state["paragraphs_per_chunk"],
            self.view.append_story_content,
            self.view.append_logs,
            self._on_chunk_complete,
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _build_chunk_query(
        self,
        state: Dict[str, Any],
        story_for_llm: str,
        current_task: str,
        rag_context: Optional[str],
    ) -> str:
        """Build query for chunk generation.

        Args:
            state: Build state
            story_for_llm: Story content
            current_task: Current plot point
            rag_context: RAG context if available

        Returns:
            Query string for LLM
        """
        parts = []

        # Show outline
        parts.append(f"COMPLETE STORY OUTLINE:\n{state['outline']}\n\n")
        parts.append(f"{'=' * 60}\n\n")

        # Story so far
        if story_for_llm:
            parts.append(f"STORY SO FAR:\n```\n{story_for_llm}\n```\n\n")
            is_first = False
        else:
            parts.append(
                "This is the BEGINNING of the story. Start from the very first plot point.\n\n"
            )
            is_first = True

        # Current task
        parts.append(f"CURRENT PLOT POINT TO ADDRESS NOW:\n{current_task}\n\n")

        # Upcoming tasks
        remaining = state["original_tasks"][state["current_task_index"] + 1 :]
        if remaining:
            parts.append("UPCOMING PLOT POINTS (do NOT write these yet):\n")
            for i, task in enumerate(remaining[:3], 1):
                parts.append(f"{i}. {task}\n")
            parts.append("\n")

        # RAG context
        if rag_context:
            parts.append(f"RELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}\n\n")

        # Notes and supplemental
        if state["notes"]:
            parts.append(f"AUTHOR'S NOTES:\n{state['notes']}\n\n")
        if state["supp_text"]:
            parts.append(f"ADDITIONAL INSTRUCTIONS:\n{state['supp_text']}\n\n")

        # Instructions
        if is_first:
            parts.append(
                f"START the story by addressing the first plot point above. "
                f"Write EXACTLY {state['paragraphs_per_chunk']} paragraphs. "
                f"Focus ONLY on the current plot point. Do NOT jump ahead to later plot points."
            )
        else:
            parts.append(
                f"CONTINUE the story by addressing the current plot point above. "
                f"Write EXACTLY {state['paragraphs_per_chunk']} paragraphs. "
                f"Focus ONLY on the current plot point while maintaining narrative flow from what came before. "
                f"Do NOT jump ahead to later plot points."
            )

        return "".join(parts)

    def _on_chunk_complete(self):
        """Handle chunk generation completion."""
        state = self.planning_model.build_state
        if not state:
            return

        self.view.set_waiting(False)

        stopped, current_story = self.llm_controller.finalize_chunk_completion(
            state["chunk_count"],
            self.view.append_logs,
            self.view.get_story_content,
        )

        if stopped:
            self._stop_build()
            return

        # Check task completion
        self.view.append_logs("🔍 Checking task completion...\n")
        current_task = state["original_tasks"][state["current_task_index"]]

        completion_status = self.rag_controller.get_outline_completion_status(
            f"- [ ] {current_task}",
            current_story,
            similarity_threshold=0.6,
        )

        task_addressed = completion_status["completion_ratio"] >= 0.5

        if task_addressed or state["task_chunk_count"] >= state["max_chunks_per_task"]:
            if not task_addressed:
                self.view.append_logs(
                    "  • Max chunks reached for this plot point; moving on...\n"
                )
            else:
                self.view.append_logs("  ✓ Plot point addressed! Moving to next...\n")
            state["current_task_index"] += 1
            state["task_chunk_count"] = 0
        else:
            self.view.append_logs("  • Continuing with current plot point...\n")

        # Check if notes should be regenerated before next chunk
        if self._should_regenerate_notes(current_story):
            self.view.set_waiting(True)
            self.notes_controller.generate_notes_async(
                current_story,
                on_complete=lambda generated_notes, _: self._continue_after_notes(
                    generated_notes
                ),
                on_error=lambda _: self._continue_after_notes(),
                clear_existing=True,
                set_waiting_on_start=False,
                set_waiting_on_finish=False,
            )
            return

        # Trigger callback if set
        if self._on_chunk_complete_callback:
            self._on_chunk_complete_callback()
        else:
            # Default: continue to next chunk
            QtCore.QTimer.singleShot(100, self.generate_next_chunk)

    def _finish_build(self):
        """Finish outline-driven build."""
        state = self.planning_model.build_state
        if not state:
            return

        self.view.append_logs(f"\n\n{'=' * 60}\n")
        self.view.append_logs("\u2705 PLANNING BUILD COMPLETE\n")
        self.view.append_logs(
            f"All {len(state['original_tasks'])} plot points addressed.\n"
        )
        self.view.append_logs(f"Total chunks generated: {state['chunk_count']}\n")
        self.view.append_logs(f"{'=' * 60}\n")

        self.view.set_waiting(False)
        self.view.set_stop_enabled(False)
        self.story_model.planning_active = False
        self.planning_model.is_active = False
        self.planning_model.reset_build_state()

    def _stop_build(self):
        """Stop outline-driven build."""
        state = self.planning_model.build_state
        if not state:
            return

        self.view.append_logs(f"\n\n{'=' * 60}\n")
        self.view.append_logs("\u23f9\ufe0f PLANNING BUILD STOPPED BY USER\n")
        self.view.append_logs(
            f"Completed {state['current_task_index']}/{len(state['original_tasks'])} plot points.\n"
        )
        self.view.append_logs(f"Generated {state['chunk_count']} chunks.\n")
        self.view.append_logs(f"{'=' * 60}\n")

        self.view.set_waiting(False)
        self.view.set_stop_enabled(False)
        self.planning_model.is_active = False

    def _on_build_error(self, error_msg: str):
        """Handle build errors.

        Args:
            error_msg: Error message
        """
        self.view.append_logs(f"\n❌ Error during planning build: {error_msg}\n")
        self.view.append_logs("Planning build stopped.\n")
        self.view.set_stop_enabled(False)
        self.planning_model.is_active = False

    def _should_regenerate_notes(self, current_story: str) -> bool:
        """Check if notes should be regenerated.

        Args:
            current_story: Current story content

        Returns:
            True if notes should be regenerated
        """
        return self.notes_controller.should_regenerate_notes(
            current_story,
            self.settings_model.auto_notes,
            self.view.notes_panel.should_regenerate_notes(),
            context_key="planning",
            first_time_regenerate=True,
        )

    @QtCore.pyqtSlot()
    def _continue_after_notes(self, generated_notes: str = None):
        """Continue planning build after notes generation."""
        if generated_notes is not None and self.planning_model.build_state:
            self.planning_model.build_state["notes"] = generated_notes

        self.view.set_waiting(False)

        if self._on_chunk_complete_callback:
            self._on_chunk_complete_callback()
        else:
            QtCore.QTimer.singleShot(100, self.generate_next_chunk)
