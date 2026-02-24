"""Planning Mode Controller

Handles all planning mode logic including:
- Interactive LLM conversation for outline creation
- Structured output generation
- Outline-driven story generation
"""

import json
import threading
import traceback
from typing import Optional, Dict, Any, Callable, List
from PyQt5 import QtCore

from models.planning_model import PlanningModel
from views.llm_panel import LLMPanel


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
        baml_controller=None,
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
        self.baml_controller = baml_controller

        # Section position tracking for redo support
        self._section_start_positions: Dict[int, int] = {}
        self._section_end_positions: Dict[int, int] = {}

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
        # Show spinner immediately so the UI is responsive while context is built
        self.view.set_waiting(True)

        # Start background thread — context building and LLM call both happen off the main thread
        thread = threading.Thread(
            target=self._planning_thread_for_panel,
            args=(user_input, current_outline, attachments_text, conversation_history),
            daemon=True,
        )
        thread.start()

    def _planning_thread_for_panel(
        self, user_input, current_outline, attachments_text, conversation_history
    ):
        """Background thread for all planning LLM interaction.

        Builds context then delegates to _run_planning_request which makes a
        single BAML Plan() call. The LLM decides whether to chat or generate
        an outline. Context building is done here (off the main thread) so the
        UI stays responsive and the progress bar spins from the start.
        """
        try:
            context = self._build_planning_context(user_input, current_outline, attachments_text)
            history_str = self._format_conversation_history(conversation_history[:-1])
            base_url = self.settings_model.base_url
            model = self.settings_model.last_model
            api_key = self.settings_model.api_key
            self._run_planning_request(
                user_input, context, history_str, base_url, model, api_key, conversation_history
            )
        except Exception as e:
            error_msg = f"\n\n❌ Error: {str(e)}\n\n"
            self._append_planning_error(error_msg)
            print(f"Planning thread error: {e}")
            traceback.print_exc()
        finally:
            QtCore.QMetaObject.invokeMethod(
                self.view,
                "set_waiting",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False),
            )

    def _run_planning_request(
        self,
        user_input: str,
        context: str,
        history_str: str,
        base_url: str,
        model: str,
        api_key: str,
        conversation_history: list,
    ) -> None:
        """Execute a BAML Plan() call. The LLM decides chat vs outline.

        - Chat: streams text deltas to panel progressively.
        - Outline: emits each new plot point as '- [ ] ...' as it arrives.
        - If action='outline' and SAP extracts nothing, retries up to 3 times.
        """
        MAX_ATTEMPTS = 3
        _prev_msg_len: List[int] = [0]
        _prev_pts_count: List[int] = [0]
        _prefix_emitted: List[bool] = [False]

        def on_partial(partial) -> None:
            action = getattr(partial, "action", None)
            if not _prefix_emitted[0]:
                if action == "chat":
                    self._emit_to_panel("\n**Assistant:** ")
                _prefix_emitted[0] = True
            if action == "chat":
                msg = getattr(partial, "message", None) or ""
                delta = msg[_prev_msg_len[0] :]
                if delta:
                    self._emit_to_panel(delta)
                _prev_msg_len[0] = len(msg)
            elif action == "outline":
                pts = getattr(partial, "plot_points", None) or []
                for point in pts[_prev_pts_count[0] :]:
                    checkbox = "[x]" if point.completed else "[ ]"
                    self._emit_to_panel(f"- {checkbox} **{point.description}**\n")
                    details = getattr(point, "details", None)
                    if details:
                        self._emit_to_panel(f"  {details}\n")
                _prev_pts_count[0] = len(pts)

        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            _prev_msg_len[0] = 0
            _prev_pts_count[0] = 0
            _prefix_emitted[0] = False
            if attempt > 1:
                self._emit_to_panel(f"\n\n⚠️ Attempt {attempt}/{MAX_ATTEMPTS}...\n\n")

            result = self.baml_controller.plan(
                context=context,
                conversation_history=history_str,
                user_message=user_input,
                base_url=base_url,
                model=model,
                api_key=api_key,
                on_partial=on_partial,
            )

            if result is None:
                continue
            action = getattr(result, "action", None)
            if action == "chat":
                break  # chat always accepted
            if action == "outline":
                if getattr(result, "plot_points", None):
                    break  # got plot points, success
                # SAP extracted nothing — retry

        if result is None:
            self._append_planning_error(
                f"\n\n❌ Planning failed after {MAX_ATTEMPTS} attempts. Please try again.\n\n"
            )
            return

        self._emit_to_panel("\n\n")
        action = getattr(result, "action", None)

        if action == "chat":
            message = getattr(result, "message", None) or ""
            # Pre-compute the full conversation HTML off the main thread so the
            # main-thread slot only needs to call setHtml() — no markdown parsing.
            rag_items = self.rag_controller.last_rag_display_items
            full_conv = list(conversation_history) + [{"role": "assistant", "content": message}]
            html = LLMPanel.build_planning_html(full_conv, rag_items, rag_collapsed=True)
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "apply_planning_message_html",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "assistant"),
                QtCore.Q_ARG(str, message),
                QtCore.Q_ARG(str, html),
            )
            self.settings_model.save_planning_conversation(
                full_conv,
                self.view.llm_panel.get_current_outline(),
            )

        elif action == "outline":
            plot_points = getattr(result, "plot_points", None) or []
            if not plot_points:
                self._append_planning_error(
                    f"\n\n❌ Failed to extract outline after {MAX_ATTEMPTS} attempts. Please try again.\n\n"
                )
                return
            outline_text = self._format_outline_checklist(plot_points)
            # Pre-compute the full conversation HTML off the main thread.
            rag_items = self.rag_controller.last_rag_display_items
            full_conv = list(conversation_history) + [
                {"role": "assistant", "content": outline_text}
            ]
            html = LLMPanel.build_planning_html(full_conv, rag_items, rag_collapsed=True)
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "apply_planning_message_html",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "assistant"),
                QtCore.Q_ARG(str, outline_text),
                QtCore.Q_ARG(str, html),
            )
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "set_current_outline",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, outline_text),
            )
            # Populate the outline tracker with structured section data
            sections_data = [
                {
                    "description": p.description,
                    "details": getattr(p, "details", None) or "",
                    "completed": p.completed,
                }
                for p in plot_points
            ]
            QtCore.QMetaObject.invokeMethod(
                self.view.llm_panel,
                "set_outline_sections_json",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, json.dumps(sections_data)),
            )
            self.settings_model.save_planning_conversation(
                full_conv,
                outline_text,
            )
        else:
            self._append_planning_error(f"\n\n❌ Unknown planner action: {action!r}\n\n")

    def _format_conversation_history(self, history: list) -> str:
        """Format conversation history as Human:/Assistant: text for BAML templates.

        Args:
            history: List of dicts with 'role' ('user'/'assistant') and 'content' keys.

        Returns:
            Newline-joined 'Human: ...' / 'Assistant: ...' string, or empty string.
        """
        lines = []
        for msg in history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _format_outline_checklist(self, plot_points) -> str:
        """Format plot points as a markdown checklist."""
        lines = []
        for point in plot_points:
            checkbox = "[x]" if point.completed else "[ ]"
            lines.append(f"- {checkbox} **{point.description}**")
            if getattr(point, "details", None):
                lines.append(f"  {point.details}")
        return "\n".join(lines)

    def _emit_to_panel(self, text: str) -> None:
        """Emit text to the planning chat panel from any thread."""
        QtCore.QMetaObject.invokeMethod(
            self.view.llm_panel,
            "append_llm_panel_text",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    def _append_planning_error(self, error_msg: str):
        """Append an error message to the LLM panel."""
        self._emit_to_panel(error_msg)

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
            self.story_model.estimate_token_count(existing_notes) if existing_notes else 0
        )
        outline_tokens = (
            self.story_model.estimate_token_count(current_outline) if current_outline else 0
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
        rag_context = self.rag_controller.query_databases(user_text, max_tokens=max_rag_tokens)

        # Build dynamic context additions. The PLANNING_PROMPT system instructions
        # live in the BAML template (models/baml_src/outline.baml); this string
        # contains only the runtime-specific sections appended after them.
        system_content = ""
        # Add story context
        if story_context:
            system_content += "\n\n=== EXISTING STORY CONTENT (WHAT HAS BEEN WRITTEN) ==="
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
            system_content += f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}"

        if attachments_text:
            system_content += f"\n\nATTACHMENTS:\n{attachments_text}"

        return system_content

    def start_outline_build(self, outline: str, notes: str, supp_text: str, system_prompt: str):
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
            "paragraphs_per_chunk": 3,
            "chunks_before_summary": 3,
            "last_rag_context": None,
        }

        self.planning_model.build_state = build_state
        self.planning_model.is_active = True

        # Reset section position tracking
        self._section_start_positions = {}
        self._section_end_positions = {}

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

        # Ensure story panel is in plain text mode for streaming
        current_story = self.view.get_story_content()
        self.view.set_story_content(current_story)

        # Initialise tracker and writing bar
        self.view.llm_panel.outline_tracker.set_active(0)
        self.view.llm_panel.set_writing_started()
        self.view.llm_panel.set_section_writing(True)

        # Start first chunk
        self.generate_next_chunk()

    def resume_writing(self):
        """Resume generation after a stop or pause.

        Resets the stop flag so generation can proceed, then calls
        ``generate_next_chunk``.  Safe to call when already active (no-op guard
        in ``generate_next_chunk``).
        """
        self.llm_controller.llm_model.reset_stop_flag()
        self.planning_model.is_active = True
        self.view.llm_panel.set_section_writing(True)
        self.view.set_stop_enabled(True)
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

        # Disable Continue button while generating
        self.view.llm_panel.set_section_writing(True)

        # Record section start position on first chunk of each section
        if state["task_chunk_count"] == 1:
            self._section_start_positions[state["current_task_index"]] = len(
                self.view.get_story_content()
            )

        # Log progress
        self.view.append_logs(f"\n{'─' * 60}\n")
        self.view.append_logs(
            f"📝 PLOT POINT {state['current_task_index'] + 1}/{len(state['original_tasks'])} "
            f"(Chunk {state['task_chunk_count']}/{self.view.llm_panel.get_chunks_per_section()})\n"
        )
        task_preview = current_task[:80] + "..." if len(current_task) > 80 else current_task
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
        self.view.append_logs("🔄 Running summarization to compress older content...\n\n")

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
            context_limit - story_tokens - outline_tokens - system_tokens - output_reserve
        )
        # Use 25% for RAG in outline-driven mode (similar to auto-build)
        max_rag_tokens = int(available_for_rag_and_story * 0.25)
        max_rag_tokens = max(500, min(max_rag_tokens, 3000))

        # Query RAG
        self.view.append_logs(
            f"🔍 Querying knowledge bases (budget: {max_rag_tokens:,} tokens)...\n"
        )
        rag_context = self.rag_controller.query_databases(current_task, max_tokens=max_rag_tokens)
        state["last_rag_context"] = rag_context

        if rag_context:
            rag_tokens = self.story_model.estimate_token_count(rag_context)
            self.view.append_logs(f"  ✓ Retrieved {rag_tokens} tokens of context\n")
        else:
            self.view.append_logs("  • No additional context found\n")

        # Build query
        query = self._build_chunk_query(state, story_for_llm, current_task, rag_context)

        # Generate
        self.view.append_logs(f"✍️ Generating {state['paragraphs_per_chunk']} paragraphs...\n\n")
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

        task_addressed = completion_status.completion_ratio >= 0.5

        max_chunks = self.view.llm_panel.get_chunks_per_section()
        task_advanced = task_addressed or state["task_chunk_count"] >= max_chunks

        if task_advanced:
            completed_index = state["current_task_index"]
            if not task_addressed:
                self.view.append_logs("  • Max chunks reached for this plot point; moving on...\n")
            else:
                self.view.append_logs("  ✓ Plot point addressed! Moving to next...\n")
            state["current_task_index"] += 1
            state["task_chunk_count"] = 0

            # Record section end position and update tracker
            self._section_end_positions[completed_index] = len(current_story)
            self.view.llm_panel.outline_tracker.mark_complete(completed_index)
            next_idx = state["current_task_index"]
            if next_idx < len(state["original_tasks"]):
                self.view.llm_panel.outline_tracker.set_active(next_idx)

            # Flag to pause after optional notes generation
            state["_should_pause"] = True
        else:
            self.view.append_logs("  • Continuing with current plot point...\n")

        # Check if notes should be regenerated before next chunk
        if self._should_regenerate_notes(current_story):
            self.notes_controller.generate_notes_async(
                current_story,
                on_complete=lambda generated_notes, _: self._continue_after_notes(generated_notes),
                on_error=lambda _: self._continue_after_notes(),
                clear_existing=True,
                set_waiting_on_start=True,
                set_waiting_on_finish=True,
            )
            return

        # Decide: pause between sections, finish, or continue same task
        if state.pop("_should_pause", False):
            if state["current_task_index"] >= len(state["original_tasks"]):
                if self._on_chunk_complete_callback:
                    self._on_chunk_complete_callback()
                else:
                    self._finish_build()
            else:
                if self._on_chunk_complete_callback:
                    self._on_chunk_complete_callback()
                else:
                    self._do_pause_between_sections()
        else:
            # Same task — continue writing
            if self._on_chunk_complete_callback:
                self._on_chunk_complete_callback()
            else:
                QtCore.QTimer.singleShot(100, self.generate_next_chunk)

    def _finish_build(self):
        """Finish outline-driven build."""
        state = self.planning_model.build_state
        if not state:
            return

        self.view.append_logs(f"\n\n{'=' * 60}\n")
        self.view.append_logs("\u2705 PLANNING BUILD COMPLETE\n")
        self.view.append_logs(f"All {len(state['original_tasks'])} plot points addressed.\n")
        self.view.append_logs(f"Total chunks generated: {state['chunk_count']}\n")
        self.view.append_logs(f"{'=' * 60}\n")

        self.view.set_waiting(False)
        self.view.set_stop_enabled(False)
        self.view.llm_panel.set_section_writing(
            True
        )  # leave button disabled — nothing left to write
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
        self.view.append_logs(
            "\u23f8\ufe0f  Section may be incomplete. "
            "Click '\u25b6 Continue' to append more text, or '\u21ba' to rewrite it from scratch.\n"
        )
        self.view.append_logs(f"{'=' * 60}\n")

        self.view.set_waiting(False)
        self.view.set_stop_enabled(False)
        self.view.llm_panel.set_section_writing(False)  # re-enable Continue so user can resume
        self.planning_model.is_active = False

    def _on_build_error(self, error_msg: str):
        """Handle build errors.

        Args:
            error_msg: Error message
        """
        self.view.append_logs(f"\n\u274c Error during planning build: {error_msg}\n")
        self.view.append_logs(
            "\u23f8\ufe0f  Section may be incomplete. "
            "Click '\u25b6 Continue' to retry, or '\u21ba' to rewrite from scratch.\n"
        )
        self.view.set_stop_enabled(False)
        self.view.llm_panel.set_section_writing(False)  # re-enable Continue so user can retry
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
            return

        state = self.planning_model.build_state
        if state and state.pop("_should_pause", False):
            if state["current_task_index"] >= len(state["original_tasks"]):
                self._finish_build()
            else:
                self._do_pause_between_sections()
        else:
            QtCore.QTimer.singleShot(100, self.generate_next_chunk)

    def _do_pause_between_sections(self):
        """Pause generation and wait for the user to click Continue."""
        self.view.set_waiting(False)
        self.view.append_logs(
            "\n⏸️  Section complete — click '\u25b6 Continue' to write the next section.\n"
        )
        self.view.llm_panel.set_section_writing(False)

    def rewrite_section_with_diff(self, section_index: int, completion_callback: Callable):
        """Rewrite a completed section using the diff overlay (red → green accept/reject).

        Args:
            section_index: Zero-based index of the outline section to redo.
            completion_callback: Called when LLM finishes streaming (shows accept/reject UI).
        """
        state = self.planning_model.build_state
        if not state:
            return

        start_pos = self._section_start_positions.get(section_index)
        if start_pos is None:
            return

        # Build query using story truncated to the section's start position so the
        # LLM only sees what came before this section.
        story_up_to_section = self.view.get_story_content()[:start_pos]

        # Point build state at this section for _build_chunk_query
        state["current_task_index"] = section_index
        state["task_chunk_count"] = 0
        current_task = state["original_tasks"][section_index]

        # Re-query RAG for fresh context
        context_limit = self.settings_model.context_limit
        outline_tokens = self.story_model.estimate_token_count(state["outline"])
        system_tokens = self.story_model.estimate_token_count(state["system_prompt"])
        story_tokens = self.story_model.estimate_token_count(story_up_to_section)
        available = context_limit - story_tokens - outline_tokens - system_tokens - 2000
        max_rag_tokens = max(500, min(int(available * 0.25), 3000))
        rag_context = self.rag_controller.query_databases(current_task, max_tokens=max_rag_tokens)
        state["last_rag_context"] = rag_context

        query = self._build_chunk_query(state, story_up_to_section, current_task, rag_context)

        self.view.append_logs(f"\n\u21ba Rewriting section {section_index + 1} via diff...\n")
        self.view.set_waiting(True)
        self.llm_controller.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)

        self.llm_controller.override_text_with_streaming(
            query=query,
            system_prompt=state["system_prompt"],
            stream_callback=self.view.stream_override_text,
            completion_callback=completion_callback,
            set_stop_enabled_callback=self.view.set_stop_enabled,
        )
