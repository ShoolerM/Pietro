"""Planning Mode Controller

Handles all planning mode logic including:
- Interactive LLM conversation for outline creation
- Structured output generation
- Outline-driven story generation
"""

import threading
import hashlib
import re
from typing import Optional, Dict, Any, Callable
from PyQt5 import QtCore

from models.planning_model import PlanningModel, OutlinePlotPoint, StoryOutline
from views.planning_mode_dialog import PlanningModeDialog


class PlanningController:
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
        view,
    ):
        """Initialize the planning controller.

        Args:
            planning_model: Planning data model
            story_model: Story data model
            settings_model: Settings model
            llm_controller: LLM controller for generation
            rag_controller: RAG controller for context retrieval
            rag_model: RAG model for configuration
            summary_model: Summary model for content compression
            view: Main view for UI interactions
        """
        self.planning_model = planning_model
        self.story_model = story_model
        self.settings_model = settings_model
        self.llm_controller = llm_controller
        self.rag_controller = rag_controller
        self.rag_model = rag_model
        self.summary_model = summary_model
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

    def start_planning_mode(self):
        """Open planning mode dialog for interactive outline creation."""
        # Create dialog
        dialog = PlanningModeDialog(
            parent=self.view,
            initial_prompt=self.settings_model.planning_prompt_template,
        )

        # Restore previous conversation
        if self.planning_model.conversation_markdown:
            dialog.set_conversation(self.planning_model.conversation_markdown)

        # Connect dialog signals
        dialog.user_input_ready.connect(lambda text: self._on_user_input(text, dialog))
        dialog.start_writing_clicked.connect(
            lambda outline: self._on_start_writing(outline)
        )
        dialog.dialog_cancelled.connect(lambda: print("Planning mode cancelled"))

        # Show dialog
        result = dialog.show_with_initial_prompt()

        # Save conversation
        self.planning_model.conversation_markdown = dialog.get_conversation_text()

        if result == dialog.Accepted:
            print("Planning mode completed successfully")
        else:
            print("Planning mode cancelled by user")

    def _on_user_input(self, user_text: str, dialog: PlanningModeDialog):
        """Handle user input in planning dialog.

        Args:
            user_text: User's message
            dialog: Planning mode dialog instance
        """
        dialog.set_waiting.emit(True)

        # Add to conversation history
        self.planning_model.add_message("user", user_text)

        # Build context
        context = self._build_planning_context(user_text, dialog.get_current_outline())

        # Start background thread for LLM
        thread = threading.Thread(
            target=self._planning_thread,
            args=(user_text, context, dialog),
            daemon=True,
        )
        thread.start()

    def _build_planning_context(
        self, user_text: str, current_outline: Optional[str]
    ) -> str:
        """Build system context for planning conversation.

        Args:
            user_text: User's current message
            current_outline: Current outline if it exists

        Returns:
            System message content with full context
        """
        # Query RAG databases
        rag_context = self.rag_controller.query_databases(user_text)

        # Get existing story content and notes
        existing_story = self.story_model.content
        existing_notes = self.story_model.notes

        # Extract story context using summarization
        story_context = ""
        if existing_story:
            max_recent_tokens = 2000
            raw_recent, split_pos = self.story_model.extract_recent_content(
                existing_story, max_recent_tokens
            )
            raw_tokens = self.story_model.estimate_token_count(raw_recent)
            story_context, context_tokens = self.summary_model.get_context_for_llm(
                raw_recent, raw_tokens
            )

        # Build system message
        system_content = """You are a creative writing assistant helping plan a story outline.

Your role is to:
1. Ask clarifying questions in the CHAT to help develop the story idea
2. When the user is ready, provide a complete outline in markdown checklist format
3. If an outline already exists, you can refine/update it based on user feedback
4. If story content already exists, analyze what's been written and mark completed plot points

IMPORTANT RULES:
- Ask questions and have discussions in the CHAT (normal conversation)
- ONLY provide a checklist outline when the user explicitly asks for it or when you have enough information
- If providing an outline, format it as a markdown checklist with ONLY actual plot points/story events:
  - [x] Completed plot point (ONLY if this specific event is already written in the existing story)
  - [ ] Remaining plot point (if not yet written OR if you're adding a new plot point to extend the story)
- If NO story content exists yet, ALL plot points MUST be marked as [ ] (unchecked/remaining)
- When story content exists, carefully analyze it and mark ONLY the plot points that describe events explicitly covered in that text as [x] completed
- When adding NEW plot points to continue or extend the story, those new points MUST be marked as [ ] (unchecked) because they haven't been written yet
- The "completed" checkbox means "this event has already been written in the story" - NOT "this event should happen" or "this is part of the plan"
- Completed items should summarize what was actually written in the existing story text
- Remaining items should describe what still needs to be written
- DO NOT mark items as completed unless they describe events that are explicitly written in the existing story content
- DO NOT include metadata as checklist items (themes, setting descriptions, character lists, tone, style, etc.)
- Each checklist item should describe a specific narrative event or action that happens in the story
- You MAY include metadata/themes in regular text OUTSIDE the checklist if helpful
- If refining an existing outline, provide the complete updated outline as a checklist
- Focus on WHAT HAPPENS in the story, not abstract concepts or meta-information"""

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

        return system_content

    def _planning_thread(
        self, user_text: str, system_content: str, dialog: PlanningModeDialog
    ):
        """Background thread for LLM conversation.

        Args:
            user_text: User's message
            system_content: System message with context
            dialog: Planning dialog for signal emission
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

            # Build messages
            messages = [SystemMessage(content=system_content)]
            for msg in self.planning_model.conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

            # Check if requesting outline
            user_msg_lower = user_text.lower()
            requesting_outline = any(
                keyword in user_msg_lower
                for keyword in [
                    "outline",
                    "plot points",
                    "story plan",
                    "structure",
                    "give me",
                    "create",
                    "write",
                    "make",
                ]
            )

            full_response = ""

            if requesting_outline:
                # Use structured output
                full_response = self._generate_structured_outline(messages, dialog)
            else:
                # Regular streaming conversation
                for chunk in self.llm_controller.llm.stream(messages):
                    if hasattr(chunk, "content"):
                        text = chunk.content
                        full_response += text
                        dialog.llm_token_received.emit(text)

            # Add to history
            self.planning_model.add_message("assistant", full_response)

            # Emit completion
            dialog.llm_response_complete.emit(full_response)
            dialog.set_waiting.emit(False)

        except Exception as e:
            print(f"Error in planning LLM: {e}")
            import traceback

            traceback.print_exc()
            dialog.set_waiting.emit(False)

    def _generate_structured_outline(
        self, messages: list, dialog: PlanningModeDialog
    ) -> str:
        """Generate outline using structured output.

        Args:
            messages: Message history for LLM
            dialog: Dialog for signal emission

        Returns:
            Formatted outline as markdown
        """
        # JSON schema for structured output
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "story_outline",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "plot_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "A specific narrative event or action that happens in the story",
                                    },
                                    "completed": {
                                        "type": "boolean",
                                        "description": "True if this plot point has already been written in the existing story, False if it still needs to be written",
                                    },
                                },
                                "required": ["description", "completed"],
                                "additionalProperties": False,
                            },
                        },
                        "discussion": {
                            "type": ["string", "null"],
                            "description": "Optional discussion, questions, or explanatory text about the outline (but not the outline itself)",
                        },
                    },
                    "required": ["plot_points"],
                    "additionalProperties": False,
                },
            },
        }

        # Bind and invoke
        structured_llm = self.llm_controller.llm.bind(response_format=json_schema)
        response = structured_llm.invoke(messages)

        # Parse JSON
        import json

        result = json.loads(response.content)

        # Convert to markdown
        full_response = ""
        if result.get("discussion"):
            full_response = result["discussion"] + "\n\n"

        full_response += "Here's your outline:\n\n"
        for plot_point in result.get("plot_points", []):
            checkbox = "[x]" if plot_point.get("completed", False) else "[ ]"
            full_response += f"- {checkbox} {plot_point['description']}\n"

        # Emit as single block
        dialog.llm_token_received.emit(full_response)
        return full_response

    def _on_start_writing(self, outline: str):
        """Handle start writing from planning dialog.

        Args:
            outline: Markdown checklist outline
        """
        # Store outline
        self.story_model.planning_outline = outline
        self.story_model.planning_active = True
        self.planning_model.current_outline = outline

        print("✓ Planning mode completed")
        print(f"Outline stored:\n{outline}")

        # Gather context
        notes = self.view.prompts_panel.get_notes_text().strip()
        supp_text = self.view.prompts_panel.gather_supplemental_text()
        system_prompt = self.view.prompts_panel.get_system_prompt_text()

        # Start outline-driven generation
        self.start_outline_build(outline, notes, supp_text, system_prompt)

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

        # Show status
        self.view.clear_thinking_text()
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

        # Query RAG
        self.view.append_logs("🔍 Querying knowledge bases for task context...\n")
        rag_context = self.rag_controller.query_databases(current_task)
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
            self.view.append_thinking_text,
            lambda: self._on_chunk_complete(),
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

        # Check stop flag
        if self.llm_controller.llm_model.stop_generation:
            self._stop_build()
            return

        self.view.append_logs(f"\n✅ Chunk {state['chunk_count']} complete!\n")

        # Update story
        current_story = self.view.get_story_content()
        self.story_model.content = current_story

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
        self.view.append_logs("✅ PLANNING BUILD COMPLETE\n")
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
        self.view.append_logs("⏹️ PLANNING BUILD STOPPED BY USER\n")
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
        self.view.append_thinking_text(
            f"\n❌ Error during planning build: {error_msg}\n"
        )
        self.view.append_thinking_text("Planning build stopped.\n")
        self.view.set_stop_enabled(False)
        self.planning_model.is_active = False
