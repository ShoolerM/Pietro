"""Main controller that coordinates all components."""

import sys
import threading
from PyQt5 import QtWidgets, QtCore

from models.story_model import StoryModel
from models.settings_model import SettingsModel
from models.prompt_model import PromptModel
from models.llm_model import LLMModel
from models.rag_model import RAGModel
from models.stylesheets import MAIN_STYLE
from models.summary_model import SummaryModel
from views.main_view import MainView
from controllers.prompt_controller import PromptController
from controllers.llm_controller import LLMController
from controllers.rag_controller import RAGController
from controllers.context_controller import ContextController
from controllers.planning_controller import PlanningController
from controllers.settings_controller import SettingsController
from controllers.notes_controller import NotesController
from models.planning_model import PlanningModel
from models.model_context_database import detect_context_window

from pathlib import Path


class CondenserSignals(QtCore.QObject):
    """Signals for background prompt condensing."""

    condense_ready = QtCore.pyqtSignal(
        str, str, str
    )  # condensed supp_text, system_prompt, notes
    condense_error = QtCore.pyqtSignal(str)  # error_message
    thinking_update = QtCore.pyqtSignal(str)  # status message for LLM Panel


class MainController:
    """Main application controller that coordinates models, views, and controllers."""

    def __init__(self):
        """Initialize the main controller and all components."""
        # Create models
        self.settings_model = SettingsModel()
        self.story_model = StoryModel()
        self.prompt_model = PromptModel()
        self.llm_model = LLMModel(
            self.settings_model.base_url, self.settings_model.temperature
        )
        self.rag_model = RAGModel()
        self.summary_model = SummaryModel()
        self.planning_model = PlanningModel()

        # Create view
        self.view = MainView()

        # Create sub-controllers
        self.prompt_controller = PromptController(
            self.prompt_model, self.view, self.settings_model
        )
        self.llm_controller = LLMController(
            self.llm_model, self.story_model, self.settings_model
        )
        self.notes_controller = NotesController(
            self.settings_model, self.llm_controller, self.view
        )
        self.rag_controller = RAGController(self.rag_model, self.view)
        self.context_controller = ContextController(
            self.story_model,
            self.settings_model,
            self.summary_model,
            self.rag_controller,
            self.rag_model,
        )
        self.settings_controller = SettingsController(
            self.view,
            self.settings_model,
            self.llm_model,
            self.rag_model,
            self._save_current_model_profile,
            self._on_refresh_models,
        )
        self.planning_controller = PlanningController(
            self.planning_model,
            self.story_model,
            self.settings_model,
            self.llm_controller,
            self.rag_controller,
            self.rag_model,
            self.summary_model,
            self.notes_controller,
            self.view,
        )

        # Track markdown content for rendering
        self._markdown_content = ""

        # Connect view signals to handlers
        self._connect_signals()

        # Connect model observers
        self._connect_observers()

        # Initialize view state
        self._initialize_view()

        # Load normal conversation history into LLM panel (navigation only)
        try:
            self.view.llm_panel.set_normal_conversation(
                self.settings_model.get_normal_conversation()
            )
        except Exception:
            pass

    def _connect_signals(self):
        """Connect view signals to controller handlers."""
        self.view.send_clicked.connect(self._on_send)
        self.view.undo_clicked.connect(self._on_undo)
        self.view.stop_clicked.connect(self._on_stop)
        self.view.clear_clicked.connect(self._on_clear)
        self.view.model_refresh_clicked.connect(self._on_refresh_models)
        self.view.model_changed.connect(self._on_model_changed)
        self.view.context_limit_changed.connect(self._on_context_limit_changed)
        self.view.toggle_summarize_prompts_requested.connect(
            self._on_toggle_summarize_prompts
        )
        self.view.toggle_smart_mode_requested.connect(self._on_toggle_smart_mode)
        self.view.rag_create_database_clicked.connect(
            self.rag_controller.create_database
        )
        self.view.rag_add_files_clicked.connect(
            self.rag_controller.add_files_to_database
        )
        self.view.rag_database_toggled.connect(self.rag_controller.toggle_database)
        self.view.rag_delete_database_clicked.connect(
            self.rag_controller.delete_database
        )
        self.view.rag_max_chunks_changed.connect(self.rag_model.set_max_chunks)
        self.view.rag_summary_chunk_size_changed.connect(
            self.rag_model.set_summary_chunk_size
        )
        self.view.rag_score_threshold_changed.connect(
            self.rag_model.set_score_variance_threshold
        )
        self.view.rag_settings_requested.connect(
            self.settings_controller.on_rag_settings_requested
        )
        self.view.prompt_selections_changed.connect(self._on_prompt_selections_changed)
        self.view.summarization_prompt_requested.connect(
            self.settings_controller.on_summarization_prompt_requested
        )
        self.view.notes_prompt_requested.connect(
            self.settings_controller.on_notes_prompt_requested
        )
        self.view.general_settings_requested.connect(
            self.settings_controller.on_general_settings_requested
        )
        self.view.model_settings_requested.connect(
            self.settings_controller.on_model_settings_requested
        )
        self.view.mode_changed.connect(self._on_mode_changed)
        self.view.inference_settings_requested.connect(
            self.settings_controller.on_inference_settings_requested
        )
        self.view.update_summary_requested.connect(self._on_update_summary_requested)
        self.view.auto_build_story_requested.connect(
            self._on_auto_build_story_requested
        )
        self.view.override_selection_requested.connect(self._on_override_selection)
        self.view.update_selection_with_prompt_requested.connect(
            self._on_update_selection_with_prompt
        )
        self.view.update_accepted.connect(self._on_update_accepted)
        self.view.update_rejected.connect(self._on_update_rejected)
        self.view.llm_panel.start_writing_requested.connect(
            self._on_start_writing_from_planning
        )

    def _connect_observers(self):
        """Connect model observers to update view."""
        self.settings_model.add_observer(self._on_settings_changed)
        self.story_model.add_observer(self._on_story_changed)
        self.llm_model.add_observer(self._on_llm_changed)

    def _initialize_view(self):
        """Initialize view with current model state."""
        # Set initial font size
        self.view.apply_font_size(self.settings_model.current_font_size)

        # Load initial models
        self._on_refresh_models()

        # Load initial RAG databases
        self.rag_controller.refresh_databases()

        # Sync summarize prompts toggle UI
        try:
            self.view.set_summarize_prompts_enabled(
                self.settings_model.summarize_prompts
            )
        except Exception:
            pass

        # Sync build with RAG toggle UI
        try:
            self.view.set_smart_mode(self.settings_model.smart_mode)
        except Exception:
            pass

    def _condense_prompts_background(self):
        """Condense prompts in background thread to avoid UI blocking."""
        if not hasattr(self, "_pending_condense_context"):
            return

        ctx = self._pending_condense_context

        # Create signals object
        signals = CondenserSignals()

        # Connect signals to UI update handlers
        signals.thinking_update.connect(self.view.append_logs)
        signals.condense_ready.connect(self._on_condense_ready)
        signals.condense_error.connect(self._on_condense_error)

        def condense_in_thread():
            try:
                condensed_supp = ctx["supp_text"]
                condensed_system = ctx["system_prompt"]
                condensed_notes = ctx["notes"]

                # Condense supplemental prompts if needed
                if ctx["supp_tokens"] > ctx["max_supp_tokens"] and condensed_supp:
                    signals.thinking_update.emit(
                        "📎 Condensing supplemental prompts...\n"
                    )
                    condensed_supp, _ = self.llm_controller.summarize_supplemental(
                        condensed_supp, ctx["max_supp_tokens"]
                    )
                    signals.thinking_update.emit("  ✓ Supplemental prompts condensed\n")

                # Condense system prompt if needed
                if ctx["system_tokens"] > ctx["max_system_tokens"] and condensed_system:
                    signals.thinking_update.emit("🔧 Condensing system prompt...\n")
                    condensed_system, _ = self.llm_controller.summarize_system_prompt(
                        condensed_system, ctx["max_system_tokens"]
                    )
                    signals.thinking_update.emit("  ✓ System prompt condensed\n")

                # Condense notes if needed
                if ctx["notes_tokens"] > ctx["max_notes_tokens"] and condensed_notes:
                    signals.thinking_update.emit("📝 Condensing notes...\n")
                    condensed_notes, _ = self.llm_controller.summarize_supplemental(
                        condensed_notes, ctx["max_notes_tokens"]
                    )
                    signals.thinking_update.emit("  ✓ Notes condensed\n\n")

                signals.condense_ready.emit(
                    condensed_supp, condensed_system, condensed_notes
                )
            except Exception as e:
                self.view.append_logs(f"Error condensing prompts: {e}")
                signals.condense_error.emit(str(e))

        # Start background thread
        thread = threading.Thread(target=condense_in_thread, daemon=True)
        thread.start()

    def _on_condense_ready(self, condensed_supp, condensed_system, condensed_notes):
        """Handle prompt condensing completion (called on main thread via signal)."""
        if not hasattr(self, "_pending_condense_context"):
            self.view.set_waiting(False)
            return

        ctx = self._pending_condense_context
        delattr(self, "_pending_condense_context")

        # Continue with story generation using condensed prompts
        self._continue_send(
            ctx["user_input"], condensed_notes, condensed_supp, condensed_system
        )

    def _on_condense_error(self, error_msg):
        """Handle prompt condensing error (called on main thread via signal)."""
        self.view.append_logs(f"  ⚠ Error condensing prompts: {error_msg}\n\n")
        self.view.set_waiting(False)

        # Continue with original uncondensed prompts
        if hasattr(self, "_pending_condense_context"):
            ctx = self._pending_condense_context
            delattr(self, "_pending_condense_context")
            self._continue_send(
                ctx["user_input"], ctx["notes"], ctx["supp_text"], ctx["system_prompt"]
            )

    def _handle_notes_ready(self, generated_notes, notes_tokens):
        """Handle notes completion and continue pending flows."""
        if hasattr(self, "_pending_notes_context"):
            ctx = self._pending_notes_context
            delattr(self, "_pending_notes_context")
            self._continue_send(
                ctx["user_input"],
                generated_notes,
                ctx["supp_text"],
                ctx["system_prompt"],
            )
            return

        if hasattr(self, "_pending_auto_build_context"):
            ctx = self._pending_auto_build_context
            delattr(self, "_pending_auto_build_context")
            self._continue_auto_build(
                ctx["initial_prompt"],
                generated_notes,
                ctx["supp_text"],
                ctx["system_prompt"],
            )
            return

        if hasattr(self, "_pending_planning_build_context"):
            ctx = self._pending_planning_build_context
            delattr(self, "_pending_planning_build_context")
            self._start_planning_build(
                ctx["outline"],
                generated_notes,
                ctx["supp_text"],
                ctx["system_prompt"],
            )
            return

        if hasattr(self, "_pending_planning_notes_continue"):
            delattr(self, "_pending_planning_notes_continue")
            try:
                if hasattr(self, "_planning_build_state"):
                    self._planning_build_state["notes"] = generated_notes
            except Exception:
                pass
            self.view.set_waiting(False)
            self._generate_next_planning_chunk()
            return

        self.view.set_waiting(False)

    def _handle_notes_error(self, error_msg):
        """Handle notes generation errors for pending flows."""
        self.view.set_waiting(False)

    def _on_settings_changed(self, event_type, data):
        """Handle settings model changes."""
        if event_type == "font_size_changed":
            self.view.apply_font_size(data)
        elif event_type == "context_limit_changed":
            # Context limit is already updated in the model
            pass
        elif event_type == "summarize_prompts_changed":
            # Update UI to reflect new setting
            try:
                self.view.set_summarize_prompts_enabled(data)
            except Exception:
                pass
        elif event_type == "smart_mode_changed":
            # Update UI to reflect new setting
            try:
                self.view.set_smart_mode(data)
            except Exception:
                pass
        elif event_type == "render_markdown_changed":
            # Update story panel markdown rendering
            try:
                self.view.story_panel.set_markdown_enabled(data)
                # Re-render story if markdown enabled
                if data and self._markdown_content:
                    self.view.render_story_markdown(self._markdown_content)
            except Exception:
                pass

    def _on_story_changed(self, event_type, data):
        """Handle story model changes."""
        if event_type == "content_changed":
            self._markdown_content = data
            self.view.set_story_content(data)
        elif event_type == "content_appended":
            self._markdown_content += data
        elif event_type == "content_cleared":
            self._markdown_content = ""
            self.view.clear_story_content()
        elif event_type == "content_restored":
            self._markdown_content = data
            self.view.set_story_content(data)

    def _on_llm_changed(self, event_type, data):
        """Handle LLM model changes."""
        if event_type == "models_fetched":
            self.view.set_models(data, self.settings_model.last_model)
            # Load profile for the restored last model
            if self.settings_model.last_model:
                profile = self.settings_model.get_model_profile(
                    self.settings_model.last_model, self.settings_model.base_url
                )
                if profile:
                    self._apply_model_profile(profile)
                # Update the LLM controller with the last model
                self.llm_controller.update_model(self.settings_model.last_model)
        elif event_type == "model_changed":
            # Model already updated
            pass

    def _on_send(self, user_input, notes, supp_text, system_prompt):
        """Handle send button click with hierarchical chunking and rolling summarization.

        Args:
            user_input: User's query input
            notes: Author's notes
            supp_text: Supplemental prompts text
            system_prompt: System prompt text
        """
        # Save user message to normal conversation history
        try:
            if not self.view.llm_panel.is_planning_mode():
                self.settings_model.append_normal_message("user", user_input)
        except Exception:
            pass

        # Check if in Planning mode
        if self.view.llm_panel.is_planning_mode():
            # Handle planning conversation
            self._handle_planning_conversation(user_input)
            return

        # Check if Build with Smart Mode is enabled
        if self.settings_model.smart_mode:
            # Trigger auto-build story mode with the user's input and context
            self._on_auto_build_story_requested(
                user_input, notes, supp_text, system_prompt
            )
            return

        # Check if notes should be regenerated
        story_context = self.view.get_story_content()
        should_regen = self.notes_controller.should_regenerate_notes(
            story_context,
            self.settings_model.auto_notes,
            self.view.notes_panel.should_regenerate_notes(),
            context_key="main",
        )

        if should_regen:
            self._pending_notes_context = {
                "user_input": user_input,
                "notes": notes,
                "supp_text": supp_text,
                "system_prompt": system_prompt,
            }

            self.notes_controller.generate_notes_async(
                story_context,
                on_complete=self._handle_notes_ready,
                on_error=self._handle_notes_error,
                clear_existing=True,
                set_waiting_on_finish=False,
            )

            return

        # Continue with story generation
        self._continue_send(user_input, notes, supp_text, system_prompt)

    def _handle_planning_conversation(self, user_input):
        """Handle user input during planning mode.

        Args:
            user_input: User's message
        """
        # Add user message to planning conversation
        self.view.llm_panel.add_planning_message("user", user_input)

        # Save conversation
        self.settings_model.save_planning_conversation(
            self.view.llm_panel.get_planning_conversation(),
            self.view.llm_panel.get_current_outline(),
        )

        # Clear the input field
        self.view.llm_panel.clear_user_input()

        # Delegate to planning controller for LLM response
        self.planning_controller.handle_planning_message(
            user_input,
            self.view.llm_panel.get_planning_conversation(),
            self.view.llm_panel.get_current_outline(),
        )

    def _continue_send(self, user_input, notes, supp_text, system_prompt):
        """Continue with story generation (after notes are ready or if not needed)."""
        # Reset stop flag and enable stop button
        self.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)

        # Reset response buffer for normal mode history
        self._current_llm_response = ""

        # Sync markdown content with any user edits
        current_story = self.view.get_story_content()
        self._markdown_content = current_story
        self.story_model.content = current_story

        # Switch to plain text mode for streaming
        self.view.set_story_content(current_story)

        # Calculate context budget
        context_limit = self.settings_model.context_limit

        # Estimate initial token counts
        supp_tokens = self.story_model.estimate_token_count(supp_text)
        notes_tokens = self.story_model.estimate_token_count(notes)
        user_tokens = self.story_model.estimate_token_count(user_input)
        system_tokens = self.story_model.estimate_token_count(system_prompt)
        safety_buffer = 500

        # Check if we need to condense supplemental/system/notes
        # Derive limits from context limit so they scale with user setting
        max_supp_tokens = max(256, int(context_limit * 0.15))
        max_system_tokens = max(256, int(context_limit * 0.15))
        max_notes_tokens = max(128, int(context_limit * 0.1))

        needs_condensing = False

        if supp_tokens > max_supp_tokens and supp_text:
            needs_condensing = True
            self.view.append_logs(
                f"\n⚠️ Supplemental prompts too large ({supp_tokens} > {max_supp_tokens} tokens)\n"
            )

        if system_tokens > max_system_tokens and system_prompt:
            needs_condensing = True
            self.view.append_logs(
                f"⚠️ System prompt too large ({system_tokens} > {max_system_tokens} tokens)\n"
            )

        if notes_tokens > max_notes_tokens and notes:
            needs_condensing = True
            self.view.append_logs(
                f"⚠️ Notes too large ({notes_tokens} > {max_notes_tokens} tokens)\n"
            )

        if needs_condensing:
            if self.settings_model.summarize_prompts:
                self.view.append_logs("🔄 Condensing oversized context elements...\n\n")
                self.view.set_waiting(True)

                # Store context for continuation after condensing
                self._pending_condense_context = {
                    "user_input": user_input,
                    "notes": notes,
                    "supp_text": supp_text,
                    "system_prompt": system_prompt,
                    "supp_tokens": supp_tokens,
                    "system_tokens": system_tokens,
                    "notes_tokens": notes_tokens,
                    "max_supp_tokens": max_supp_tokens,
                    "max_system_tokens": max_system_tokens,
                    "max_notes_tokens": max_notes_tokens,
                }

                # Condense in background thread to avoid UI blocking
                self._condense_prompts_background()
                return
            else:
                # Summarization disabled - inform the user we're skipping condensing
                self.view.append_logs(
                    "⚠️ Prompt summarization disabled; skipping condensing of oversized prompts.\n"
                )

        fixed_costs = (
            supp_tokens + notes_tokens + user_tokens + system_tokens + safety_buffer
        )

        # Calculate available space for story context
        available_for_story = context_limit - fixed_costs

        # Reserve space for rolling summary (up to 1000 tokens)
        max_rolling_summary_tokens = min(1000, int(available_for_story * 0.4))

        # Remaining space for raw recent content
        max_raw_tokens = min(
            self.rag_model.summary_chunk_size,
            available_for_story - max_rolling_summary_tokens,
        )

        if max_raw_tokens < 0:
            max_raw_tokens = 500  # Emergency minimum

        # Get current story token count
        story_tokens = self.story_model.estimate_token_count(current_story)

        # Determine if we need chunking/summarization
        needs_chunking = story_tokens > max_raw_tokens and current_story

        if needs_chunking:
            self.view.append_logs(f"\n{'=' * 60}\n")
            self.view.append_logs("📊 HIERARCHICAL SUMMARIZATION ACTIVE\n")
            self.view.append_logs(
                f"Story tokens: {story_tokens} | Context limit: {context_limit}\n"
            )
            self.view.append_logs(f"Max raw content: {max_raw_tokens} tokens\n")
            self.view.append_logs(
                f"Max rolling summary: {max_rolling_summary_tokens} tokens\n"
            )
            self.view.append_logs(f"{'=' * 60}\n\n")

            self.view.set_waiting(True)

            # Store context for the completion callback
            self._pending_send_context = {
                "user_input": user_input,
                "notes": notes,
                "supp_text": supp_text,
                "system_prompt": system_prompt,
                "current_story": current_story,
            }

            # Process in background thread
            self.llm_controller.process_story_with_summarization(
                current_story,
                max_raw_tokens,
                max_rolling_summary_tokens,
                self.summary_model,
                self.view.append_logs,
                self._on_summarization_complete,
                self._on_summarization_error,
                self.view.set_waiting,
            )
            # Return early - completion will trigger the actual LLM call
            return
        else:
            # Story fits entirely in context
            story_for_llm = current_story

        # Build final query
        if story_for_llm:
            final_query = (
                "Based on this story so far:\n```\n",
                story_for_llm,
                "\n```\nthe following should happen next (user input):\n",
                user_input,
            )
            final_query = "".join([p for p in final_query if p is not None])
        else:
            final_query = user_input

        # Prepend planning outline if active
        if self.story_model.planning_active and self.story_model.planning_outline:
            outline_context = f"""STORY OUTLINE (must address all tasks):
{self.story_model.planning_outline}

"""
            final_query = outline_context + final_query
            self.view.append_logs("\n📋 Using planning outline as context\n")

        # Save to history before appending new content
        self.story_model.save_to_history()

        # Calculate dynamic RAG token budget (30% of available context after fixed costs)
        context_limit = self.settings_model.context_limit
        user_tokens = self.story_model.estimate_token_count(user_input)
        supp_tokens = (
            self.story_model.estimate_token_count(supp_text) if supp_text else 0
        )
        notes_tokens = self.story_model.estimate_token_count(notes) if notes else 0
        system_tokens = (
            self.story_model.estimate_token_count(system_prompt) if system_prompt else 0
        )
        output_reserve = 2000  # Reserve tokens for model output

        fixed_costs = (
            user_tokens + supp_tokens + notes_tokens + system_tokens + output_reserve
        )
        available_for_rag_and_story = context_limit - fixed_costs

        # Allocate 30% for RAG, rest for story
        max_rag_tokens = int(available_for_rag_and_story * 0.3)
        max_rag_tokens = max(500, min(max_rag_tokens, 4000))  # Clamp between 500-4000

        # Query RAG databases with dynamic token budget
        rag_context = self.rag_controller.query_databases(
            user_input, max_tokens=max_rag_tokens
        )
        if rag_context:
            rag_tokens = self.story_model.estimate_token_count(rag_context)
            final_query = (
                final_query
                + "\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
                + rag_context
            )
            self.view.append_logs(
                f"\n🔍 Including RAG context ({rag_tokens:,}/{max_rag_tokens:,} tokens)\n"
            )

        # Always append supplemental text
        if supp_text:
            final_query = final_query + "\n\n" + supp_text
            self.view.append_logs(
                f"\n📎 Including {len(supp_text)} chars of supplemental prompts\n"
            )

        # Always append notes text
        if notes:
            final_query = final_query + "\n\nAUTHOR'S NOTES (for context):\n" + notes
            self.view.append_logs(f"📝 Including {len(notes)} chars of author notes\n")

        # Show system prompt info
        if system_prompt:
            self.view.append_logs(
                f"🔧 Using system prompt ({len(system_prompt)} chars)\n"
            )

        # Start waiting animation
        self.view.set_waiting(True)

        # Add newline before response
        if current_story:
            self.view.append_story_content("\n")

        # Invoke LLM in background
        self.llm_controller.invoke_llm(
            final_query,
            system_prompt,
            self._on_text_appended,
            self.view.append_llm_panel_text,
            self._on_render_markdown,
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _on_text_appended(self, text):
        """Handle text appended from LLM stream."""
        try:
            if hasattr(self, "_current_llm_response"):
                self._current_llm_response += text
        except Exception:
            pass
        self.story_model.append_content(text)
        self.view.append_story_content(text)

    def _on_render_markdown(self):
        """Handle markdown rendering request."""
        self.view.render_story_markdown(self._markdown_content)

    def _update_markdown_content(self, current_story):
        """Sync markdown content cache with current story text."""
        self._markdown_content = current_story

    def _render_story_markdown_from_view(self):
        """Render markdown using current story content from the view."""
        self._markdown_content = self.view.get_story_content()
        self.view.render_story_markdown(self._markdown_content)

    def _on_summarization_complete(self, story_for_llm, story_context_tokens):
        """Handle completion of background summarization.

        Args:
            story_for_llm: Processed story context for LLM
            story_context_tokens: Token count
        """
        # Auto-save summary state
        self.save_summary_state()

        # Retrieve the pending context
        if not hasattr(self, "_pending_send_context"):
            self.view.append_logs("\n❌ Error: Lost context during summarization\n")
            self.view.set_waiting(False)
            return

        ctx = self._pending_send_context
        user_input = ctx["user_input"]
        notes = ctx["notes"]
        supp_text = ctx["supp_text"]
        system_prompt = ctx["system_prompt"]
        current_story = ctx["current_story"]

        # Build final query
        if story_for_llm:
            final_query = (
                "Based on this story so far:\n```\n",
                story_for_llm,
                "\n```\nthe following should happen next (user input):\n",
                user_input,
            )
            final_query = "".join([p for p in final_query if p is not None])
        else:
            final_query = user_input

        # Save to history before appending new content
        self.story_model.save_to_history()

        # Query RAG databases for relevant context
        rag_context = self.rag_controller.query_databases(user_input)
        if rag_context:
            rag_tokens = self.story_model.estimate_token_count(rag_context)
            max_rag_tokens = 600

            # Check if RAG context is too large
            if rag_tokens > max_rag_tokens:
                self.view.append_logs(
                    f"\n⚠️ RAG context too large ({rag_tokens} > {max_rag_tokens} tokens)\n"
                )
                self.view.append_logs("🔄 Condensing RAG context...\n")

                rag_context, rag_tokens = self.llm_controller.summarize_rag_context(
                    rag_context, max_rag_tokens
                )

                self.view.append_logs(f"  ✓ Reduced to {rag_tokens} tokens\n")

            final_query = (
                final_query
                + "\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
                + rag_context
            )
            self.view.append_logs(f"\n🔍 Including RAG context ({rag_tokens} tokens)\n")

        # Always append supplemental text
        if supp_text:
            final_query = final_query + "\n\n" + supp_text
            self.view.append_logs(
                f"\n📎 Including {len(supp_text)} chars of supplemental prompts\n"
            )

        # Always append notes text
        if notes:
            final_query = final_query + "\n\nAUTHOR'S NOTES (for context):\n" + notes
            self.view.append_logs(f"📝 Including {len(notes)} chars of author notes\n")

        # Show system prompt info
        if system_prompt:
            self.view.append_logs(
                f"🔧 Using system prompt ({len(system_prompt)} chars)\n"
            )

        # Start waiting animation
        self.view.set_waiting(True)

        # Add newline before response
        if current_story:
            self.view.append_story_content("\n")

        # Invoke LLM in background
        self.llm_controller.invoke_llm(
            final_query,
            system_prompt,
            self._on_text_appended,
            self.view.append_llm_panel_text,
            self._on_render_markdown,
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _on_summarization_error(self, error_message):
        """Handle error during background summarization.

        Args:
            error_message: Error description
        """
        self.view.append_logs(f"\n❌ Chunking/summarization error: {error_message}\n")
        self.view.append_logs("Falling back to recent content only...\n\n")

        # Get recent content as fallback
        if not hasattr(self, "_pending_send_context"):
            self.view.set_waiting(False)
            return

        ctx = self._pending_send_context
        current_story = ctx["current_story"]

        # Calculate how much we can fit
        context_limit = self.settings_model.context_limit
        supp_tokens = self.story_model.estimate_token_count(ctx["supp_text"])
        notes_tokens = self.story_model.estimate_token_count(ctx["notes"])
        user_tokens = self.story_model.estimate_token_count(ctx["user_input"])
        system_tokens = self.story_model.estimate_token_count(ctx["system_prompt"])
        fixed_costs = supp_tokens + notes_tokens + user_tokens + system_tokens + 500
        max_raw_tokens = min(
            self.rag_model.summary_chunk_size, context_limit - fixed_costs
        )

        story_for_llm, _ = self.story_model.extract_recent_content(
            current_story, max_raw_tokens
        )

        # Continue with the send process
        self._on_summarization_complete(story_for_llm, max_raw_tokens)

    def save_summary_state(self):
        """Save the current summary state to disk."""
        try:
            settings_dir = Path("settings")
            settings_dir.mkdir(exist_ok=True)
            summary_file = settings_dir / "story_summary_state.json"
            self.summary_model.save_to_file(str(summary_file))
        except Exception as e:
            self.view.append_logs(f"Warning: Could not save summary state: {e}")

    def load_summary_state(self):
        """Load the summary state from disk."""
        try:
            settings_dir = Path("settings")
            summary_file = settings_dir / "story_summary_state.json"
            if summary_file.exists():
                self.summary_model.load_from_file(str(summary_file))
        except Exception as e:
            self.view.append_logs(f"Warning: Could not load summary state: {e}")

    def _on_undo(self):
        """Handle undo button click."""
        if self.story_model.undo():
            # Story model will notify observers and update view
            pass

    def _on_stop(self):
        """Handle stop button click."""
        self.llm_model.request_stop()
        self.view.set_stop_enabled(False)
        self.view.append_story_content("\n[Generation stopped by user]\n")

    def _on_clear(self):
        """Handle clear button click."""
        self.story_model.clear_content()
        self.summary_model.clear()
        # Clear planning mode state
        self.story_model.planning_outline = ""
        self.story_model.planning_active = False
        # Reset story hash tracker so next generation doesn't think story changed
        self.notes_controller.reset_story_hash()

    def _on_refresh_models(self):
        """Handle refresh models button click."""
        success, result = self.llm_model.fetch_available_models()
        if not success:
            self.view.append_logs(result)

    def _on_model_changed(self, model_name):
        """Handle model selection change."""
        # Save profile for current model before switching
        self._save_current_model_profile()

        # Persist last selected model
        if model_name:
            self.settings_model.last_model = model_name

        # Apply profile settings for the newly selected model (if any)
        profile = self.settings_model.get_model_profile(
            model_name, self.settings_model.base_url
        )
        if profile:
            self._apply_model_profile(profile)
        else:
            # No saved profile - try auto-detection
            try:
                context_length, confidence, source = detect_context_window(model_name)

                # Update settings and UI with detected value
                self.settings_model.context_limit = context_length
                self.view.set_context_limit(context_length)

                # Save the detected value to profile for future use
                self._save_current_model_profile()
            except Exception as e:
                self.view.append_logs(
                    f"Warning: Could not auto-detect context window: {e}"
                )
                # Fall back to existing behavior (keep current value)

        # Update the LLM model
        self.llm_controller.update_model(model_name)

    def _on_context_limit_changed(self, limit):
        """Handle context limit change."""
        self.settings_model.context_limit = limit
        self._save_current_model_profile()

    def _save_current_model_profile(self):
        """Persist current settings for the active model + base URL."""
        try:
            model_name = self.llm_model.current_model
            if not model_name:
                return

            self.settings_model.save_model_profile(
                model_name=model_name,
                base_url=self.settings_model.base_url,
                context_limit=self.settings_model.context_limit,
                inference_ip=self.settings_model.inference_ip,
                inference_port=self.settings_model.inference_port,
            )
        except Exception:
            pass

    def _apply_model_profile(self, profile):
        """Apply a saved model profile to current settings."""
        try:
            # Apply inference settings if they differ
            profile_ip = profile.get("inference_ip")
            profile_port = profile.get("inference_port")
            if profile_ip and profile_port:
                self.settings_model.inference_ip = profile_ip
                self.settings_model.inference_port = profile_port
                self.llm_model.base_url = self.settings_model.base_url

            # Apply context limit
            context_limit = profile.get("context_limit")
            if context_limit:
                self.settings_model.context_limit = context_limit
                self.view.set_context_limit(context_limit)
        except Exception:
            pass

    def _on_prompt_selections_changed(self, supplemental_files, system_prompt):
        """Handle prompt selection changes from the view.

        Args:
            supplemental_files: List of selected supplemental file paths
            system_prompt: Selected system prompt file path
        """
        try:
            self.settings_model.set_selected_supplemental_files(supplemental_files)
            self.settings_model.set_selected_system_prompt(system_prompt)
        except Exception as e:
            print(f"Error saving prompt selections: {e}")

    def _on_toggle_summarize_prompts(self):
        """Toggle the summarize prompts setting (triggered from UI)."""
        try:
            new_val = not self.settings_model.summarize_prompts
            self.settings_model.summarize_prompts = new_val
        except Exception:
            pass

    def _on_toggle_smart_mode(self):
        """Toggle the build with RAG setting (triggered from UI)."""
        try:
            new_val = not self.settings_model.smart_mode
            self.settings_model.smart_mode = new_val
        except Exception:
            pass

    def _on_override_selection(self, selected_text, start_pos, end_pos):
        """Handle override selection request.

        Args:
            selected_text: The bottom_text that was selected
            start_pos: Start position of selection
            end_pos: End position of selection
        """
        # Get current user input from LLM panel
        user_input = self.view.llm_panel.get_user_input().strip()

        if not user_input:
            self.view.show_warning(
                "No Prompt",
                "Please enter a prompt in the input box before overriding text.",
            )
            return

        # Get system prompt
        system_prompt = self.view.utilities_panel.get_system_prompt_text()

        # Reset stop flag and enable stop button
        self.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)

        # Save story to history for undo
        current_story = self.view.get_story_content()
        self._markdown_content = current_story
        self.story_model.content = current_story
        self.story_model.save_to_history()

        # Build query for text override
        query = f"""Rewrite the following text according to the instruction.

TEXT TO REWRITE:
{selected_text}

INSTRUCTION:
{user_input}

REWRITTEN VERSION (output only the rewritten text, nothing else):"""

        # Initialize streaming replacement
        self.view.start_text_override(start_pos, end_pos)

        # Invoke LLM with streaming override
        self.llm_controller.override_text_with_streaming(
            query=query,
            system_prompt=system_prompt,
            stream_callback=self.view.stream_override_text,
            completion_callback=self._on_update_complete,
            set_stop_enabled_callback=self.view.set_stop_enabled,
        )

    def _on_update_selection_with_prompt(
        self, selected_text, start_pos, end_pos, prompt
    ):
        """Handle update selection request with prompt from dialog.

        Args:
            selected_text: The text that was selected
            start_pos: Start position of selection
            end_pos: End position of selection
            prompt: The change instruction from dialog
        """
        # Get system prompt
        system_prompt = self.view.utilities_panel.get_system_prompt_text()

        # Reset stop flag and enable stop button
        self.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)
        self.view.set_waiting(True)

        # Save story to history for undo
        current_story = self.view.get_story_content()
        self._markdown_content = current_story
        self.story_model.content = current_story
        self.story_model.save_to_history()

        # Gather context using context controller
        context = self.context_controller.gather_context_for_edit(
            selected_text, start_pos, end_pos, prompt
        )

        # Build query with surrounding context and RAG
        query = "Rewrite the following text according to the instruction."

        # Add context before if available
        if context["context_before"]:
            query += f"""

CONTEXT BEFORE (do not modify this):
{context["context_before"]}"""

        query += f"""

TEXT TO REWRITE:
{selected_text}"""

        # Add context after if available
        if context["context_after"]:
            query += f"""

CONTEXT AFTER (do not modify this):
{context["context_after"]}"""

        query += f"""

INSTRUCTION:
{prompt}"""

        # Add RAG context if available
        if context["rag_context"]:
            query += f"""

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{context["rag_context"]}"""

        # Add notes if available
        if context["notes"]:
            query += f"""

ADDITIONAL CONTEXT (author's notes):
{context["notes"]}"""

        query += """

REWRITTEN VERSION (output only the rewritten text, nothing else):"""

        # Initialize streaming replacement
        self.view.start_text_update(start_pos, end_pos)

        # Invoke LLM with streaming update
        self.llm_controller.override_text_with_streaming(
            query=query,
            system_prompt=system_prompt,
            stream_callback=self.view.stream_override_text,
            completion_callback=self._on_update_complete,
            set_stop_enabled_callback=self.view.set_stop_enabled,
        )

    def _on_update_complete(self):
        """Handle completion of text update operation."""
        # Finalize the update (show accept/reject UI)
        self.view.finish_text_update()
        self.view.set_waiting(False)
        # Note: Story model will be updated when user accepts the change

    def _on_update_accepted(self):
        """Handle user accepting the override."""
        # Update story model with new content
        new_story = self.view.get_story_content()
        self._markdown_content = new_story
        self.story_model.content = new_story

    def _on_update_rejected(self):
        """Handle user rejecting the override."""
        # Story text already restored by view, just sync model
        restored_story = self.view.get_story_content()
        self._markdown_content = restored_story
        self.story_model.content = restored_story

    def _on_update_summary_requested(self):
        """Handle request to regenerate story summary after user edits.

        This clears the existing summary and re-processes the current story
        to rebuild the rolling summary from scratch.
        """
        # Get current story content
        current_story = self.view.get_story_content()

        if not current_story or len(current_story.strip()) == 0:
            return

        # Clear existing summary
        self.summary_model.clear()

        # Show notification in LLM Panel
        self.view.append_logs(f"\n{'=' * 60}\n")
        self.view.append_logs("🔄 REGENERATING STORY SUMMARY\n")
        self.view.append_logs("Processing entire story from scratch...\n")
        self.view.append_logs(f"{'=' * 60}\n\n")

        # Calculate context budget
        context_limit = self.settings_model.context_limit

        # Use conservative estimates for fixed costs
        safety_buffer = 2000  # Large buffer since we don't know what user will send

        # Calculate available space for story context
        available_for_story = context_limit - safety_buffer

        # Reserve space for rolling summary
        max_rolling_summary_tokens = min(1000, int(available_for_story * 0.4))

        # Remaining space for raw recent content
        max_raw_tokens = min(
            self.rag_model.summary_chunk_size,
            available_for_story - max_rolling_summary_tokens,
        )

        if max_raw_tokens < 0:
            max_raw_tokens = 500  # Emergency minimum

        # Get story token count
        story_tokens = self.story_model.estimate_token_count(current_story)

        self.view.append_logs(f"Story size: {story_tokens} tokens\n")
        self.view.append_logs(f"Target raw content: {max_raw_tokens} tokens\n")
        self.view.append_logs(
            f"Target rolling summary: {max_rolling_summary_tokens} tokens\n\n"
        )

        # Start processing
        self.view.set_waiting(True)

        # Process in background thread
        def on_complete(story_for_llm, tokens):
            self.view.append_logs("\n✅ Summary regeneration complete!\n")
            self.view.append_logs("Summary will be used for next generation.\n")
            self.view.set_waiting(False)

        def on_error(error_msg):
            self.view.append_logs(f"\n❌ Error regenerating summary: {error_msg}\n")
            self.view.set_waiting(False)

        self.llm_controller.process_story_with_summarization(
            current_story,
            max_raw_tokens,
            max_rolling_summary_tokens,
            self.summary_model,
            self.view.append_logs,
            on_complete,
            on_error,
            self.view.set_waiting,
        )

    def _on_auto_build_story_requested(
        self, initial_prompt=None, notes=None, supp_text=None, system_prompt=None
    ):
        """Handle request to automatically build a complete story with iterative RAG and summarization.

        This mode:
        1. Uses provided prompt or prompts user for an initial story prompt
        2. Generates story in chunks (3 paragraphs at a time)
        3. After each chunk: re-runs RAG with latest content + initial prompt
        4. Every 2-3 chunks: summarizes older content to maintain context window
        5. Continues until user stops or a reasonable story length is reached

        Args:
            initial_prompt: Optional initial prompt. If None, prompts user via dialog.
            notes: Optional author's notes. If None, will gather from view.
            supp_text: Optional supplemental text. If None, will gather from view.
            system_prompt: Optional system prompt. If None, will gather from view.
        """
        # Get initial prompt from parameter or user via dialog
        if initial_prompt is None:
            initial_prompt, ok = QtWidgets.QInputDialog.getText(
                self.view,
                "Auto Build Story with RAG",
                "Enter the initial story prompt:\n(This will guide the entire story generation)",
                QtWidgets.QLineEdit.Normal,
                "",
            )

            if not ok or not initial_prompt.strip():
                return

            initial_prompt = initial_prompt.strip()
        elif not initial_prompt.strip():
            # If provided but empty, abort
            return

        # Get other context elements - use provided values or gather from view
        if notes is None:
            notes = self.view.notes_panel.get_notes_text().strip()
        if supp_text is None:
            supp_text = self.view.utilities_panel.gather_supplemental_text()
        if system_prompt is None:
            system_prompt = self.view.utilities_panel.get_system_prompt_text()

        # Check if notes should be regenerated
        story_context = self.view.get_story_content()
        should_regen = self.notes_controller.should_regenerate_notes(
            story_context,
            self.settings_model.auto_notes,
            self.view.notes_panel.should_regenerate_notes(),
            context_key="main",
        )

        if should_regen:
            self._pending_auto_build_context = {
                "initial_prompt": initial_prompt,
                "notes": notes,
                "supp_text": supp_text,
                "system_prompt": system_prompt,
            }

            self.notes_controller.generate_notes_async(
                story_context,
                on_complete=self._handle_notes_ready,
                on_error=self._handle_notes_error,
                clear_existing=True,
                set_waiting_on_finish=False,
            )

            return

        # Continue with auto-build mode
        self._continue_auto_build(initial_prompt, notes, supp_text, system_prompt)

    def _continue_auto_build(self, initial_prompt, notes, supp_text, system_prompt):
        """Continue with auto-build mode (after notes are ready or if not needed)."""
        # Reset stop flag and enable stop button
        self.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)

        # Clear LLM Panel and provide instructions
        self.view.clear_thinking_text()
        self.view.append_logs(f"\n{'=' * 60}\n")
        self.view.append_logs("🤖 AUTO STORY BUILD MODE ACTIVATED\n")
        self.view.append_logs(f"{'=' * 60}\n\n")
        self.view.append_logs(f"Initial Prompt: {initial_prompt[:100]}...\n\n")
        self.view.append_logs("Configuration:\n")
        self.view.append_logs("  • Chunk size: 3 paragraphs\n")
        self.view.append_logs("  • Summarize every: 2-3 chunks\n")
        self.view.append_logs("  • Max chunks: 10 (configurable)\n")
        self.view.append_logs("  • RAG: Enabled (refresh after each chunk)\n\n")
        self.view.append_logs("Press STOP to end generation at any time.\n")
        self.view.append_logs(f"{'=' * 60}\n\n")

        # Sync markdown content with any user edits
        current_story = self.view.get_story_content()
        self._markdown_content = current_story
        self.story_model.content = current_story

        # Switch to plain text mode for streaming
        self.view.set_story_content(current_story)

        # Save to history before starting
        self.story_model.save_to_history()

        # Initialize state for iterative generation
        self._auto_build_state = {
            "initial_prompt": initial_prompt,
            "notes": notes,
            "supp_text": supp_text,
            "system_prompt": system_prompt,
            "chunk_count": 0,
            "max_chunks": self.rag_model.max_chunks,
            "paragraphs_per_chunk": 3,
            "chunks_before_summary": 3,
            "last_rag_context": None,
        }

        # Start first chunk generation
        self._generate_next_chunk()

    def _generate_next_chunk(self):
        """Generate the next chunk in auto-build mode."""
        state = self._auto_build_state

        # Check if we should stop
        if state["chunk_count"] >= state["max_chunks"]:
            self.view.append_logs(f"\n\n{'=' * 60}\n")
            self.view.append_logs("✅ AUTO BUILD COMPLETE\n")
            self.view.append_logs(f"Generated {state['chunk_count']} chunks total.\n")
            self.view.append_logs(f"{'=' * 60}\n")
            self.view.set_stop_enabled(False)
            # Render markdown
            self._markdown_content = self.view.get_story_content()
            self.view.render_story_markdown(self._markdown_content)
            return

        if self.llm_model.stop_generation:
            self.view.append_logs(f"\n\n{'=' * 60}\n")
            self.view.append_logs("⏹️ AUTO BUILD STOPPED BY USER\n")
            self.view.append_logs(f"Generated {state['chunk_count']} chunks.\n")
            self.view.append_logs(f"{'=' * 60}\n")
            self.view.set_stop_enabled(False)
            # Render markdown
            self._markdown_content = self.view.get_story_content()
            self.view.render_story_markdown(self._markdown_content)
            return

        state["chunk_count"] += 1
        chunk_num = state["chunk_count"]

        self.view.append_logs(f"\n{'─' * 60}\n")
        self.view.append_logs(
            f"📝 GENERATING CHUNK {chunk_num}/{state['max_chunks']}\n"
        )
        self.view.append_logs(f"{'─' * 60}\n\n")

        # Get current story content
        current_story = self.view.get_story_content()

        # Calculate context budget and condense if needed
        context_limit = self.settings_model.context_limit

        # Estimate token counts
        supp_tokens = self.story_model.estimate_token_count(state["supp_text"])
        notes_tokens = self.story_model.estimate_token_count(state["notes"])
        system_tokens = self.story_model.estimate_token_count(state["system_prompt"])
        safety_buffer = 500

        # Check if we need to condense supplemental/system/notes
        max_supp_tokens = max(256, int(context_limit * 0.15))
        max_system_tokens = max(256, int(context_limit * 0.15))
        max_notes_tokens = max(128, int(context_limit * 0.1))

        # Condense if needed (same logic as _on_send)
        if (
            supp_tokens > max_supp_tokens
            and state["supp_text"]
            and self.settings_model.summarize_prompts
        ):
            self.view.append_logs("🔄 Condensing supplemental prompts...\n")
            state["supp_text"], supp_tokens = (
                self.llm_controller.summarize_supplemental(
                    state["supp_text"], max_supp_tokens
                )
            )
            self.view.append_logs(f"  ✓ Reduced to {supp_tokens} tokens\n")

        if (
            system_tokens > max_system_tokens
            and state["system_prompt"]
            and self.settings_model.summarize_prompts
        ):
            self.view.append_logs("🔄 Condensing system prompt...\n")
            state["system_prompt"], system_tokens = (
                self.llm_controller.summarize_system_prompt(
                    state["system_prompt"], max_system_tokens
                )
            )
            self.view.append_logs(f"  ✓ Reduced to {system_tokens} tokens\n")

        if (
            notes_tokens > max_notes_tokens
            and state["notes"]
            and self.settings_model.summarize_prompts
        ):
            self.view.append_logs("🔄 Condensing notes...\n")
            state["notes"], notes_tokens = self.llm_controller.summarize_supplemental(
                state["notes"], max_notes_tokens
            )
            self.view.append_logs(f"  ✓ Reduced to {notes_tokens} tokens\n")

        # Query RAG with initial prompt + recent story content
        rag_query = state["initial_prompt"]
        if current_story:
            # Use last 500 chars of story for RAG context
            recent_story = (
                current_story[-500:] if len(current_story) > 500 else current_story
            )
            rag_query = (
                f"{state['initial_prompt']}\n\nRecent story content:\n{recent_story}"
            )

        # Calculate dynamic RAG budget for auto-build (25% allocation)
        context_limit = self.settings_model.context_limit
        output_reserve = 2000
        available_for_rag_and_story = (
            context_limit - supp_tokens - notes_tokens - system_tokens - output_reserve
        )
        max_rag_tokens = int(
            available_for_rag_and_story * 0.25
        )  # 25% for RAG in auto-build
        max_rag_tokens = max(500, min(max_rag_tokens, 3000))

        self.view.append_logs(
            f"🔍 Querying RAG databases (budget: {max_rag_tokens:,} tokens)...\n"
        )
        rag_context = self.rag_controller.query_databases(
            rag_query, max_tokens=max_rag_tokens
        )

        if rag_context:
            rag_tokens = self.story_model.estimate_token_count(rag_context)
            self.view.append_logs(f"  ✓ Retrieved {rag_tokens:,} tokens from RAG\n")
            state["last_rag_context"] = rag_context
        else:
            self.view.append_logs("  ℹ️ No RAG results\n")
            state["last_rag_context"] = None

        # Check if we need to summarize story
        story_tokens = self.story_model.estimate_token_count(current_story)
        fixed_costs = supp_tokens + notes_tokens + system_tokens + safety_buffer
        available_for_story = context_limit - fixed_costs
        max_raw_tokens = min(
            self.rag_model.summary_chunk_size, int(available_for_story * 0.6)
        )

        if story_tokens > max_raw_tokens and current_story:
            self.view.append_logs(f"\n📊 Story getting large ({story_tokens} tokens)\n")
            self.view.append_logs(
                "🔄 Running summarization to compress older content...\n\n"
            )

            # Store context and run summarization, then continue in callback
            self._auto_build_pending_continue = True

            max_rolling_summary_tokens = min(1000, int(available_for_story * 0.4))

            self.llm_controller.process_story_with_summarization(
                current_story,
                max_raw_tokens,
                max_rolling_summary_tokens,
                self.summary_model,
                self.view.append_llm_panel_text,
                self._on_auto_build_summarization_complete,
                self._on_auto_build_error,
                self.view.set_waiting,
            )
            return  # Will continue in callback
        else:
            story_for_llm = current_story

        # Build query for this chunk
        self._execute_chunk_generation(story_for_llm)

    def _execute_chunk_generation(self, story_for_llm):
        """Execute the actual chunk generation with the prepared context."""
        state = self._auto_build_state

        # Build final query
        query_parts = []

        if story_for_llm:
            query_parts.append(f"Story so far:\n```\n{story_for_llm}\n```\n\n")

        if state["last_rag_context"]:
            query_parts.append(f"Relevant context:\n{state['last_rag_context']}\n\n")

        if state["notes"]:
            query_parts.append(f"Author notes:\n{state['notes']}\n\n")

        if state["supp_text"]:
            query_parts.append(f"Additional instructions:\n{state['supp_text']}\n\n")

        query_parts.append(f"Initial prompt: {state['initial_prompt']}\n\n")
        query_parts.append(
            f"Continue the story. Write EXACTLY {state['paragraphs_per_chunk']} paragraphs. "
            f"Maintain narrative flow and character consistency."
        )

        final_query = "".join(query_parts)

        # Generate chunk with paragraph limit
        self.view.append_logs(
            f"✍️ Generating {state['paragraphs_per_chunk']} paragraphs...\n\n"
        )

        # Start waiting animation before LLM call
        self.view.set_waiting(True)

        self.llm_controller.generate_story_chunk(
            final_query,
            state["system_prompt"],
            state["paragraphs_per_chunk"],
            self.view.append_story_content,
            self.view.append_llm_panel_text,
            self._on_auto_build_chunk_complete,
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _on_auto_build_chunk_complete(self):
        """Called when a chunk generation completes (auto-build)."""
        state = self._auto_build_state

        stopped, current_story = self.llm_controller.finalize_chunk_completion(
            state["chunk_count"],
            self.view.append_logs,
            self.view.get_story_content,
            set_stop_enabled=self.view.set_stop_enabled,
            on_story_updated=self._update_markdown_content,
            render_markdown_callback=self._render_story_markdown_from_view,
            stop_log_lines=[
                f"\n\n{'=' * 60}\n",
                "⏹️ AUTO BUILD STOPPED BY USER\n",
                f"Generated {state['chunk_count']} chunks.\n",
                f"{'=' * 60}\n",
            ],
        )

        if stopped:
            return

        if current_story is not None:
            self._markdown_content = current_story

        QtCore.QTimer.singleShot(100, self._generate_next_chunk)

    def _on_auto_build_summarization_complete(self, story_for_llm, tokens):
        """Called when summarization completes during auto-build."""
        # Check if user requested stop
        if self.llm_model.stop_generation:
            state = self._auto_build_state
            self.view.append_logs(f"\n\n{'=' * 60}\n")
            self.view.append_logs(
                "⏹️ AUTO BUILD STOPPED BY USER (during summarization)\n"
            )
            self.view.append_logs(f"Generated {state['chunk_count']} chunks.\n")
            self.view.append_logs(f"{'=' * 60}\n")
            self.view.set_stop_enabled(False)
            # Render markdown
            self._markdown_content = self.view.get_story_content()
            self.view.render_story_markdown(self._markdown_content)
            return

        self.view.append_logs(f"\n✅ Summarization complete ({tokens} tokens)\n")
        self.view.append_logs("Continuing with chunk generation...\n\n")

        # Continue with chunk generation
        self._execute_chunk_generation(story_for_llm)

    def _on_auto_build_error(self, error_msg):
        """Handle errors during auto-build mode."""
        self.view.append_logs(f"\n❌ Error during auto-build: {error_msg}\n")
        self.view.append_logs("Auto-build stopped.\n")
        self.view.set_stop_enabled(False)

    def _on_mode_changed(self, mode):
        """Handle mode change from LLM panel."""
        self.view.append_logs(f"Mode changed to: {mode}")

        # If Smart Mode is selected, enable smart_mode
        if mode == "Smart Mode":
            if not self.settings_model.smart_mode:
                self.settings_model.smart_mode = True
                self.view.set_smart_mode(True)
                self.view.append_logs(
                    "✓ Smart Mode enabled (continuous writing with RAG)"
                )
            # Disable planning mode if it was active
            self.view.llm_panel.set_planning_mode(False)
        # If Normal mode, disable smart_mode and planning mode
        elif mode == "Normal":
            if self.settings_model.smart_mode:
                self.settings_model.smart_mode = False
                self.view.set_smart_mode(False)
                self.view.append_logs("✓ Normal mode")
            # Disable planning mode if it was active
            self.view.llm_panel.set_planning_mode(False)
        # If Planning mode, initialize planning
        elif mode == "Planning":
            # Disable smart mode if it was active
            if self.settings_model.smart_mode:
                self.settings_model.smart_mode = False
                self.view.set_smart_mode(False)
            # Initialize planning mode
            self._initialize_planning_mode()

    def _initialize_planning_mode(self):
        """Initialize planning mode in LLM Panel."""
        # Enable planning mode in panel
        self.view.llm_panel.set_planning_mode(True)

        # Load saved conversation ONLY for user message history (arrow key navigation)
        saved_conversation = self.settings_model.get_planning_conversation()

        if saved_conversation:
            # Extract only user messages for arrow key recall
            user_messages = [
                msg["content"] for msg in saved_conversation if msg["role"] == "user"
            ]
            self.view.llm_panel.user_message_history = user_messages

        # Always show welcome message (fresh start, don't restore old conversations)
        welcome_text = (
            self.settings_model.planning_prompt_template
            or "Let's plan your story! Describe what you'd like to write about."
        )
        self.view.llm_panel.display_planning_welcome(welcome_text)
        self.view.llm_panel.append_llm_panel_text(
            '*When ready, type "Start Writing" to begin story generation.*\n\n'
        )

    def _on_start_writing_from_planning(self, outline):
        """Handle start writing request from planning mode.

        Args:
            outline: The outline text to use for generation
        """
        self.view.llm_panel.append_llm_panel_text(
            "\n\n🚀 **Starting Story Generation...**\n\n"
        )

        # Store outline in story model
        self.story_model.planning_outline = outline
        self.story_model.planning_active = True
        self.planning_model.current_outline = outline

        # Save planning state
        self.settings_model.save_planning_conversation(
            self.view.llm_panel.get_planning_conversation(), outline
        )

        # Gather context
        notes = self.view.notes_panel.get_notes_text().strip()
        supp_text = self.view.utilities_panel.gather_supplemental_text()
        system_prompt = self.view.utilities_panel.get_system_prompt_text()

        # Disable planning mode in panel (keep conversation visible)
        self.view.llm_panel.set_planning_mode(False)

        # Switch mode selector back to Normal
        self.view.llm_panel.set_mode("Normal")

        # Start outline-driven generation
        self.planning_controller.start_outline_build(
            outline, notes, supp_text, system_prompt
        )

    def show(self):
        """Show the main view."""
        # Initialize story panel markdown state from settings
        self.view.story_panel.set_markdown_enabled(self.settings_model.render_markdown)

        self.view.resize(800, 600)
        self.view.show()


def main():
    """Main application entry point."""
    app = QtWidgets.QApplication(sys.argv)

    # Set dark theme stylesheet
    app.setStyleSheet(MAIN_STYLE)

    controller = MainController()
    controller.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
