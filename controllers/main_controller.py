"""Main controller that coordinates all components."""

import hashlib
import sys
import threading
from PyQt5 import QtWidgets, QtCore

from models.story_model import StoryModel
from models.settings_model import SettingsModel
from models.prompt_model import PromptModel
from models.llm_model import LLMModel
from models.rag_model import RAGModel
from models.summary_model import SummaryModel
from views.main_view import MainView
from controllers.prompt_controller import PromptController
from controllers.llm_controller import LLMController
from controllers.rag_controller import RAGController
from controllers.context_controller import ContextController
from controllers.planning_controller import PlanningController
from models.planning_model import PlanningModel
from models.model_context_database import detect_context_window

from pathlib import Path


class NotesGeneratorSignals(QtCore.QObject):
    """Signals for background notes generation."""

    notes_chunk = QtCore.pyqtSignal(str)  # chunk of notes being streamed
    notes_ready = QtCore.pyqtSignal(str, int)  # generated_notes, notes_tokens
    notes_error = QtCore.pyqtSignal(str)  # error_message


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
        self.rag_controller = RAGController(self.rag_model, self.view)
        self.context_controller = ContextController(
            self.story_model,
            self.settings_model,
            self.summary_model,
            self.rag_controller,
            self.rag_model,
        )
        self.planning_controller = PlanningController(
            self.planning_model,
            self.story_model,
            self.settings_model,
            self.llm_controller,
            self.rag_controller,
            self.rag_model,
            self.summary_model,
            self.view,
        )

        # Track markdown content for rendering
        self._markdown_content = ""

        # Track story content for notes regeneration
        self._last_story_content_hash = None

        # Connect view signals to handlers
        self._connect_signals()

        # Connect model observers
        self._connect_observers()

        # Initialize view state
        self._initialize_view()

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
        self.view.rag_settings_requested.connect(self._on_rag_settings_requested)
        self.view.prompt_selections_changed.connect(self._on_prompt_selections_changed)
        self.view.summarization_prompt_requested.connect(
            self._on_summarization_prompt_requested
        )
        self.view.notes_prompt_requested.connect(self._on_notes_prompt_requested)
        self.view.general_settings_requested.connect(
            self._on_general_settings_requested
        )
        self.view.model_settings_requested.connect(self._on_model_settings_requested)
        self.view.mode_changed.connect(self._on_mode_changed)
        self.view.inference_settings_requested.connect(
            self._on_inference_settings_requested
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
        self.view.planning_mode_requested.connect(self._on_planning_mode_requested)

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

    def _generate_notes_background(self, story_context):
        """Generate notes in background thread and update UI safely via signals.

        Args:
            story_context: The current story content for notes generation
        """
        # Create signals object
        signals = NotesGeneratorSignals()

        # Connect signals to UI update handlers
        signals.notes_chunk.connect(self._on_notes_chunk)
        signals.notes_ready.connect(self._on_notes_generated)
        signals.notes_error.connect(self._on_notes_generation_error)

        def generate_in_thread():
            try:
                notes_prompt = self.settings_model.notes_prompt_template

                # Define callback for streaming chunks
                def on_chunk(chunk):
                    signals.notes_chunk.emit(chunk)

                generated_notes, notes_tokens = self.llm_controller.generate_notes(
                    story_context, notes_prompt, on_chunk
                )
                signals.notes_ready.emit(generated_notes, notes_tokens)
            except Exception as e:
                self.view.append_logs(f"Error generating notes: {e}")
                signals.notes_error.emit(str(e))

        # Start background thread
        thread = threading.Thread(target=generate_in_thread, daemon=True)

        # Start background thread
        thread = threading.Thread(target=generate_in_thread, daemon=True)
        thread.start()

    def _on_notes_chunk(self, chunk):
        """Handle streaming notes chunk (called on main thread via signal)."""
        # Append chunk to notes section
        self.view.prompts_panel.append_notes(chunk)

    def _on_notes_generated(self, generated_notes, notes_tokens):
        """Handle notes generation completion (called on main thread via signal)."""
        # Mark notes as LLM-generated (they've already been streamed in)
        self.view.prompts_panel.mark_notes_as_llm_generated(generated_notes)
        self.view.append_logs(f"  ✓ Generated {notes_tokens} tokens of notes\n\n")

        # Continue with story generation if we have pending context
        if hasattr(self, "_pending_notes_context"):
            ctx = self._pending_notes_context
            delattr(self, "_pending_notes_context")
            # Continue with the story generation by calling _continue_send
            self._continue_send(
                ctx["user_input"], ctx["notes"], ctx["supp_text"], ctx["system_prompt"]
            )
        elif hasattr(self, "_pending_auto_build_context"):
            ctx = self._pending_auto_build_context
            delattr(self, "_pending_auto_build_context")
            # Continue with auto-build mode
            self._continue_auto_build(
                ctx["initial_prompt"],
                ctx["notes"],
                ctx["supp_text"],
                ctx["system_prompt"],
            )
        elif hasattr(self, "_pending_planning_build_context"):
            ctx = self._pending_planning_build_context
            delattr(self, "_pending_planning_build_context")
            # Continue with planning build mode
            self._start_planning_build(
                ctx["outline"], ctx["notes"], ctx["supp_text"], ctx["system_prompt"]
            )
        elif hasattr(self, "_pending_planning_notes_continue"):
            delattr(self, "_pending_planning_notes_continue")
            try:
                if hasattr(self, "_planning_build_state"):
                    self._planning_build_state["notes"] = generated_notes
            except Exception:
                pass
            self.view.set_waiting(False)
            self._generate_next_planning_chunk()
        else:
            self.view.set_waiting(False)

    def _on_notes_generation_error(self, error_msg):
        """Handle notes generation error (called on main thread via signal)."""
        self.view.append_logs(f"  ⚠ Error generating notes: {error_msg}\n\n")
        self.view.set_waiting(False)

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
        # Check if Build with Smart Mode is enabled
        if self.settings_model.smart_mode:
            # Trigger auto-build story mode with the user's input and context
            self._on_auto_build_story_requested(
                user_input, notes, supp_text, system_prompt
            )
            return

        # Check if notes should be regenerated
        story_context = self.view.get_story_content()
        current_story_hash = hashlib.md5(story_context.encode()).hexdigest()

        # Regenerate notes if:
        # 1. Story is NOT blank (has content), AND
        # 2. Either story content has changed OR notes are unmodified LLM content
        story_changed = (
            self._last_story_content_hash is not None
            and current_story_hash != self._last_story_content_hash
        )

        should_regen = (
            self.settings_model.auto_notes
            and story_context.strip()  # Only if story has content
            and (story_changed or self.view.prompts_panel.should_regenerate_notes())
        )

        if should_regen:
            self.view.append_logs("📝 Generating scene notes...\n")
            self.view.set_waiting(True)

            # Clear notes section before regenerating
            self.view.prompts_panel.clear_notes()

            # Store context to continue story generation after notes are ready
            self._pending_notes_context = {
                "user_input": user_input,
                "notes": notes,
                "supp_text": supp_text,
                "system_prompt": system_prompt,
            }

            # Generate notes in background thread using signals for thread-safe UI updates
            self._generate_notes_background(story_context)

            # Store current story hash for next comparison
            self._last_story_content_hash = current_story_hash

            # Return here - will continue in _on_notes_generated callback
            return

        # Store current story hash for next comparison
        self._last_story_content_hash = current_story_hash

        # Continue with story generation
        self._continue_send(user_input, notes, supp_text, system_prompt)

    def _continue_send(self, user_input, notes, supp_text, system_prompt):
        """Continue with story generation (after notes are ready or if not needed)."""
        # Reset stop flag and enable stop button
        self.llm_model.reset_stop_flag()
        self.view.set_stop_enabled(True)

        # Clear LLM Panel
        self.view.clear_thinking_text()

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
            self.view.append_thinking_text,
            self._on_render_markdown,
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _on_text_appended(self, text):
        """Handle text appended from LLM stream."""
        self.story_model.append_content(text)
        self.view.append_story_content(text)

    def _on_render_markdown(self):
        """Handle markdown rendering request."""
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
            self.view.append_thinking_text,
            self._on_render_markdown,
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _on_summarization_error(self, error_message):
        """Handle error during background summarization.

        Args:
            error_message: Error description
        """
        self.view.append_thinking_text(
            f"\n❌ Chunking/summarization error: {error_message}\n"
        )
        self.view.append_thinking_text("Falling back to recent content only...\n\n")

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
        self._last_story_content_hash = None

    def _on_refresh_models(self):
        """Handle refresh models button click."""
        success, result = self.llm_model.fetch_available_models()
        if not success:
            self.view.set_model_error(result)

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
        system_prompt = self.view.prompts_panel.get_system_prompt_text()

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
        system_prompt = self.view.prompts_panel.get_system_prompt_text()

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
        self.view.clear_thinking_text()
        self.view.append_thinking_text(f"\n{'=' * 60}\n")
        self.view.append_thinking_text("🔄 REGENERATING STORY SUMMARY\n")
        self.view.append_thinking_text("Processing entire story from scratch...\n")
        self.view.append_thinking_text(f"{'=' * 60}\n\n")

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

        self.view.append_thinking_text(f"Story size: {story_tokens} tokens\n")
        self.view.append_thinking_text(f"Target raw content: {max_raw_tokens} tokens\n")
        self.view.append_thinking_text(
            f"Target rolling summary: {max_rolling_summary_tokens} tokens\n\n"
        )

        # Start processing
        self.view.set_waiting(True)

        # Process in background thread
        def on_complete(story_for_llm, tokens):
            self.view.append_thinking_text("\n✅ Summary regeneration complete!\n")
            self.view.append_thinking_text(
                "Summary will be used for next generation.\n"
            )
            self.view.set_waiting(False)

        def on_error(error_msg):
            self.view.append_thinking_text(
                f"\n❌ Error regenerating summary: {error_msg}\n"
            )
            self.view.set_waiting(False)

        self.llm_controller.process_story_with_summarization(
            current_story,
            max_raw_tokens,
            max_rolling_summary_tokens,
            self.summary_model,
            self.view.append_thinking_text,
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
            notes = self.view.prompts_panel.get_notes_text().strip()
        if supp_text is None:
            supp_text = self.view.prompts_panel.gather_supplemental_text()
        if system_prompt is None:
            system_prompt = self.view.prompts_panel.get_system_prompt_text()

        # Check if notes should be regenerated
        story_context = self.view.get_story_content()
        current_story_hash = hashlib.md5(story_context.encode()).hexdigest()

        # Regenerate notes if:
        # 1. Story is NOT blank (has content), AND
        # 2. Either story content has changed OR notes are unmodified LLM content
        story_changed = (
            self._last_story_content_hash is not None
            and current_story_hash != self._last_story_content_hash
        )

        should_regen = (
            self.settings_model.auto_notes
            and story_context.strip()  # Only if story has content
            and (story_changed or self.view.prompts_panel.should_regenerate_notes())
        )

        if should_regen:
            self.view.append_logs("📝 Generating scene notes...\n")
            self.view.set_waiting(True)

            # Clear notes section before regenerating
            self.view.prompts_panel.clear_notes()

            # Store context to continue auto-build after notes are ready
            self._pending_auto_build_context = {
                "initial_prompt": initial_prompt,
                "notes": notes,
                "supp_text": supp_text,
                "system_prompt": system_prompt,
            }

            # Generate notes in background thread using signals for thread-safe UI updates
            self._generate_notes_background(story_context)

            # Store current story hash for next comparison
            self._last_story_content_hash = current_story_hash

            # Return here - will continue in _on_notes_generated callback
            return

        # Store current story hash for next comparison
        self._last_story_content_hash = current_story_hash

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
                self.view.append_thinking_text,
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
            self.view.append_thinking_text,
            lambda: self._on_chunk_complete(),
            self.view.set_waiting,
            self.view.set_stop_enabled,
        )

    def _on_chunk_complete(self):
        """Called when a chunk generation completes."""
        state = self._auto_build_state

        # Check if user requested stop
        if self.llm_model.stop_generation:
            self.view.append_thinking_text(f"\n\n{'=' * 60}\n")
            self.view.append_thinking_text("⏹️ AUTO BUILD STOPPED BY USER\n")
            self.view.append_thinking_text(
                f"Generated {state['chunk_count']} chunks.\n"
            )
            self.view.append_thinking_text(f"{'=' * 60}\n")
            self.view.set_stop_enabled(False)
            # Render markdown
            self._markdown_content = self.view.get_story_content()
            self.view.render_story_markdown(self._markdown_content)
            return

        self.view.append_logs(f"\n✅ Chunk {state['chunk_count']} complete!\n")

        # Update story model with new content
        current_story = self.view.get_story_content()
        self._markdown_content = current_story
        self.story_model.content = current_story

        # Generate next chunk after a brief moment
        # Use QTimer to schedule on the main thread
        QtCore.QTimer.singleShot(100, self._generate_next_chunk)

    def _on_auto_build_summarization_complete(self, story_for_llm, tokens):
        """Called when summarization completes during auto-build."""
        # Check if user requested stop
        if self.llm_model.stop_generation:
            state = self._auto_build_state
            self.view.append_logs(f"\n\n{'=' * 60}\n")
            self.view.append_logs("⏹️ LD STOPPED BY USER (during summarization)\n")
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

    def _on_summarization_prompt_requested(self):
        """Handle summarization prompt settings menu action."""
        saved, new_prompt = self.view.show_summarization_prompt_dialog(
            self.settings_model.summary_prompt_template
        )

        if saved and new_prompt is not None:
            success = self.settings_model.save_summary_prompt(new_prompt)
            if not success:
                self.view.show_warning(
                    "Save Error", "Failed to save summarization prompt"
                )

    def _on_planning_mode_requested(self):
        """Handle Planning Mode menu action.

        Delegates to planning controller.
        """
        self.planning_controller.start_planning_mode()

    def _on_notes_prompt_requested(self):
        """Handle notes prompt settings menu action."""
        saved, new_prompt = self.view.show_notes_prompt_dialog(
            self.settings_model.notes_prompt_template
        )

        if saved and new_prompt is not None:
            success = self.settings_model.save_notes_prompt(new_prompt)
            if not success:
                self.view.show_warning("Save Error", "Failed to save notes prompt")

    def _on_general_settings_requested(self):
        """Handle general settings menu action."""
        result = self.view.show_general_settings_dialog(
            self.settings_model.auto_notes, self.settings_model.render_markdown
        )

        if result["saved"]:
            if result["auto_notes"] is not None:
                self.settings_model.auto_notes = result["auto_notes"]
                self.view.append_logs(
                    f"✓ Auto Notes: {'enabled' if result['auto_notes'] else 'disabled'}"
                )

            if result["render_markdown"] is not None:
                self.settings_model.render_markdown = result["render_markdown"]
                # Update story panel markdown state immediately
                self.view.story_panel.set_markdown_enabled(result["render_markdown"])
                # Re-render current story with new setting
                current_story = self.view.get_story_content()
                if result["render_markdown"] and current_story:
                    self.view.render_story_markdown(self._markdown_content)
                self.view.append_logs(
                    f"✓ Render Markdown: {'enabled' if result['render_markdown'] else 'disabled'}"
                )

    def _on_settings_opened(self):
        """Handle settings menu action.
        DEPRECATED: Use _on_summarization_prompt_requested instead.
        """
        self._on_summarization_prompt_requested()

    def _on_rag_settings_requested(self):
        """Handle RAG settings dialog request."""
        # Get current settings from model and show dialog
        current_threshold_percent = self.rag_model.score_variance_threshold * 100.0
        self.view.show_rag_settings_dialog(
            current_max_chunks=self.rag_model.max_chunks,
            current_summary_chunk_size=self.rag_model.summary_chunk_size,
            current_score_threshold=current_threshold_percent,
        )

    def _on_model_settings_requested(self):
        """Handle model settings dialog request."""
        result = self.view.show_model_settings_dialog(
            current_context_limit=self.settings_model.context_limit
        )

        if result:
            self.settings_model.context_limit = result
            self.view.append_logs(f"✓ Context limit set to: {result} tokens")

    def _on_mode_changed(self, mode):
        """Handle mode change from bottom control panel."""
        self.view.append_logs(f"Mode changed to: {mode}")

        # If Smart Mode is selected, enable smart_mode
        if mode == "Smart Mode":
            if not self.settings_model.smart_mode:
                self.settings_model.smart_mode = True
                self.view.set_smart_mode(True)
                self.view.append_logs("✓ Smart Mode enabled (build with RAG)")
        # If Normal mode, disable smart_mode
        elif mode == "Normal":
            if self.settings_model.smart_mode:
                self.settings_model.smart_mode = False
                self.view.set_smart_mode(False)
                self.view.append_logs("✓ Normal mode (build with RAG disabled)")
        # Planning mode is handled by MainView opening the planning dialog

    def _on_model_settings_requested(self):
        """Handle model settings dialog request."""
        result = self.view.show_model_settings_dialog(
            current_context_limit=self.settings_model.context_limit
        )

        if result:
            self.settings_model.context_limit = result
            self.view.append_logs(f"✓ Context limit set to: {result} tokens")

    def _on_mode_changed(self, mode):
        """Handle mode change from bottom control panel."""
        self.view.append_logs(f"Mode changed to: {mode}")

        # If Smart Mode is selected, enable smart_mode
        if mode == "Smart Mode":
            if not self.settings_model.smart_mode:
                self.settings_model.smart_mode = True
                self.view.set_smart_mode(True)
                self.view.append_logs("✓ Smart Mode enabled (build with RAG)")
        # If Normal mode, disable smart_mode
        elif mode == "Normal":
            if self.settings_model.smart_mode:
                self.settings_model.smart_mode = False
                self.view.set_smart_mode(False)
                self.view.append_logs("✓ Normal mode (build with RAG disabled)")
        # Planning mode is handled by MainView opening the planning dialog

    def _on_inference_settings_requested(self):
        """Handle inference settings dialog request."""
        # Get current settings from model and show dialog
        result = self.view.show_inference_settings_dialog(
            current_ip=self.settings_model.inference_ip,
            current_port=self.settings_model.inference_port,
        )

        if result:
            ip, port = result

            # Test the connection before saving
            test_url = f"http://{ip}:{port}/v1"
            self.view.append_logs(f"Testing connection to: {test_url}")

            # Temporarily update the LLM model URL for testing
            old_url = self.llm_model.base_url
            self.llm_model.base_url = test_url

            # Try to fetch models to verify connection
            success, result_data = self.llm_model.fetch_available_models()

            if success:
                # Connection successful, save settings
                self.settings_model.inference_ip = ip
                self.settings_model.inference_port = port
                self.settings_model.save_inference_settings()
                self.view.append_logs(f"✓ Inference server updated to: {ip}:{port}")
                self.view.append_logs(f"  Base URL: {self.settings_model.base_url}")

                # Ensure LLM model uses updated base URL
                self.llm_model.base_url = self.settings_model.base_url

                # Save profile for current model with new inference settings
                self._save_current_model_profile()

                # Show success message
                QtWidgets.QMessageBox.information(
                    self.view,
                    "Connection Successful",
                    f"Successfully connected to inference server at {ip}:{port}\n\n"
                    f"Found {len(result_data)} model(s).",
                )

                # Refresh models in the view
                self._on_refresh_models()
            else:
                # Connection failed, revert to old URL
                self.llm_model.base_url = old_url

                # Show error message
                QtWidgets.QMessageBox.critical(
                    self.view,
                    "Connection Failed",
                    f"Could not connect to inference server at {ip}:{port}\n\n"
                    f"Error: {result_data}\n\n"
                    f"Please check:\n"
                    f"• The IP address and port are correct\n"
                    f"• The inference server is running\n"
                    f"• There are no firewall issues",
                )
                self.view.append_logs(
                    f"✗ Failed to connect to {ip}:{port}: {result_data}"
                )
            settings_dir = Path("settings")
            settings_dir.mkdir(exist_ok=True)
            filepath = str(settings_dir / "current_summary.json")

        return self.summary_model.save_to_file(filepath)

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
    app.setStyleSheet("""
        QWidget {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        QTextEdit {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            selection-background-color: #3a3a3a;
        }
        QPushButton {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #303030;
        }
        QPushButton:checkable {
            border: 2px solid #555555;
        }
        QPushButton:checked {
            background-color: #2d5a2d;
            border: 2px solid #4a9e4a;
            color: #ffffff;
            font-weight: bold;
        }
        QPushButton:checked:hover {
            background-color: #346a34;
        }
        QPushButton:!checked {
            background-color: #404040;
            border: 2px solid #555555;
        }
        QPushButton:!checked:hover {
            background-color: #505050;
        }
        QListWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            alternate-background-color: #252525;
        }
        QListWidget::item {
            padding: 3px;
            color: #ffffff;
        }
        QListWidget::item:selected {
            background-color: #3a3a3a;
            color: #ffffff;
        }
        QListWidget::item:hover {
            color: #ffffff;
        }
        QTreeWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            alternate-background-color: #252525;
        }
        QTreeWidget::item {
            padding: 3px;
            color: #ffffff;
        }
        QTreeWidget::item:selected {
            background-color: #3a3a3a;
            color: #ffffff;
        }
        QTreeWidget::item:hover {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        QTreeWidget::branch {
            background-color: #1e1e1e;
        }
        QTreeWidget::branch:has-siblings:!adjoins-item {
            border-image: none;
        }
        QTreeWidget::branch:has-siblings:adjoins-item {
            border-image: none;
        }
        QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
            border-image: none;
        }
        QTreeWidget::branch:closed:has-children:has-siblings,
        QTreeWidget::branch:has-children:!has-siblings:closed {
            background: #1e1e1e;
            border-image: none;
        }
        QTreeWidget::branch:open:has-children:has-siblings,
        QTreeWidget::branch:open:has-children:!has-siblings {
            background: #1e1e1e;
            border-image: none;
        }
        QTreeWidget::indicator {
            width: 13px;
            height: 13px;
        }
        QTreeWidget::indicator:unchecked {
            background-color: #2a2a2a;
            border: 1px solid #555555;
        }
        QTreeWidget::indicator:checked {
            background-color: #4a9eff;
            border: 1px solid #3a8eef;
            image: none;
        }
        QTreeWidget::indicator:checked::after {
            content: "✓";
        }
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #cccccc;
        }
        QProgressBar {
            border: 1px solid #555555;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
        }
        QProgressBar::chunk {
            background-color: #505050;
        }
        QLabel {
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #2a2a2a;
        }
        QTabBar::tab {
            background-color: #303030;
            color: #ffffff;
            border: 1px solid #555555;
            border-bottom: none;
            padding: 6px 12px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #404040;
            border-bottom: 1px solid #404040;
        }
        QTabBar::tab:hover {
            background-color: #3a3a3a;
        }
        QTabBar::close-button {
            image: url(none);
            subcontrol-position: right;
        }
        QTabBar::close-button:hover {
            background-color: #555555;
        }
        QComboBox {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
            min-height: 20px;
        }
        QComboBox:hover {
            background-color: #505050;
        }
        QComboBox:on {
            background-color: #303030;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
            margin-right: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #2a2a2a;
            color: #ffffff;
            selection-background-color: #3a3a3a;
            border: 1px solid #555555;
        }
        QSplitter::handle {
            background-color: #555555;
            height: 3px;
        }
        QSplitter::handle:hover {
            background-color: #777777;
        }
        QSplitter::handle:pressed {
            background-color: #999999;
        }
        QSpinBox {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QSpinBox:hover {
            background-color: #505050;
        }
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: #505050;
            border: 1px solid #555555;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background-color: #606060;
        }
    """)

    controller = MainController()
    controller.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
