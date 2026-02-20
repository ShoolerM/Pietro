"""Controller for notes generation and regeneration logic."""

import hashlib
import threading
from PyQt5 import QtCore


class NotesSignals(QtCore.QObject):
    """Signals for background notes generation."""

    notes_chunk = QtCore.pyqtSignal(str)
    notes_ready = QtCore.pyqtSignal(str, int)
    notes_error = QtCore.pyqtSignal(str)


class NotesController:
    """Handles notes generation, streaming, and regeneration checks."""

    def __init__(self, settings_model, llm_controller, view):
        self.settings_model = settings_model
        self.llm_controller = llm_controller
        self.view = view
        self._last_story_hashes = {}

    def reset_story_hash(self, context_key=None):
        """Reset stored story hash for a context key (or all)."""
        if context_key is None:
            self._last_story_hashes.clear()
        else:
            self._last_story_hashes.pop(context_key, None)

    def should_regenerate_notes(
        self,
        story_context,
        auto_notes_enabled,
        should_regenerate_flag,
        context_key="main",
        first_time_regenerate=False,
    ):
        """Determine whether notes should be regenerated and update hash.

        Args:
            story_context: Current story content
            auto_notes_enabled: Whether auto-notes is enabled
            should_regenerate_flag: View-level indicator for regen (LLM notes)
            context_key: Key for separate regeneration contexts
            first_time_regenerate: If True, regenerate when no prior hash exists

        Returns:
            bool: True if notes should be regenerated
        """
        if not auto_notes_enabled or not story_context.strip():
            return False

        story_hash = hashlib.md5(story_context.encode()).hexdigest()
        previous = self._last_story_hashes.get(context_key)

        if previous is None:
            self._last_story_hashes[context_key] = story_hash
            return True if first_time_regenerate else should_regenerate_flag

        story_changed = previous != story_hash
        self._last_story_hashes[context_key] = story_hash
        return story_changed or should_regenerate_flag

    def generate_notes_async(
        self,
        story_context,
        on_complete=None,
        on_error=None,
        on_chunk=None,
        clear_existing=False,
        log_start=True,
        set_waiting_on_start=True,
        set_waiting_on_finish=True,
    ):
        """Generate notes in a background thread with streaming UI updates.

        Args:
            story_context: Current story content
            on_complete: Callback invoked with (generated_notes, notes_tokens)
            on_error: Callback invoked with error message
            on_chunk: Optional callback for each streamed chunk
            clear_existing: Whether to clear notes before generation
            log_start: Whether to log the start message
            set_waiting_on_start: Toggle waiting state at start
            set_waiting_on_finish: Toggle waiting state on completion
        """
        if clear_existing:
            self.view.notes_panel.clear_notes()

        if log_start:
            self.view.append_logs("📝 Generating scene notes...\n")

        def _set_waiting(waiting):
            QtCore.QMetaObject.invokeMethod(
                self.view,
                "set_waiting",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, waiting),
            )

        if set_waiting_on_start:
            _set_waiting(True)

        signals = NotesSignals()

        def default_chunk_handler(chunk):
            self.view.notes_panel.append_notes(chunk)

        def default_ready_handler(generated_notes, notes_tokens):
            self.view.notes_panel.mark_notes_as_llm_generated(generated_notes)
            self.view.append_logs(f"  ✓ Generated {notes_tokens} tokens of notes\n\n")

        def default_error_handler(error_msg):
            self.view.append_logs(f"  ⚠ Error generating notes: {error_msg}\n\n")

        if on_chunk is None:
            on_chunk = default_chunk_handler

        def on_ready(generated_notes, notes_tokens):
            default_ready_handler(generated_notes, notes_tokens)
            if on_complete:
                on_complete(generated_notes, notes_tokens)
            if set_waiting_on_finish:
                _set_waiting(False)

        def on_error_internal(error_msg):
            default_error_handler(error_msg)
            if on_error:
                on_error(error_msg)
            if set_waiting_on_finish:
                _set_waiting(False)

        signals.notes_chunk.connect(on_chunk, QtCore.Qt.QueuedConnection)
        signals.notes_ready.connect(on_ready, QtCore.Qt.QueuedConnection)
        signals.notes_error.connect(on_error_internal, QtCore.Qt.QueuedConnection)

        def generate_in_thread():
            try:
                notes_prompt = self.settings_model.notes_prompt_template

                def stream_chunk(chunk):
                    signals.notes_chunk.emit(chunk)

                generated_notes, notes_tokens = self.llm_controller.generate_notes(
                    story_context, notes_prompt, stream_chunk
                )
                signals.notes_ready.emit(generated_notes, notes_tokens)
            except Exception as e:
                signals.notes_error.emit(str(e))

        thread = threading.Thread(target=generate_in_thread, daemon=True)
        thread.start()
