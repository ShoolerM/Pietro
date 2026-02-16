"""Notes panel view for author notes with markdown streaming."""

import hashlib
from PyQt5 import QtWidgets, QtCore


class NotesPanel(QtWidgets.QWidget):
    """Panel for managing author notes (LLM context)."""

    font_size_changed = QtCore.pyqtSignal(int)  # delta

    def __init__(self):
        super().__init__()

        # Notes tracking for modification detection
        self._notes_last_set_by_llm = False
        self._notes_llm_content = ""
        self._notes_content_hash = None
        self._notes_streaming_content = ""
        self._notes_update_pending = False

        # Timer for throttled markdown updates during streaming
        self._notes_update_timer = QtCore.QTimer()
        self._notes_update_timer.setSingleShot(True)
        self._notes_update_timer.setInterval(100)
        self._notes_update_timer.timeout.connect(self._apply_notes_markdown_update)

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        notes_layout = QtWidgets.QVBoxLayout()
        notes_layout.setContentsMargins(5, 5, 5, 5)

        self.notes_text = QtWidgets.QTextEdit()
        self.notes_text.setAcceptRichText(True)
        self.notes_text.setMarkdown("")
        self.notes_text.setPlaceholderText(
            "Add notes here for LLM context (e.g., character details, plot points, reminders)...\n"
            "Added to LLM context.\nSupports Markdown formatting."
        )
        self.notes_text.installEventFilter(self)

        # Track user modifications to notes
        self.notes_text.textChanged.connect(self._on_notes_text_changed)

        notes_layout.addWidget(self.notes_text)
        self.setLayout(notes_layout)

    def eventFilter(self, obj, event):
        """Event filter for font resizing with Ctrl+Wheel."""
        if event.type() == QtCore.QEvent.Wheel:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.font_size_changed.emit(1)
                elif delta < 0:
                    self.font_size_changed.emit(-1)
                return True
        return False

    def apply_font_size(self, size):
        """Apply font size to notes text widget."""
        font = self.notes_text.font()
        font.setPointSize(size)
        self.notes_text.setFont(font)

    def get_notes_text(self):
        """Get the current notes text for LLM context."""
        return self.notes_text.toPlainText()

    def clear_notes(self):
        """Clear the notes section and reset tracking state."""
        try:
            self.notes_text.textChanged.disconnect(self._on_notes_text_changed)
        except Exception:
            pass
        self.notes_text.clear()
        self.notes_text.textChanged.connect(self._on_notes_text_changed)
        self._notes_last_set_by_llm = False
        self._notes_content_hash = None
        self._notes_llm_content = ""
        self._notes_streaming_content = ""

    def set_notes_from_llm(self, text: str):
        """Set notes text from LLM generation, marking it as LLM-generated."""
        try:
            self.notes_text.textChanged.disconnect(self._on_notes_text_changed)
        except Exception:
            pass

        self._notes_last_set_by_llm = True
        self._notes_llm_content = text
        self._notes_content_hash = hashlib.md5(text.encode()).hexdigest()
        self.notes_text.setMarkdown(text)

        self.notes_text.textChanged.connect(self._on_notes_text_changed)

    @QtCore.pyqtSlot(str)
    def append_notes(self, text: str):
        """Append text chunk to notes (for streaming) with throttled markdown."""
        self._notes_streaming_content += text
        self._notes_update_pending = True

        if not self._notes_update_timer.isActive():
            self._notes_update_timer.start()

    def _apply_notes_markdown_update(self):
        """Apply pending markdown update to notes (called by timer)."""
        if not self._notes_update_pending:
            return

        try:
            self.notes_text.textChanged.disconnect(self._on_notes_text_changed)
        except Exception:
            pass

        self.notes_text.setMarkdown(self._notes_streaming_content)

        cursor = self.notes_text.textCursor()
        cursor.movePosition(cursor.End)
        self.notes_text.setTextCursor(cursor)
        self.notes_text.ensureCursorVisible()

        self.notes_text.textChanged.connect(self._on_notes_text_changed)
        self._notes_update_pending = False

    def mark_notes_as_llm_generated(self, text: str):
        """Mark the current notes as LLM-generated after streaming completes."""
        self._notes_update_timer.stop()
        if self._notes_update_pending:
            self._apply_notes_markdown_update()

        self._notes_last_set_by_llm = True
        self._notes_llm_content = text
        self._notes_content_hash = hashlib.md5(text.encode()).hexdigest()

    def is_notes_user_modified(self) -> bool:
        """Check if notes have been modified by the user since last LLM generation."""
        if not self._notes_last_set_by_llm:
            return True

        current_text = self.notes_text.toPlainText()
        return current_text != self._notes_llm_content

    def _on_notes_text_changed(self):
        """Handle notes text changes to detect user modifications."""
        if self._notes_last_set_by_llm:
            current_text = self.notes_text.toPlainText()
            if current_text != self._notes_llm_content:
                self._notes_last_set_by_llm = False

    def should_regenerate_notes(self) -> bool:
        """Check if notes should be regenerated (blank or unmodified LLM content)."""
        current_text = self.notes_text.toPlainText()

        if not current_text.strip():
            return True

        if self._notes_last_set_by_llm and self._notes_content_hash:
            current_hash = hashlib.md5(current_text.encode()).hexdigest()
            return current_hash == self._notes_content_hash

        return False
