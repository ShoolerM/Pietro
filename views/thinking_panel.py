"""LLM Panel view for thinking process and notes."""

from PyQt5 import QtWidgets, QtCore, QtGui
from views.search_widget import SearchWidget


class ThinkingPanel(QtWidgets.QWidget):
    """Panel for thinking process and notes sidebar."""

    # Signals
    font_size_changed = QtCore.pyqtSignal(int)  # delta

    def __init__(self):
        super().__init__()
        self.search_widget = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thinking Process section
        thinking_container = QtWidgets.QWidget()
        thinking_container_layout = QtWidgets.QVBoxLayout()
        thinking_container_layout.setContentsMargins(0, 0, 0, 0)
        thinking_container_layout.setSpacing(0)

        thinking_label = QtWidgets.QLabel("Thinking Process")
        thinking_container_layout.addWidget(thinking_label)

        self.thinking_text = QtWidgets.QTextEdit()
        self.thinking_text.setReadOnly(True)
        self.thinking_text.setAcceptRichText(False)
        self.thinking_text.setPlaceholderText(
            "LLM thinking process will appear here..."
        )
        self.thinking_text.installEventFilter(self)
        thinking_container_layout.addWidget(self.thinking_text)

        thinking_container.setLayout(thinking_container_layout)

        # Create search widget
        self.search_widget = SearchWidget(self.thinking_text, thinking_container)
        self.search_widget.hide()
        self.search_widget.close_requested.connect(self.search_widget.hide)
        thinking_container_layout.insertWidget(0, self.search_widget)

        layout.addWidget(thinking_container)
        self.setLayout(layout)

        # Set size constraints
        self.setMinimumWidth(250)
        self.setMaximumWidth(500)

        # Add Ctrl+F shortcut for search
        self.search_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        self.search_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.search_shortcut.activated.connect(self._show_search)

    def _show_search(self):
        """Show the search widget."""
        if self.search_widget:
            self.search_widget.show_and_focus()

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

    # Public methods

    def append_thinking_text(self, text):
        """Append text to LLM Panel."""
        try:
            self.thinking_text.moveCursor(QtGui.QTextCursor.End)
        except Exception:
            pass
        self.thinking_text.insertPlainText(text)
        try:
            self.thinking_text.verticalScrollBar().setValue(
                self.thinking_text.verticalScrollBar().maximum()
            )
        except Exception:
            pass

    def clear_thinking_text(self):
        """Clear LLM Panel."""
        self.thinking_text.clear()

    def apply_font_size(self, size):
        """Apply font size to text widgets."""
        font = QtGui.QFont()
        font.setPointSize(size)
        self.thinking_text.setFont(font)
