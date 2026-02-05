"""Search widget for text editors with inline search bar."""
import re
from PyQt5 import QtWidgets, QtCore, QtGui


class SearchWidget(QtWidgets.QWidget):
    """Inline search widget with find/replace functionality."""
    
    # Signal emitted when widget should be closed
    close_requested = QtCore.pyqtSignal()
    
    def __init__(self, text_edit, parent=None):
        super().__init__(parent)
        self.text_edit = text_edit
        self.current_match_index = -1
        self.matches = []
        
        self._init_ui()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        # Install event filter on text_edit to catch Escape there too
        text_edit.installEventFilter(self)
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Search input
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Find...")
        self.search_input.textChanged.connect(self._on_search_text_changed)
        self.search_input.returnPressed.connect(self._find_next)
        layout.addWidget(self.search_input)
        
        # Match counter label
        self.match_label = QtWidgets.QLabel("0/0")
        self.match_label.setMinimumWidth(50)
        layout.addWidget(self.match_label)
        
        # Previous button
        prev_btn = QtWidgets.QPushButton("↑")
        prev_btn.setToolTip("Previous (Shift+Enter)")
        prev_btn.setMaximumWidth(30)
        prev_btn.clicked.connect(self._find_previous)
        layout.addWidget(prev_btn)
        
        # Next button
        next_btn = QtWidgets.QPushButton("↓")
        next_btn.setToolTip("Next (Enter)")
        next_btn.setMaximumWidth(30)
        next_btn.clicked.connect(self._find_next)
        layout.addWidget(next_btn)
        
        # Case sensitive checkbox
        self.case_sensitive_cb = QtWidgets.QCheckBox("Match case")
        self.case_sensitive_cb.stateChanged.connect(self._on_search_text_changed)
        layout.addWidget(self.case_sensitive_cb)
        
        # Regex checkbox
        self.regex_cb = QtWidgets.QCheckBox("Regex")
        self.regex_cb.stateChanged.connect(self._on_search_text_changed)
        layout.addWidget(self.regex_cb)
        
        # Close button
        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setToolTip("Close (Esc)")
        close_btn.setMaximumWidth(30)
        close_btn.clicked.connect(self._close)
        layout.addWidget(close_btn)
        
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Style the widget
        self.setStyleSheet("""
            SearchWidget {
                background-color: palette(window);
                border: 1px solid palette(mid);
                border-radius: 3px;
            }
        """)
        
        # Install event filter on search input for Shift+Enter
        self.search_input.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle Shift+Enter for previous search, Esc in search input, and Esc in text edit."""
        if event.type() == QtCore.QEvent.KeyPress:
            # Handle Esc in search input or when search widget is visible and text edit has Escape
            if event.key() == QtCore.Qt.Key_Escape:
                # Close search if it's visible
                if self.isVisible():
                    self._close()
                    return True
            # Handle Shift+Enter in search input
            elif obj == self.search_input and event.key() == QtCore.Qt.Key_Return and event.modifiers() & QtCore.Qt.ShiftModifier:
                self._find_previous()
                return True
        return super().eventFilter(obj, event)
    
    def keyPressEvent(self, event):
        """Handle key press events on the widget itself."""
        if event.key() == QtCore.Qt.Key_Escape:
            # Don't close here - let it be handled by the search input's event filter
            pass
        else:
            super().keyPressEvent(event)
    
    def hideEvent(self, event):
        """Override hide event to clear highlights when widget is hidden."""
        self._clear_highlights()
        super().hideEvent(event)
    
    def _close(self):
        """Close the search widget and position cursor at beginning of last selected match."""
        # Position cursor at the beginning of the last selected match before hiding
        if self.matches and self.current_match_index >= 0:
            start, end = self.matches[self.current_match_index]
            cursor = self.text_edit.textCursor()
            cursor.setPosition(start)
            self.text_edit.setTextCursor(cursor)
            # Focus the text edit so cursor is visible
            self.text_edit.setFocus()
        
        self.close_requested.emit()
        self.hide()
    
    def _on_search_text_changed(self):
        """Handle search text or options change."""
        self._perform_search()
    
    def _perform_search(self):
        """Perform the search and highlight matches."""
        search_text = self.search_input.text()
        
        # Clear previous highlights
        self._clear_highlights()
        self.matches = []
        self.current_match_index = -1
        
        if not search_text:
            self.match_label.setText("0/0")
            return
        
        # Get document content
        document = self.text_edit.document()
        content = document.toPlainText()
        
        # Search for matches
        case_sensitive = self.case_sensitive_cb.isChecked()
        use_regex = self.regex_cb.isChecked()
        
        try:
            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(search_text, flags)
                for match in pattern.finditer(content):
                    self.matches.append((match.start(), match.end()))
            else:
                # Simple text search
                if not case_sensitive:
                    content_lower = content.lower()
                    search_lower = search_text.lower()
                    start = 0
                    while True:
                        pos = content_lower.find(search_lower, start)
                        if pos == -1:
                            break
                        self.matches.append((pos, pos + len(search_text)))
                        start = pos + 1
                else:
                    start = 0
                    while True:
                        pos = content.find(search_text, start)
                        if pos == -1:
                            break
                        self.matches.append((pos, pos + len(search_text)))
                        start = pos + 1
        except re.error as e:
            # Invalid regex
            self.match_label.setText(f"Invalid regex: {e}")
            return
        
        # Highlight all matches
        if self.matches:
            self._highlight_all_matches()
            self.current_match_index = 0
            self._highlight_current_match()
            self._update_match_label()
        else:
            self.match_label.setText("0/0")
    
    def _clear_highlights(self):
        """Clear all search highlights."""
        # Create a cursor that selects all text
        cursor = QtGui.QTextCursor(self.text_edit.document())
        cursor.select(QtGui.QTextCursor.Document)
        
        # Reset format - explicitly clear background
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QtGui.QBrush())  # Clear background
        cursor.mergeCharFormat(fmt)
        
        # Reset text cursor position
        cursor = self.text_edit.textCursor()
        cursor.clearSelection()
        self.text_edit.setTextCursor(cursor)
    
    def _highlight_all_matches(self):
        """Highlight all matches in light color."""
        # Dimmer light yellow background for non-current matches
        highlight_format = QtGui.QTextCharFormat()
        highlight_format.setBackground(QtGui.QColor(100, 100, 0, 100))  # Dimmer yellow
        
        for start, end in self.matches:
            cursor = self.text_edit.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(highlight_format)
    
    def _highlight_current_match(self):
        """Highlight the current match more prominently."""
        if not self.matches or self.current_match_index < 0:
            return
        
        start, end = self.matches[self.current_match_index]
        
        # Re-highlight all matches first
        self._highlight_all_matches()
        
        # Then highlight current match with bright yellow
        current_format = QtGui.QTextCharFormat()
        current_format.setBackground(QtGui.QColor(200, 200, 0, 150))  # Bright yellow, fully opaque
        
        cursor = self.text_edit.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        cursor.mergeCharFormat(current_format)
        
        # Scroll to the match by positioning cursor without selection
        cursor.setPosition(start)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
    
    def _update_match_label(self):
        """Update the match counter label."""
        if self.matches:
            self.match_label.setText(f"{self.current_match_index + 1}/{len(self.matches)}")
        else:
            self.match_label.setText("0/0")
    
    def _find_next(self):
        """Find next match."""
        if not self.matches:
            return
        
        self.current_match_index = (self.current_match_index + 1) % len(self.matches)
        self._highlight_current_match()
        self._update_match_label()
    
    def _find_previous(self):
        """Find previous match."""
        if not self.matches:
            return
        
        self.current_match_index = (self.current_match_index - 1) % len(self.matches)
        self._highlight_current_match()
        self._update_match_label()
    
    def show_and_focus(self):
        """Show the widget and focus the search input."""
        self.show()
        self.search_input.setFocus()
        self.search_input.selectAll()
