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
        self.replace_mode = False
        
        self._init_ui()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        # Install event filter on text_edit to catch Escape there too
        text_edit.installEventFilter(self)
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QtWidgets.QGridLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Row 0: Search
        # Search input
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Find...")
        self.search_input.textChanged.connect(self._on_search_text_changed)
        self.search_input.returnPressed.connect(self._find_next)
        main_layout.addWidget(self.search_input, 0, 0)
        
        # Match counter label
        self.match_label = QtWidgets.QLabel("0/0")
        self.match_label.setMinimumWidth(50)
        main_layout.addWidget(self.match_label, 0, 1)
        
        # Previous button
        prev_btn = QtWidgets.QPushButton("↑")
        prev_btn.setToolTip("Previous (Shift+Enter)")
        prev_btn.setMaximumWidth(30)
        prev_btn.clicked.connect(self._find_previous)
        main_layout.addWidget(prev_btn, 0, 2)
        
        # Next button
        next_btn = QtWidgets.QPushButton("↓")
        next_btn.setToolTip("Next (Enter)")
        next_btn.setMaximumWidth(30)
        next_btn.clicked.connect(self._find_next)
        main_layout.addWidget(next_btn, 0, 3)
        
        # Case sensitive checkbox
        self.case_sensitive_cb = QtWidgets.QCheckBox("Match case")
        self.case_sensitive_cb.stateChanged.connect(self._on_search_text_changed)
        main_layout.addWidget(self.case_sensitive_cb, 0, 4)
        
        # Regex checkbox
        self.regex_cb = QtWidgets.QCheckBox("Regex")
        self.regex_cb.stateChanged.connect(self._on_search_text_changed)
        main_layout.addWidget(self.regex_cb, 0, 5)
        
        # Close button
        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setToolTip("Close (Esc)")
        close_btn.setMaximumWidth(30)
        close_btn.clicked.connect(self._close)
        main_layout.addWidget(close_btn, 0, 6)
        
        # Row 1: Replace (initially hidden)
        # Replace input
        self.replace_input = QtWidgets.QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        main_layout.addWidget(self.replace_input, 1, 0)
        
        # Replace single button
        replace_single_btn = QtWidgets.QPushButton("Replace")
        replace_single_btn.setToolTip("Replace current (Enter)")
        replace_single_btn.setMaximumWidth(70)
        replace_single_btn.clicked.connect(self._replace_current)
        main_layout.addWidget(replace_single_btn, 1, 3)
        
        # Replace all button
        replace_all_btn = QtWidgets.QPushButton("Replace All")
        replace_all_btn.setToolTip("Replace all (Ctrl+Enter)")
        replace_all_btn.setMaximumWidth(85)
        replace_all_btn.clicked.connect(self._replace_all)
        main_layout.addWidget(replace_all_btn, 1, 4)
        
        # Store reference to replace row for showing/hiding
        self.replace_row_index = 1
        
        self.setLayout(main_layout)
        
        # Hide replace row initially
        self._set_replace_row_visible(False)
        
        # Style the widget
        self.setStyleSheet("""
            SearchWidget {
                background-color: palette(window);
                border: 1px solid palette(mid);
                border-radius: 3px;
            }
        """)
        
        # Install event filter on search input for Tab and other keys
        self.search_input.installEventFilter(self)
        self.replace_input.installEventFilter(self)
    
    def _set_replace_row_visible(self, visible):
        """Show or hide the replace row."""
        layout = self.layout()
        for col in range(layout.columnCount()):
            item = layout.itemAtPosition(self.replace_row_index, col)
            if item and item.widget():
                item.widget().setVisible(visible)
    
    def eventFilter(self, obj, event):
        """Handle Shift+Enter for previous search, Esc in search input, and Esc in text edit."""
        if event.type() == QtCore.QEvent.KeyPress:
            # Handle Esc in search input or when search widget is visible and text edit has Escape
            if event.key() == QtCore.Qt.Key_Escape:
                # Close search if it's visible
                if self.isVisible():
                    self._close()
                    return True
            # Handle Tab in search input - go directly to replace input
            elif obj == self.search_input and event.key() == QtCore.Qt.Key_Tab:
                if self.replace_input.isVisible():
                    self.replace_input.setFocus()
                    return True
            # Handle Shift+Tab in replace input - go back to search input
            elif obj == self.replace_input and event.key() == QtCore.Qt.Key_Tab and event.modifiers() & QtCore.Qt.ShiftModifier:
                self.search_input.setFocus()
                return True
            # Handle Shift+Enter in search input
            elif obj == self.search_input and event.key() == QtCore.Qt.Key_Return and event.modifiers() & QtCore.Qt.ShiftModifier:
                self._find_previous()
                return True
            # Handle Enter/Ctrl+Enter in replace input
            elif obj == self.replace_input and event.key() == QtCore.Qt.Key_Return:
                if event.modifiers() & QtCore.Qt.ControlModifier:
                    self._replace_all()
                else:
                    self._replace_current()
                return True
            # Handle Shift+Enter in replace input to go to next match
            elif obj == self.replace_input and event.key() == QtCore.Qt.Key_Return and event.modifiers() & QtCore.Qt.ShiftModifier:
                self._find_next()
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
    
    def _perform_search(self, prefer_pos=None):
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
            if prefer_pos is not None:
                self.current_match_index = 0
                for idx, (start, end) in enumerate(self.matches):
                    if start >= prefer_pos:
                        self.current_match_index = idx
                        break
            else:
                self.current_match_index = 0
            self._highlight_current_match()
            self._update_match_label()
        else:
            self.match_label.setText("0/0")
    
    def _clear_highlights(self):
        """Clear all search highlights."""
        self.text_edit.setExtraSelections([])
    
    def _highlight_all_matches(self):
        """Highlight all matches in light color."""
        selections = []
        highlight_format = QtGui.QTextCharFormat()
        highlight_format.setBackground(QtGui.QColor(100, 100, 0, 100))  # Dimmer yellow
        
        for start, end in self.matches:
            cursor = QtGui.QTextCursor(self.text_edit.document())
            cursor.setPosition(start)
            cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = highlight_format
            selections.append(sel)
        return selections
    
    def _highlight_current_match(self):
        """Highlight the current match more prominently."""
        if not self.matches or self.current_match_index < 0:
            return
        
        start, end = self.matches[self.current_match_index]
        
        # Re-highlight all matches first
        selections = self._highlight_all_matches()
        
        # Then highlight current match with bright yellow
        current_format = QtGui.QTextCharFormat()
        current_format.setBackground(QtGui.QColor(200, 200, 0, 150))  # Bright yellow, fully opaque
        
        cursor = QtGui.QTextCursor(self.text_edit.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        current_sel = QtWidgets.QTextEdit.ExtraSelection()
        current_sel.cursor = cursor
        current_sel.format = current_format
        selections.append(current_sel)
        self.text_edit.setExtraSelections(selections)
        
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
        # Re-perform search with existing text
        if self.search_input.text():
            self._perform_search()
        self.search_input.setFocus()
        self.search_input.selectAll()
    
    def toggle_replace_mode(self):
        """Toggle replace mode on/off."""
        self.replace_mode = not self.replace_mode
        if self.replace_mode:
            self._set_replace_row_visible(True)
            self.replace_input.setFocus()
        else:
            self._set_replace_row_visible(False)
            self.search_input.setFocus()
    
    def show_replace(self):
        """Show the search and replace widget."""
        self.replace_mode = True
        self._set_replace_row_visible(True)
        self.show()
        # Re-perform search with existing text
        if self.search_input.text():
            self._perform_search()
        self.search_input.setFocus()
    
    def _replace_current(self):
        """Replace the current match with the replacement text."""
        if not self.matches or self.current_match_index < 0:
            return
        
        start, end = self.matches[self.current_match_index]
        replacement_text = self.replace_input.text()
        
        # Update document with undo block (replace only current match)
        cursor = self.text_edit.textCursor()
        cursor.beginEditBlock()
        try:
            cursor.setPosition(start)
            cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
            cursor.insertText(replacement_text)
        finally:
            cursor.endEditBlock()
        
        # Re-search to update matches and position
        next_pos = start + len(replacement_text)
        self._perform_search(prefer_pos=next_pos)
        
        # Position cursor at the next match if available
        if self.matches and self.current_match_index < len(self.matches):
            start, end = self.matches[self.current_match_index]
            cursor = self.text_edit.textCursor()
            cursor.setPosition(start)
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()
    
    def _replace_all(self):
        """Replace all matches with the replacement text."""
        if not self.matches:
            return
        
        replacement_text = self.replace_input.text()
        search_text = self.search_input.text()
        
        # Get document content
        document = self.text_edit.document()
        content = document.toPlainText()
        
        # Perform replacement
        case_sensitive = self.case_sensitive_cb.isChecked()
        use_regex = self.regex_cb.isChecked()
        
        try:
            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                new_content = re.sub(search_text, replacement_text, content, flags=flags)
            else:
                if not case_sensitive:
                    # Case-insensitive replacement
                    new_content = content
                    pattern = re.compile(re.escape(search_text), re.IGNORECASE)
                    new_content = pattern.sub(replacement_text, new_content)
                else:
                    new_content = content.replace(search_text, replacement_text)
        except re.error:
            return
        
        # Update document with undo block
        cursor = self.text_edit.textCursor()
        cursor.beginEditBlock()
        try:
            cursor.select(QtGui.QTextCursor.Document)
            cursor.insertText(new_content)
        finally:
            cursor.endEditBlock()
        
        # Re-search to update matches
        self._perform_search()
