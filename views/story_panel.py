"""Story panel view for displaying story/output content."""

import os
from PyQt5 import QtWidgets, QtCore, QtGui
import markdown
from views.search_widget import SearchWidget


class StoryPanel(QtWidgets.QWidget):
    """Panel for story output and file editing tabs."""

    # Signals
    file_saved = QtCore.pyqtSignal(str, str)  # file_path, content
    font_size_changed = QtCore.pyqtSignal(int)  # delta
    toggle_thinking_requested = QtCore.pyqtSignal()  # request to toggle LLM Panel
    update_summary_requested = QtCore.pyqtSignal()  # request to regenerate summary
    toggle_summarize_prompts_requested = (
        QtCore.pyqtSignal()
    )  # request to toggle prompt summarization
    toggle_smart_mode_requested = (
        QtCore.pyqtSignal()
    )  # request to toggle build with Smart Mode
    auto_build_story_requested = (
        QtCore.pyqtSignal()
    )  # request to automatically build complete story
    override_selection_requested = QtCore.pyqtSignal(
        str, int, int
    )  # selected_text, start_pos, end_pos
    update_selection_with_prompt_requested = QtCore.pyqtSignal(
        str, int, int, str
    )  # selected_text, start_pos, end_pos, prompt
    update_accepted = QtCore.pyqtSignal()  # user accepted the update
    update_rejected = QtCore.pyqtSignal()  # user rejected the update
    send_requested = QtCore.pyqtSignal()  # Ctrl+Enter: Send prompt
    undo_requested = QtCore.pyqtSignal()  # Alt+Z: Undo last story change
    stop_requested = QtCore.pyqtSignal()  # Escape: Stop generation
    clear_requested = QtCore.pyqtSignal()  # Alt+Shift+X: Clear story

    def __init__(self):
        super().__init__()

        # Setup keyboard shortcuts
        self._setup_shortcuts()

        # Track open file tabs
        self.open_file_tabs = {}

        # Markdown rendering state
        self._markdown_enabled = True

        # LLM Panel visibility state (tracked externally, updated via setter)
        self._thinking_visible = True
        # Summarization of prompts enabled state
        self._summarize_prompts_enabled = True
        # Build with RAG enabled state
        self._smart_mode = False

        # Text update state for streaming replacement
        self._update_cursor = None
        self._update_start_pos = None
        self._update_end_pos = None
        self._update_new_text_start = None  # Start position of new green text
        self._update_new_text_end = None  # End position of new green text
        self._update_active = False
        self._original_text = None  # Store original text for rejection
        self._accept_reject_widget = None  # Widget with Accept/Reject buttons

        # Search widget (created per tab)
        self.search_widgets = {}  # Maps tab index to search widget

        self._init_ui()

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for story panel actions."""
        # Send: Ctrl+Enter
        send_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        send_shortcut.activated.connect(lambda: self.send_requested.emit())

        # Undo: Alt+Z
        undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Alt+Z"), self)
        undo_shortcut.activated.connect(lambda: self.undo_requested.emit())

        # Stop: Escape
        stop_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self)
        stop_shortcut.activated.connect(lambda: self.stop_requested.emit())

        # Clear: Alt+Shift+X
        clear_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Alt+Shift+X"), self)
        clear_shortcut.activated.connect(lambda: self.clear_requested.emit())

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Tabbed interface for story output and file editing
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)

        # Create the story/output tab (always first, not closeable)
        story_tab_container = QtWidgets.QWidget()
        story_tab_layout = QtWidgets.QVBoxLayout()
        story_tab_layout.setContentsMargins(0, 0, 0, 0)
        story_tab_layout.setSpacing(0)

        self.story_text = QtWidgets.QTextEdit()
        self.story_text.setAcceptRichText(True)
        self.story_text.setPlaceholderText(
            "Response output (appended as streamed output)"
        )
        self.story_text.installEventFilter(self)
        self.story_text.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.story_text.customContextMenuRequested.connect(self._show_context_menu)
        self.story_text.textChanged.connect(self._mark_story_modified)
        self.story_file_path = None
        self.story_modified = False

        story_tab_layout.addWidget(self.story_text)
        story_tab_container.setLayout(story_tab_layout)

        self.tab_widget.addTab(story_tab_container, "Story")

        # Create search widget for story tab
        story_search = SearchWidget(self.story_text, story_tab_container)
        story_search.hide()
        story_search.close_requested.connect(story_search.hide)
        story_tab_layout.insertWidget(0, story_search)
        self.search_widgets[0] = story_search

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

        # Add Ctrl+S shortcut for saving files
        self.save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.save_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.save_shortcut.activated.connect(self._save_current_file)

        # Add Ctrl+R shortcut for updating selected text
        self.update_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        self.update_shortcut.activated.connect(self._on_update_selection_requested)

        # Add Ctrl+F shortcut for search and replace
        self.search_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        self.search_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.search_shortcut.activated.connect(self.show_replace)

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

    def _show_search(self):
        """Show search widget for current active tab."""
        current_index = self.tab_widget.currentIndex()

        if current_index in self.search_widgets:
            search_widget = self.search_widgets[current_index]
            search_widget.show_and_focus()
        else:
            # No search widget for this tab (shouldn't happen but handle gracefully)
            pass

    def show_search(self):
        """Public method to show search widget."""
        self._show_search()

    def show_replace(self):
        """Show search and replace widget for current active tab."""
        current_index = self.tab_widget.currentIndex()

        if current_index in self.search_widgets:
            search_widget = self.search_widgets[current_index]
            search_widget.show_replace()
        else:
            # No search widget for this tab (shouldn't happen but handle gracefully)
            pass

    def isReadOnly(self):
        """Return read-only state of the story text editor."""
        return self.story_text.isReadOnly()

    def _show_context_menu(self, position):
        """Show context menu for the story text widget."""
        menu = QtWidgets.QMenu(self)

        # Send action (Ctrl+Enter)
        send_action = menu.addAction("Send Prompt")
        send_action.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        send_action.triggered.connect(lambda: self.send_requested.emit())

        # Undo action (Alt+Z)
        undo_action = menu.addAction("Undo")
        undo_action.setShortcut(QtGui.QKeySequence("Alt+Z"))
        undo_action.triggered.connect(lambda: self.undo_requested.emit())

        # Stop action (Escape)
        stop_action = menu.addAction("Stop Generation")
        stop_action.setShortcut(QtGui.QKeySequence("Escape"))
        stop_action.triggered.connect(lambda: self.stop_requested.emit())

        # Clear action (Alt+Shift+X)
        clear_action = menu.addAction("Clear Story")
        clear_action.setShortcut(QtGui.QKeySequence("Alt+Shift+X"))
        clear_action.triggered.connect(lambda: self.clear_requested.emit())

        menu.addSeparator()

        # Check if text is selected
        cursor = self.story_text.textCursor()
        has_selection = cursor.hasSelection()

        # Update Selected Text action (only if text is selected)
        if has_selection:
            update_action = menu.addAction("✨ Update Selected Text")
            update_action.triggered.connect(self._on_update_selection_requested)
            menu.addSeparator()

        # Update Summary action
        update_summary_action = menu.addAction("🔄 Update Summary")
        update_summary_action.triggered.connect(
            lambda: self.update_summary_requested.emit()
        )

        # Summarize Prompts toggle
        summarize_text = (
            "Summarize Prompts: ON"
            if self._summarize_prompts_enabled
            else "Summarize Prompts: OFF"
        )
        summarize_action = menu.addAction(summarize_text)
        summarize_action.triggered.connect(
            lambda: self.toggle_summarize_prompts_requested.emit()
        )

        menu.addSeparator()

        # Build with RAG toggle
        build_rag_text = (
            "Build with RAG: ON" if self._smart_mode else "Build with RAG: OFF"
        )
        build_rag_action = menu.addAction(build_rag_text)
        build_rag_action.triggered.connect(
            lambda: self.toggle_smart_mode_requested.emit()
        )

        menu.addSeparator()

        # Toggle LLM Panel action
        thinking_text = "Hide LLM Panel" if self._thinking_visible else "Show LLM Panel"
        thinking_action = menu.addAction(thinking_text)
        thinking_action.triggered.connect(lambda: self.toggle_thinking_requested.emit())

        # Show menu at cursor position
        menu.exec_(self.story_text.mapToGlobal(position))

    def _save_current_file(self):
        """Save the currently active file tab when Ctrl+S is pressed."""
        current_index = self.tab_widget.currentIndex()

        # Save the Story tab (index 0)
        if current_index == 0:
            if not self.story_file_path:
                self.save_story_file_as()
            else:
                self.save_story_file()
            return

        # Find the file path for the current tab
        for file_path, data in self.open_file_tabs.items():
            if data["index"] == current_index:
                # Get the content from the editor and save
                self._save_file(file_path, data["editor"])
                break

    def _mark_file_modified(self, file_path):
        """Mark a file as modified (add asterisk to tab name)."""
        if file_path not in self.open_file_tabs:
            return

        data = self.open_file_tabs[file_path]

        # Only update if not already marked as modified
        if not data["modified"]:
            data["modified"] = True
            tab_index = data["index"]
            original_name = data["original_filename"]
            self.tab_widget.setTabText(tab_index, f"{original_name} *")

    def _save_file(self, file_path, editor):
        """Save a file and remove the modified indicator."""
        content = editor.toPlainText()

        # Emit the save signal
        self.file_saved.emit(file_path, content)

        # Mark as not modified and update tab name
        if file_path in self.open_file_tabs:
            data = self.open_file_tabs[file_path]
            data["modified"] = False
            tab_index = data["index"]
            original_name = data["original_filename"]
            self.tab_widget.setTabText(tab_index, original_name)

    def _mark_story_modified(self):
        """Mark story tab as modified."""
        if not self.story_modified:
            self.story_modified = True
            current_label = self.tab_widget.tabText(0)
            if not current_label.endswith(" *"):
                self.tab_widget.setTabText(0, f"{current_label} *")

    def load_story_file(self):
        """Load a file into the story tab."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not open file:\n{e}")
            return
        self.story_text.blockSignals(True)
        try:
            self.story_text.setPlainText(content)
        finally:
            self.story_text.blockSignals(False)
        self.story_file_path = file_path
        self.story_modified = False
        self.tab_widget.setTabText(0, QtCore.QFileInfo(file_path).fileName())

    def save_story_file(self):
        """Save story tab content. If no file is loaded, perform Save As."""
        if not self.story_file_path:
            self.save_story_file_as()
            return
        content = self.story_text.toPlainText()
        self.file_saved.emit(self.story_file_path, content)
        self.story_modified = False
        self.tab_widget.setTabText(0, QtCore.QFileInfo(self.story_file_path).fileName())

    def save_story_file_as(self):
        """Save story tab content to a new file path."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File As", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return
        self.story_file_path = file_path
        self.save_story_file()

    def _close_tab(self, index):
        """Close a tab (but not the Story tab at index 0)."""
        if index == 0:
            self.clear_story_file()
            return

        tab_widget = self.tab_widget.widget(index)

        file_to_remove = None
        for file_path, data in self.open_file_tabs.items():
            if data["container"] == tab_widget:
                file_to_remove = file_path
                # Check if file has unsaved changes
                if data.get("modified", False):
                    filename = data["original_filename"]
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Unsaved Changes",
                        f"'{filename}' has unsaved changes. Do you want to close it anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        return  # Don't close the tab
                break

        if file_to_remove:
            del self.open_file_tabs[file_to_remove]

        # Clean up search widget for this tab
        if index in self.search_widgets:
            del self.search_widgets[index]

        # Update search widget indices
        new_search_widgets = {}
        for idx, widget in self.search_widgets.items():
            if idx > index:
                new_search_widgets[idx - 1] = widget
            else:
                new_search_widgets[idx] = widget
        self.search_widgets = new_search_widgets

        for file_path, data in self.open_file_tabs.items():
            if data["index"] > index:
                data["index"] -= 1

        self.tab_widget.removeTab(index)

    def clear_story_file(self):
        """Clear the story tab to an in-memory document."""
        self.story_text.blockSignals(True)
        try:
            self.story_text.clear()
        finally:
            self.story_text.blockSignals(False)
        self.story_file_path = None
        self.story_modified = False
        self.tab_widget.setTabText(0, "Story")

    # Public methods

    def get_story_content(self):
        """Get current story content as plain text."""
        return self.story_text.toPlainText()

    def set_story_content(self, content):
        """Set story content."""
        if self._markdown_enabled:
            self.story_text.setHtml(self._markdown_to_html(content))
        else:
            self.story_text.setPlainText(content)

        # Scroll to bottom
        try:
            self.story_text.verticalScrollBar().setValue(
                self.story_text.verticalScrollBar().maximum()
            )
        except Exception:
            pass

    def append_story_content(self, text):
        """Append text to story content."""
        cursor = self.story_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.story_text.setTextCursor(cursor)
        cursor.insertText(text)

        try:
            self.story_text.verticalScrollBar().setValue(
                self.story_text.verticalScrollBar().maximum()
            )
        except Exception:
            pass

    def render_story_markdown(self, markdown_content):
        """Render markdown content in story view."""
        if self._markdown_enabled:
            try:
                styled_html = self._markdown_to_html(markdown_content)
                self.story_text.setHtml(styled_html)
                # Scroll to bottom after rendering
                self.story_text.verticalScrollBar().setValue(
                    self.story_text.verticalScrollBar().maximum()
                )
            except Exception:
                pass

    def clear_story_content(self):
        """Clear story content."""
        self.story_text.clear()

    def set_markdown_enabled(self, enabled):
        """Set whether markdown rendering is enabled."""
        self._markdown_enabled = enabled

    def is_markdown_enabled(self):
        """Check if markdown rendering is enabled."""
        return self._markdown_enabled

    def set_thinking_visible(self, visible):
        """Set LLM Panel visibility state (for context menu display)."""
        self._thinking_visible = visible

    def set_summarize_prompts_enabled(self, enabled: bool):
        """Update internal state for whether prompt summarization is enabled.

        This updates the label shown in the context menu next time it's opened.
        """
        self._summarize_prompts_enabled = bool(enabled)

    def set_smart_mode(self, enabled: bool):
        """Update internal state for whether build with Smart Mode is enabled.

        This updates the label shown in the context menu next time it's opened.
        """
        self._smart_mode = bool(enabled)

    def open_file_tab(self, file_path):
        """Open a file in a new editable tab.

        Args:
            file_path: Path to the file to open

        Returns:
            bool: True if successful, False otherwise
        """
        # If file is already open, switch to that tab
        if file_path in self.open_file_tabs:
            tab_index = self.open_file_tabs[file_path]["index"]
            self.tab_widget.setCurrentIndex(tab_index)
            return True

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not open file:\n{e}")
            return False

        # Create new tab with text editor
        file_editor = QtWidgets.QTextEdit()
        file_editor.setAcceptRichText(False)
        file_editor.setPlainText(content)
        file_editor.installEventFilter(self)

        # Connect to detect changes
        file_editor.textChanged.connect(lambda: self._mark_file_modified(file_path))

        # Create container with save button
        tab_container = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(lambda: self._save_file(file_path, file_editor))

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(save_button)
        button_layout.addStretch()

        tab_layout.addLayout(button_layout)
        tab_layout.addWidget(file_editor)
        tab_container.setLayout(tab_layout)

        # Add tab
        filename = os.path.basename(file_path)
        tab_index = self.tab_widget.addTab(tab_container, filename)

        # Create search widget for this file tab
        file_search = SearchWidget(file_editor, tab_container)
        file_search.hide()
        file_search.close_requested.connect(file_search.hide)
        tab_layout.insertWidget(0, file_search)
        self.search_widgets[tab_index] = file_search

        # Track the tab
        self.open_file_tabs[file_path] = {
            "index": tab_index,
            "editor": file_editor,
            "container": tab_container,
            "modified": False,
            "original_filename": filename,
        }

        self.tab_widget.setCurrentIndex(tab_index)
        return True

    def apply_font_size(self, size):
        """Apply font size to text widgets."""
        font = QtGui.QFont()
        font.setPointSize(size)

        try:
            self.story_text.setFont(font)
        except Exception:
            pass

        for file_path, data in self.open_file_tabs.items():
            try:
                data["editor"].setFont(font)
            except Exception:
                pass

    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """Convert markdown to styled HTML."""
        html_content = markdown.markdown(
            markdown_text, extensions=["fenced_code", "tables", "nl2br"]
        )
        styled_html = f"""
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #ffffff;
                background-color: #2b2b2b;
                max-width: 100%;
                padding: 10px;
                white-space: pre-wrap;
            }}
            code {{
                background-color: #3c3c3c;
                color: #e0e0e0;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            }}
            pre {{
                background-color: #3c3c3c;
                color: #e0e0e0;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            pre code {{
                background-color: transparent;
                color: #e0e0e0;
                padding: 0;
            }}
            blockquote {{
                border-left: 4px solid #555;
                padding-left: 15px;
                margin-left: 0;
                color: #aaa;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid #555;
                padding: 8px;
                text-align: left;
                color: #ffffff;
            }}
            th {{
                background-color: #3c3c3c;
                color: #ffffff;
                font-weight: bold;
            }}
            h1, h2, h3, h4, h5, h6 {{
                margin-top: 20px;
                margin-bottom: 10px;
                color: #ffffff;
            }}
            p {{
                color: #ffffff;
                white-space: pre-wrap;
            }}
            li {{
                color: #ffffff;
            }}
            a {{
                color: #6cb6ff;
            }}
            strong {{
                color: #ffffff;
            }}
            em {{
                color: #ffffff;
            }}
        </style>
        {html_content}
        """
        return styled_html

    def _on_update_selection_requested(self):
        """Handle update selection request from context menu."""
        cursor = self.story_text.textCursor()
        if not cursor.hasSelection():
            return

        selected_text = cursor.selectedText()
        start_pos = cursor.selectionStart()
        end_pos = cursor.selectionEnd()

        # Show dialog asking for change instructions
        prompt = self._show_update_prompt_dialog()

        if prompt is not None:  # None means user cancelled
            # Emit signal with selection info and prompt
            self.update_selection_with_prompt_requested.emit(
                selected_text, start_pos, end_pos, prompt
            )

    def _show_update_prompt_dialog(self):
        """Show dialog asking user for change instructions.

        Returns:
            str: The prompt entered by user, or None if cancelled
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Update Text")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(200)

        layout = QtWidgets.QVBoxLayout()

        # Instruction label
        label = QtWidgets.QLabel(
            "What changes would you like to make to the selected text?"
        )
        layout.addWidget(label)

        # Text input
        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlaceholderText(
            "e.g., Make this all caps, Rewrite as a question, Add more detail..."
        )
        layout.addWidget(text_edit)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        update_button = QtWidgets.QPushButton("Update")
        cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(update_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        result = {"prompt": None}

        def on_update():
            prompt = text_edit.toPlainText().strip()
            if prompt:
                result["prompt"] = prompt
                dialog.accept()
            else:
                # Show warning if empty
                QtWidgets.QMessageBox.warning(
                    dialog, "Empty Prompt", "Please enter change instructions."
                )

        update_button.clicked.connect(on_update)
        cancel_button.clicked.connect(dialog.reject)

        # Create event filter for Enter key handling
        class EnterKeyFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.KeyPress:
                    if (
                        event.key() == QtCore.Qt.Key_Return
                        and event.modifiers() == QtCore.Qt.NoModifier
                    ):
                        on_update()
                        return True
                return False

        # Install event filter for Enter key
        filter_obj = EnterKeyFilter()
        text_edit.installEventFilter(filter_obj)

        text_edit.setFocus()
        update_button.setDefault(True)

        dialog.exec_()

        return result["prompt"]

    def start_text_update(self, start_pos, end_pos):
        """Initialize streaming text replacement at the given position.

        Args:
            start_pos: Start position of text to replace
            end_pos: End position of text to replace
        """
        # Create cursor at selection and store original text
        cursor = self.story_text.textCursor()
        cursor.setPosition(start_pos)
        cursor.setPosition(end_pos, QtGui.QTextCursor.KeepAnchor)

        # Store original text for potential rejection
        self._original_text = cursor.selectedText()

        # Apply RED color to original text to show it's being replaced
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(255, 100, 100))  # Red for original text
        cursor.mergeCharFormat(fmt)

        # Position cursor after the original text
        cursor.setPosition(end_pos)

        # Insert separator for visual distinction
        cursor.insertText(" → ")

        # Store the position where new text will start
        self._update_new_text_start = cursor.position()

        # Store cursor and positions for streaming new text
        self._update_cursor = cursor
        self._update_start_pos = start_pos
        self._update_end_pos = end_pos
        self._update_active = True

    def stream_override_text(self, text_chunk):
        """Stream replacement text after the original text.

        Args:
            text_chunk: Text chunk to insert
        """
        if not self._update_active or self._update_cursor is None:
            return

        # Apply GREEN color to new text
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(100, 200, 100))  # Green for new text

        # Insert text with green formatting
        self._update_cursor.insertText(text_chunk, fmt)

        # Keep cursor at end of inserted text
        self.story_text.setTextCursor(self._update_cursor)

    def finish_text_update(self):
        """Finalize the text update operation and show accept/reject UI."""
        if not self._update_active:
            return

        # Get the end position of the new text
        if self._update_cursor:
            self._update_new_text_end = self._update_cursor.position()

        # Show accept/reject widget
        self._show_accept_reject_widget()

    def _show_accept_reject_widget(self):
        """Show accept/reject buttons for the update."""
        # Remove existing widget if any
        if self._accept_reject_widget:
            self._accept_reject_widget.deleteLater()

        # Create widget with buttons
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        label = QtWidgets.QLabel("Update complete:")
        accept_btn = QtWidgets.QPushButton("✓ Accept")
        reject_btn = QtWidgets.QPushButton("✗ Reject")

        # Style buttons
        accept_btn.setStyleSheet(
            "background-color: #28a745; color: white; padding: 5px 15px;"
        )
        reject_btn.setStyleSheet(
            "background-color: #dc3545; color: white; padding: 5px 15px;"
        )

        layout.addWidget(label)
        layout.addWidget(accept_btn)
        layout.addWidget(reject_btn)
        layout.addStretch()

        widget.setLayout(layout)
        widget.setStyleSheet("background-color: #3c3c3c; border-radius: 3px;")

        # Connect buttons
        accept_btn.clicked.connect(self._on_accept_update)
        reject_btn.clicked.connect(self._on_reject_update)

        # Add widget to layout below story text
        self.layout().insertWidget(1, widget)
        self._accept_reject_widget = widget

    def _on_accept_update(self):
        """Handle accept button click - keep green text, remove red original."""
        # Store the length of new text before deletion
        new_text_length = self._update_new_text_end - self._update_new_text_start

        # Remove the red original text and separator
        if (
            self._update_start_pos is not None
            and self._update_new_text_start is not None
        ):
            cursor = self.story_text.textCursor()
            # Select from start to the beginning of new text (includes red original + separator)
            cursor.setPosition(self._update_start_pos)
            cursor.setPosition(
                self._update_new_text_start, QtGui.QTextCursor.KeepAnchor
            )
            # Delete the red original text and separator
            cursor.removeSelectedText()

        # Clear formatting from the new text (make it normal white)
        # After deletion, the new text is now at _update_start_pos to _update_start_pos + new_text_length
        if self._update_start_pos is not None and new_text_length > 0:
            cursor = self.story_text.textCursor()
            cursor.setPosition(self._update_start_pos)
            cursor.setPosition(
                self._update_start_pos + new_text_length, QtGui.QTextCursor.KeepAnchor
            )

            # Reset to default white text
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(255, 255, 255))  # White text
            cursor.mergeCharFormat(fmt)

        # Remove accept/reject widget
        if self._accept_reject_widget:
            self._accept_reject_widget.deleteLater()
            self._accept_reject_widget = None

        # Reset state
        self._update_cursor = None
        self._update_start_pos = None
        self._update_end_pos = None
        self._update_new_text_start = None
        self._update_new_text_end = None
        self._update_active = False
        self._original_text = None

        # Emit signal
        self.update_accepted.emit()

    def _on_reject_update(self):
        """Handle reject button click - keep red original, remove green new text."""
        # Remove the separator (" → ") and green new text
        if (
            self._update_new_text_start is not None
            and self._update_new_text_end is not None
        ):
            cursor = self.story_text.textCursor()
            # Select from 3 chars before (the separator " → ") to end of new text
            separator_start = self._update_new_text_start - 3
            cursor.setPosition(separator_start)
            cursor.setPosition(self._update_new_text_end, QtGui.QTextCursor.KeepAnchor)
            # Delete the separator and green new text
            cursor.removeSelectedText()

        # Clear red coloring from original text (make it normal white)
        if (
            self._update_start_pos is not None
            and self._update_new_text_start is not None
        ):
            cursor = self.story_text.textCursor()
            cursor.setPosition(self._update_start_pos)
            # Select original text (without the separator)
            cursor.setPosition(
                self._update_new_text_start - 3, QtGui.QTextCursor.KeepAnchor
            )

            # Reset to default white text
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(255, 255, 255))  # White text
            cursor.mergeCharFormat(fmt)

        # Remove accept/reject widget
        if self._accept_reject_widget:
            self._accept_reject_widget.deleteLater()
            self._accept_reject_widget = None

        # Reset state
        self._update_cursor = None
        self._update_start_pos = None
        self._update_end_pos = None
        self._update_new_text_start = None
        self._update_new_text_end = None
        self._update_active = False
        self._original_text = None

        # Emit signal
        self.update_rejected.emit()
