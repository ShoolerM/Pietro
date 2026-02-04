"""Story panel view for displaying story/output content."""
import os
from PyQt5 import QtWidgets, QtCore, QtGui
import markdown


class StoryPanel(QtWidgets.QWidget):
    """Panel for story output and file editing tabs."""
    
    # Signals
    file_saved = QtCore.pyqtSignal(str, str)  # file_path, content
    font_size_changed = QtCore.pyqtSignal(int)  # delta
    toggle_thinking_requested = QtCore.pyqtSignal()  # request to toggle thinking panel
    toggle_markdown_requested = QtCore.pyqtSignal()  # request to toggle markdown
    update_summary_requested = QtCore.pyqtSignal()  # request to regenerate summary
    toggle_summarize_prompts_requested = QtCore.pyqtSignal()  # request to toggle prompt summarization
    toggle_build_with_rag_requested = QtCore.pyqtSignal()  # request to toggle build with RAG mode
    auto_build_story_requested = QtCore.pyqtSignal()  # request to automatically build complete story
    
    def __init__(self):
        super().__init__()
        
        # Track open file tabs
        self.open_file_tabs = {}
        
        # Markdown rendering state
        self._markdown_enabled = True
        
        # Thinking panel visibility state (tracked externally, updated via setter)
        self._thinking_visible = True
        # Summarization of prompts enabled state
        self._summarize_prompts_enabled = True
        # Build with RAG enabled state
        self._build_with_rag_enabled = False
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tabbed interface for story output and file editing
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        
        # Create the story/output tab (always first, not closeable)
        self.story_text = QtWidgets.QTextEdit()
        self.story_text.setAcceptRichText(True)
        self.story_text.setPlaceholderText('Response output (appended as streamed output)')
        self.story_text.installEventFilter(self)
        self.story_text.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.story_text.customContextMenuRequested.connect(self._show_context_menu)
        self.tab_widget.addTab(self.story_text, "Story")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
        # Add Ctrl+S shortcut for saving files
        self.save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self._save_current_file)
    
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
    
    def _show_context_menu(self, position):
        """Show context menu for the story text widget."""
        menu = QtWidgets.QMenu(self)
        
        # Update Summary action
        update_summary_action = menu.addAction("🔄 Update Summary")
        update_summary_action.triggered.connect(lambda: self.update_summary_requested.emit())

        # Summarize Prompts toggle
        summarize_text = "Summarize Prompts: ON" if self._summarize_prompts_enabled else "Summarize Prompts: OFF"
        summarize_action = menu.addAction(summarize_text)
        summarize_action.triggered.connect(lambda: self.toggle_summarize_prompts_requested.emit())
        
        menu.addSeparator()
        
        # Build with RAG toggle
        build_rag_text = "Build with RAG: ON" if self._build_with_rag_enabled else "Build with RAG: OFF"
        build_rag_action = menu.addAction(build_rag_text)
        build_rag_action.triggered.connect(lambda: self.toggle_build_with_rag_requested.emit())
        
        menu.addSeparator()
        
        # Toggle Thinking Panel action
        thinking_text = "Hide Thinking Panel" if self._thinking_visible else "Show Thinking Panel"
        thinking_action = menu.addAction(thinking_text)
        thinking_action.triggered.connect(lambda: self.toggle_thinking_requested.emit())
        
        # Toggle Markdown action
        markdown_text = "Markdown: ON" if self._markdown_enabled else "Markdown: OFF"
        markdown_action = menu.addAction(markdown_text)
        markdown_action.triggered.connect(lambda: self.toggle_markdown_requested.emit())
        
        # Show menu at cursor position
        menu.exec_(self.story_text.mapToGlobal(position))
    
    def _save_current_file(self):
        """Save the currently active file tab when Ctrl+S is pressed."""
        current_index = self.tab_widget.currentIndex()
        
        # Don't try to save the Story tab (index 0)
        if current_index == 0:
            return
        
        # Find the file path for the current tab
        for file_path, data in self.open_file_tabs.items():
            if data['index'] == current_index:
                # Get the content from the editor and save
                self._save_file(file_path, data['editor'])
                break
    
    def _mark_file_modified(self, file_path):
        """Mark a file as modified (add asterisk to tab name)."""
        if file_path not in self.open_file_tabs:
            return
        
        data = self.open_file_tabs[file_path]
        
        # Only update if not already marked as modified
        if not data['modified']:
            data['modified'] = True
            tab_index = data['index']
            original_name = data['original_filename']
            self.tab_widget.setTabText(tab_index, f"{original_name} *")
    
    def _save_file(self, file_path, editor):
        """Save a file and remove the modified indicator."""
        content = editor.toPlainText()
        
        # Emit the save signal
        self.file_saved.emit(file_path, content)
        
        # Mark as not modified and update tab name
        if file_path in self.open_file_tabs:
            data = self.open_file_tabs[file_path]
            data['modified'] = False
            tab_index = data['index']
            original_name = data['original_filename']
            self.tab_widget.setTabText(tab_index, original_name)
    
    def _close_tab(self, index):
        """Close a tab (but not the Story tab at index 0)."""
        if index == 0:
            return
        
        tab_widget = self.tab_widget.widget(index)
        
        file_to_remove = None
        for file_path, data in self.open_file_tabs.items():
            if data['container'] == tab_widget:
                file_to_remove = file_path
                # Check if file has unsaved changes
                if data.get('modified', False):
                    filename = data['original_filename']
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        'Unsaved Changes',
                        f"'{filename}' has unsaved changes. Do you want to close it anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        return  # Don't close the tab
                break
        
        if file_to_remove:
            del self.open_file_tabs[file_to_remove]
        
        for file_path, data in self.open_file_tabs.items():
            if data['index'] > index:
                data['index'] -= 1
        
        self.tab_widget.removeTab(index)
    
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
            self.story_text.verticalScrollBar().setValue(self.story_text.verticalScrollBar().maximum())
        except Exception:
            pass
    
    def append_story_content(self, text):
        """Append text to story content."""
        cursor = self.story_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.story_text.setTextCursor(cursor)
        cursor.insertText(text)
        
        try:
            self.story_text.verticalScrollBar().setValue(self.story_text.verticalScrollBar().maximum())
        except Exception:
            pass
    
    def render_story_markdown(self, markdown_content):
        """Render markdown content in story view."""
        if self._markdown_enabled:
            try:
                styled_html = self._markdown_to_html(markdown_content)
                self.story_text.setHtml(styled_html)
                # Scroll to bottom after rendering
                self.story_text.verticalScrollBar().setValue(self.story_text.verticalScrollBar().maximum())
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
        """Set thinking panel visibility state (for context menu display)."""
        self._thinking_visible = visible

    def set_summarize_prompts_enabled(self, enabled: bool):
        """Update internal state for whether prompt summarization is enabled.

        This updates the label shown in the context menu next time it's opened.
        """
        self._summarize_prompts_enabled = bool(enabled)
    
    def set_build_with_rag_enabled(self, enabled: bool):
        """Update internal state for whether build with RAG mode is enabled.

        This updates the label shown in the context menu next time it's opened.
        """
        self._build_with_rag_enabled = bool(enabled)
    
    def open_file_tab(self, file_path):
        """Open a file in a new editable tab.
        
        Args:
            file_path: Path to the file to open
            
        Returns:
            bool: True if successful, False otherwise
        """
        # If file is already open, switch to that tab
        if file_path in self.open_file_tabs:
            tab_index = self.open_file_tabs[file_path]['index']
            self.tab_widget.setCurrentIndex(tab_index)
            return True
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
        
        save_button = QtWidgets.QPushButton('Save')
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
        
        # Track the tab
        self.open_file_tabs[file_path] = {
            'index': tab_index,
            'editor': file_editor,
            'container': tab_container,
            'modified': False,
            'original_filename': filename
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
                data['editor'].setFont(font)
            except Exception:
                pass
    
    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """Convert markdown to styled HTML."""
        html_content = markdown.markdown(
            markdown_text,
            extensions=['fenced_code', 'tables', 'nl2br']
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
