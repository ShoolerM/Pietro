"""Planning Mode Dialog for interactive story outline creation."""
from PyQt5 import QtWidgets, QtCore, QtGui


class PlanningInputField(QtWidgets.QTextEdit):
    """Custom text input field with Enter to send, Shift+Enter for newline."""
    
    send_requested = QtCore.pyqtSignal(str)  # Emitted when user presses Enter
    
    def keyPressEvent(self, event):
        """Handle key press events.
        
        Enter: Send message
        Shift+Enter: Add newline
        """
        if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
            # Check if Shift is held
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                # Shift+Enter: Add newline
                super().keyPressEvent(event)
            else:
                # Enter: Send message
                text = self.toPlainText().strip()
                if text:
                    self.send_requested.emit(text)
                    self.clear()
                event.accept()
        else:
            super().keyPressEvent(event)


class PlanningModeDialog(QtWidgets.QDialog):
    """Dialog for interactive planning mode with LLM conversation.
    
    Users converse with LLM to build a story outline as a markdown checklist,
    then click 'Start Writing' to proceed with story generation guided by outline.
    """
    
    # Signals
    user_input_ready = QtCore.pyqtSignal(str)  # User submitted input
    start_writing_clicked = QtCore.pyqtSignal(str)  # User clicked Start Writing, emit final outline
    dialog_cancelled = QtCore.pyqtSignal()  # User clicked Cancel
    llm_token_received = QtCore.pyqtSignal(str)  # Streaming token from LLM (thread-safe)
    llm_response_complete = QtCore.pyqtSignal(str)  # Full LLM response received
    set_waiting = QtCore.pyqtSignal(bool)  # Control waiting state
    
    def __init__(self, parent=None, initial_prompt="What story would you like to plan?"):
        """Initialize the Planning Mode Dialog.
        
        Args:
            parent: Parent widget
            initial_prompt: The initial prompt to show the user (from settings)
        """
        super().__init__(parent)
        self.setWindowTitle('Story Planning Mode')
        self.resize(800, 600)
        self.setModal(True)
        
        self._initial_prompt = initial_prompt
        self._outline_items = []  # Track outline tasks as they're generated
        self._current_llm_response = ""  # Accumulate full response
        self._current_llm_response_chunk = ""  # Current message being displayed
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title_label = QtWidgets.QLabel('Story Planning Assistant')
        title_font = title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        info_label = QtWidgets.QLabel(
            'Have a conversation with the LLM to develop your story outline.\n'
            'When you\'re satisfied with the outline, click "Start Writing" to begin generation.\n'
            'Tip: Press Enter to send, Shift+Enter to add a newline.'
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet('color: #666666; font-style: italic;')
        layout.addWidget(info_label)
        
        # Main content area with two columns
        content_layout = QtWidgets.QHBoxLayout()
        
        # Left column: Conversation
        left_layout = QtWidgets.QVBoxLayout()
        left_label = QtWidgets.QLabel('Conversation:')
        left_layout.addWidget(left_label)
        
        # Conversation display (read-only)
        self.conversation_display = QtWidgets.QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setStyleSheet(
            'QTextEdit { background-color: #f5f5f5; border: 1px solid #cccccc; }'
        )
        left_layout.addWidget(self.conversation_display)
        
        # User input field
        input_label = QtWidgets.QLabel('Your response:')
        left_layout.addWidget(input_label)
        
        self.user_input_field = PlanningInputField()
        self.user_input_field.setMaximumHeight(80)
        self.user_input_field.setPlaceholderText('Type your response or ideas for the outline...')
        left_layout.addWidget(self.user_input_field)
        
        # Send button (can still click if preferred)
        send_button = QtWidgets.QPushButton('Send (or press Enter)')
        send_button.clicked.connect(self._on_send_input)
        left_layout.addWidget(send_button)
        
        content_layout.addLayout(left_layout, 2)
        
        # Right column: Outline display
        right_layout = QtWidgets.QVBoxLayout()
        right_label = QtWidgets.QLabel('Story Outline:')
        right_layout.addWidget(right_label)
        
        self.outline_display = QtWidgets.QTextEdit()
        self.outline_display.setReadOnly(False)
        self.outline_display.setStyleSheet(
            'QTextEdit { background-color: #fffaf0; border: 1px solid #cccccc; color: #000000; }'
        )
        self.outline_display.setPlaceholderText(
            'Outline will appear here as the conversation develops...'
        )
        right_layout.addWidget(self.outline_display)
        
        content_layout.addLayout(right_layout, 1)
        
        layout.addLayout(content_layout, 1)
        
        # Bottom buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.start_writing_button = QtWidgets.QPushButton('Start Writing')
        self.start_writing_button.setStyleSheet(
            'QPushButton { background-color: #4CAF50; color: white; '
            'padding: 6px; font-weight: bold; }'
        )
        self.start_writing_button.clicked.connect(self._on_start_writing)
        button_layout.addWidget(self.start_writing_button)
        
        cancel_button = QtWidgets.QPushButton('Cancel')
        cancel_button.clicked.connect(self._on_cancel)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect internal signals."""
        # Connect custom input field send signal to handler
        self.user_input_field.send_requested.connect(self._on_send_input)
        
        # Connect streaming signals for thread-safe updates
        self.llm_token_received.connect(self._on_llm_token, QtCore.Qt.QueuedConnection)
        self.llm_response_complete.connect(self._on_llm_response_complete, QtCore.Qt.QueuedConnection)
        self.set_waiting.connect(self._set_waiting_state, QtCore.Qt.QueuedConnection)

        # Enable Start Writing when outline content changes (manual edits allowed)
        self.outline_display.textChanged.connect(self._update_start_writing_state)
        
        # Enable starting to write only when there's outline content
        self.start_writing_button.setEnabled(False)
    
    def _on_send_input(self):
        """Handle user input submission."""
        user_text = self.user_input_field.toPlainText().strip()
        if not user_text:
            return
        
        # Append user message to conversation
        self._append_conversation(f"You: {user_text}", is_user=True)
        
        # Clear input field
        self.user_input_field.clear()

        # Reset per-response accumulators
        self._current_llm_response = ""
        self._current_llm_response_chunk = ""
        
        # Emit signal for controller to handle LLM response
        self.user_input_ready.emit(user_text)
    
    def _on_start_writing(self):
        """Handle Start Writing button click."""
        outline = self.outline_display.toPlainText().strip()
        if not outline:
            QtWidgets.QMessageBox.warning(
                self,
                'No Outline',
                'Please develop an outline with the LLM before starting to write.'
            )
            return
        
        self.start_writing_clicked.emit(outline)
        self.accept()
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        reply = QtWidgets.QMessageBox.question(
            self,
            'Cancel Planning',
            'Are you sure you want to cancel planning mode?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.dialog_cancelled.emit()
            self.reject()
    
    def append_llm_response(self, text):
        """Accumulate LLM response token (called from signal, not thread).
        
        Args:
            text: The text token from LLM to append
        """
        self._current_llm_response += text
        self._current_llm_response_chunk += text
    
    def _on_llm_token(self, token):
        """Handle individual token from LLM streaming (thread-safe via signal).
        
        Args:
            token: Single token from LLM
        """
        self._current_llm_response += token
        self._current_llm_response_chunk += token
    
    def _on_llm_response_complete(self, full_response):
        """Handle completion of LLM response (thread-safe via signal).
        
        Args:
            full_response: The complete response from LLM
        """
        # Check if response is an outline (starts with dash and checkbox)
        response_text = full_response if full_response is not None else self._current_llm_response
        clean_response = response_text.strip()
        if clean_response.startswith('- ['):
            # This is an outline - update outline display (don't show in conversation)
            self.set_outline_from_llm(clean_response)
            # Notify user that outline was generated
            self._append_conversation("Assistant: Outline generated! Review it on the right and click 'Start Writing' when ready.", is_user=False)
        else:
            # This is a regular response - display in conversation
            display_text = self._current_llm_response_chunk.strip() or clean_response
            self._append_conversation(f"Assistant: {display_text}", is_user=False)
        
        # Reset chunk accumulator for next response
        self._current_llm_response_chunk = ""
        self._current_llm_response = ""
    
    def _set_waiting_state(self, waiting):
        """Set the waiting state (thread-safe via signal).
        
        Args:
            waiting: True if waiting for LLM, False otherwise
        """
        self.user_input_field.setEnabled(not waiting)
    
    def _append_conversation(self, text, is_user=False):
        """Append text to conversation display with styling.
        
        Args:
            text: Text to append
            is_user: True if this is user input, False if LLM response
        """
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        
        char_format = QtGui.QTextCharFormat()
        if is_user:
            char_format.setForeground(QtGui.QColor('#1a73e8'))  # Blue for user
            char_format.setFontWeight(QtGui.QFont.Bold)
        else:
            char_format.setForeground(QtGui.QColor('#202124'))  # Dark gray for assistant
        
        cursor.setCharFormat(char_format)
        cursor.insertText(text + '\n\n')
        
        # Auto-scroll to bottom
        self.conversation_display.setTextCursor(cursor)
        self.conversation_display.ensureCursorVisible()
    
    def set_outline_from_llm(self, outline_text):
        """Update outline display based on LLM suggestion.
        
        When LLM generates or suggests a markdown checklist outline,
        this method shows it in the outline display. User can approve
        by clicking "Start Writing" or request modifications by continuing
        the conversation.
        
        Args:
            outline_text: Markdown checklist outline from LLM
        """
        self.outline_display.setPlainText(outline_text)
        
        # Enable Start Writing button when we have an outline
        self._update_start_writing_state()

    def _update_start_writing_state(self):
        """Enable Start Writing button when outline is non-empty."""
        outline_text = self.outline_display.toPlainText().strip()
        self.start_writing_button.setEnabled(bool(outline_text))
    
    def get_conversation_text(self):
        """Get the full conversation text.
        
        Returns:
            str: The conversation history
        """
        return self.conversation_display.toPlainText()
    
    def get_current_outline(self):
        """Get the current outline text.
        
        Returns:
            str: The markdown checklist outline
        """
        return self.outline_display.toPlainText()
    
    def show_with_initial_prompt(self):
        """Show the dialog and display the initial prompt."""
        self._append_conversation(f"Assistant: {self._initial_prompt}", is_user=False)
        self.user_input_field.setFocus()
        return self.exec_()
