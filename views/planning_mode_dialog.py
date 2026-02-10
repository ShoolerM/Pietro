"""Planning Mode Dialog for interactive story outline creation."""

import re
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
    start_writing_clicked = QtCore.pyqtSignal(
        str
    )  # User clicked Start Writing, emit final outline
    dialog_cancelled = QtCore.pyqtSignal()  # User clicked Cancel
    llm_token_received = QtCore.pyqtSignal(
        str
    )  # Streaming token from LLM (thread-safe)
    llm_response_complete = QtCore.pyqtSignal(str)  # Full LLM response received
    set_waiting = QtCore.pyqtSignal(bool)  # Control waiting state

    def __init__(
        self, parent=None, initial_prompt="What story would you like to plan?"
    ):
        """Initialize the Planning Mode Dialog.

        Args:
            parent: Parent widget
            initial_prompt: The initial prompt to show the user (from settings)
        """
        super().__init__(parent)
        self.setWindowTitle("Story Planning Mode")
        self.resize(800, 600)
        self.setModal(True)

        self._initial_prompt = initial_prompt
        self._current_llm_response = ""  # Accumulate full response
        self._conversation_markdown = ""  # Full conversation as markdown
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()

        # Title
        title_label = QtWidgets.QLabel("Story Planning Assistant")
        title_font = title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        info_label = QtWidgets.QLabel(
            "Have a conversation with the LLM to develop your story outline.\n"
            'When you\'re satisfied with the outline, click "Start Writing" to begin generation.\n'
            "Tip: Press Enter to send, Shift+Enter to add a newline."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(info_label)

        # Main conversation area (renders markdown)
        conversation_label = QtWidgets.QLabel("Planning Conversation:")
        layout.addWidget(conversation_label)

        # Use QTextBrowser for markdown rendering
        self.conversation_display = QtWidgets.QTextBrowser()
        self.conversation_display.setOpenExternalLinks(False)
        self.conversation_display.setStyleSheet(
            "QTextBrowser { background-color: #f5f5f5; border: 1px solid #cccccc; color: #000000; }"
        )
        layout.addWidget(self.conversation_display)

        # User input field
        input_label = QtWidgets.QLabel("Your response:")
        layout.addWidget(input_label)

        self.user_input_field = PlanningInputField()
        self.user_input_field.setMaximumHeight(80)
        self.user_input_field.setPlaceholderText(
            "Type your response or ideas for the outline..."
        )
        layout.addWidget(self.user_input_field)

        # Send button (can still click if preferred)
        send_button = QtWidgets.QPushButton("Send (or press Enter)")
        send_button.clicked.connect(self._on_send_input)
        layout.addWidget(send_button)

        # Bottom buttons
        button_layout = QtWidgets.QHBoxLayout()

        clear_button = QtWidgets.QPushButton("Clear Conversation")
        clear_button.setStyleSheet(
            "QPushButton { background-color: #ff9800; color: white; padding: 6px; }"
        )
        clear_button.clicked.connect(self._on_clear_conversation)
        button_layout.addWidget(clear_button)

        button_layout.addStretch()

        self.start_writing_button = QtWidgets.QPushButton("Start Writing")
        self.start_writing_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 6px; font-weight: bold; }"
        )
        self.start_writing_button.clicked.connect(self._on_start_writing)
        button_layout.addWidget(self.start_writing_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
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
        self.llm_response_complete.connect(
            self._on_llm_response_complete, QtCore.Qt.QueuedConnection
        )
        self.set_waiting.connect(self._set_waiting_state, QtCore.Qt.QueuedConnection)

    def _on_send_input(self):
        """Handle user input submission."""
        user_text = self.user_input_field.toPlainText().strip()
        if not user_text:
            return

        # Append user message to conversation as markdown
        self._conversation_markdown += f"\n\n**You:** {user_text}\n"
        self._render_conversation()

        # Clear input field
        self.user_input_field.clear()

        # Reset per-response accumulator and add assistant prefix
        self._current_llm_response = ""
        self._conversation_markdown += "\n\n**Assistant:**\n"
        self._render_conversation()

        # Emit signal for controller to handle LLM response
        self.user_input_ready.emit(user_text)

    def _on_start_writing(self):
        """Handle Start Writing button click."""
        # Extract outline from conversation (look for markdown checklist)
        outline = self._extract_outline_from_conversation()
        if not outline:
            QtWidgets.QMessageBox.warning(
                self,
                "No Outline Found",
                "Please develop an outline with the LLM before starting to write.\n\n"
                "The LLM can provide an outline in various formats:\n"
                "• Markdown checklist (- [ ] Plot point)\n"
                "• Bullet points (- Plot point)\n"
                "• Numbered list (1. Plot point)\n"
                "• Acts/Chapters structure",
            )
            return

        self.start_writing_clicked.emit(outline)
        self.accept()

    def _on_clear_conversation(self):
        """Handle Clear Conversation button click."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Conversation",
            "Are you sure you want to clear this conversation and start fresh?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._conversation_markdown = f"**Assistant:**\n{self._initial_prompt}\n"
            self._current_llm_response = ""
            self._render_conversation()

    def _on_cancel(self):
        """Handle Cancel button click."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Cancel Planning",
            "Are you sure you want to cancel planning mode?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
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

    def _on_llm_token(self, token):
        """Handle individual token from LLM streaming (thread-safe via signal).

        Args:
            token: Single token from LLM
        """
        self._current_llm_response += token
        # Stream the token to the display immediately
        self._conversation_markdown += token
        self._render_conversation()

    def _on_llm_response_complete(self, full_response):
        """Handle completion of LLM response (thread-safe via signal).

        Args:
            full_response: The complete response from LLM
        """
        # Add newline at end of response
        self._conversation_markdown += "\n"
        self._render_conversation()

        # Reset accumulator for next response
        self._current_llm_response = ""

    def _set_waiting_state(self, waiting):
        """Set the waiting state (thread-safe via signal).

        Args:
            waiting: True if waiting for LLM, False otherwise
        """
        self.user_input_field.setEnabled(not waiting)

    def _render_conversation(self):
        """Render the conversation markdown to the display."""
        self.conversation_display.setMarkdown(self._conversation_markdown)
        # Auto-scroll to bottom
        scrollbar = self.conversation_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _extract_outline_from_conversation(self):
        """Extract outline from conversation in various formats.

        Looks for structured outline content including:
        - Markdown checklists (- [ ] or - [x])
        - Bullet points (- )
        - Numbered lists (1., 2., etc.)
        - Act/Chapter headings (## Act I, Act 1:, etc.)

        Returns the most recent complete outline found.

        Returns:
            str: Outline text or empty string if none found
        """
        lines = self._conversation_markdown.split("\n")
        outline_lines = []
        in_outline = False

        # Find the last contiguous block of outline-like content
        for line in reversed(lines):
            stripped = line.strip()

            # Check if this line looks like outline content
            is_outline_line = (
                re.match(r"^- \[[x ]\]", stripped)  # Checklist item
                or re.match(r"^- ", stripped)  # Bullet point
                or re.match(r"^\d+\.", stripped)  # Numbered item
                or re.match(r"^#{1,3}\s+", stripped)  # Heading (##, ###)
                or re.match(
                    r"^(Act|Chapter|Part)\s+", stripped, re.IGNORECASE
                )  # Act/Chapter
                or re.match(r"^\*\*", stripped)  # Bold text (often used for headings)
            )

            if is_outline_line:
                outline_lines.insert(0, stripped)
                in_outline = True
            elif (
                in_outline
                and stripped
                and not stripped.startswith("**Assistant:**")
                and not stripped.startswith("**You:**")
            ):
                # Hit non-outline content after finding outline - but keep going if it's just empty lines
                # Stop if we hit a conversational marker
                break
            elif in_outline and not stripped:
                # Empty line within outline - keep it for formatting
                outline_lines.insert(0, "")

        # Clean up - remove leading/trailing empty lines
        while outline_lines and not outline_lines[0]:
            outline_lines.pop(0)
        while outline_lines and not outline_lines[-1]:
            outline_lines.pop()

        outline = "\n".join(outline_lines)

        # Filter out non-plot-point items (metadata, themes, notes, etc.)
        # These are typically descriptive headers that aren't actual story events
        filtered_lines = []
        for line in outline_lines:
            stripped = line.strip()

            # Skip empty lines temporarily (we'll add them back if between plot points)
            if not stripped:
                continue

            # Check if this is a metadata/thematic header (not an actual plot point)
            # Look for patterns like "**Themes:**", "**Setting:**", "Themes", etc.
            is_metadata = False

            # Pattern 1: Bold metadata headers (e.g., "**Themes:**")
            if re.match(
                r"^\*\*\s*(Themes?|Setting|Characters?|Tone|Style|Notes?|Genre|Mood|Atmosphere)s?\s*[:]*\s*\*\*",
                stripped,
                re.IGNORECASE,
            ):
                is_metadata = True

            # Pattern 2: Checklist items that are just metadata labels
            if re.match(
                r"^- \[[ x]\]\s*\*\*\s*(Themes?|Setting|Characters?|Tone|Style|Notes?|Genre|Mood|Atmosphere)s?\s*[:]*",
                stripped,
                re.IGNORECASE,
            ):
                is_metadata = True

            # Pattern 3: Lines that look like thematic descriptions rather than plot events
            # These often contain abstract concepts without concrete actions
            if re.search(
                r"(Loyalty|Perseverance|Trust|Hope|Fear|Courage|Love|Friendship|Betrayal|Redemption|Loss|Growth)",
                stripped,
                re.IGNORECASE,
            ):
                # Check if it's ONLY themes (no concrete action verbs)
                has_action = re.search(
                    r"\b(introduce|encounter|discover|find|chase|escape|reach|arrive|meet|fight|overcome|realize|decide)\b",
                    stripped,
                    re.IGNORECASE,
                )
                if (
                    not has_action and len(stripped.split()) < 15
                ):  # Short lines that are just themes
                    is_metadata = True

            if not is_metadata:
                filtered_lines.append(line)

        outline_lines = filtered_lines

        # Convert to checklist format if it's not already
        if outline_lines and not re.search(r"- \[[x ]\]", "\n".join(outline_lines)):
            # Convert bullet points or numbered lists to checklists
            converted_lines = []
            for line in outline_lines:
                stripped = line.strip()
                if re.match(r"^- ", stripped):
                    # Already a bullet, convert to checklist
                    converted_lines.append(re.sub(r"^- ", "- [ ] ", stripped))
                elif re.match(r"^\d+\.\s+", stripped):
                    # Numbered item, convert to checklist
                    converted_lines.append(re.sub(r"^\d+\.\s+", "- [ ] ", stripped))
                elif stripped:
                    # Heading or other content, keep as-is (or convert to checklist if it seems like a task)
                    if not re.match(r"^#{1,3}\s+", stripped):
                        # Not a heading, treat as task
                        converted_lines.append(f"- [ ] {stripped}")
                    else:
                        converted_lines.append(stripped)
                else:
                    converted_lines.append(line)

            outline = "\n".join(converted_lines)

        return outline

    def get_conversation_text(self):
        """Get the full conversation text.

        Returns:
            str: The conversation history
        """
        return self._conversation_markdown

    def get_current_outline(self):
        """Get the current outline from the conversation.

        Returns:
            str: The markdown checklist outline extracted from conversation
        """
        return self._extract_outline_from_conversation()

    def set_conversation(self, conversation_markdown):
        """Set the conversation content (for persistence).

        Args:
            conversation_markdown: Previously saved conversation markdown
        """
        if conversation_markdown:
            self._conversation_markdown = conversation_markdown
        else:
            self._conversation_markdown = f"**Assistant:**\n{self._initial_prompt}\n"

    def show_with_initial_prompt(self):
        """Show the dialog and display the initial prompt."""
        if not self._conversation_markdown:
            self._conversation_markdown = f"**Assistant:**\n{self._initial_prompt}\n"
        self._render_conversation()
        self.user_input_field.setFocus()
        return self.exec_()
