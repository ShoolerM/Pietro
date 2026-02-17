"""LLM panel view combining thinking process, prompt input, and controls."""

from PyQt5 import QtWidgets, QtCore, QtGui
from models.stylesheets import PLANNING_MODE
from views.search_widget import SearchWidget
from views.custom_widgets import AutoGrowTextEdit


class LLMPanel(QtWidgets.QWidget):
    """Panel for LLM interaction: thinking process, prompt input, and model controls."""

    # Signals
    font_size_changed = QtCore.pyqtSignal(int)  # delta
    send_clicked = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal(str)  # "Normal", "Planning", "Smart Mode"
    model_changed = QtCore.pyqtSignal(str)
    model_refresh_clicked = QtCore.pyqtSignal()
    start_writing_requested = QtCore.pyqtSignal(
        str
    )  # Emits outline when user types "Start Writing"
    outline_changed = QtCore.pyqtSignal(str)  # Emits when outline is edited

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.search_widget = None
        self.message_history = []  # Store message history as list of (type, text) tuples
        self.user_message_history = []  # Store only user messages for navigation
        self.history_index = (
            -1
        )  # Current position in history (-1 = current/new message)
        self.current_draft = ""  # Store current unsent message when navigating history
        self._attached_files = []
        self._rag_selected = []
        self._rag_items = []
        self._rag_items_collapsed = False

        # Planning mode state
        self._in_planning_mode = False
        self._planning_conversation = []  # List of {"role": str, "content": str}
        self._current_outline = ""  # Current outline text (editable)

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thinking Process section
        thinking_container = QtWidgets.QWidget()
        thinking_container_layout = QtWidgets.QVBoxLayout()
        thinking_container_layout.setContentsMargins(5, 5, 5, 5)
        thinking_container_layout.setSpacing(5)

        self.thinking_label = QtWidgets.QLabel("LLM Panel")
        thinking_container_layout.addWidget(self.thinking_label)

        self._rag_message_index = None

        self.thinking_text = QtWidgets.QTextBrowser()
        self.thinking_text.setReadOnly(True)
        self.thinking_text.setAcceptRichText(True)  # Enable rich text for formatting
        self.thinking_text.setPlaceholderText("Message history will appear here...")
        self.thinking_text.installEventFilter(self)
        self.thinking_text.setOpenExternalLinks(False)
        self.thinking_text.setOpenLinks(False)
        self.thinking_text.anchorClicked.connect(self._on_thinking_anchor_clicked)
        self.thinking_text.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.thinking_text.customContextMenuRequested.connect(
            self._show_output_context_menu
        )
        thinking_container_layout.addWidget(self.thinking_text, stretch=1)

        thinking_container.setLayout(thinking_container_layout)

        # Create search widget
        self.search_widget = SearchWidget(self.thinking_text, thinking_container)
        self.search_widget.hide()
        self.search_widget.close_requested.connect(self.search_widget.hide)
        thinking_container_layout.insertWidget(0, self.search_widget)

        layout.addWidget(thinking_container, stretch=1)

        # Progress bar (between thinking text and input field)
        self.wait_progress = QtWidgets.QProgressBar()
        self.wait_progress.setFixedHeight(12)
        self.wait_progress.setTextVisible(False)
        self.wait_progress.setRange(0, 0)  # Indeterminate progress
        self.wait_progress.hide()
        layout.addWidget(self.wait_progress)

        # Prompt input section
        prompt_container = QtWidgets.QWidget()
        prompt_layout = QtWidgets.QVBoxLayout()
        prompt_layout.setContentsMargins(5, 5, 5, 5)
        prompt_layout.setSpacing(0)

        self.input_field = AutoGrowTextEdit(min_lines=2, max_lines=10)
        self.input_field.setPlaceholderText(
            "Type your prompt here. Enter to send, Shift+Enter for newline"
        )
        self.input_field.setAcceptDrops(False)
        self.input_field.send_signal.connect(self._on_send)
        self.input_field.installEventFilter(
            self
        )  # Install event filter for arrows + drop

        input_row = QtWidgets.QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)
        input_row.addWidget(self.input_field, stretch=1)

        buttons_container = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QVBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(4)

        button_style = (
            "QPushButton { border: 1px solid rgba(255,255,255,0.2); "
            "border-radius: 4px; padding: 0px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.08); }"
            "QPushButton:pressed { background: rgba(255,255,255,0.16); }"
        )

        self.attach_button = QtWidgets.QPushButton("+")
        self.attach_button.setToolTip("Attach files")
        self.attach_button.setFixedSize(24, 24)
        self.attach_button.setFlat(True)
        self.attach_button.setStyleSheet(button_style)
        self.attach_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.attach_button.clicked.connect(self._attach_files)

        self.send_button = QtWidgets.QPushButton(">")
        self.send_button.setToolTip("Send")
        self.send_button.setFixedSize(24, 24)
        self.send_button.setFlat(True)
        self.send_button.setStyleSheet(button_style)
        self.send_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.send_button.clicked.connect(self._on_send)

        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.attach_button)
        buttons_layout.addWidget(self.send_button)
        buttons_layout.addStretch(1)
        buttons_container.setLayout(buttons_layout)

        input_row.addWidget(buttons_container, stretch=0)
        prompt_layout.addLayout(input_row)

        prompt_container.setLayout(prompt_layout)
        layout.addWidget(prompt_container, stretch=0)

        # Control bar (no labels, minimal style)
        control_bar = QtWidgets.QWidget()
        control_bar.setStyleSheet("background: transparent;")
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(5, 2, 5, 2)
        control_layout.setSpacing(5)

        # Mode dropdown (no label)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Normal", "Planning", "Smart Mode"])
        self.mode_combo.setToolTip(
            "Normal: Single response mode\n"
            "Planning: Open planning dialog\n"
            "Smart Mode: Continuous writing with RAG (N chunks)"
        )
        self.mode_combo.currentTextChanged.connect(
            lambda text: self.mode_changed.emit(text)
        )
        control_layout.addWidget(self.mode_combo)

        # Model dropdown (no label)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.currentTextChanged.connect(
            lambda text: self.model_changed.emit(text)
        )
        control_layout.addWidget(self.model_combo, stretch=1)

        # Refresh button
        self.model_refresh_button = QtWidgets.QPushButton("↻")
        self.model_refresh_button.setToolTip("Refresh Models")
        self.model_refresh_button.setFixedWidth(30)
        self.model_refresh_button.clicked.connect(
            lambda: self.model_refresh_clicked.emit()
        )
        control_layout.addWidget(self.model_refresh_button)

        control_bar.setLayout(control_layout)
        layout.addWidget(control_bar, stretch=0)

        self.setLayout(layout)

        # Set size constraints
        self.setMinimumWidth(250)

        # Drag-and-drop overlay (hidden by default)
        self._drop_overlay = QtWidgets.QLabel(self)
        self._drop_overlay.setText("Drop image to attach")
        self._drop_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self._drop_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._drop_overlay.setStyleSheet(
            "background: rgba(255,255,255,0.12);"
            "color: rgba(255,255,255,0.9);"
            "border: 2px dashed rgba(255,255,255,0.4);"
            "border-radius: 8px;"
            "font-size: 14px;"
        )
        self._drop_overlay.setGeometry(
            8, 8, max(0, self.width() - 16), max(0, self.height() - 16)
        )
        self._drop_overlay.hide()

        # Add Ctrl+F shortcut for search
        self.search_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        self.search_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.search_shortcut.activated.connect(self._show_search)

        self._update_attachment_ui()

    def _show_search(self):
        """Show the search widget."""
        if self.search_widget:
            self.search_widget.show_and_focus()

    def set_rag_selection(self, databases):
        self._rag_selected = list(databases or [])

    def set_rag_items(self, items):
        self._rag_items = list(items or [])
        self._rag_items_collapsed = True
        self._update_rag_items_message()

    def collapse_rag_selection(self):
        return

    def collapse_rag_items(self):
        if not self._rag_items:
            return
        self._rag_items_collapsed = True
        self._update_rag_items_message()

    def _update_rag_items_message(self):
        # Remove existing RAG message if any
        if self._rag_message_index is not None:
            try:
                self.message_history.pop(self._rag_message_index)
            except Exception:
                pass
            self._rag_message_index = None

        if not self._rag_items:
            self._render_message_history()
            return

        arrow = ">" if self._rag_items_collapsed else "v"
        header = f"{arrow} RAG Items ({len(self._rag_items)})"
        body = "\n".join(f"• {item}" for item in self._rag_items)
        text = header if self._rag_items_collapsed else f"{header}\n{body}"
        self.message_history.append(("rag", text))
        self._rag_message_index = len(self.message_history) - 1
        self._render_message_history()

    def _on_thinking_anchor_clicked(self, url):
        if url.toString() == "rag-toggle":
            self._rag_items_collapsed = not self._rag_items_collapsed
            self._update_rag_items_message()

    def _on_send(self):
        """Handle send signal from input field."""
        # Get the user's message before clearing
        user_message = self.get_user_input()
        if user_message:
            # Check if in planning mode and message is "Start Writing"
            if (
                self._in_planning_mode
                and user_message.strip().lower() == "start writing"
            ):
                # Use stored outline (structured) instead of parsing conversation
                outline = self._current_outline
                if not outline or not outline.strip():
                    # No valid outline found
                    self.append_llm_panel_text(
                        "\n⚠️ **No outline found.** Please ask me to create an outline first before typing 'Start Writing'.\n\n"
                    )
                    self.clear_user_input()
                    return

                # Validate outline has at least one unchecked item
                has_unchecked = "- [ ]" in outline
                if not has_unchecked:
                    self.append_llm_panel_text(
                        "\n⚠️ **Outline has no remaining tasks.** The outline should contain unchecked items (- [ ]) to write.\n\n"
                    )
                    self.clear_user_input()
                    return

                # Valid outline exists, proceed with start writing
                self.start_writing_requested.emit(outline)
                # Clear input but don't add to history for "Start Writing" command
                self.clear_user_input()
                return

            if self._in_planning_mode:
                self.user_message_history.append(user_message)
            else:
                self.add_user_message(user_message)
            # Reset history navigation
            self.history_index = -1
            self.current_draft = ""
            # Show progress bar immediately
            self.wait_progress.show()
        self.send_clicked.emit()

    def _navigate_history_up(self):
        """Navigate to previous message in history."""
        if not self.user_message_history:
            return

        # Save current draft if we're at the bottom
        if self.history_index == -1:
            self.current_draft = self.input_field.toPlainText()

        # Move up in history
        if self.history_index < len(self.user_message_history) - 1:
            self.history_index += 1
            # Get message from end of list (most recent first)
            message = self.user_message_history[-(self.history_index + 1)]
            self.input_field.setPlainText(message)
            # Move cursor to end
            self.input_field.moveCursor(QtGui.QTextCursor.End)

    def _navigate_history_down(self):
        """Navigate to next (newer) message in history."""
        if self.history_index <= -1:
            return

        # Move down in history
        self.history_index -= 1

        if self.history_index == -1:
            # Back to current draft
            self.input_field.setPlainText(self.current_draft)
        else:
            # Get message from end of list
            message = self.user_message_history[-(self.history_index + 1)]
            self.input_field.setPlainText(message)

        # Move cursor to end
        self.input_field.moveCursor(QtGui.QTextCursor.End)

    def eventFilter(self, obj, event):
        """Event filter for font resizing with Ctrl+Wheel and history navigation."""
        # Handle arrow key navigation in input field
        if (
            hasattr(self, "input_field")
            and obj == self.input_field
            and event.type() == QtCore.QEvent.KeyPress
        ):
            if event.key() == QtCore.Qt.Key_Up:
                self._navigate_history_up()
                return True
            elif event.key() == QtCore.Qt.Key_Down:
                self._navigate_history_down()
                return True

        # Handle font resizing in thinking text
        if obj == self.thinking_text and event.type() == QtCore.QEvent.Wheel:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.font_size_changed.emit(1)
                elif delta < 0:
                    self.font_size_changed.emit(-1)
                return True
        return False

    def dragEnterEvent(self, event):
        image_paths = self._get_image_paths_from_mime(event.mimeData())
        if image_paths:
            self._show_drop_overlay(True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):
        image_paths = self._get_image_paths_from_mime(event.mimeData())
        if image_paths:
            self._show_drop_overlay(True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._show_drop_overlay(False)
        event.accept()

    def dropEvent(self, event):
        image_paths = self._get_image_paths_from_mime(event.mimeData())
        if image_paths:
            self._add_attachments(image_paths)
            event.acceptProposedAction()
        else:
            if event.mimeData().hasUrls():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Unsupported File",
                    "Only image files (png/jpg/jpeg) can be dropped here.",
                )
                event.acceptProposedAction()
            else:
                event.ignore()
        self._show_drop_overlay(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_drop_overlay"):
            margin = 8
            self._drop_overlay.setGeometry(
                margin,
                margin,
                max(0, self.width() - margin * 2),
                max(0, self.height() - margin * 2),
            )

    def _show_drop_overlay(self, show):
        if not hasattr(self, "_drop_overlay"):
            return
        if show:
            self._drop_overlay.show()
            self._drop_overlay.raise_()
        else:
            self._drop_overlay.hide()

    # Public methods

    @QtCore.pyqtSlot(str)
    def append_logs(self, text):
        """Append text to LLM Panel."""
        try:
            self.thinking_text.moveCursor(QtGui.QTextCursor.End)
            self.thinking_text.insertPlainText(text)
            self.thinking_text.moveCursor(QtGui.QTextCursor.End)
        except Exception:
            pass

    def set_thinking_text(self, text):
        """Set thinking text."""
        try:
            self.thinking_text.setPlainText(text)
        except Exception:
            pass

    def clear_thinking_text(self):
        """Clear thinking text."""
        try:
            self.thinking_text.clear()
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def append_llm_panel_text(self, text):
        """Append text to LLM Panel thinking text area."""
        try:
            self.thinking_text.moveCursor(QtGui.QTextCursor.End)
            self.thinking_text.insertPlainText(text)
            self.thinking_text.moveCursor(QtGui.QTextCursor.End)
        except Exception:
            pass

    def clear_output_display(self):
        """Clear the output display without clearing saved history."""
        try:
            self.thinking_text.clear()
            self.message_history.clear()
        except Exception:
            pass

    def apply_font_size(self, size):
        """Apply font size to thinking text."""
        try:
            font = self.thinking_text.font()
            font.setPointSize(size)
            self.thinking_text.setFont(font)
        except Exception:
            pass

    def get_user_input(self):
        """Get user input text."""
        return self.input_field.toPlainText().strip()

    def clear_user_input(self):
        """Clear user input field."""
        self.input_field.clear()

    def _attach_files(self):
        """Open file dialog to attach files."""
        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Attach Files",
            "",
            "Text/Docs/Images (*.txt *.md *.json *.yaml *.yml *.csv *.docx *.png *.jpg *.jpeg);;All Files (*.*)",
        )
        if not file_paths:
            return

        added = False
        newly_added = []
        for path in file_paths:
            if self._is_supported_attachment(path):
                if path not in self._attached_files:
                    self._attached_files.append(path)
                    added = True
                    newly_added.append(path)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Unsupported File",
                    f"{path}\n\nOnly text files and images (png/jpg/jpeg) are supported."
                    " GIFs and videos are not allowed.",
                )

        if added:
            self._update_attachment_ui()
            self._notify_attachments_added(newly_added)

    def _add_attachments(self, paths):
        """Add attachments without UI list."""
        added = False
        newly_added = []
        for path in paths:
            if self._is_supported_attachment(path):
                if path not in self._attached_files:
                    self._attached_files.append(path)
                    added = True
                    newly_added.append(path)
        if added:
            self._update_attachment_ui()
            self._notify_attachments_added(newly_added)
        return added

    def _update_attachment_ui(self):
        count = len(self._attached_files)
        if count:
            self.attach_button.setToolTip(f"Attach files ({count} attached)")
        else:
            self.attach_button.setToolTip("Attach files")

    def _notify_attachments_added(self, paths):
        if not paths:
            return
        for path in paths:
            self.append_llm_panel_text(f"Added: {path}\n")

    def _get_image_paths_from_mime(self, mime_data):
        if not mime_data or not mime_data.hasUrls():
            return []
        paths = []
        for url in mime_data.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if self._is_supported_drop_image(path):
                    paths.append(path)
        return paths

    def _is_supported_drop_image(self, path):
        lower = path.lower()
        return lower.endswith((".png", ".jpg", ".jpeg"))

    def _is_supported_attachment(self, path):
        lower = path.lower()
        if lower.endswith((".gif", ".mp4", ".mov", ".avi", ".mkv", ".webm")):
            return False
        if lower.endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".docx",
            )
        ):
            return True

        import mimetypes

        mime = mimetypes.guess_type(path)[0] or ""
        if mime.startswith("text/"):
            return True
        if mime in ("application/json", "application/x-yaml", "text/yaml"):
            return True

        # Fall back to allowing unknown types; they will be validated on read
        return True

    def get_attached_files(self):
        """Return list of attached files."""
        return list(self._attached_files)

    def clear_attachments(self):
        """Clear all attachments from UI and state."""
        self._attached_files = []
        self._update_attachment_ui()

    def set_models(self, models):
        """Set available models in the dropdown."""
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        # Restore selection if it exists
        index = self.model_combo.findText(current)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def set_model(self, model_name):
        """Set the current model."""
        index = self.model_combo.findText(model_name)
        if index >= 0:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)

    def get_current_model(self):
        """Get currently selected model."""
        return self.model_combo.currentText()

    def get_current_mode(self):
        """Get currently selected mode."""
        return self.mode_combo.currentText()

    def set_mode(self, mode):
        """Set the current mode.

        Args:
            mode: Mode name ("Normal", "Planning", "Smart Mode")
        """
        index = self.mode_combo.findText(mode)
        if index >= 0:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)

    def set_waiting(self, waiting):
        """Show or hide the progress bar.

        Args:
            waiting: True to show progress bar, False to hide
        """
        if waiting:
            self.wait_progress.show()
        else:
            self.wait_progress.hide()

    def add_user_message(self, message):
        """Add a user message to the history and display it."""
        self.message_history.append(("user", message))
        self.user_message_history.append(message)  # Add to navigation history
        self._render_message_history()

    def add_ai_message(self, message):
        """Add an AI message to the history and display it."""
        self.message_history.append(("ai", message))
        self._render_message_history()

    def clear_message_history(self):
        """Clear all message history."""
        self.message_history.clear()
        self.thinking_text.clear()
        self._rag_message_index = None

    def set_normal_conversation(self, conversation):
        """Load and render normal-mode conversation history.

        Args:
            conversation: List of {"role": str, "content": str}
        """
        self.message_history.clear()
        self.user_message_history.clear()

        for msg in conversation or []:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                self.user_message_history.append(content)

    def _show_output_context_menu(self, position):
        """Show context menu for output area."""
        menu = self.thinking_text.createStandardContextMenu()
        menu.addSeparator()
        clear_action = menu.addAction("Clear Output")
        clear_action.triggered.connect(self.clear_output_display)
        menu.exec_(self.thinking_text.viewport().mapToGlobal(position))

    def _render_message_history(self):
        """Render the message history with formatting."""
        html_parts = []

        for msg_type, msg_text in self.message_history:
            if msg_type == "user":
                # User messages in bold with blue color
                html_parts.append(
                    f'<div style="margin-bottom: 10px;">'
                    f'<span style="color: #4a9eff; font-weight: bold;">User:</span><br>'
                    f'<span style="margin-left: 10px;">{self._escape_html(msg_text)}</span>'
                    f"</div>"
                )
            elif msg_type == "ai":
                # AI messages in italic with green color
                html_parts.append(
                    f'<div style="margin-bottom: 10px;">'
                    f'<span style="color: #4eff9e; font-weight: bold;">AI:</span><br>'
                    f'<span style="margin-left: 10px; font-style: italic;">{self._escape_html(msg_text)}</span>'
                    f"</div>"
                )
            elif msg_type == "rag":
                parts = msg_text.split("\n", 1)
                header = parts[0]
                body = parts[1] if len(parts) > 1 else ""
                html_parts.append(
                    f'<div style="margin-bottom: 10px;">'
                    f'<a href="rag-toggle" style="color: #d9a6ff; font-weight: bold; text-decoration: none;">{self._escape_html(header)}</a>'
                    f"</div>"
                )
                if body and not self._rag_items_collapsed:
                    html_parts.append(
                        f'<div style="margin-bottom: 10px; margin-left: 10px; color: #cbb6dd;">'
                        f"{self._escape_html(body)}"
                        f"</div>"
                    )

        self.thinking_text.setHtml("".join(html_parts))
        # Scroll to bottom
        self.thinking_text.moveCursor(QtGui.QTextCursor.End)

    def _escape_html(self, text):
        """Escape HTML special characters and preserve newlines."""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&#39;")
        text = text.replace("\n", "<br>")
        return text

    # === Planning Mode Methods ===

    def set_planning_mode(self, enabled):
        """Enable or disable planning mode.

        Args:
            enabled: True for planning mode, False for normal mode
        """
        self._in_planning_mode = enabled
        if enabled:
            self.thinking_label.setText("📋 Planning Mode")
            self.thinking_label.setStyleSheet(PLANNING_MODE)
            self.thinking_text.setStyleSheet("border: 2px solid #4a7a4a;")
        else:
            self.thinking_label.setText("LLM Panel")
            self.thinking_label.setStyleSheet("")
            self.thinking_text.setStyleSheet("")

    def is_planning_mode(self):
        """Check if currently in planning mode."""
        return self._in_planning_mode

    @QtCore.pyqtSlot(str, str)
    def add_planning_message(self, role, content):
        """Add a message to planning conversation.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self._planning_conversation.append({"role": role, "content": content})
        # Re-render the complete conversation with markdown
        self._render_planning_conversation()

    def _render_planning_conversation(self):
        """Render the complete planning conversation with markdown."""
        if not self._in_planning_mode:
            return

        try:
            import markdown

            html_parts = []
            html_parts.append(
                '<div style="font-family: sans-serif; line-height: 1.6;">'
            )

            for msg in self._planning_conversation:
                role = msg["role"]
                content = msg["content"]

                if role == "user":
                    # User messages in blue
                    html_parts.append(
                        '<div style="margin-bottom: 15px;">'
                        '<strong style="color: #4a9eff;">You:</strong><br>'
                        f'<div style="margin-left: 10px;">{self._escape_html(content)}</div>'
                        "</div>"
                    )
                else:
                    # Assistant messages with markdown rendering
                    markdown_html = markdown.markdown(
                        content, extensions=["fenced_code", "tables", "nl2br"]
                    )
                    html_parts.append(
                        '<div style="margin-bottom: 15px;">'
                        '<strong style="color: #4eff9e;">Assistant:</strong><br>'
                        f'<div style="margin-left: 10px;">{markdown_html}</div>'
                        "</div>"
                    )

            html_parts.append("</div>")

            self.thinking_text.setHtml("".join(html_parts))
            # Scroll to bottom
            self.thinking_text.moveCursor(QtGui.QTextCursor.End)
        except Exception as e:
            print(f"Error rendering planning conversation: {e}")

    def get_planning_conversation(self):
        """Get the planning conversation history."""
        return self._planning_conversation

    def set_planning_conversation(self, conversation):
        """Set the planning conversation history."""
        self._planning_conversation = conversation
        # Populate user message history for up-arrow navigation
        self.user_message_history = [
            msg["content"] for msg in conversation if msg["role"] == "user"
        ]

    def clear_planning_conversation(self):
        """Clear the planning conversation."""
        self._planning_conversation = []
        self.clear_thinking_text()

    def get_current_outline(self):
        """Get the current outline."""
        return self._current_outline

    @QtCore.pyqtSlot(str)
    def set_current_outline(self, outline):
        """Set the current outline.

        Args:
            outline: Outline text in markdown checklist format
        """
        self._current_outline = outline

    def display_planning_welcome(self, welcome_text):
        """Display welcome message for planning mode."""
        self.clear_thinking_text()
        self.append_logs(f"📋 **Planning Mode Activated**\n\n{welcome_text}\n\n")
        self.append_logs("---\n\n")
