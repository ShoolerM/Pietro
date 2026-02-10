"""Control panel view for main UI controls."""

from PyQt5 import QtWidgets, QtCore, QtGui

from views.custom_widgets import AutoGrowTextEdit


class ControlPanel(QtWidgets.QWidget):
    """Panel for main UI controls (input, buttons, model selection)."""

    # Signals
    send_clicked = QtCore.pyqtSignal(
        str, str, str, str
    )  # user_input, notes, supp_text, system_prompt
    undo_clicked = QtCore.pyqtSignal()
    stop_clicked = QtCore.pyqtSignal()
    clear_clicked = QtCore.pyqtSignal()
    toggle_thinking_clicked = QtCore.pyqtSignal()
    model_refresh_clicked = QtCore.pyqtSignal()
    model_changed = QtCore.pyqtSignal(str)
    context_limit_changed = QtCore.pyqtSignal(int)
    font_size_changed = QtCore.pyqtSignal(int)  # delta

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Waiting indicator
        self.wait_progress = QtWidgets.QProgressBar(self)
        self.wait_progress.setFixedHeight(12)
        self.wait_progress.setTextVisible(False)
        self.wait_progress.setRange(0, 0)
        self.wait_progress.hide()
        layout.addWidget(self.wait_progress)

        # Bottom row: Send | Undo | Entry | Stop | Clear
        input_layout = QtWidgets.QHBoxLayout()

        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_button)

        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.clicked.connect(lambda: self.undo_clicked.emit())
        input_layout.addWidget(self.undo_button)

        self.input_field = AutoGrowTextEdit(min_lines=1, max_lines=8)
        self.input_field.setPlaceholderText(
            "Type your main query here. Enter to send, Shift+Enter for newline"
        )
        self.input_field.send_signal.connect(self._on_send)
        self.input_field.installEventFilter(self)
        input_layout.addWidget(self.input_field, stretch=1)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.stop_clicked.emit())
        input_layout.addWidget(self.stop_button)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.clear_clicked.emit())
        input_layout.addWidget(self.clear_button)

        layout.addLayout(input_layout)

        # Model selection row
        model_layout = QtWidgets.QHBoxLayout()

        model_label = QtWidgets.QLabel("Model:")
        model_layout.addWidget(model_label)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(
            lambda text: self.model_changed.emit(text)
        )
        model_layout.addWidget(self.model_combo, stretch=1)

        self.model_refresh_button = QtWidgets.QPushButton("Refresh Models")
        self.model_refresh_button.clicked.connect(
            lambda: self.model_refresh_clicked.emit()
        )
        model_layout.addWidget(self.model_refresh_button)

        context_label = QtWidgets.QLabel("Context Limit:")
        model_layout.addWidget(context_label)

        self.context_limit_spinbox = QtWidgets.QSpinBox()
        self.context_limit_spinbox.setMinimum(1024)
        self.context_limit_spinbox.setMaximum(1000000)
        self.context_limit_spinbox.setValue(4096)
        self.context_limit_spinbox.setSingleStep(1024)
        self.context_limit_spinbox.setSuffix(" tokens")
        self.context_limit_spinbox.setToolTip(
            "Maximum context size. Story will be auto-summarized if it exceeds this limit."
        )
        self.context_limit_spinbox.valueChanged.connect(
            lambda val: self.context_limit_changed.emit(val)
        )
        model_layout.addWidget(self.context_limit_spinbox)

        layout.addLayout(model_layout)
        self.setLayout(layout)

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

    def _on_send(self):
        """Handle send signal from input field."""
        # The actual gathering of supplemental/system prompts will be done by main_view
        # This just forwards the signal
        self.send_clicked.emit(
            "", "", "", ""
        )  # Empty strings - main_view will fill these

    # Public methods

    def get_user_input(self):
        """Get user input text."""
        return self.input_field.toPlainText().strip()

    def set_context_limit(self, value):
        """Set context limit spinbox without emitting signals."""
        try:
            self.context_limit_spinbox.blockSignals(True)
            self.context_limit_spinbox.setValue(int(value))
        finally:
            self.context_limit_spinbox.blockSignals(False)

    def clear_user_input(self):
        """Clear user input field."""
        self.input_field.clear()

    def set_waiting(self, waiting):
        """Set waiting state (show/hide progress bar, enable/disable input)."""
        try:
            if waiting:
                self.input_field.setReadOnly(True)
                self.wait_progress.show()
            else:
                self.wait_progress.hide()
                self.input_field.setReadOnly(False)
        except RuntimeError:
            pass

    def set_stop_enabled(self, enabled):
        """Enable or disable the stop button."""
        self.stop_button.setEnabled(enabled)

    def set_models(self, models, selected_model=None):
        """Set available models in dropdown."""
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            if selected_model and selected_model in models:
                self.model_combo.setCurrentText(selected_model)
            elif self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
        else:
            self.model_combo.addItem("No models available")

    def set_model_error(self, error_message):
        """Set error message in model dropdown."""
        self.model_combo.clear()
        self.model_combo.addItem(error_message)

    def apply_font_size(self, size):
        """Apply font size to input field."""
        font = QtGui.QFont()
        font.setPointSize(size)
        try:
            self.input_field.setFont(font)
        except Exception:
            pass
