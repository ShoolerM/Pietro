"""Bottom control panel for IDE-style layout with input and mode selection."""

from PyQt5 import QtWidgets, QtCore

from views.custom_widgets import AutoGrowTextEdit


class BottomControlPanel(QtWidgets.QWidget):
    """Bottom panel for prompt input, mode selection, and model selection."""

    # Signals
    send_clicked = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal(str)  # "Normal", "Planning", "Smart Mode"
    model_changed = QtCore.pyqtSignal(str)
    model_refresh_clicked = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Left side: Input field
        self.input_field = AutoGrowTextEdit(min_lines=1, max_lines=8)
        self.input_field.setPlaceholderText(
            "Type your prompt here. Enter to send, Shift+Enter for newline"
        )
        self.input_field.send_signal.connect(self._on_send)
        layout.addWidget(self.input_field, stretch=3)

        # Right side: Mode and Model selection
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setSpacing(5)

        # Mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Mode:")
        mode_layout.addWidget(mode_label)

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
        mode_layout.addWidget(self.mode_combo, stretch=1)
        controls_layout.addLayout(mode_layout)

        # Model selection
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model:")
        model_layout.addWidget(model_label)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.currentTextChanged.connect(
            lambda text: self.model_changed.emit(text)
        )
        model_layout.addWidget(self.model_combo, stretch=1)

        self.model_refresh_button = QtWidgets.QPushButton("↻")
        self.model_refresh_button.setToolTip("Refresh Models")
        self.model_refresh_button.setFixedWidth(30)
        self.model_refresh_button.clicked.connect(
            lambda: self.model_refresh_clicked.emit()
        )
        model_layout.addWidget(self.model_refresh_button)
        controls_layout.addLayout(model_layout)

        layout.addLayout(controls_layout, stretch=1)
        self.setLayout(layout)

    def _on_send(self):
        """Handle send signal from input field."""
        self.send_clicked.emit()

    # Public methods

    def get_user_input(self):
        """Get user input text."""
        return self.input_field.toPlainText().strip()

    def clear_user_input(self):
        """Clear user input field."""
        self.input_field.clear()

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
        """Set the current mode."""
        index = self.mode_combo.findText(mode)
        if index >= 0:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)
