"""Logs panel view."""

from PyQt5 import QtWidgets, QtCore, QtGui


class LogsPanel(QtWidgets.QWidget):
    """Panel for displaying logs."""

    clear_requested = QtCore.pyqtSignal()
    font_size_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        self.logs_text = QtWidgets.QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setPlaceholderText(
            "Status messages and logs will appear here..."
        )
        layout.addWidget(self.logs_text)

        clear_layout = QtWidgets.QHBoxLayout()
        clear_layout.addStretch()
        clear_logs_button = QtWidgets.QPushButton("Clear Logs")
        clear_logs_button.clicked.connect(self.clear_requested.emit)
        clear_layout.addWidget(clear_logs_button)
        layout.addLayout(clear_layout)

        self.setLayout(layout)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.font_size_changed.emit(1)
                elif delta < 0:
                    self.font_size_changed.emit(-1)
                return True
        return False

    def apply_font_size(self, size):
        font = QtGui.QFont()
        font.setPointSize(size)
        self.logs_text.setFont(font)

    def append_logs(self, text: str):
        self.logs_text.moveCursor(QtGui.QTextCursor.End)
        self.logs_text.insertPlainText(text)
        self.logs_text.moveCursor(QtGui.QTextCursor.End)

    def clear_logs(self):
        self.logs_text.clear()
