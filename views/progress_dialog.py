"""Progress dialog for long-running operations."""
from PyQt5 import QtWidgets, QtCore


class ProgressDialog(QtWidgets.QDialog):
    """Dialog showing progress for long-running operations."""
    
    def __init__(self, title="Processing", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        
        # Prevent closing with X button
        self.setWindowFlags(
            QtCore.Qt.Window | 
            QtCore.Qt.WindowTitleHint | 
            QtCore.Qt.CustomizeWindowHint
        )
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout()
        
        # Main message
        self.message_label = QtWidgets.QLabel("Processing...")
        self.message_label.setWordWrap(True)
        font = self.message_label.font()
        font.setPointSize(10)
        self.message_label.setFont(font)
        layout.addWidget(self.message_label)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate by default
        layout.addWidget(self.progress_bar)
        
        # Detail text area (for logs)
        self.detail_text = QtWidgets.QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(150)
        font = self.detail_text.font()
        font.setFamily("Consolas")
        font.setPointSize(8)
        self.detail_text.setFont(font)
        layout.addWidget(self.detail_text)
        
        # Cancel button (initially hidden)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
    
    def set_message(self, message):
        """Set the main message."""
        self.message_label.setText(message)
        QtWidgets.QApplication.processEvents()
    
    def set_progress(self, current, maximum=None):
        """Set progress value.
        
        Args:
            current: Current progress value
            maximum: Maximum value (if None, keeps current maximum)
        """
        if maximum is not None:
            self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(current)
        QtWidgets.QApplication.processEvents()
    
    def set_indeterminate(self, indeterminate=True):
        """Set whether progress bar is indeterminate.
        
        Args:
            indeterminate: If True, shows busy indicator. If False, shows normal progress.
        """
        if indeterminate:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(0)
        else:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
        QtWidgets.QApplication.processEvents()
    
    def append_detail(self, text):
        """Append text to detail area."""
        self.detail_text.append(text)
        # Auto-scroll to bottom
        scrollbar = self.detail_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QtWidgets.QApplication.processEvents()
    
    def clear_details(self):
        """Clear the detail text area."""
        self.detail_text.clear()
        QtWidgets.QApplication.processEvents()
    
    def show_cancel_button(self, show=True):
        """Show or hide the cancel button."""
        if show:
            self.cancel_button.show()
        else:
            self.cancel_button.hide()
        QtWidgets.QApplication.processEvents()
    
    def finish_success(self, message="Operation completed successfully"):
        """Finish with success message."""
        self.set_message(message)
        self.set_progress(100, 100)
        self.append_detail(f"\n✓ {message}")
        QtCore.QTimer.singleShot(1000, self.accept)  # Auto-close after 1 second
    
    def finish_error(self, message="Operation failed"):
        """Finish with error message."""
        self.set_message(f"Error: {message}")
        self.append_detail(f"\n❌ {message}")
        
        # Show OK button to dismiss
        self.cancel_button.setText("OK")
        self.cancel_button.show()
