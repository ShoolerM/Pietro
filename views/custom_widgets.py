"""Custom Qt widgets for the application."""
from pathlib import Path
import shutil
from PyQt5 import QtWidgets, QtCore


class FileTreeWidget(QtWidgets.QTreeWidget):
    """Custom tree widget that moves files in the filesystem when dragged."""
    
    def dropEvent(self, event):
        """Handle drop event to move files in the filesystem."""
        # Get the item being dragged
        dragged_items = self.selectedItems()
        if not dragged_items:
            super().dropEvent(event)
            return
        
        dragged_item = dragged_items[0]
        source_path = dragged_item.data(0, QtCore.Qt.UserRole)
        
        if not source_path:
            super().dropEvent(event)
            return
        
        # Get the drop target
        drop_indicator = self.dropIndicatorPosition()
        target_item = self.itemAt(event.pos())
        
        # Determine the destination directory
        dest_dir = None
        
        if target_item:
            target_path = target_item.data(0, QtCore.Qt.UserRole)
            if target_path:
                target_path_obj = Path(target_path)
                # If dropping on a directory, move into it
                if target_path_obj.is_dir():
                    dest_dir = target_path_obj
                else:
                    # If dropping on a file, move to its parent directory
                    dest_dir = target_path_obj.parent
        
        # If no valid target or dropping at root level
        if dest_dir is None:
            # Get the root directory (supplemental or system_prompts)
            source_path_obj = Path(source_path)
            if 'supplemental' in str(source_path_obj):
                dest_dir = Path('supplemental')
            elif 'system_prompts' in str(source_path_obj):
                dest_dir = Path('system_prompts')
            else:
                super().dropEvent(event)
                return
        
        source_path_obj = Path(source_path)
        dest_path = dest_dir / source_path_obj.name
        
        # Don't move if source and destination are the same
        if source_path_obj.parent == dest_dir:
            super().dropEvent(event)
            return
        
        # Move the file/directory in the filesystem
        try:
            shutil.move(str(source_path_obj), str(dest_path))
            # Update the item's stored path
            dragged_item.setData(0, QtCore.Qt.UserRole, str(dest_path))
            super().dropEvent(event)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Move Failed",
                f"Failed to move '{source_path_obj.name}':\n{str(e)}"
            )


class AutoGrowTextEdit(QtWidgets.QTextEdit):
    """QTextEdit that grows/shrinks with content, wraps text, and treats Enter as send
    while Shift+Enter inserts a newline.
    Emits send_signal when Enter (without Shift) is pressed.
    """
    send_signal = QtCore.pyqtSignal()

    def __init__(self, min_lines=1, max_lines=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.setAcceptRichText(False)
        self.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.document().contentsChanged.connect(self._adjust_height)
        # initial size
        self._adjust_height()

    def _adjust_height(self):
        # Use the document size (which accounts for wrapped lines) when possible
        try:
            doc_height = int(self.document().size().height())
            # add some padding for frame/margins
            padding = 12
            height = doc_height + padding
            # clamp between min and max lines
            line_h = self.fontMetrics().height()
            min_h = max(24, self.min_lines * line_h + 8)
            max_h = max(100, self.max_lines * line_h + 8)
            height = max(min_h, min(max_h, height))
            self.setFixedHeight(height)
        except Exception:
            pass

    def keyPressEvent(self, event):
        # Enter (without Shift) => send
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                # insert newline
                super().keyPressEvent(event)
                return
            # emit send signal instead of inserting newline
            self.send_signal.emit()
            return
        super().keyPressEvent(event)
