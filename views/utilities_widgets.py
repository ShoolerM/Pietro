"""Shared widgets for utilities panels."""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore
import shutil


class OrderTrackingTreeWidget(QtWidgets.QTreeWidget):
    """QTreeWidget that emits a signal when items are reordered via drag and drop."""

    items_reordered = QtCore.pyqtSignal()
    file_moved = QtCore.pyqtSignal(str, str)  # old_path, new_path

    def __init__(self, base_path=""):
        super().__init__()
        self.base_path = base_path
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event):
        """Override dropEvent to actually move files on disk."""
        dragged_items = self.selectedItems()
        if not dragged_items:
            event.ignore()
            return

        dragged_item = dragged_items[0]
        drop_position = event.pos()
        target_item = self.itemAt(drop_position)
        dragged_path = dragged_item.data(0, QtCore.Qt.UserRole)

        if not dragged_path:
            super().dropEvent(event)
            return

        dragged_path = Path(dragged_path)

        if target_item:
            target_path = target_item.data(0, QtCore.Qt.UserRole)
            if target_path:
                target_dir = Path(target_path).parent
            else:
                target_dir = self._get_folder_path(target_item)
        else:
            target_dir = Path(self.base_path)

        new_path = target_dir / dragged_path.name

        if dragged_path == new_path:
            event.accept()
            return

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dragged_path), str(new_path))
            event.accept()
            self.file_moved.emit(str(dragged_path), str(new_path))
        except Exception as e:
            event.ignore()
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to move file:\n{str(e)}"
            )

    def _get_folder_path(self, folder_item):
        """Get the full path of a folder item by traversing up the tree."""
        parts = []
        current = folder_item

        while current is not None:
            parts.insert(0, current.text(0))
            current = current.parent()

        result = Path(self.base_path)
        for part in parts:
            result = result / part

        return result
