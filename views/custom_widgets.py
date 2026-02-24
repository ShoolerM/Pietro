"""Custom Qt widgets for the application."""

from pathlib import Path
import shutil
from PyQt5 import QtWidgets, QtCore, QtGui


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
            if "supplemental" in str(source_path_obj):
                dest_dir = Path("supplemental")
            elif "system_prompts" in str(source_path_obj):
                dest_dir = Path("system_prompts")
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
                self, "Move Failed", f"Failed to move '{source_path_obj.name}':\n{str(e)}"
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


# Stylesheet for the section-edit dialog
_EDIT_DIALOG_STYLE: str = (
    "QDialog { background: #1e1e1e; }"
    "QLabel { color: #cccccc; }"
    "QLineEdit { background: #2d2d2d; color: #e0e0e0; border: 1px solid #555;"
    "  border-radius: 3px; padding: 4px; }"
    "QTextEdit { background: #2d2d2d; color: #e0e0e0; border: 1px solid #555;"
    "  border-radius: 3px; padding: 4px; }"
    "QPushButton { background: #3a3a3a; color: #cccccc; border: 1px solid #555;"
    "  border-radius: 3px; padding: 4px 12px; }"
    "QPushButton:hover { background: #4a4a4a; border-color: #888; }"
    "QPushButton:pressed { background: #555; }"
)


class OutlineSectionRow(QtWidgets.QWidget):
    """A single row in the OutlineTrackerWidget representing one outline section."""

    redo_clicked = QtCore.pyqtSignal()
    # Emitted when the user clicks the [✓] icon on a completed section to un-check it
    uncheck_clicked = QtCore.pyqtSignal()
    # Emitted when the user edits the section title/details; carries (new_title, new_details)
    section_edited = QtCore.pyqtSignal(str, str)

    _STATUS_ICONS = {
        "pending": "[ ]",
        "active": "[▶]",
        "done": "[✓]",
    }
    _STATUS_LABEL_COLORS = {
        "pending": "#666666",
        "active": "#4a9eff",
        "done": "#4eff9e",
    }
    _STATUS_TITLE_COLORS = {
        "pending": "#888888",
        "active": "#e0e0e0",
        "done": "#cccccc",
    }

    def __init__(self, index: int, title: str, details: str, status: str = "pending", parent=None):
        super().__init__(parent)
        self._status = status
        self._init_ui(title, details)

    def _init_ui(self, title: str, details: str) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Status icon – displayed as a flat QPushButton so it becomes clickable
        # when the section is done, letting the user un-check it back to pending.
        self._status_label = QtWidgets.QPushButton(self._STATUS_ICONS.get(self._status, "[ ]"))
        self._status_label.setFixedWidth(28)
        self._status_label.setFlat(True)
        font = self._status_label.font()
        font.setFamily("monospace")
        self._status_label.setFont(font)
        self._status_label.clicked.connect(self._on_status_clicked)
        layout.addWidget(self._status_label)

        # Text column (title + details). Details label is always created but
        # hidden when empty, so update_content can show/hide it cleanly.
        self._text_layout = QtWidgets.QVBoxLayout()
        self._text_layout.setContentsMargins(0, 0, 0, 0)
        self._text_layout.setSpacing(1)

        self._title_label = QtWidgets.QLabel(title)
        self._title_label.setWordWrap(True)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._text_layout.addWidget(self._title_label)

        # Always create the details label; hide it when there is no detail text
        self._details_label = QtWidgets.QLabel(details or "")
        self._details_label.setWordWrap(True)
        self._details_label.setStyleSheet("color: #777777; font-size: 11px;")
        self._text_layout.addWidget(self._details_label)
        if not details:
            self._details_label.hide()

        layout.addLayout(self._text_layout, stretch=1)

        # Edit button – always visible, lets the user modify section title/details
        self._edit_button = QtWidgets.QPushButton("✏")
        self._edit_button.setFixedSize(24, 24)
        self._edit_button.setFlat(True)
        self._edit_button.setToolTip("Edit this section")
        self._edit_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._edit_button.setStyleSheet(
            "QPushButton { color: #777777; border: 1px solid #444; border-radius: 3px; }"
            "QPushButton:hover { color: #cccccc; border-color: #888; }"
        )
        self._edit_button.clicked.connect(self._on_edit_clicked)
        layout.addWidget(self._edit_button)

        # Redo button (only visible for completed/active sections)
        self._redo_button = QtWidgets.QPushButton("↺")
        self._redo_button.setFixedSize(24, 24)
        self._redo_button.setFlat(True)
        self._redo_button.setToolTip("Rewrite this section")
        self._redo_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._redo_button.setStyleSheet(
            "QPushButton { color: #aaaaaa; border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { color: #ffffff; border-color: #888; }"
        )
        self._redo_button.clicked.connect(self.redo_clicked)
        self._redo_button.setVisible(self._status in ("done", "active"))
        layout.addWidget(self._redo_button)

        self._apply_colors()

    def _apply_colors(self) -> None:
        label_color: str = self._STATUS_LABEL_COLORS.get(self._status, "#666666")
        title_color: str = self._STATUS_TITLE_COLORS.get(self._status, "#888888")

        # Base button style – mimics a plain label (no border, transparent background)
        base_style: str = (
            f"QPushButton {{ color: {label_color}; font-family: monospace; border: none;"
            f" background: transparent; text-align: left; padding: 0; }}"
        )

        if self._status == "done":
            # Add red hover tint to signal that clicking will un-check the section
            self._status_label.setStyleSheet(base_style + "QPushButton:hover { color: #ff8888; }")
            self._status_label.setCursor(QtCore.Qt.PointingHandCursor)
            self._status_label.setToolTip("Click to mark as not yet written")
        else:
            self._status_label.setStyleSheet(base_style)
            self._status_label.setCursor(QtCore.Qt.ArrowCursor)
            self._status_label.setToolTip("")

        self._title_label.setStyleSheet(f"color: {title_color};")

    def _on_status_clicked(self) -> None:
        """Emit uncheck_clicked when the [✓] icon is clicked on a completed section."""
        if self._status == "done":
            self.uncheck_clicked.emit()

    def _on_edit_clicked(self) -> None:
        """Open a modal dialog to let the user edit the section title and details."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Edit Section")
        dialog.setMinimumWidth(420)
        dialog.setStyleSheet(_EDIT_DIALOG_STYLE)

        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.setSpacing(8)
        dialog_layout.setContentsMargins(12, 12, 12, 12)

        # Title field
        title_label = QtWidgets.QLabel("Title:")
        dialog_layout.addWidget(title_label)
        title_edit = QtWidgets.QLineEdit(self._title_label.text())
        dialog_layout.addWidget(title_edit)

        # Details field
        details_label = QtWidgets.QLabel("Details:")
        dialog_layout.addWidget(details_label)
        details_edit = QtWidgets.QTextEdit()
        details_edit.setPlainText(
            self._details_label.text() if self._details_label.isVisible() else ""
        )
        details_edit.setFixedHeight(90)
        details_edit.setAcceptRichText(False)
        dialog_layout.addWidget(details_edit)

        # OK / Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_title: str = title_edit.text().strip()
            new_details: str = details_edit.toPlainText().strip()
            # Require a non-empty title before accepting the edit
            if new_title:
                self.update_content(new_title, new_details)
                self.section_edited.emit(new_title, new_details)

    def update_content(self, title: str, details: str) -> None:
        """Update the display labels with new title and details text.

        Args:
            title: New section title (must be non-empty).
            details: New section detail text; hides the details label when empty.
        """
        self._title_label.setText(title)
        if details:
            self._details_label.setText(details)
            self._details_label.show()
        else:
            self._details_label.hide()
        # Let the layout recalculate the preferred size for this widget
        self.updateGeometry()

    def set_status(self, status: str) -> None:
        """Update the row's visual status."""
        self._status = status
        self._status_label.setText(self._STATUS_ICONS.get(status, "[ ]"))
        self._redo_button.setVisible(status in ("done", "active"))
        self._apply_colors()


class OutlineTrackerWidget(QtWidgets.QWidget):
    """Displays the story outline as an interactive checklist during writing.

    Each section shows its current status (pending / active / done) and a redo
    button for completed sections.  Emits ``redo_requested(index)`` when the user
    clicks the ↺ button on a completed row.
    """

    redo_requested = QtCore.pyqtSignal(int)
    # Emitted when the user un-checks a completed section; carries the section index
    uncheck_requested = QtCore.pyqtSignal(int)
    # Emitted when the user edits a section's title/details; carries (index, title, details)
    section_edited = QtCore.pyqtSignal(int, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections: list = []  # list of dicts: {title, details, status}
        self._row_widgets: list = []  # parallel list of OutlineSectionRow
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        header = QtWidgets.QLabel("📋 Writing Progress")
        header.setStyleSheet("color: #aaaaaa; font-size: 11px; padding: 0 6px;")
        layout.addWidget(header)

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._list.setFocusPolicy(QtCore.Qt.NoFocus)
        self._list.setMaximumHeight(210)
        self._list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._list.setStyleSheet(
            "QListWidget {"
            "  background: #1e1e1e;"
            "  border: 1px solid #333;"
            "  border-radius: 4px;"
            "}"
            "QListWidget::item {"
            "  border-bottom: 1px solid #2a2a2a;"
            "  padding: 0;"
            "}"
        )
        layout.addWidget(self._list)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_sections(self, sections: list):
        """Populate tracker from a list of section dicts.

        Args:
            sections: List of dicts with 'description', 'details', 'completed' keys.
        """
        self._sections = [
            {
                "title": s.get("description", ""),
                "details": s.get("details", "") or "",
                "status": "done" if s.get("completed") else "pending",
            }
            for s in sections
        ]
        self._rebuild()

    def set_active(self, index: int):
        """Mark a section as currently being written."""
        for i, sec in enumerate(self._sections):
            if sec["status"] == "active":
                sec["status"] = "pending"
                if i < len(self._row_widgets):
                    self._row_widgets[i].set_status("pending")
        if 0 <= index < len(self._sections):
            self._sections[index]["status"] = "active"
            if index < len(self._row_widgets):
                self._row_widgets[index].set_status("active")
                self._list.scrollToItem(self._list.item(index))

    def mark_complete(self, index: int):
        """Mark a section as fully written."""
        if 0 <= index < len(self._sections):
            self._sections[index]["status"] = "done"
            if index < len(self._row_widgets):
                self._row_widgets[index].set_status("done")

    def reset(self):
        """Reset all sections to pending state."""
        for sec in self._sections:
            sec["status"] = "pending"
        for row in self._row_widgets:
            row.set_status("pending")

    def reset_from(self, index: int):
        """Reset all sections from *index* onward to pending, then set index as active.

        Args:
            index: The section index to mark as active; all sections at or after
                   this index are reset to pending first.
        """
        for i in range(index, len(self._sections)):
            self._sections[i]["status"] = "pending"
            if i < len(self._row_widgets):
                self._row_widgets[i].set_status("pending")
        self.set_active(index)

    def section_count(self) -> int:
        return len(self._sections)

    # ── Private ────────────────────────────────────────────────────────────────

    def _rebuild(self):
        self._list.clear()
        self._row_widgets = []
        for i, sec in enumerate(self._sections):
            row = OutlineSectionRow(i, sec["title"], sec["details"], sec["status"])
            row.redo_clicked.connect(lambda _checked=False, idx=i: self.redo_requested.emit(idx))
            row.uncheck_clicked.connect(lambda idx=i: self.uncheck_requested.emit(idx))
            # Wire section edit signal to internal handler so we can update _sections data
            row.section_edited.connect(
                lambda title, details, idx=i: self._on_row_edited(idx, title, details)
            )
            item = QtWidgets.QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            self._list.setItemWidget(item, row)
            self._row_widgets.append(row)

    def get_sections(self) -> list:
        """Return a copy of the current sections list.

        Returns:
            List of dicts with 'title', 'details', and 'status' keys.
        """
        return [dict(sec) for sec in self._sections]

    def _on_row_edited(self, index: int, title: str, details: str) -> None:
        """Update internal section data when a row is edited and propagate the signal.

        Also refreshes the QListWidgetItem size hint so the row resizes correctly.

        Args:
            index: Zero-based index of the edited section.
            title: New section title.
            details: New section details.
        """
        # Update internal sections data to keep it in sync with the UI
        if 0 <= index < len(self._sections):
            self._sections[index]["title"] = title
            self._sections[index]["details"] = details

        # Refresh the list item's size hint after the row content changed
        if index < self._list.count():
            item = self._list.item(index)
            row_widget = self._row_widgets[index]
            item.setSizeHint(row_widget.sizeHint())

        # Bubble the edit event up to the LLMPanel
        self.section_edited.emit(index, title, details)
