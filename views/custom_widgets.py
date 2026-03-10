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


# Debounce interval for in-place edit signals to avoid firing on every keystroke
_EDIT_DEBOUNCE_MS: int = 300

# Maximum number of characters used when auto-generating a title from body text
_TITLE_MAX_CHARS: int = 40


def _title_from_body(body: str) -> str:
    """Generate a short title from the first sentence or words of a body string.

    Args:
        body: The full body/guidelines text to derive a title from.

    Returns:
        A string of at most _TITLE_MAX_CHARS characters ending on a word boundary.
    """
    # Use the first sentence up to _TITLE_MAX_CHARS chars
    first_sentence: str = body.split(".")[0].strip()
    if len(first_sentence) <= _TITLE_MAX_CHARS:
        return first_sentence or body.strip()[:_TITLE_MAX_CHARS]
    # Truncate at the last word boundary within the limit
    truncated: str = first_sentence[:_TITLE_MAX_CHARS]
    last_space: int = truncated.rfind(" ")
    return truncated[:last_space].rstrip() if last_space > 0 else truncated


class OutlineSectionRow(QtWidgets.QWidget):
    """A single row in the OutlineTrackerWidget representing one outline section."""

    # Emitted when the user clicks the [✓] icon on a completed section to un-check it
    uncheck_clicked = QtCore.pyqtSignal()
    # Emitted when the user clicks [ ] / [▶] on a pending/active section to mark it done
    check_clicked = QtCore.pyqtSignal()
    # Emitted when the user edits the section details; carries (title, new_details)
    section_edited = QtCore.pyqtSignal(str, str)
    # Emitted when the user clicks the delete button on this row
    delete_clicked = QtCore.pyqtSignal()

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
    _STATUS_DETAILS_COLORS = {
        "pending": "#777777",
        "active": "#aaaaaa",
        "done": "#888888",
    }

    def __init__(self, index: int, title: str, details: str, status: str = "pending", parent=None):
        super().__init__(parent)
        # Title is stored for internal markdown reconstruction but not displayed
        self._title: str = title
        self._status: str = status
        # Debounce timer so text-change signals are not fired on every keystroke
        self._edit_debounce_timer = QtCore.QTimer(self)
        self._edit_debounce_timer.setSingleShot(True)
        self._edit_debounce_timer.setInterval(_EDIT_DEBOUNCE_MS)
        self._edit_debounce_timer.timeout.connect(self._emit_section_edited)
        self._init_ui(details)

    def _init_ui(self, details: str) -> None:
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

        # Inline editable body text — replaces the separate title label and read-only
        # details label. The title is stored internally but not shown to the user.
        self._details_edit = QtWidgets.QPlainTextEdit(details or "")
        self._details_edit.setPlaceholderText("Section guidelines...")
        self._details_edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._details_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._details_edit.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self._details_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self._details_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._details_edit, stretch=1)

        # Delete button — always visible; removes this section from the outline
        self._delete_button = QtWidgets.QPushButton("\u2715")
        self._delete_button.setFixedSize(24, 24)
        self._delete_button.setFlat(True)
        self._delete_button.setToolTip("Delete this section")
        self._delete_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._delete_button.setStyleSheet(
            "QPushButton { color: #666; border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { color: #ff8888; border-color: #ff6666; }"
        )
        self._delete_button.clicked.connect(self.delete_clicked)
        layout.addWidget(self._delete_button)

        self._apply_colors()

    def _apply_colors(self) -> None:
        label_color: str = self._STATUS_LABEL_COLORS.get(self._status, "#666666")
        details_color: str = self._STATUS_DETAILS_COLORS.get(self._status, "#777777")

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
        elif self._status in ("pending", "active"):
            # Add green hover tint to signal that clicking will mark the section done
            self._status_label.setStyleSheet(base_style + "QPushButton:hover { color: #4eff9e; }")
            self._status_label.setCursor(QtCore.Qt.PointingHandCursor)
            self._status_label.setToolTip("Click to mark as written")
        else:
            self._status_label.setStyleSheet(base_style)
            self._status_label.setCursor(QtCore.Qt.ArrowCursor)
            self._status_label.setToolTip("")

        # Update the details edit text color to reflect the current section status
        self._details_edit.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background: transparent;"
            f"  color: {details_color};"
            f"  font-size: 11px;"
            f"  border: none;"
            f"  padding: 0;"
            f"}}"
            f"QPlainTextEdit:focus {{"
            f"  background: #252525;"
            f"  border-bottom: 1px solid #444;"
            f"}}"
        )

    def _on_status_clicked(self) -> None:
        """Emit the appropriate signal when the status icon is clicked."""
        if self._status == "done":
            self.uncheck_clicked.emit()
        elif self._status in ("pending", "active"):
            self.check_clicked.emit()

    def _on_text_changed(self) -> None:
        """Restart the debounce timer so section_edited is not fired on every keystroke."""
        self._edit_debounce_timer.stop()
        self._edit_debounce_timer.start()
        # Ask the layout to recalculate the row height as the content grows/shrinks
        self.updateGeometry()

    def _emit_section_edited(self) -> None:
        """Emit section_edited with the current details text after the debounce period.

        The internal title is always kept in sync with the body text so the
        markdown outline reconstruction always has a meaningful heading.
        """
        current_details: str = self._details_edit.toPlainText().strip()
        # Keep the internal title derived from whatever the user has typed
        self._title = _title_from_body(current_details) if current_details else ""
        self.section_edited.emit(self._title, current_details)

    def update_content(self, title: str, details: str) -> None:
        """Update the stored title and visible details text.

        Args:
            title: New section title (stored internally for markdown; not displayed).
            details: New section detail text.
        """
        self._title = title
        # Block signals to avoid triggering _on_text_changed while setting the text
        self._details_edit.blockSignals(True)
        self._details_edit.setPlainText(details)
        self._details_edit.blockSignals(False)
        self.updateGeometry()

    def set_status(self, status: str) -> None:
        """Update the row's visual status."""
        self._status = status
        self._status_label.setText(self._STATUS_ICONS.get(status, "[ ]"))
        self._apply_colors()

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        """Tell Qt that this widget's preferred height depends on its width."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        """Calculate the correct wrapped height for the given widget width.

        Uses QFontMetrics.boundingRect() with word-wrap to estimate the text height
        for the inline QPlainTextEdit without triggering recursive layout updates.

        Args:
            width: The widget width to measure against.

        Returns:
            The minimum height needed to display all content at this width.
        """
        h_layout = self.layout()
        margins = h_layout.contentsMargins()
        spacing: int = h_layout.spacing()

        # Sum up fixed-width items that eat into the text column:
        #   left-margin + status-button + spacing + spacing + delete-button + right-margin
        fixed_w: int = margins.left() + margins.right()
        fixed_w += 28 + spacing  # status button + gap
        fixed_w += spacing + 24  # gap + delete button

        text_w: int = max(1, width - fixed_w)

        # Measure wrapped text height via font metrics (safe: does not touch the doc layout)
        fm: QtGui.QFontMetrics = self._details_edit.fontMetrics()
        text: str = self._details_edit.toPlainText() or " "
        text_h: int = (
            fm.boundingRect(0, 0, text_w, 10000, QtCore.Qt.TextWordWrap, text).height()
            + 8  # QPlainTextEdit top/bottom internal padding
        )

        # Ensure the row is tall enough for the fixed-size delete button (24 px)
        min_button_h: int = 24
        return max(text_h, min_button_h) + margins.top() + margins.bottom()


class OutlineTrackerWidget(QtWidgets.QWidget):
    """Displays the story outline as an interactive checklist during writing.

    Each section shows its current status (pending / active / done).
    """

    # Emitted when the user un-checks a completed section; carries the section index
    uncheck_requested = QtCore.pyqtSignal(int)
    # Emitted when the user manually checks a pending/active section; carries the section index
    check_requested = QtCore.pyqtSignal(int)
    # Emitted when the user edits a section's title/details; carries (index, title, details)
    section_edited = QtCore.pyqtSignal(int, str, str)
    # Emitted when the user adds a brand-new section; carries (title, details)
    section_added = QtCore.pyqtSignal(str, str)
    # Emitted when the user deletes a section; carries the original index
    section_deleted = QtCore.pyqtSignal(int)
    # Emitted when the user clears all sections at once via the context menu
    all_sections_cleared = QtCore.pyqtSignal()
    # Emitted when the user clicks the close button on the tracker header
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections: list = []  # list of dicts: {title, details, status}
        self._row_widgets: list = []  # parallel list of OutlineSectionRow
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        # Header row: title label on the left, close button on the right
        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(0)

        header_label = QtWidgets.QLabel("📋 Outline")
        header_label.setStyleSheet("color: #aaaaaa; font-size: 11px; padding: 0 6px;")
        header_row.addWidget(header_label, stretch=1)

        # Close button lets the user hide the tracker without leaving planning mode
        self._close_button = QtWidgets.QPushButton("×")
        self._close_button.setFixedSize(16, 16)
        self._close_button.setFlat(True)
        self._close_button.setToolTip("Hide outline")
        self._close_button.setStyleSheet(
            "QPushButton { color: #666; border: none; background: transparent; font-size: 14px; }"
            "QPushButton:hover { color: #aaa; }"
        )
        self._close_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._close_button.clicked.connect(self._on_close_clicked)
        header_row.addWidget(self._close_button)

        layout.addLayout(header_row)

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._list.setFocusPolicy(QtCore.Qt.NoFocus)
        self._list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Re-lay-out items whenever the viewport is resized so word-wrap stays correct
        self._list.setResizeMode(QtWidgets.QListView.Adjust)
        # Slow mouse-wheel scrolling to one line per tick for fine-grained navigation
        self._list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._list.verticalScrollBar().setSingleStep(15)
        self._list.installEventFilter(self)
        # Right-click context menu for bulk operations
        self._list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_list_context_menu)
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

        # Button at the bottom of the list so the user can add new sections manually
        self._add_section_button = QtWidgets.QPushButton("＋ Add Section")
        self._add_section_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._add_section_button.setToolTip("Add a new outline section")
        self._add_section_button.setStyleSheet(
            "QPushButton { color: #888; background: transparent; border: 1px dashed #444;"
            " border-radius: 4px; padding: 4px 10px; font-size: 11px; margin: 2px 4px; }"
            "QPushButton:hover { color: #bbb; border-color: #777; background: #2a2a2a; }"
        )
        self._add_section_button.clicked.connect(self._on_add_section_clicked)
        layout.addWidget(self._add_section_button)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Intercept wheel events on the list to scroll one line at a time."""
        if obj is self._list and event.type() == QtCore.QEvent.Wheel:
            # Scroll by a fixed pixel amount per wheel notch instead of a full page
            delta: int = event.angleDelta().y()
            scroll_amount: int = -40 if delta > 0 else 40
            self._list.verticalScrollBar().setValue(
                self._list.verticalScrollBar().value() + scroll_amount
            )
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Recalculate item size hints on resize so word-wrapped labels fit."""
        super().resizeEvent(event)
        self._update_item_size_hints()

    def _update_item_size_hints(self) -> None:
        """Constrain each row to the current viewport width and update its size hint.

        Uses heightForWidth() on each row rather than sizeHint() because
        QLabel.sizeHint() always returns the single-line (unwrapped) height,
        causing rows with long text to be clipped vertically.
        """
        available_width: int = self._list.viewport().width()
        if available_width <= 0:
            return
        for i, row in enumerate(self._row_widgets):
            if i >= self._list.count():
                break
            # Pin width so the row's layout knows how wide it is for button visibility
            row.setFixedWidth(available_width)
            # Compute the correct wrapped height directly
            h: int = row.heightForWidth(available_width)
            hint = QtCore.QSize(available_width, h)
            # Release the fixed-width constraint so normal layout can take over
            row.setMinimumWidth(0)
            row.setMaximumWidth(16_777_215)  # Qt's QWIDGETSIZE_MAX
            self._list.item(i).setSizeHint(hint)

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

    def clear_all_sections(self) -> None:
        """Remove every section from the tracker and emit all_sections_cleared."""
        self._sections.clear()
        self._rebuild()
        self.all_sections_cleared.emit()

    # ── Private ──────────────────────────────────────────────────────────

    def _on_list_context_menu(self, pos: QtCore.QPoint) -> None:
        """Show a right-click context menu over the section list.

        Args:
            pos: Cursor position in list-widget local coordinates.
        """
        if not self._sections:
            return

        menu = QtWidgets.QMenu(self)
        clear_action = menu.addAction("🗑️  Clear all sections")
        clear_action.setToolTip("Remove every section from the outline")

        action = menu.exec_(self._list.viewport().mapToGlobal(pos))
        if action == clear_action:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Clear Outline",
                "Remove all sections from the outline?\nThis cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.clear_all_sections()

    def _rebuild(self):
        self._list.clear()
        self._row_widgets = []
        for i, sec in enumerate(self._sections):
            row = OutlineSectionRow(i, sec["title"], sec["details"], sec["status"])
            row.uncheck_clicked.connect(lambda idx=i: self.uncheck_requested.emit(idx))
            row.check_clicked.connect(lambda idx=i: self.check_requested.emit(idx))
            # Wire section edit signal to internal handler so we can update _sections data
            row.section_edited.connect(
                lambda title, details, idx=i: self._on_row_edited(idx, title, details)
            )
            # Wire delete signal so the section is removed and indices refreshed
            row.delete_clicked.connect(lambda idx=i: self._on_row_deleted(idx))
            item = QtWidgets.QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            self._list.setItemWidget(item, row)
            self._row_widgets.append(row)
        # Defer the size-hint recalculation to the next event loop tick so the
        # viewport has a real width by the time we measure the wrapped row heights.
        QtCore.QTimer.singleShot(0, self._update_item_size_hints)

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

        # Refresh all item size hints (wrapping may change the height of edited row)
        self._update_item_size_hints()

        # Bubble the edit event up to the LLMPanel
        self.section_edited.emit(index, title, details)

    def _on_close_clicked(self) -> None:
        """Hide the tracker and emit the closed signal."""
        self.hide()
        self.closed.emit()

    def _on_row_deleted(self, index: int) -> None:
        """Remove a section from the tracker and propagate the deletion.

        After removal the entire widget list is rebuilt so all row-index
        closures remain correct.

        Args:
            index: Zero-based index of the section to remove.
        """
        if 0 <= index < len(self._sections):
            self._sections.pop(index)
        self._rebuild()
        self.section_deleted.emit(index)

    def _on_add_section_clicked(self) -> None:
        """Immediately add an empty section row and move focus to its text field."""
        # Add a placeholder section and let the user type directly in the row
        self.add_section("", "")
        # Focus the newly created row's text edit so the user can type right away
        if self._row_widgets:
            new_row: OutlineSectionRow = self._row_widgets[-1]
            self._list.scrollToBottom()
            QtCore.QTimer.singleShot(0, new_row._details_edit.setFocus)

    def add_section(self, title: str, details: str) -> None:
        """Append a new pending section to the tracker and emit section_added.

        Args:
            title: Non-empty section title.
            details: Optional section detail text.
        """
        # Add the section to internal state and rebuild the widget list
        self._sections.append({"title": title, "details": details, "status": "pending"})
        self._rebuild()
        self.section_added.emit(title, details)
