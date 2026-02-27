"""RAG File Browser Dialog

A searchable, modeless dialog that lists all source files registered in a
RAG database.  Double-clicking (or pressing Open) a file emits
``file_selected`` with the absolute file path so the caller can open it for
editing.
"""

from pathlib import Path

from PyQt5 import QtCore, QtWidgets


class RagFileBrowserDialog(QtWidgets.QDialog):
    """Modeless dialog for browsing and opening files in a RAG database.

    Emits ``file_selected(file_path: str)`` when the user wants to open a
    file for editing.  The dialog stays open after a file is selected so
    the user can open multiple files in sequence.
    """

    # Emitted with the absolute file path the user wants to open.
    file_selected = QtCore.pyqtSignal(str)

    def __init__(self, db_name: str, file_paths: list, parent=None):
        """Initialise the dialog.

        Args:
            db_name: Display name of the RAG database (used in the title).
            file_paths: Absolute file path strings registered in the database.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Files in '{db_name}'")
        self.setMinimumSize(520, 400)
        # Prevent the dialog from blocking the main window
        self.setWindowModality(QtCore.Qt.NonModal)

        # Keep the full unfiltered list so the search can re-filter
        self._all_file_paths: list = list(file_paths)

        self._init_ui()
        self._populate_list(self._all_file_paths)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the dialog layout."""
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(6)

        # Search / filter bar
        search_row = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel("Search:")
        self._search_input = QtWidgets.QLineEdit()
        self._search_input.setPlaceholderText("Type to filter files…")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_changed)
        search_row.addWidget(search_label)
        search_row.addWidget(self._search_input)
        layout.addLayout(search_row)

        # File list
        self._file_list = QtWidgets.QListWidget()
        self._file_list.setAlternatingRowColors(True)
        self._file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # Double-click opens the file immediately
        self._file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._file_list)

        # Count label shown below the list
        self._count_label = QtWidgets.QLabel()
        self._count_label.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(self._count_label)

        # Button row
        button_row = QtWidgets.QHBoxLayout()
        self._open_button = QtWidgets.QPushButton("Open in Editor")
        self._open_button.setDefault(True)
        self._open_button.clicked.connect(self._on_open_clicked)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row.addStretch()
        button_row.addWidget(self._open_button)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # List population
    # ------------------------------------------------------------------

    def _populate_list(self, paths: list) -> None:
        """Clear and repopulate the file list widget from *paths*.

        Files are sorted alphabetically by filename (case-insensitive).
        The full path is stored as UserRole data and shown as a tooltip.

        Args:
            paths: Absolute file path strings to display.
        """
        self._file_list.clear()

        # Sort by filename, case-insensitive
        sorted_paths = sorted(paths, key=lambda p: Path(p).name.lower())

        for path in sorted_paths:
            item = QtWidgets.QListWidgetItem(Path(path).name)
            # Store the full path so we can retrieve it without parsing the label
            item.setData(QtCore.Qt.UserRole, path)
            item.setToolTip(path)
            self._file_list.addItem(item)

        # Update the count label
        total = len(self._all_file_paths)
        shown = len(sorted_paths)
        if shown == total:
            self._count_label.setText(f"{total} file(s)")
        else:
            self._count_label.setText(f"{shown} of {total} file(s) shown")

        # Pre-select the first item so pressing Open/Enter works immediately
        if self._file_list.count() > 0:
            self._file_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_search_changed(self, text: str) -> None:
        """Re-filter the file list as the user types.

        Args:
            text: Current search bar text.
        """
        query = text.strip().lower()
        if query:
            filtered = [p for p in self._all_file_paths if query in Path(p).name.lower()]
        else:
            # Empty query — show everything
            filtered = self._all_file_paths
        self._populate_list(filtered)

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Emit ``file_selected`` for the double-clicked item.

        Args:
            item: The QListWidgetItem that was double-clicked.
        """
        full_path: str = item.data(QtCore.Qt.UserRole)
        if full_path:
            self.file_selected.emit(full_path)

    def _on_open_clicked(self) -> None:
        """Emit ``file_selected`` for the currently highlighted item."""
        current = self._file_list.currentItem()
        if current:
            full_path: str = current.data(QtCore.Qt.UserRole)
            if full_path:
                self.file_selected.emit(full_path)
