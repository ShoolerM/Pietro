"""Supplemental prompts panel view."""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
import shutil

from models.stylesheets import SUPPLEMENTAL_PROMPTS
from views.utilities_widgets import OrderTrackingTreeWidget


class SupplementalPanel(QtWidgets.QWidget):
    """Panel for managing supplemental prompts."""

    refresh_clicked = QtCore.pyqtSignal()
    add_clicked = QtCore.pyqtSignal()
    file_opened = QtCore.pyqtSignal(str)
    selections_changed = QtCore.pyqtSignal(list)
    font_size_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.supp_list = OrderTrackingTreeWidget(base_path="supplemental")
        self.supp_list.setHeaderHidden(True)
        self.supp_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.supp_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.supp_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.supp_list.setAlternatingRowColors(True)
        fm = self.supp_list.fontMetrics()
        row_h = fm.lineSpacing() or fm.height()
        min_h = row_h * 2 + 8
        self.supp_list.setMinimumHeight(min_h)
        self.supp_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.supp_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.supp_list.setUniformRowHeights(True)
        self.supp_list.installEventFilter(self)
        self.supp_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.supp_list.customContextMenuRequested.connect(self._show_context_menu)

        self.supp_list.itemChanged.connect(self._on_item_changed)
        self.supp_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        self.supp_list.file_moved.connect(self._on_file_moved)

        layout.addWidget(self.supp_list)
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
        try:
            self.supp_list.setFont(font)
        except Exception:
            pass

    def _on_file_moved(self, old_path, new_path):
        self.refresh_clicked.emit()

    def _on_item_changed(self, item, column):
        self.selections_changed.emit(self.get_checked_paths())

    def _on_file_double_clicked(self, item, column=0):
        file_path = item.data(0, QtCore.Qt.UserRole)
        if file_path:
            self.file_opened.emit(file_path)

    def _show_context_menu(self, position):
        item = self.supp_list.itemAt(position)

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(SUPPLEMENTAL_PROMPTS)

        if item:
            item_data = item.data(0, QtCore.Qt.UserRole)
            if item_data:
                delete_action = menu.addAction("Delete File")
                delete_action.triggered.connect(lambda: self._on_delete_clicked(item))
            else:
                create_file_action = menu.addAction("Create File")
                create_file_action.triggered.connect(lambda: self._on_create_file(item))
                create_folder_action = menu.addAction("Create Folder")
                create_folder_action.triggered.connect(
                    lambda: self._on_create_directory(item)
                )
                menu.addSeparator()
                delete_action = menu.addAction("Delete Folder")
                delete_action.triggered.connect(lambda: self._on_delete_clicked(item))
        else:
            create_file_action = menu.addAction("Create File")
            create_file_action.triggered.connect(lambda: self._on_create_file(None))
            create_folder_action = menu.addAction("Create Folder")
            create_folder_action.triggered.connect(
                lambda: self._on_create_directory(None)
            )

        menu.addSeparator()
        refresh_action = menu.addAction("Refresh")
        refresh_action.triggered.connect(self.refresh_clicked.emit)
        add_new_action = menu.addAction("Add New")
        add_new_action.triggered.connect(self.add_clicked.emit)

        menu.exec_(self.supp_list.viewport().mapToGlobal(position))

    def _on_create_file(self, parent_item):
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            parent_path = self._get_item_path(parent_item)
        else:
            parent_path = Path("supplemental")

        filename, ok = QtWidgets.QInputDialog.getText(
            self, "Create File", "Enter filename:"
        )

        if ok and filename:
            if not filename.endswith(".txt"):
                filename += ".txt"

            file_path = parent_path / filename

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                self.refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create file: {str(e)}"
                )

    def _on_create_directory(self, parent_item):
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            parent_path = self._get_item_path(parent_item)
        else:
            parent_path = Path("supplemental")

        dirname, ok = QtWidgets.QInputDialog.getText(
            self, "Create Folder", "Enter folder name:"
        )

        if ok and dirname:
            dir_path = parent_path / dirname

            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create folder: {str(e)}"
                )

    def _on_delete_clicked(self, item):
        if not item:
            return

        item_path = item.data(0, QtCore.Qt.UserRole)
        if item_path:
            path = Path(item_path)
            item_type = "file"
        else:
            path = self._get_item_path(item)
            item_type = "folder"

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete this {item_type}?\n{path}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )

        if reply == QtWidgets.QMessageBox.Yes:
            try:
                if item_type == "file":
                    path.unlink()
                else:
                    shutil.rmtree(path)

                QtWidgets.QMessageBox.information(self, "Success", f"Deleted: {path}")
                self.refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to delete: {str(e)}"
                )

    def _get_item_path(self, item):
        parts = []
        current = item

        while current is not None:
            parts.insert(0, current.text(0))
            current = current.parent()

        root = Path("supplemental")
        for part in parts:
            root = root / part

        return root

    def get_checked_paths(self):
        checked_paths = []

        def collect_checked(item):
            if item is None:
                for i in range(self.supp_list.topLevelItemCount()):
                    collect_checked(self.supp_list.topLevelItem(i))
            else:
                if item.checkState(0) == QtCore.Qt.Checked:
                    path = item.data(0, QtCore.Qt.UserRole)
                    if path:
                        checked_paths.append(str(path))

                for i in range(item.childCount()):
                    collect_checked(item.child(i))

        collect_checked(None)
        return checked_paths

    def gather_supplemental_text(self):
        parts = []

        def collect_checked_items(parent):
            if parent is None:
                count = self.supp_list.topLevelItemCount()
                for i in range(count):
                    item = self.supp_list.topLevelItem(i)
                    collect_checked_items(item)
            else:
                if parent.checkState(0) == QtCore.Qt.Checked:
                    path = parent.data(0, QtCore.Qt.UserRole)
                    if path:
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                txt = f.read().strip()
                                if txt:
                                    parts.append(txt)
                        except Exception:
                            pass

                child_count = parent.childCount()
                for i in range(child_count):
                    child = parent.child(i)
                    collect_checked_items(child)

        collect_checked_items(None)
        return "\n".join(parts)

    def load_files(self, files, selected_files=None):
        expanded_folders = self._save_tree_expanded_state()

        try:
            self.supp_list.itemChanged.disconnect(self._on_item_changed)
        except Exception:
            pass

        self.supp_list.clear()
        self._populate_tree(files, checkable=True, draggable=True)

        if selected_files:
            self._restore_tree_selections(selected_files)

        self._restore_tree_expanded_state(expanded_folders)
        self.supp_list.itemChanged.connect(self._on_item_changed)

    def _populate_tree(self, files, checkable=True, draggable=False):
        parent_items = {}

        for file_path, is_dir in files:
            path = Path(file_path)
            parent_path = path.parent

            parent_item = parent_items.get(parent_path)

            if is_dir:
                folder_item = QtWidgets.QTreeWidgetItem(
                    parent_item if parent_item else self.supp_list
                )
                folder_item.setText(0, path.name)
                folder_item.setData(0, QtCore.Qt.UserRole, None)
                folder_item.setFlags(folder_item.flags() | QtCore.Qt.ItemIsEnabled)
                font = folder_item.font(0)
                font.setBold(True)
                folder_item.setFont(0, font)
                parent_items[path] = folder_item
            else:
                file_item = QtWidgets.QTreeWidgetItem(
                    parent_item if parent_item else self.supp_list
                )
                file_item.setText(0, path.name)
                flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                if checkable:
                    flags |= QtCore.Qt.ItemIsUserCheckable
                    file_item.setCheckState(0, QtCore.Qt.Unchecked)
                if draggable:
                    flags |= QtCore.Qt.ItemIsDragEnabled
                file_item.setFlags(flags)
                file_item.setData(0, QtCore.Qt.UserRole, str(file_path))

    def _restore_tree_selections(self, selected_paths):
        if not selected_paths:
            return

        selected_paths_set = {str(p) for p in selected_paths}

        def check_item(item):
            if item is None:
                for i in range(self.supp_list.topLevelItemCount()):
                    check_item(self.supp_list.topLevelItem(i))
            else:
                item_path = item.data(0, QtCore.Qt.UserRole)
                if item_path and str(item_path) in selected_paths_set:
                    item.setCheckState(0, QtCore.Qt.Checked)

                for i in range(item.childCount()):
                    check_item(item.child(i))

        check_item(None)

    def _save_tree_expanded_state(self):
        expanded_folders = set()

        def collect_expanded(item):
            if item is None:
                for i in range(self.supp_list.topLevelItemCount()):
                    collect_expanded(self.supp_list.topLevelItem(i))
            else:
                if item.childCount() > 0 and item.isExpanded():
                    path_parts = []
                    current = item
                    while current is not None:
                        path_parts.insert(0, current.text(0))
                        current = current.parent()
                    folder_path = "/".join(path_parts)
                    expanded_folders.add(folder_path)

                for i in range(item.childCount()):
                    collect_expanded(item.child(i))

        collect_expanded(None)
        return expanded_folders

    def _restore_tree_expanded_state(self, expanded_folders):
        if not expanded_folders:
            return

        def expand_matching(item, path_parts):
            if item is None:
                for i in range(self.supp_list.topLevelItemCount()):
                    expand_matching(
                        self.supp_list.topLevelItem(i),
                        [self.supp_list.topLevelItem(i).text(0)],
                    )
            else:
                current_path = "/".join(path_parts)
                if current_path in expanded_folders:
                    item.setExpanded(True)

                for i in range(item.childCount()):
                    child = item.child(i)
                    child_path_parts = path_parts + [child.text(0)]
                    expand_matching(child, child_path_parts)

        expand_matching(None, [])

    def get_file_order(self):
        ordered_files = []

        def collect_files(item):
            if item is None:
                for i in range(self.supp_list.topLevelItemCount()):
                    collect_files(self.supp_list.topLevelItem(i))
            else:
                file_path = item.data(0, QtCore.Qt.UserRole)
                if file_path:
                    ordered_files.append(str(file_path))

                for i in range(item.childCount()):
                    collect_files(item.child(i))

        collect_files(None)
        return ordered_files
