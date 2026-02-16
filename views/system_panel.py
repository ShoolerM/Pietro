"""System prompts panel view."""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
import shutil

from models.stylesheets import SYSTEM_PROMPTS
from views.utilities_widgets import OrderTrackingTreeWidget


class SystemPanel(QtWidgets.QWidget):
    """Panel for managing system prompts."""

    refresh_clicked = QtCore.pyqtSignal()
    add_clicked = QtCore.pyqtSignal()
    file_opened = QtCore.pyqtSignal(str)
    selection_changed = QtCore.pyqtSignal(str)
    font_size_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.sys_list = OrderTrackingTreeWidget(base_path="system_prompts")
        self.sys_list.setHeaderHidden(True)
        self.sys_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.sys_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.sys_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sys_list.setAlternatingRowColors(True)
        fm_sys = self.sys_list.fontMetrics()
        row_h_sys = fm_sys.lineSpacing() or fm_sys.height()
        min_h_sys = row_h_sys * 2 + 8
        self.sys_list.setMinimumHeight(min_h_sys)
        self.sys_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.sys_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.sys_list.setUniformRowHeights(True)
        self.sys_list.installEventFilter(self)
        self.sys_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.sys_list.customContextMenuRequested.connect(self._show_context_menu)

        self.sys_list.itemChanged.connect(self._on_item_changed)
        self.sys_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        self.sys_list.file_moved.connect(self._on_file_moved)

        layout.addWidget(self.sys_list)
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
            self.sys_list.setFont(font)
        except Exception:
            pass

    def _on_file_moved(self, old_path, new_path):
        self.refresh_clicked.emit()

    def _on_item_changed(self, item, column):
        if not item.data(0, QtCore.Qt.UserRole):
            return

        if item.checkState(0) == QtCore.Qt.Checked:

            def uncheck_all_except(item_to_skip, parent=None):
                if parent is None:
                    count = self.sys_list.topLevelItemCount()
                    for i in range(count):
                        item = self.sys_list.topLevelItem(i)
                        uncheck_all_except(item_to_skip, item)
                else:
                    if (
                        parent is not item_to_skip
                        and parent.checkState(0) == QtCore.Qt.Checked
                    ):
                        parent.setCheckState(0, QtCore.Qt.Unchecked)

                    child_count = parent.childCount()
                    for i in range(child_count):
                        child = parent.child(i)
                        uncheck_all_except(item_to_skip, child)

            uncheck_all_except(item)

        self.selection_changed.emit(self.get_checked_path())

    def _on_file_double_clicked(self, item, column=0):
        file_path = item.data(0, QtCore.Qt.UserRole)
        if file_path:
            self.file_opened.emit(file_path)

    def _show_context_menu(self, position):
        item = self.sys_list.itemAt(position)

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(SYSTEM_PROMPTS)

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

        menu.exec_(self.sys_list.viewport().mapToGlobal(position))

    def _on_create_file(self, parent_item):
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            parent_path = self._get_item_path(parent_item)
        else:
            parent_path = Path("system_prompts")

        filename, ok = QtWidgets.QInputDialog.getText(
            self, "Create File", "Enter filename:"
        )

        if ok and filename:
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
            parent_path = Path("system_prompts")

        dirname, ok = QtWidgets.QInputDialog.getText(
            self, "Create Folder", "Enter folder name:"
        )

        if ok and dirname:
            dir_path = parent_path / dirname

            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Folder created: {dir_path}"
                )
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

        root = Path("system_prompts")
        for part in parts:
            root = root / part

        return root

    def get_checked_path(self):
        def find_checked(item):
            if item is None:
                for i in range(self.sys_list.topLevelItemCount()):
                    result = find_checked(self.sys_list.topLevelItem(i))
                    if result:
                        return result
            else:
                if item.checkState(0) == QtCore.Qt.Checked:
                    path = item.data(0, QtCore.Qt.UserRole)
                    if path:
                        return str(path)

                for i in range(item.childCount()):
                    result = find_checked(item.child(i))
                    if result:
                        return result
            return None

        return find_checked(None)

    def get_system_prompt_text(self):
        checked_path = self.get_checked_path()
        if checked_path:
            try:
                with open(checked_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""

    def load_files(self, files, selected_file=None):
        expanded_folders = self._save_tree_expanded_state()

        try:
            self.sys_list.itemChanged.disconnect(self._on_item_changed)
        except Exception:
            pass

        self.sys_list.clear()
        self._populate_tree(files, checkable=True, draggable=True)

        if selected_file:
            self._restore_tree_selections([selected_file])

        self._restore_tree_expanded_state(expanded_folders)
        self.sys_list.itemChanged.connect(self._on_item_changed)

    def _populate_tree(self, files, checkable=True, draggable=False):
        parent_items = {}

        for file_path, is_dir in files:
            path = Path(file_path)
            parent_path = path.parent

            parent_item = parent_items.get(parent_path)

            if is_dir:
                folder_item = QtWidgets.QTreeWidgetItem(
                    parent_item if parent_item else self.sys_list
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
                    parent_item if parent_item else self.sys_list
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
                for i in range(self.sys_list.topLevelItemCount()):
                    check_item(self.sys_list.topLevelItem(i))
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
                for i in range(self.sys_list.topLevelItemCount()):
                    collect_expanded(self.sys_list.topLevelItem(i))
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
                for i in range(self.sys_list.topLevelItemCount()):
                    expand_matching(
                        self.sys_list.topLevelItem(i),
                        [self.sys_list.topLevelItem(i).text(0)],
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
