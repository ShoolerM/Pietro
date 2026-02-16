"""RAG databases panel view."""

from PyQt5 import QtWidgets, QtCore
import traceback


class RagPanel(QtWidgets.QWidget):
    """Panel for managing RAG databases."""

    create_database_clicked = QtCore.pyqtSignal()
    add_files_clicked = QtCore.pyqtSignal(str)
    database_toggled = QtCore.pyqtSignal(str)
    refresh_clicked = QtCore.pyqtSignal()
    delete_database_clicked = QtCore.pyqtSignal(str)
    max_chunks_changed = QtCore.pyqtSignal(int)
    summary_chunk_size_changed = QtCore.pyqtSignal(int)
    score_threshold_changed = QtCore.pyqtSignal(float)
    settings_requested = QtCore.pyqtSignal()
    font_size_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.rag_list = QtWidgets.QTreeWidget()
        self.rag_list.setHeaderHidden(True)
        self.rag_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.rag_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.rag_list.setAlternatingRowColors(True)
        fm_rag = self.rag_list.fontMetrics()
        row_h_rag = fm_rag.lineSpacing() or fm_rag.height()
        min_h_rag = row_h_rag * 2 + 8
        self.rag_list.setMinimumHeight(min_h_rag)
        self.rag_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.rag_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.rag_list.setUniformRowHeights(True)
        self.rag_list.installEventFilter(self)

        self.rag_list.itemChanged.connect(self._on_item_changed)
        self.rag_list.itemDoubleClicked.connect(self._on_item_double_clicked)

        rag_controls = QtWidgets.QVBoxLayout()
        self.rag_create_db = QtWidgets.QPushButton("New Database")
        self.rag_create_db.clicked.connect(self.create_database_clicked.emit)
        rag_controls.addWidget(self.rag_create_db)

        self.rag_delete_db = QtWidgets.QPushButton("Delete Selected")
        self.rag_delete_db.setToolTip("Delete the selected database")
        self.rag_delete_db.clicked.connect(self._on_delete_database_clicked)
        rag_controls.addWidget(self.rag_delete_db)
        rag_controls.addStretch(1)

        rag_main_layout = QtWidgets.QHBoxLayout()
        rag_main_layout.addWidget(self.rag_list)
        rag_main_layout.addLayout(rag_controls)

        layout.addLayout(rag_main_layout)
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
        font = self.rag_list.font()
        font.setPointSize(size)
        self.rag_list.setFont(font)

    def _on_item_changed(self, item, column):
        try:
            user_data = item.data(0, QtCore.Qt.UserRole)
            if user_data == "database":
                db_name = item.text(0).split(" (")[0]
                self.database_toggled.emit(db_name)
        except Exception as e:
            print(f"Error in _on_rag_item_changed: {e}")
            traceback.print_exc()

    def _on_item_double_clicked(self, item, column=0):
        user_data = item.data(0, QtCore.Qt.UserRole)
        if user_data and user_data.startswith("add_files:"):
            db_name = user_data.split(":", 1)[1]
            self.add_files_clicked.emit(db_name)

    def _on_delete_database_clicked(self):
        checked_databases = []

        for i in range(self.rag_list.topLevelItemCount()):
            top_item = self.rag_list.topLevelItem(i)
            user_data = top_item.data(0, QtCore.Qt.UserRole)

            if user_data == "database" and top_item.checkState(0) == QtCore.Qt.Checked:
                db_name = top_item.text(0).split(" (")[0]
                checked_databases.append(db_name)

        if not checked_databases:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please check at least one database to delete."
            )
            return

        if len(checked_databases) == 1:
            message = (
                f'Are you sure you want to delete the database "{checked_databases[0]}"?\n\n'
                "This will remove all stored files and embeddings."
            )
            title = "Confirm Deletion"
        else:
            db_list = "\n  • ".join(checked_databases)
            message = (
                f"Are you sure you want to delete {len(checked_databases)} databases?\n\n"
                f"  • {db_list}\n\n"
                "This will remove all stored files and embeddings from these databases."
            )
            title = f"Confirm Deletion of {len(checked_databases)} Databases"

        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply == QtWidgets.QMessageBox.Yes:
            for db_name in checked_databases:
                self.delete_database_clicked.emit(db_name)

    def load_databases(self, databases):
        try:
            self.rag_list.clear()

            try:
                self.rag_list.itemChanged.disconnect(self._on_item_changed)
            except Exception:
                pass

            for db_name, file_count, is_selected in databases:
                db_item = QtWidgets.QTreeWidgetItem([f"{db_name} ({file_count} files)"])
                db_item.setData(0, QtCore.Qt.UserRole, "database")

                db_item.setFlags(
                    db_item.flags()
                    | QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsUserCheckable
                    | QtCore.Qt.ItemIsSelectable
                )
                db_item.setCheckState(
                    0, QtCore.Qt.Checked if is_selected else QtCore.Qt.Unchecked
                )

                add_button_item = QtWidgets.QTreeWidgetItem(["+ Add Files"])
                add_button_item.setData(0, QtCore.Qt.UserRole, f"add_files:{db_name}")
                add_button_item.setFlags(
                    add_button_item.flags()
                    | QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable
                )

                db_item.addChild(add_button_item)
                self.rag_list.addTopLevelItem(db_item)
                db_item.setExpanded(True)

            self.rag_list.itemChanged.connect(self._on_item_changed)

        except Exception as e:
            print(f"Error loading RAG databases: {e}")
            traceback.print_exc()

    def show_settings_dialog(
        self,
        current_max_chunks=10,
        current_summary_chunk_size=1500,
        current_score_threshold=5.0,
    ):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("RAG Settings")
        dialog.setMinimumWidth(450)

        layout = QtWidgets.QVBoxLayout()

        max_chunks_group = QtWidgets.QGroupBox("Generated Chunk Amount (Auto-Build)")
        max_chunks_layout = QtWidgets.QVBoxLayout()

        max_chunks_desc = QtWidgets.QLabel(
            "Number of chunks to generate during auto-build Smart Mode.\nDefault: 10"
        )
        max_chunks_desc.setWordWrap(True)
        max_chunks_layout.addWidget(max_chunks_desc)

        max_chunks_input_layout = QtWidgets.QHBoxLayout()
        max_chunks_label = QtWidgets.QLabel("Max Chunks:")
        max_chunks_input_layout.addWidget(max_chunks_label)

        max_chunks_spinbox = QtWidgets.QSpinBox()
        max_chunks_spinbox.setMinimum(1)
        max_chunks_spinbox.setMaximum(50)
        max_chunks_spinbox.setValue(current_max_chunks)
        max_chunks_spinbox.setToolTip(
            "Number of chunks to generate in auto-build mode (1-50)"
        )
        max_chunks_input_layout.addWidget(max_chunks_spinbox)
        max_chunks_input_layout.addStretch()

        max_chunks_layout.addLayout(max_chunks_input_layout)
        max_chunks_group.setLayout(max_chunks_layout)

        summary_group = QtWidgets.QGroupBox("Summarization Chunk Size")
        summary_layout = QtWidgets.QVBoxLayout()

        summary_desc = QtWidgets.QLabel(
            "Maximum raw token chunk size used when summarizing the story.\n"
            "Larger values keep more recent story content before summarizing.\n"
            "Default: 1500"
        )
        summary_desc.setWordWrap(True)
        summary_layout.addWidget(summary_desc)

        summary_input_layout = QtWidgets.QHBoxLayout()
        summary_label = QtWidgets.QLabel("Max Raw Tokens:")
        summary_input_layout.addWidget(summary_label)

        summary_spinbox = QtWidgets.QSpinBox()
        summary_spinbox.setMinimum(256)
        summary_spinbox.setMaximum(200000)
        summary_spinbox.setSingleStep(256)
        summary_spinbox.setValue(current_summary_chunk_size)
        summary_spinbox.setToolTip(
            "Max raw tokens kept before summarizing (256-200000)"
        )
        summary_input_layout.addWidget(summary_spinbox)
        summary_input_layout.addStretch()

        summary_layout.addLayout(summary_input_layout)
        summary_group.setLayout(summary_layout)

        threshold_group = QtWidgets.QGroupBox("Score Variance Threshold")
        threshold_layout = QtWidgets.QVBoxLayout()

        threshold_desc = QtWidgets.QLabel(
            "Filter RAG results based on similarity score variance.\n"
            "Results varying from the top score by more than this percentage will be excluded.\n"
            "Top 3 results are always included. Lower values = stricter filtering.\n"
            "Default: 5%"
        )
        threshold_desc.setWordWrap(True)
        threshold_layout.addWidget(threshold_desc)

        threshold_input_layout = QtWidgets.QHBoxLayout()
        threshold_label = QtWidgets.QLabel("Variance Threshold:")
        threshold_input_layout.addWidget(threshold_label)

        threshold_spinbox = QtWidgets.QDoubleSpinBox()
        threshold_spinbox.setMinimum(5.0)
        threshold_spinbox.setMaximum(30.0)
        threshold_spinbox.setSingleStep(1.0)
        threshold_spinbox.setDecimals(0)
        threshold_spinbox.setValue(current_score_threshold)
        threshold_spinbox.setSuffix("%")
        threshold_spinbox.setToolTip(
            "Score variance threshold for filtering results (5%-30%)"
        )
        threshold_input_layout.addWidget(threshold_spinbox)
        threshold_input_layout.addStretch()

        threshold_layout.addLayout(threshold_input_layout)
        threshold_group.setLayout(threshold_layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        layout.addWidget(max_chunks_group)
        layout.addWidget(summary_group)
        layout.addWidget(threshold_group)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            max_chunks = max_chunks_spinbox.value()
            summary_chunk_size = summary_spinbox.value()
            score_threshold = threshold_spinbox.value()
            self.max_chunks_changed.emit(max_chunks)
            self.summary_chunk_size_changed.emit(summary_chunk_size)
            self.score_threshold_changed.emit(score_threshold)
