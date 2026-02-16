"""Utilities panel container view."""

from PyQt5 import QtWidgets, QtCore

from views.notes_panel import NotesPanel
from views.supplemental_panel import SupplementalPanel
from views.system_panel import SystemPanel
from views.rag_panel import RagPanel
from views.logs_panel import LogsPanel
from models.supplemental_panel_model import SupplementalPanelModel
from models.system_panel_model import SystemPanelModel
from models.rag_panel_model import RagPanelModel
from models.logs_panel_model import LogsPanelModel
from controllers.supplemental_controller import SupplementalController
from controllers.system_controller import SystemController
from controllers.rag_panel_controller import RagPanelController
from controllers.logs_controller import LogsController


class UtilitiesPanel(QtWidgets.QWidget):
    """Panel for managing notes, supplemental prompts, system prompts, RAG, and logs."""

    supplemental_refresh_clicked = QtCore.pyqtSignal()
    supplemental_add_clicked = QtCore.pyqtSignal()
    supplemental_file_opened = QtCore.pyqtSignal(str)
    system_refresh_clicked = QtCore.pyqtSignal()
    system_add_clicked = QtCore.pyqtSignal()
    system_file_opened = QtCore.pyqtSignal(str)
    rag_create_database_clicked = QtCore.pyqtSignal()
    rag_add_files_clicked = QtCore.pyqtSignal(str)
    rag_database_toggled = QtCore.pyqtSignal(str)
    rag_refresh_clicked = QtCore.pyqtSignal()
    rag_delete_database_clicked = QtCore.pyqtSignal(str)
    rag_max_chunks_changed = QtCore.pyqtSignal(int)
    rag_summary_chunk_size_changed = QtCore.pyqtSignal(int)
    rag_score_threshold_changed = QtCore.pyqtSignal(float)
    rag_settings_requested = QtCore.pyqtSignal()
    prompt_selections_changed = QtCore.pyqtSignal(list, str)
    font_size_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.notes_panel = NotesPanel()
        self.supplemental_panel = SupplementalPanel()
        self.system_panel = SystemPanel()
        self.rag_panel = RagPanel()
        self.logs_panel = LogsPanel()

        self._init_models()
        self._init_controllers()
        self._init_ui()
        self._connect_signals()

    def _init_models(self):
        self.supplemental_model = SupplementalPanelModel()
        self.system_model = SystemPanelModel()
        self.rag_model = RagPanelModel()
        self.logs_model = LogsPanelModel()

    def _init_controllers(self):
        self.supplemental_controller = SupplementalController(
            self.supplemental_model, self.supplemental_panel
        )
        self.system_controller = SystemController(self.system_model, self.system_panel)
        self.rag_controller = RagPanelController(self.rag_model, self.rag_panel)
        self.logs_controller = LogsController(self.logs_model, self.logs_panel)

    def _init_ui(self):
        container = QtWidgets.QGroupBox("Utilities")
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(5, 5, 5, 5)

        self.utilities_tab_widget = QtWidgets.QTabWidget()
        self.utilities_tab_widget.addTab(self.notes_panel, "Notes")
        self.utilities_tab_widget.addTab(self.supplemental_panel, "Supplemental")
        self.utilities_tab_widget.addTab(self.system_panel, "System")
        self.utilities_tab_widget.addTab(self.rag_panel, "RAG")
        self.utilities_tab_widget.addTab(self.logs_panel, "Logs")

        container_layout.addWidget(self.utilities_tab_widget)
        container.setLayout(container_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def _connect_signals(self):
        self.notes_panel.font_size_changed.connect(self.font_size_changed.emit)
        self.supplemental_panel.font_size_changed.connect(self.font_size_changed.emit)
        self.system_panel.font_size_changed.connect(self.font_size_changed.emit)
        self.rag_panel.font_size_changed.connect(self.font_size_changed.emit)
        self.logs_panel.font_size_changed.connect(self.font_size_changed.emit)

        self.supplemental_panel.refresh_clicked.connect(
            self.supplemental_refresh_clicked.emit
        )
        self.supplemental_panel.add_clicked.connect(self.supplemental_add_clicked.emit)
        self.supplemental_panel.file_opened.connect(self.supplemental_file_opened.emit)
        self.supplemental_panel.selections_changed.connect(
            self._on_prompt_selections_changed
        )

        self.system_panel.refresh_clicked.connect(self.system_refresh_clicked.emit)
        self.system_panel.add_clicked.connect(self.system_add_clicked.emit)
        self.system_panel.file_opened.connect(self.system_file_opened.emit)
        self.system_panel.selection_changed.connect(self._on_prompt_selections_changed)

        self.rag_panel.create_database_clicked.connect(
            self.rag_create_database_clicked.emit
        )
        self.rag_panel.add_files_clicked.connect(self.rag_add_files_clicked.emit)
        self.rag_panel.database_toggled.connect(self.rag_database_toggled.emit)
        self.rag_panel.refresh_clicked.connect(self.rag_refresh_clicked.emit)
        self.rag_panel.delete_database_clicked.connect(
            self.rag_delete_database_clicked.emit
        )
        self.rag_panel.max_chunks_changed.connect(self.rag_max_chunks_changed.emit)
        self.rag_panel.summary_chunk_size_changed.connect(
            self.rag_summary_chunk_size_changed.emit
        )
        self.rag_panel.score_threshold_changed.connect(
            self.rag_score_threshold_changed.emit
        )
        self.rag_panel.settings_requested.connect(self.rag_settings_requested.emit)

    def _on_prompt_selections_changed(self, *_):
        supplemental_files = self.supplemental_controller.get_selected_files()
        system_prompt = self.system_controller.get_selected_file()
        self.prompt_selections_changed.emit(supplemental_files, system_prompt)

    def apply_font_size(self, size):
        self.supplemental_panel.apply_font_size(size)
        self.system_panel.apply_font_size(size)
        self.rag_panel.apply_font_size(size)
        self.logs_panel.apply_font_size(size)
        self.notes_panel.apply_font_size(size)

    def gather_supplemental_text(self):
        return self.supplemental_controller.gather_supplemental_text()

    def get_system_prompt_text(self):
        return self.system_controller.get_system_prompt_text()

    def load_supplemental_files(self, files, selected_files=None):
        self.supplemental_controller.load_files(files, selected_files)

    def load_system_prompt_files(self, files, selected_file=None):
        self.system_controller.load_files(files, selected_file)

    def load_rag_databases(self, databases):
        self.rag_controller.load_databases(databases)

    def show_rag_settings_dialog(
        self,
        current_max_chunks=10,
        current_summary_chunk_size=1500,
        current_score_threshold=5.0,
    ):
        return self.rag_controller.show_settings_dialog(
            current_max_chunks,
            current_summary_chunk_size,
            current_score_threshold,
        )

    def append_logs(self, text: str):
        self.logs_controller.append_logs(text)

    def clear_logs(self):
        self.logs_controller.clear_logs()
