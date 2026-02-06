"""Main application view - refactored to use panel components."""
from PyQt5 import QtWidgets, QtCore

from views.thinking_panel import ThinkingPanel
from views.story_panel import StoryPanel
from views.prompts_panel import PromptsPanel
from views.control_panel import ControlPanel


class MainView(QtWidgets.QWidget):
    """Main application window view - coordinates all panels."""
    
    # Signals for user actions (forwarded from panels)
    send_clicked = QtCore.pyqtSignal(str, str, str, str)  # user_input, notes, supp_text, system_prompt
    undo_clicked = QtCore.pyqtSignal()
    stop_clicked = QtCore.pyqtSignal()
    clear_clicked = QtCore.pyqtSignal()
    model_refresh_clicked = QtCore.pyqtSignal()
    model_changed = QtCore.pyqtSignal(str)
    context_limit_changed = QtCore.pyqtSignal(int)
    supplemental_refresh_clicked = QtCore.pyqtSignal()
    supplemental_add_clicked = QtCore.pyqtSignal()
    supplemental_file_opened = QtCore.pyqtSignal(str)
    system_refresh_clicked = QtCore.pyqtSignal()
    system_add_clicked = QtCore.pyqtSignal()
    system_file_opened = QtCore.pyqtSignal(str)
    rag_create_database_clicked = QtCore.pyqtSignal()
    rag_add_files_clicked = QtCore.pyqtSignal(str)  # database name
    rag_database_toggled = QtCore.pyqtSignal(str)  # database name
    rag_refresh_clicked = QtCore.pyqtSignal()
    rag_delete_database_clicked = QtCore.pyqtSignal(str)  # database name
    rag_similarity_threshold_changed = QtCore.pyqtSignal(float)  # threshold value
    rag_max_docs_changed = QtCore.pyqtSignal(int)  # max documents per database
    rag_settings_requested = QtCore.pyqtSignal()  # request to show settings dialog
    prompt_selections_changed = QtCore.pyqtSignal(list, str)  # supplemental_files, system_prompt
    settings_opened = QtCore.pyqtSignal()
    file_saved = QtCore.pyqtSignal(str, str)
    font_size_changed = QtCore.pyqtSignal(int)
    inference_settings_requested = QtCore.pyqtSignal()  # request to show inference settings dialog
    update_summary_requested = QtCore.pyqtSignal()  # request to regenerate story summary
    toggle_summarize_prompts_requested = QtCore.pyqtSignal()  # forwarded from story panel
    toggle_build_with_rag_requested = QtCore.pyqtSignal()  # forwarded from story panel
    auto_build_story_requested = QtCore.pyqtSignal()  # request to automatically build complete story
    override_selection_requested = QtCore.pyqtSignal(str, int, int)  # selected_text, start_pos, end_pos
    update_selection_with_prompt_requested = QtCore.pyqtSignal(str, int, int, str)  # selected_text, start_pos, end_pos, prompt
    update_accepted = QtCore.pyqtSignal()  # user accepted the update
    update_rejected = QtCore.pyqtSignal()  # user rejected the update
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Chat UI')
        
        # Create panels
        self.thinking_panel = ThinkingPanel()
        self.story_panel = StoryPanel()
        self.prompts_panel = PromptsPanel()
        self.control_panel = ControlPanel()
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create menu bar
        menu_bar = QtWidgets.QMenuBar()
        file_menu = menu_bar.addMenu('File')
        load_action = file_menu.addAction('Load...')
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(lambda: self.story_panel.load_story_file())
        save_action = file_menu.addAction('Save')
        save_action.triggered.connect(lambda: self.story_panel._save_current_file())
        save_as_action = file_menu.addAction('Save As...')
        save_as_action.triggered.connect(lambda: self.story_panel.save_story_file_as())
        file_menu.addSeparator()
        settings_action = file_menu.addAction('Settings')
        settings_action.triggered.connect(lambda: self.settings_opened.emit())
        
        inference_menu = menu_bar.addMenu('Inference')
        inference_settings_action = inference_menu.addAction('Server Settings...')
        inference_settings_action.triggered.connect(lambda: self.inference_settings_requested.emit())
        
        rag_menu = menu_bar.addMenu('RAG')
        rag_settings_action = rag_menu.addAction('Similarity Threshold...')
        rag_settings_action.triggered.connect(lambda: self.rag_settings_requested.emit())
        
        # Main vertical splitter - story panel on top, prompts on bottom
        main_vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # Top container: story panel + control panel
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.story_panel, stretch=1)  # Give story panel stretch factor to fill space
        top_layout.addWidget(self.control_panel, stretch=0)  # Control panel stays at fixed size at bottom
        top_container.setLayout(top_layout)
        
        main_vertical_splitter.addWidget(top_container)
        main_vertical_splitter.addWidget(self.prompts_panel)
        main_vertical_splitter.setSizes([400, 200])
        
        # Main horizontal layout with thinking panel on left
        main_hbox = QtWidgets.QHBoxLayout()
        main_hbox.addWidget(self.thinking_panel)
        main_hbox.addWidget(main_vertical_splitter, stretch=1)
        
        # Create main widget and layout
        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setMenuBar(menu_bar)
        central_layout.addLayout(main_hbox)
        central_widget.setLayout(central_layout)
        
        # Set central widget with wrapper layout
        wrapper = QtWidgets.QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(central_widget)
        self.setLayout(wrapper)
    
    def _connect_signals(self):
        """Connect panel signals to main view signals."""
        # Story panel signals
        self.story_panel.file_saved.connect(self.file_saved.emit)
        self.story_panel.font_size_changed.connect(self.font_size_changed.emit)
        self.story_panel.toggle_thinking_requested.connect(self._toggle_thinking_panel)
        self.story_panel.toggle_markdown_requested.connect(self._toggle_markdown)
        self.story_panel.update_summary_requested.connect(self.update_summary_requested.emit)
        self.story_panel.toggle_summarize_prompts_requested.connect(self.toggle_summarize_prompts_requested.emit)
        self.story_panel.toggle_build_with_rag_requested.connect(self.toggle_build_with_rag_requested.emit)
        self.story_panel.auto_build_story_requested.connect(self.auto_build_story_requested.emit)
        self.story_panel.override_selection_requested.connect(self.override_selection_requested.emit)
        self.story_panel.update_selection_with_prompt_requested.connect(self.update_selection_with_prompt_requested.emit)
        self.story_panel.update_accepted.connect(self.update_accepted.emit)
        self.story_panel.update_rejected.connect(self.update_rejected.emit)
        
        # Thinking panel signals
        self.thinking_panel.font_size_changed.connect(self.font_size_changed.emit)
        
        # Control panel signals
        self.control_panel.send_clicked.connect(self._on_send)
        self.control_panel.undo_clicked.connect(self.undo_clicked.emit)
        self.control_panel.clear_clicked.connect(self.clear_clicked.emit)
        self.control_panel.stop_clicked.connect(self.stop_clicked.emit)
        self.control_panel.model_refresh_clicked.connect(self.model_refresh_clicked.emit)
        self.control_panel.model_changed.connect(self.model_changed.emit)
        self.control_panel.context_limit_changed.connect(self.context_limit_changed.emit)
        self.control_panel.font_size_changed.connect(self.font_size_changed.emit)
        
        # Prompts panel signals
        self.prompts_panel.supplemental_refresh_clicked.connect(self.supplemental_refresh_clicked.emit)
        self.prompts_panel.supplemental_add_clicked.connect(self.supplemental_add_clicked.emit)
        self.prompts_panel.supplemental_file_opened.connect(self.supplemental_file_opened.emit)
        self.prompts_panel.system_refresh_clicked.connect(self.system_refresh_clicked.emit)
        self.prompts_panel.system_add_clicked.connect(self.system_add_clicked.emit)
        self.prompts_panel.system_file_opened.connect(self.system_file_opened.emit)
        self.prompts_panel.rag_create_database_clicked.connect(self.rag_create_database_clicked.emit)
        self.prompts_panel.rag_add_files_clicked.connect(self.rag_add_files_clicked.emit)
        self.prompts_panel.rag_database_toggled.connect(self.rag_database_toggled.emit)
        self.prompts_panel.rag_refresh_clicked.connect(self.rag_refresh_clicked.emit)
        self.prompts_panel.rag_delete_database_clicked.connect(self.rag_delete_database_clicked.emit)
        self.prompts_panel.rag_similarity_threshold_changed.connect(self.rag_similarity_threshold_changed.emit)
        self.prompts_panel.rag_max_docs_changed.connect(self.rag_max_docs_changed.emit)
        self.prompts_panel.rag_settings_requested.connect(self.rag_settings_requested.emit)
        self.prompts_panel.prompt_selections_changed.connect(self.prompt_selections_changed.emit)
        self.prompts_panel.font_size_changed.connect(self.font_size_changed.emit)
    
    def _toggle_thinking_panel(self):
        """Toggle visibility of the thinking panel."""
        if self.thinking_panel.isVisible():
            self.thinking_panel.hide()
            self.story_panel.set_thinking_visible(False)
        else:
            self.thinking_panel.show()
            self.story_panel.set_thinking_visible(True)
    
    def _toggle_markdown(self):
        """Toggle markdown rendering in story panel."""
        current_state = self.story_panel.is_markdown_enabled()
        new_state = not current_state
        self.story_panel.set_markdown_enabled(new_state)
        
        # Re-render the current content with new setting
        current_content = self.story_panel.get_story_content()
        if current_content:
            self.story_panel.set_story_content(current_content)
    
    def _on_send(self):
        """Handle send button click - gather data from all panels."""
        user_input = self.control_panel.get_user_input().strip()
        if not user_input:
            return
        
        notes = self.prompts_panel.get_notes_text().strip()
        supp_text = self.prompts_panel.gather_supplemental_text()
        system_prompt = self.prompts_panel.get_system_prompt_text()
        
        self.send_clicked.emit(user_input, notes, supp_text, system_prompt)
    
    # === Public API methods for controller interaction ===
    
    def get_story_content(self):
        """Get current story content as plain text."""
        return self.story_panel.get_story_content()
    
    def set_story_content(self, content):
        """Set story content."""
        self.story_panel.set_story_content(content)
    
    def append_story_content(self, text):
        """Append text to story content."""
        self.story_panel.append_story_content(text)
    
    def render_story_markdown(self, markdown_content):
        """Render markdown content in story view."""
        self.story_panel.render_story_markdown(markdown_content)
    
    def clear_story_content(self):
        """Clear story content."""
        self.story_panel.clear_story_content()
    
    def append_thinking_text(self, text):
        """Append text to thinking panel."""
        self.thinking_panel.append_thinking_text(text)
    
    def clear_thinking_text(self):
        """Clear thinking panel."""
        self.thinking_panel.clear_thinking_text()
    
    def set_waiting(self, waiting):
        """Set waiting state (show/hide progress bar, enable/disable input)."""
        self.control_panel.set_waiting(waiting)

    def set_summarize_prompts_enabled(self, enabled: bool):
        """Update the StoryPanel UI state for summarization toggle."""
        try:
            self.story_panel.set_summarize_prompts_enabled(enabled)
        except Exception:
            pass
    
    def set_build_with_rag_enabled(self, enabled: bool):
        """Update the StoryPanel UI state for build with RAG toggle."""
        try:
            self.story_panel.set_build_with_rag_enabled(enabled)
        except Exception:
            pass
    
    def set_stop_enabled(self, enabled):
        """Enable or disable the stop button."""
        self.control_panel.set_stop_enabled(enabled)
    
    def set_models(self, models):
        """Set available models in dropdown."""
        self.control_panel.set_models(models)
    
    def set_model_error(self, error_message):
        """Set error message in model dropdown."""
        self.control_panel.set_model_error(error_message)
    
    def load_supplemental_files(self, files, selected_files=None):
        """Load supplemental files into tree widget."""
        self.prompts_panel.load_supplemental_files(files, selected_files)
    
    def load_system_prompt_files(self, files, selected_file=None):
        """Load system prompt files into tree widget."""
        self.prompts_panel.load_system_prompt_files(files, selected_file)
    
    def load_rag_databases(self, databases):
        """Load RAG databases into tree widget."""
        self.prompts_panel.load_rag_databases(databases)
    
    def show_file_chooser(self, title, multiple=False, allow_directory=False):
        """Show file chooser dialog.
        
        Args:
            title: Dialog title
            multiple: Allow selecting multiple files
            allow_directory: Allow selecting files or directories
        
        Returns:
            list: Selected file/directory paths
        """
        if allow_directory:
            # Use file dialog that allows selecting both files and directories
            dialog = QtWidgets.QFileDialog(self, title, "")
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
            dialog.setOption(QtWidgets.QFileDialog.DontResolveSymlinks, True)
            
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                return dialog.selectedFiles()
            return []
        elif multiple:
            file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, title, "", "All Files (*.*)"
            )
            return file_paths
        else:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, title, "", "All Files (*.*)"
            )
            return [file_path] if file_path else []
    
    def open_file_tab(self, file_path):
        """Open a file in a new editable tab."""
        return self.story_panel.open_file_tab(file_path)
    
    def apply_font_size(self, size):
        """Apply font size to all text widgets."""
        self.story_panel.apply_font_size(size)
        self.thinking_panel.apply_font_size(size)
        self.prompts_panel.apply_font_size(size)
        self.control_panel.apply_font_size(size)
    
    def show_rag_settings_dialog(self, current_max_docs=3, current_threshold=0.0):
        """Show RAG settings dialog with current values."""
        return self.prompts_panel.show_rag_settings_dialog(current_max_docs, current_threshold)
    
    def show_inference_settings_dialog(self, current_ip='192.168.0.1', current_port=1234):
        """Show inference server settings dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Inference Server Settings')
        dialog.setMinimumWidth(400)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Description
        desc_label = QtWidgets.QLabel(
            'Configure the IP address and port of your inference server.\n'
            'These settings will be saved and persist across application restarts.'
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # IP Address
        ip_layout = QtWidgets.QHBoxLayout()
        ip_label = QtWidgets.QLabel('IP Address:')
        ip_label.setMinimumWidth(80)
        ip_layout.addWidget(ip_label)
        
        ip_input = QtWidgets.QLineEdit()
        ip_input.setText(current_ip)
        ip_input.setPlaceholderText('e.g., 192.168.0.1 or localhost')
        ip_layout.addWidget(ip_input)
        layout.addLayout(ip_layout)
        
        # Port
        port_layout = QtWidgets.QHBoxLayout()
        port_label = QtWidgets.QLabel('Port:')
        port_label.setMinimumWidth(80)
        port_layout.addWidget(port_label)
        
        port_input = QtWidgets.QSpinBox()
        port_input.setMinimum(1)
        port_input.setMaximum(65535)
        port_input.setValue(current_port)
        port_layout.addWidget(port_input)
        port_layout.addStretch()
        layout.addLayout(port_layout)
        
        # Current URL preview
        url_preview_label = QtWidgets.QLabel()
        url_preview_label.setStyleSheet('color: #888; font-style: italic;')
        
        def update_url_preview():
            url = f"http://{ip_input.text()}:{port_input.value()}/v1"
            url_preview_label.setText(f'Full URL: {url}')
        
        update_url_preview()
        ip_input.textChanged.connect(lambda: update_url_preview())
        port_input.valueChanged.connect(lambda: update_url_preview())
        
        layout.addWidget(url_preview_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            ip = ip_input.text().strip()
            port = port_input.value()
            return (ip, port)
        
        return None
    
    def show_settings_dialog(self, current_prompt):
        """Show settings dialog and return new prompt if saved."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Settings')
        dialog.resize(600, 400)
        
        layout = QtWidgets.QVBoxLayout()
        
        label = QtWidgets.QLabel('Summary Prompt Template:')
        layout.addWidget(label)
        
        info_label = QtWidgets.QLabel(
            'Note: "STORY TO SUMMARIZE:\\n{story_text}\\n\\n" will be automatically appended.'
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet('color: #888888; font-style: italic;')
        layout.addWidget(info_label)
        
        prompt_edit = QtWidgets.QTextEdit()
        prompt_edit.setPlainText(current_prompt)
        layout.addWidget(prompt_edit)
        
        button_layout = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton('Save')
        cancel_button = QtWidgets.QPushButton('Cancel')
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        result = {'saved': False, 'prompt': None}
        
        def on_save():
            result['saved'] = True
            result['prompt'] = prompt_edit.toPlainText()
            dialog.accept()
        
        save_button.clicked.connect(on_save)
        cancel_button.clicked.connect(dialog.reject)
        
        dialog.exec_()
        
        return result['saved'], result['prompt']
    
    def show_input_dialog(self, title, label):
        """Show input dialog and return result."""
        return QtWidgets.QInputDialog.getText(self, title, label)
    
    def show_warning(self, title, message):
        """Show warning message box."""
        QtWidgets.QMessageBox.warning(self, title, message)
    
    def start_text_update(self, start_pos, end_pos):
        """Initialize streaming text replacement at the given position.
        
        Args:
            start_pos: Start position of text to replace
            end_pos: End position of text to replace
        """
        self.story_panel.start_text_update(start_pos, end_pos)
    
    def stream_override_text(self, text_chunk):
        """Stream replacement text at the update position.
        
        Args:
            text_chunk: Text chunk to insert
        """
        self.story_panel.stream_override_text(text_chunk)
    
    def finish_text_update(self):
        """Finalize the text update operation and clear formatting."""
        self.story_panel.finish_text_update()
