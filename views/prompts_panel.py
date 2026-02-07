"""Prompts panel view for supplemental, system, and RAG prompts."""
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
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
        # Get the item being dragged
        dragged_items = self.selectedItems()
        if not dragged_items:
            event.ignore()
            return
        
        dragged_item = dragged_items[0]
        
        # Get the target item (where we're dropping)
        drop_position = event.pos()
        target_item = self.itemAt(drop_position)
        
        # Get the file path of the dragged item
        dragged_path = dragged_item.data(0, QtCore.Qt.UserRole)
        
        if not dragged_path:
            # It's a folder being dragged - allow default behavior
            super().dropEvent(event)
            return
        
        dragged_path = Path(dragged_path)
        
        # Determine the target directory
        if target_item:
            target_path = target_item.data(0, QtCore.Qt.UserRole)
            if target_path:
                # Dropped on a file - use its parent directory
                target_dir = Path(target_path).parent
            else:
                # Dropped on a folder - get the folder path
                target_dir = self._get_folder_path(target_item)
        else:
            # Dropped on empty space - use base directory
            target_dir = Path(self.base_path)
        
        # Calculate new path
        new_path = target_dir / dragged_path.name
        
        # Don't move if it's the same location
        if dragged_path == new_path:
            event.accept()
            return
        
        # Move the actual file
        try:
            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(dragged_path), str(new_path))
            
            print(f"✓ Moved file: {dragged_path.name} -> {target_dir}")
            
            # Accept the event
            event.accept()
            
            # Emit signal to refresh the view
            self.file_moved.emit(str(dragged_path), str(new_path))
            
        except Exception as e:
            print(f"✗ Error moving file: {e}")
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
        
        # Build the path
        result = Path(self.base_path)
        for part in parts:
            result = result / part
        
        return result


class PromptsPanel(QtWidgets.QWidget):
    """Panel for managing supplemental prompts, system prompts, and RAG databases."""
    
    # Signals
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
    rag_max_chunks_changed = QtCore.pyqtSignal(int)  # max chunks for auto-build
    rag_summary_chunk_size_changed = QtCore.pyqtSignal(int)  # max raw tokens for summarization
    rag_settings_requested = QtCore.pyqtSignal()  # request to show settings dialog
    prompt_selections_changed = QtCore.pyqtSignal(list, str)  # supplemental_files, system_prompt
    font_size_changed = QtCore.pyqtSignal(int)  # delta
    
    def __init__(self):
        super().__init__()
        
        # Notes tracking for modification detection
        self._notes_last_set_by_llm = False
        self._notes_llm_content = ""
        self._notes_content_hash = None  # Hash of last LLM-generated content
        self._notes_streaming_content = ""  # Accumulates content during streaming
        self._notes_update_pending = False  # Flag for pending markdown update
        
        # Timer for throttled markdown updates during streaming
        self._notes_update_timer = QtCore.QTimer()
        self._notes_update_timer.setSingleShot(True)
        self._notes_update_timer.setInterval(100)  # Update every 100ms
        self._notes_update_timer.timeout.connect(self._apply_notes_markdown_update)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main container
        container = QtWidgets.QGroupBox('Prompts')
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget for switching between prompt types
        self.prompts_tab_widget = QtWidgets.QTabWidget()
        
        # === Notes Tab (first) ===
        notes_tab = self._create_notes_tab()
        
        # === Supplemental Prompts Tab ===
        supp_tab = self._create_supplemental_tab()
        
        # === System Prompts Tab ===
        sys_tab = self._create_system_tab()
        
        # === RAG Tab ===
        rag_tab = self._create_rag_tab()
        
        # Add tabs to the tab widget
        self.prompts_tab_widget.addTab(notes_tab, "Notes")
        self.prompts_tab_widget.addTab(supp_tab, "Supplemental")
        self.prompts_tab_widget.addTab(sys_tab, "System")
        self.prompts_tab_widget.addTab(rag_tab, "RAG")
        
        container_layout.addWidget(self.prompts_tab_widget)
        container.setLayout(container_layout)
        
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)
        self.setLayout(main_layout)
    
    def _create_notes_tab(self):
        """Create the notes tab for LLM context."""
        notes_tab = QtWidgets.QWidget()
        notes_layout = QtWidgets.QVBoxLayout()
        notes_layout.setContentsMargins(5, 5, 5, 5)
        
        self.notes_text = QtWidgets.QTextEdit()
        self.notes_text.setAcceptRichText(True)
        self.notes_text.setMarkdown('')  # Enable markdown support
        self.notes_text.setPlaceholderText('Add notes here for LLM context (e.g., character details, plot points, reminders)...\nAdded to LLM context.\nSupports Markdown formatting.')
        self.notes_text.installEventFilter(self)
        
        # Track user modifications to notes
        self.notes_text.textChanged.connect(self._on_notes_text_changed)
        
        notes_layout.addWidget(self.notes_text)
        
        notes_tab.setLayout(notes_layout)
        return notes_tab
    
    def _create_supplemental_tab(self):
        """Create the supplemental prompts tab."""
        supp_tab = QtWidgets.QWidget()
        supp_layout = QtWidgets.QVBoxLayout()
        supp_layout.setContentsMargins(0, 0, 0, 0)
        
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
        self.supp_list.customContextMenuRequested.connect(self._show_supp_context_menu)
        
        # Connect signals
        self.supp_list.itemChanged.connect(self._on_supp_item_changed)
        self.supp_list.itemDoubleClicked.connect(self._on_supplemental_file_double_clicked)
        
        # Connect file moved signal to refresh the view
        self.supp_list.file_moved.connect(self._on_supp_file_moved)
        
        supp_layout.addWidget(self.supp_list)
        supp_tab.setLayout(supp_layout)
        
        return supp_tab
    
    def _create_system_tab(self):
        """Create the system prompts tab."""
        sys_tab = QtWidgets.QWidget()
        sys_layout = QtWidgets.QVBoxLayout()
        sys_layout.setContentsMargins(0, 0, 0, 0)
        
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
        self.sys_list.customContextMenuRequested.connect(self._show_sys_context_menu)
        
        # Connect signals
        self.sys_list.itemChanged.connect(self._on_sys_item_changed)
        self.sys_list.itemDoubleClicked.connect(self._on_system_file_double_clicked)
        
        # Connect file moved signal to refresh the view
        self.sys_list.file_moved.connect(self._on_sys_file_moved)
        
        sys_layout.addWidget(self.sys_list)
        sys_tab.setLayout(sys_layout)
        
        return sys_tab
    
    def _create_rag_tab(self):
        """Create the RAG databases tab."""
        rag_tab = QtWidgets.QWidget()
        rag_layout = QtWidgets.QVBoxLayout()
        rag_layout.setContentsMargins(0, 0, 0, 0)
        
        # RAG tree widget for databases
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
        
        # Connect signals
        self.rag_list.itemChanged.connect(self._on_rag_item_changed)
        self.rag_list.itemDoubleClicked.connect(self._on_rag_item_double_clicked)
        
        # RAG controls
        rag_controls = QtWidgets.QVBoxLayout()
        self.rag_create_db = QtWidgets.QPushButton('New Database')
        self.rag_create_db.clicked.connect(lambda: self.rag_create_database_clicked.emit())
        rag_controls.addWidget(self.rag_create_db)
        
        self.rag_delete_db = QtWidgets.QPushButton('Delete Selected')
        self.rag_delete_db.setToolTip('Delete the selected database')
        self.rag_delete_db.clicked.connect(self._on_delete_database_clicked)
        rag_controls.addWidget(self.rag_delete_db)
        rag_controls.addStretch(1)
        
        rag_main_layout = QtWidgets.QHBoxLayout()
        rag_main_layout.addWidget(self.rag_list)
        rag_main_layout.addLayout(rag_controls)
        
        rag_layout.addLayout(rag_main_layout)
        rag_tab.setLayout(rag_layout)
        
        return rag_tab
    
    def eventFilter(self, obj, event):
        """Event filter for font resizing with Ctrl+Wheel."""
        if event.type() == QtCore.QEvent.Wheel:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.font_size_changed.emit(1)
                elif delta < 0:
                    self.font_size_changed.emit(-1)
                return True
        return False
    
    # Signal handlers
    
    def _on_supp_file_moved(self, old_path, new_path):
        """Handle when a supplemental file is moved via drag and drop.
        
        This refreshes the view to reflect the new file location.
        """
        # Refresh the supplemental files list
        self.supplemental_refresh_clicked.emit()
    
    def _on_sys_file_moved(self, old_path, new_path):
        """Handle when a system prompt file is moved via drag and drop.
        
        This refreshes the view to reflect the new file location.
        """
        # Refresh the system prompts list
        self.system_refresh_clicked.emit()
    
    def _on_supp_item_changed(self, item, column):
        """Track supplemental item checkbox state changes."""
        self._emit_prompt_selections()
    
    def _on_sys_item_changed(self, item, column):
        """Track system item checkbox state changes and ensure only one is checked."""
        if not item.data(0, QtCore.Qt.UserRole):
            return
        
        if item.checkState(0) == QtCore.Qt.Checked:
            # Uncheck all other items
            def uncheck_all_except(item_to_skip, parent=None):
                if parent is None:
                    count = self.sys_list.topLevelItemCount()
                    for i in range(count):
                        item = self.sys_list.topLevelItem(i)
                        uncheck_all_except(item_to_skip, item)
                else:
                    if parent is not item_to_skip and parent.checkState(0) == QtCore.Qt.Checked:
                        parent.setCheckState(0, QtCore.Qt.Unchecked)
                    
                    child_count = parent.childCount()
                    for i in range(child_count):
                        child = parent.child(i)
                        uncheck_all_except(item_to_skip, child)
            
            uncheck_all_except(item)
        
        self._emit_prompt_selections()
    
    def _on_rag_item_changed(self, item, column):
        """Handle RAG database checkbox state changes."""
        try:
            user_data = item.data(0, QtCore.Qt.UserRole)
            
            if user_data == 'database':
                # Extract db name without file count
                db_name = item.text(0).split(' (')[0]
                print(f"RAG database toggled: {db_name}")
                self.rag_database_toggled.emit(db_name)
        except Exception as e:
            print(f"Error in _on_rag_item_changed: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_supplemental_file_double_clicked(self, item, column=0):
        """Handle double-click on supplemental file."""
        file_path = item.data(0, QtCore.Qt.UserRole)
        if file_path:
            self.supplemental_file_opened.emit(file_path)
    
    def _on_system_file_double_clicked(self, item, column=0):
        """Handle double-click on system prompt file."""
        file_path = item.data(0, QtCore.Qt.UserRole)
        if file_path:
            self.system_file_opened.emit(file_path)
    
    def _on_rag_item_double_clicked(self, item, column=0):
        """Handle double-click on RAG list item."""
        user_data = item.data(0, QtCore.Qt.UserRole)
        if user_data and user_data.startswith('add_files:'):
            # Extract database name
            db_name = user_data.split(':', 1)[1]
            self.rag_add_files_clicked.emit(db_name)
    
    def _on_delete_database_clicked(self):
        """Handle delete database button click."""
        print("=" * 80)
        print("DELETE BUTTON CLICKED!")
        print("=" * 80)
        
        # Collect all checked databases
        checked_databases = []
        print("Searching for checked databases...")
        
        for i in range(self.rag_list.topLevelItemCount()):
            top_item = self.rag_list.topLevelItem(i)
            user_data = top_item.data(0, QtCore.Qt.UserRole)
            
            if user_data == 'database' and top_item.checkState(0) == QtCore.Qt.Checked:
                db_name = top_item.text(0).split(' (')[0]
                checked_databases.append(db_name)
                print(f"  Found checked database: {db_name}")
        
        if not checked_databases:
            print("No checked databases found - showing warning")
            self.show_warning("No Selection", "Please check at least one database to delete.")
            return
        
        # Create confirmation message
        if len(checked_databases) == 1:
            message = f'Are you sure you want to delete the database "{checked_databases[0]}"?\n\nThis will remove all stored files and embeddings.'
            title = 'Confirm Deletion'
        else:
            db_list = '\n  • '.join(checked_databases)
            message = f'Are you sure you want to delete {len(checked_databases)} databases?\n\n  • {db_list}\n\nThis will remove all stored files and embeddings from these databases.'
            title = f'Confirm Deletion of {len(checked_databases)} Databases'
        
        print(f"Asking user to confirm deletion of {len(checked_databases)} database(s)")
        
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            print(f"User confirmed - deleting {len(checked_databases)} database(s)")
            for db_name in checked_databases:
                print(f"  Emitting delete signal for: {db_name}")
                self.rag_delete_database_clicked.emit(db_name)
    
    def _emit_prompt_selections(self):
        """Emit signal with current prompt selections."""
        supplemental_files = self._get_checked_supplemental_paths()
        system_prompt = self._get_checked_system_prompt_path()
        self.prompt_selections_changed.emit(supplemental_files, system_prompt)
    
    def _get_checked_supplemental_paths(self):
        """Get list of checked supplemental file paths."""
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
    
    def _get_checked_system_prompt_path(self):
        """Get the path of the checked system prompt file."""
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
    
    # Public methods
    
    def gather_supplemental_text(self):
        """Read checked supplemental files and concatenate their contents."""
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
                            with open(path, 'r', encoding='utf-8') as f:
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
    
    def get_notes_text(self):
        """Get the current notes text for LLM context."""
        return self.notes_text.toPlainText()
    
    def clear_notes(self):
        """Clear the notes section and reset tracking state."""
        self.notes_text.textChanged.disconnect(self._on_notes_text_changed)
        self.notes_text.clear()
        self.notes_text.textChanged.connect(self._on_notes_text_changed)
        self._notes_last_set_by_llm = False
        self._notes_content_hash = None
        self._notes_llm_content = ""
        self._notes_streaming_content = ""  # Reset streaming accumulator
    
    def set_notes_from_llm(self, text: str):
        """Set notes text from LLM generation, marking it as LLM-generated.
        
        Args:
            text: The LLM-generated notes content
        """
        # Temporarily disconnect signal to avoid triggering user-modification detection
        self.notes_text.textChanged.disconnect(self._on_notes_text_changed)
        
        import hashlib
        
        self._notes_last_set_by_llm = True
        self._notes_llm_content = text
        # Compute hash of the content for modification detection
        self._notes_content_hash = hashlib.md5(text.encode()).hexdigest()
        self.notes_text.setMarkdown(text)  # Use markdown rendering
        
        # Reconnect signal
        self.notes_text.textChanged.connect(self._on_notes_text_changed)
    
    def append_notes(self, text: str):
        """Append text chunk to notes (for streaming) with throttled markdown rendering.
        
        Args:
            text: Text chunk to append
        """
        # Accumulate the streaming content
        self._notes_streaming_content += text
        self._notes_update_pending = True
        
        # Start/restart the timer for throttled updates
        # This ensures we only re-render markdown every 100ms, not on every chunk
        if not self._notes_update_timer.isActive():
            self._notes_update_timer.start()
    
    def _apply_notes_markdown_update(self):
        """Apply pending markdown update to notes (called by timer)."""
        if not self._notes_update_pending:
            return
        
        # Temporarily disconnect signal to avoid triggering user-modification detection
        self.notes_text.textChanged.disconnect(self._on_notes_text_changed)
        
        # Render the accumulated content as markdown
        self.notes_text.setMarkdown(self._notes_streaming_content)
        
        # Move cursor to end to show latest content
        cursor = self.notes_text.textCursor()
        cursor.movePosition(cursor.End)
        self.notes_text.setTextCursor(cursor)
        self.notes_text.ensureCursorVisible()
        
        # Reconnect signal
        self.notes_text.textChanged.connect(self._on_notes_text_changed)
        
        self._notes_update_pending = False
    
    def mark_notes_as_llm_generated(self, text: str):
        """Mark the current notes as LLM-generated after streaming completes.
        
        Args:
            text: The complete LLM-generated notes content
        """
        import hashlib
        
        # Stop the timer and apply any final pending update
        self._notes_update_timer.stop()
        if self._notes_update_pending:
            self._apply_notes_markdown_update()
        
        self._notes_last_set_by_llm = True
        self._notes_llm_content = text
        # Compute hash of the content for modification detection
        self._notes_content_hash = hashlib.md5(text.encode()).hexdigest()
    
    def is_notes_user_modified(self) -> bool:
        """Check if notes have been modified by the user since last LLM generation.
        
        Returns:
            True if user has modified the notes or if notes were never LLM-generated
        """
        if not self._notes_last_set_by_llm:
            # User-created content or blank - consider as user-modified
            return True
        
        current_text = self.notes_text.toPlainText()
        # Check if current text differs from last LLM-generated content
        return current_text != self._notes_llm_content
    
    def _on_notes_text_changed(self):
        """Handle notes text changes to detect user modifications."""
        if self._notes_last_set_by_llm:
            current_text = self.notes_text.toPlainText()
            # If text has changed from LLM content, mark as user-modified
            if current_text != self._notes_llm_content:
                self._notes_last_set_by_llm = False
    
    def should_regenerate_notes(self) -> bool:
        """Check if notes should be regenerated (blank or unmodified LLM content).
        
        Returns:
            True if notes are blank or LLM-generated and unmodified (hash match)
        """
        import hashlib
        
        current_text = self.notes_text.toPlainText()
        
        # Always regenerate if notes are blank
        if not current_text.strip():
            return True
        
        # Check if notes are LLM-generated and haven't been modified by comparing hash
        if self._notes_last_set_by_llm and self._notes_content_hash:
            # Compute hash of current text
            current_hash = hashlib.md5(current_text.encode()).hexdigest()
            # If hash matches, notes are unmodified LLM content - should regenerate
            return current_hash == self._notes_content_hash
        
        # Notes were user-created or modified - preserve them
        return False
    
    def get_system_prompt_text(self):
        """Read the checked system prompt file and return its contents."""
        def find_checked_item(parent):
            if parent is None:
                count = self.sys_list.topLevelItemCount()
                for i in range(count):
                    item = self.sys_list.topLevelItem(i)
                    result = find_checked_item(item)
                    if result:
                        return result
            else:
                if parent.checkState(0) == QtCore.Qt.Checked:
                    path = parent.data(0, QtCore.Qt.UserRole)
                    if path:
                        return path
                
                child_count = parent.childCount()
                for i in range(child_count):
                    child = parent.child(i)
                    result = find_checked_item(child)
                    if result:
                        return result
            return None
        
        checked_path = find_checked_item(None)
        if checked_path:
            try:
                with open(checked_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""
    
    def load_supplemental_files(self, files, selected_files=None):
        """Load supplemental files into tree widget.
        
        Args:
            files: List of (path, is_dir) tuples
            selected_files: Optional list of file paths that should be checked
        """
        # Save the current expanded state before clearing
        expanded_folders = self._save_tree_expanded_state(self.supp_list)
        
        # Temporarily disconnect signal to avoid triggering during setup
        try:
            self.supp_list.itemChanged.disconnect(self._on_supp_item_changed)
        except:
            pass
        
        self.supp_list.clear()
        self._populate_tree(self.supp_list, files, checkable=True, draggable=True)
        
        # Restore selections if provided
        if selected_files:
            self._restore_tree_selections(self.supp_list, selected_files)
        
        # Restore the expanded state
        self._restore_tree_expanded_state(self.supp_list, expanded_folders)
        
        # Reconnect signal
        self.supp_list.itemChanged.connect(self._on_supp_item_changed)
    
    def load_system_prompt_files(self, files, selected_file=None):
        """Load system prompt files into tree widget.
        
        Args:
            files: List of (path, is_dir) tuples
            selected_file: Optional file path that should be checked
        """
        # Save the current expanded state before clearing
        expanded_folders = self._save_tree_expanded_state(self.sys_list)
        
        # Temporarily disconnect signal to avoid triggering during setup
        try:
            self.sys_list.itemChanged.disconnect(self._on_sys_item_changed)
        except:
            pass
        
        self.sys_list.clear()
        self._populate_tree(self.sys_list, files, checkable=True, draggable=True)
        
        # Restore selection if provided
        if selected_file:
            self._restore_tree_selections(self.sys_list, [selected_file])
        
        # Restore the expanded state
        self._restore_tree_expanded_state(self.sys_list, expanded_folders)
        
        # Reconnect signal
        self.sys_list.itemChanged.connect(self._on_sys_item_changed)
    
    def load_rag_databases(self, databases):
        """Load RAG databases into tree widget.
        
        Args:
            databases: List of (db_name, file_count, is_selected) tuples
        """
        try:
            self.rag_list.clear()
            
            # Disconnect signal to avoid triggering during setup
            try:
                self.rag_list.itemChanged.disconnect(self._on_rag_item_changed)
            except:
                pass  # Might not be connected yet
            
            for db_name, file_count, is_selected in databases:
                # Create database item
                db_item = QtWidgets.QTreeWidgetItem([f"{db_name} ({file_count} files)"])
                db_item.setData(0, QtCore.Qt.UserRole, 'database')
                
                # Make it checkable
                db_item.setFlags(db_item.flags() | QtCore.Qt.ItemIsEnabled | 
                               QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
                db_item.setCheckState(0, QtCore.Qt.Checked if is_selected else QtCore.Qt.Unchecked)
                
                # Add "+" button as a child item
                add_button_item = QtWidgets.QTreeWidgetItem(["+ Add Files"])
                add_button_item.setData(0, QtCore.Qt.UserRole, f'add_files:{db_name}')
                add_button_item.setFlags(add_button_item.flags() | QtCore.Qt.ItemIsEnabled | 
                                       QtCore.Qt.ItemIsSelectable)
                
                db_item.addChild(add_button_item)
                self.rag_list.addTopLevelItem(db_item)
                db_item.setExpanded(True)
            
            # Reconnect signal
            self.rag_list.itemChanged.connect(self._on_rag_item_changed)
            
        except Exception as e:
            print(f"Error loading RAG databases: {e}")
            import traceback
            traceback.print_exc()
    
    def show_rag_settings_dialog(self, current_max_docs=3, current_threshold=0.0, current_max_chunks=10, current_summary_chunk_size=1500):
        """Show RAG settings dialog with current values.
        
        Args:
            current_max_docs: Current max documents setting
            current_threshold: Current similarity threshold setting
            current_max_chunks: Current max chunks setting
            current_summary_chunk_size: Current summarization chunk size
        """
        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('RAG Settings')
        dialog.setMinimumWidth(450)
        
        layout = QtWidgets.QVBoxLayout()

        # === Max Chunks Section ===
        max_chunks_group = QtWidgets.QGroupBox('Generated Chunk Amount (Auto-Build)')
        max_chunks_layout = QtWidgets.QVBoxLayout()

        max_chunks_desc = QtWidgets.QLabel(
            'Number of chunks to generate during auto-build RAG mode.\n'
            'Default: 10'
        )
        max_chunks_desc.setWordWrap(True)
        max_chunks_layout.addWidget(max_chunks_desc)

        max_chunks_input_layout = QtWidgets.QHBoxLayout()
        max_chunks_label = QtWidgets.QLabel('Max Chunks:')
        max_chunks_input_layout.addWidget(max_chunks_label)

        max_chunks_spinbox = QtWidgets.QSpinBox()
        max_chunks_spinbox.setMinimum(1)
        max_chunks_spinbox.setMaximum(50)
        max_chunks_spinbox.setValue(current_max_chunks)
        max_chunks_spinbox.setToolTip('Number of chunks to generate in auto-build mode (1-50)')
        max_chunks_input_layout.addWidget(max_chunks_spinbox)
        max_chunks_input_layout.addStretch()

        max_chunks_layout.addLayout(max_chunks_input_layout)
        max_chunks_group.setLayout(max_chunks_layout)
        
        # === Max Documents Section ===
        max_docs_group = QtWidgets.QGroupBox('Maximum Documents per Retrieval')
        max_docs_layout = QtWidgets.QVBoxLayout()
        
        max_docs_desc = QtWidgets.QLabel(
            'Maximum number of document chunks to retrieve from each database.\n'
            'Default: 3'
        )
        max_docs_desc.setWordWrap(True)
        max_docs_layout.addWidget(max_docs_desc)
        
        max_docs_input_layout = QtWidgets.QHBoxLayout()
        max_docs_label = QtWidgets.QLabel('Max Documents:')
        max_docs_input_layout.addWidget(max_docs_label)
        
        max_docs_spinbox = QtWidgets.QSpinBox()
        max_docs_spinbox.setMinimum(1)
        max_docs_spinbox.setMaximum(20)
        max_docs_spinbox.setValue(current_max_docs)
        max_docs_spinbox.setToolTip('Number of chunks to retrieve per database (1-20)')
        max_docs_input_layout.addWidget(max_docs_spinbox)
        max_docs_input_layout.addStretch()
        
        max_docs_layout.addLayout(max_docs_input_layout)
        max_docs_group.setLayout(max_docs_layout)

        # === Summarization Chunk Size Section ===
        summary_group = QtWidgets.QGroupBox('Summarization Chunk Size')
        summary_layout = QtWidgets.QVBoxLayout()

        summary_desc = QtWidgets.QLabel(
            'Maximum raw token chunk size used when summarizing the story.\n'
            'Larger values keep more recent story content before summarizing.\n'
            'Default: 1500'
        )
        summary_desc.setWordWrap(True)
        summary_layout.addWidget(summary_desc)

        summary_input_layout = QtWidgets.QHBoxLayout()
        summary_label = QtWidgets.QLabel('Max Raw Tokens:')
        summary_input_layout.addWidget(summary_label)

        summary_spinbox = QtWidgets.QSpinBox()
        summary_spinbox.setMinimum(256)
        summary_spinbox.setMaximum(200000)
        summary_spinbox.setSingleStep(256)
        summary_spinbox.setValue(current_summary_chunk_size)
        summary_spinbox.setToolTip('Max raw tokens kept before summarizing (256-200000)')
        summary_input_layout.addWidget(summary_spinbox)
        summary_input_layout.addStretch()

        summary_layout.addLayout(summary_input_layout)
        summary_group.setLayout(summary_layout)

        
        # === Similarity Threshold Section ===
        threshold_group = QtWidgets.QGroupBox('Similarity Threshold')
        threshold_layout = QtWidgets.QVBoxLayout()
        
        threshold_desc = QtWidgets.QLabel(
            'Minimum similarity score (0.0 - 1.0) for RAG search results.\n'
            'Lower values return more results but may be less relevant.\n'
            'Higher values return fewer but more relevant results.\n'
            'Default: 0.0 (no filtering)'
        )
        threshold_desc.setWordWrap(True)
        threshold_layout.addWidget(threshold_desc)
        
        # Slider
        slider_layout = QtWidgets.QHBoxLayout()
        slider_label = QtWidgets.QLabel('Threshold:')
        slider_layout.addWidget(slider_label)
        
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(int(current_threshold * 100))
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider_layout.addWidget(slider)
        
        value_label = QtWidgets.QLabel(f'{current_threshold:.2f}')
        value_label.setMinimumWidth(40)
        slider_layout.addWidget(value_label)
        
        threshold_layout.addLayout(slider_layout)
        threshold_group.setLayout(threshold_layout)
        
        # Update label when slider changes
        def update_label(value):
            value_label.setText(f'{value / 100:.2f}')
        
        slider.valueChanged.connect(update_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        
        
        layout.addWidget(max_chunks_group)
        layout.addWidget(summary_group)
        layout.addWidget(max_docs_group)
        layout.addWidget(threshold_group)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            max_docs = max_docs_spinbox.value()
            max_chunks = max_chunks_spinbox.value()
            summary_chunk_size = summary_spinbox.value()
            threshold = slider.value() / 100.0
            print(f"RAG max documents set to: {max_docs}")
            print(f"RAG max chunks set to: {max_chunks}")
            print(f"RAG summary chunk size set to: {summary_chunk_size}")
            print(f"RAG similarity threshold set to: {threshold}")
            self.rag_max_docs_changed.emit(max_docs)
            self.rag_max_chunks_changed.emit(max_chunks)
            self.rag_summary_chunk_size_changed.emit(summary_chunk_size)
            self.rag_similarity_threshold_changed.emit(threshold)
    
    def show_warning(self, title, message):
        """Show warning message box."""
        QtWidgets.QMessageBox.warning(self, title, message)
    
    def apply_font_size(self, size):
        """Apply font size to tree widgets."""
        font = QtGui.QFont()
        font.setPointSize(size)
        
        for widget in [self.supp_list, self.sys_list]:
            try:
                widget.setFont(font)
            except Exception:
                pass
    
    def _populate_tree(self, tree_widget, files, checkable=True, draggable=False):
        """Populate a tree widget with files and directories.
        
        Args:
            tree_widget: The QTreeWidget to populate
            files: List of (path, is_dir) tuples
            checkable: Whether items should be checkable
            draggable: Whether items should be draggable
        """
        # Track parent items by path
        parent_items = {}
        
        for file_path, is_dir in files:
            path = Path(file_path)
            parent_path = path.parent
            
            # Find or create parent item
            if parent_path in parent_items:
                parent_item = parent_items[parent_path]
            else:
                parent_item = None
            
            # Create item
            if is_dir:
                folder_item = QtWidgets.QTreeWidgetItem(parent_item if parent_item else tree_widget)
                folder_item.setText(0, path.name)
                folder_item.setData(0, QtCore.Qt.UserRole, None)
                folder_item.setFlags(folder_item.flags() | QtCore.Qt.ItemIsEnabled)
                font = folder_item.font(0)
                font.setBold(True)
                folder_item.setFont(0, font)
                parent_items[path] = folder_item
            else:
                file_item = QtWidgets.QTreeWidgetItem(parent_item if parent_item else tree_widget)
                file_item.setText(0, path.name)
                flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                if checkable:
                    flags |= QtCore.Qt.ItemIsUserCheckable
                    file_item.setCheckState(0, QtCore.Qt.Unchecked)
                if draggable:
                    flags |= QtCore.Qt.ItemIsDragEnabled
                file_item.setFlags(flags)
                file_item.setData(0, QtCore.Qt.UserRole, str(file_path))
    
    def _restore_tree_selections(self, tree_widget, selected_paths):
        """Restore checkbox states for items in a tree widget.
        
        Args:
            tree_widget: The QTreeWidget to update
            selected_paths: List of file paths that should be checked
        """
        if not selected_paths:
            return
        
        # Convert paths to strings for comparison
        selected_paths_set = {str(p) for p in selected_paths}
        
        def check_item(item):
            """Recursively check items that match selected paths."""
            if item is None:
                # Process top-level items
                for i in range(tree_widget.topLevelItemCount()):
                    check_item(tree_widget.topLevelItem(i))
            else:
                # Check this item if it matches
                item_path = item.data(0, QtCore.Qt.UserRole)
                if item_path and str(item_path) in selected_paths_set:
                    item.setCheckState(0, QtCore.Qt.Checked)
                
                # Process children
                for i in range(item.childCount()):
                    check_item(item.child(i))
        
        check_item(None)

    def _save_tree_expanded_state(self, tree_widget):
        """Save which folders are currently expanded in the tree.
        
        Args:
            tree_widget: The QTreeWidget to save state from
            
        Returns:
            Set of folder paths that are currently expanded
        """
        expanded_folders = set()
        
        def collect_expanded(item):
            """Recursively collect expanded folder paths."""
            if item is None:
                # Process top-level items
                for i in range(tree_widget.topLevelItemCount()):
                    collect_expanded(tree_widget.topLevelItem(i))
            else:
                # If this is a folder and it's expanded, save its path
                if item.childCount() > 0 and item.isExpanded():
                    # Build the folder path from the tree hierarchy
                    path_parts = []
                    current = item
                    while current is not None:
                        path_parts.insert(0, current.text(0))
                        current = current.parent()
                    folder_path = "/".join(path_parts)
                    expanded_folders.add(folder_path)
                
                # Process children
                for i in range(item.childCount()):
                    collect_expanded(item.child(i))
        
        collect_expanded(None)
        return expanded_folders
    
    def _restore_tree_expanded_state(self, tree_widget, expanded_folders):
        """Restore which folders should be expanded in the tree.
        
        Args:
            tree_widget: The QTreeWidget to update
            expanded_folders: Set of folder paths that should be expanded
        """
        if not expanded_folders:
            return
        
        def expand_matching(item, path_parts):
            """Recursively expand items that match saved paths."""
            if item is None:
                # Process top-level items
                for i in range(tree_widget.topLevelItemCount()):
                    expand_matching(tree_widget.topLevelItem(i), [tree_widget.topLevelItem(i).text(0)])
            else:
                # Check if this folder should be expanded
                current_path = "/".join(path_parts)
                if current_path in expanded_folders:
                    item.setExpanded(True)
                
                # Process children
                for i in range(item.childCount()):
                    child = item.child(i)
                    child_path_parts = path_parts + [child.text(0)]
                    expand_matching(child, child_path_parts)
        
        expand_matching(None, [])

    def _get_supp_file_order(self):
        """Get the current order of supplemental files in the tree."""
        ordered_files = []
        
        def collect_files(item):
            """Recursively collect file paths in tree order."""
            if item is None:
                # Process top-level items
                for i in range(self.supp_list.topLevelItemCount()):
                    collect_files(self.supp_list.topLevelItem(i))
            else:
                # If this item is a file, add it
                file_path = item.data(0, QtCore.Qt.UserRole)
                if file_path:
                    ordered_files.append(str(file_path))
                
                # Process children (folders)
                for i in range(item.childCount()):
                    collect_files(item.child(i))
        
        collect_files(None)
        return ordered_files

    def _show_supp_context_menu(self, position):
        """Show context menu for supplemental prompts tree."""
        item = self.supp_list.itemAt(position)
        
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
        """)
        
        if item:
            # Context menu for existing item
            item_data = item.data(0, QtCore.Qt.UserRole)
            if item_data:  # It's a file
                delete_action = menu.addAction("Delete File")
                delete_action.triggered.connect(lambda: self._on_supplemental_delete_clicked(item))
            else:  # It's a folder
                create_file_action = menu.addAction("Create File")
                create_file_action.triggered.connect(lambda: self._on_create_supp_file(item))
                create_folder_action = menu.addAction("Create Folder")
                create_folder_action.triggered.connect(lambda: self._on_create_supp_directory(item))
                menu.addSeparator()
                delete_action = menu.addAction("Delete Folder")
                delete_action.triggered.connect(lambda: self._on_supplemental_delete_clicked(item))
        else:
            # Context menu for empty space (root level)
            create_file_action = menu.addAction("Create File")
            create_file_action.triggered.connect(lambda: self._on_create_supp_file(None))
            create_folder_action = menu.addAction("Create Folder")
            create_folder_action.triggered.connect(lambda: self._on_create_supp_directory(None))
        
        # Always add refresh and add new at the bottom
        menu.addSeparator()
        refresh_action = menu.addAction("Refresh")
        refresh_action.triggered.connect(lambda: self.supplemental_refresh_clicked.emit())
        add_new_action = menu.addAction("Add New")
        add_new_action.triggered.connect(lambda: self.supplemental_add_clicked.emit())
        
        menu.exec_(self.supp_list.viewport().mapToGlobal(position))

    def _show_sys_context_menu(self, position):
        """Show context menu for system prompts tree."""
        item = self.sys_list.itemAt(position)
        
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
        """)
        
        if item:
            # Context menu for existing item
            item_data = item.data(0, QtCore.Qt.UserRole)
            if item_data:  # It's a file
                delete_action = menu.addAction("Delete File")
                delete_action.triggered.connect(lambda: self._on_system_delete_clicked(item))
            else:  # It's a folder
                create_file_action = menu.addAction("Create File")
                create_file_action.triggered.connect(lambda: self._on_create_sys_file(item))
                create_folder_action = menu.addAction("Create Folder")
                create_folder_action.triggered.connect(lambda: self._on_create_sys_directory(item))
                menu.addSeparator()
                delete_action = menu.addAction("Delete Folder")
                delete_action.triggered.connect(lambda: self._on_system_delete_clicked(item))
        else:
            # Context menu for empty space (root level)
            create_file_action = menu.addAction("Create File")
            create_file_action.triggered.connect(lambda: self._on_create_sys_file(None))
            create_folder_action = menu.addAction("Create Folder")
            create_folder_action.triggered.connect(lambda: self._on_create_sys_directory(None))
        
        # Always add refresh and add new at the bottom
        menu.addSeparator()
        refresh_action = menu.addAction("Refresh")
        refresh_action.triggered.connect(lambda: self.system_refresh_clicked.emit())
        add_new_action = menu.addAction("Add New")
        add_new_action.triggered.connect(lambda: self.system_add_clicked.emit())
        
        menu.exec_(self.sys_list.viewport().mapToGlobal(position))

    def _on_create_supp_file(self, parent_item):
        """Create a new supplemental prompt file."""
        # Determine parent directory
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            # Parent is a folder
            parent_path = self._get_item_path(parent_item, self.supp_list)
        else:
            # Root level
            parent_path = Path("supplemental")
        
        # Get filename from user
        filename, ok = QtWidgets.QInputDialog.getText(
            self, "Create File", "Enter filename:"
        )
        
        if ok and filename:
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            file_path = parent_path / filename
            
            # Create the file
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                # Refresh the tree
                self.supplemental_refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create file: {str(e)}"
                )

    def _on_create_supp_directory(self, parent_item):
        """Create a new supplemental prompt directory."""
        # Determine parent directory
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            # Parent is a folder
            parent_path = self._get_item_path(parent_item, self.supp_list)
        else:
            # Root level
            parent_path = Path("supplemental")
        
        # Get directory name from user
        dirname, ok = QtWidgets.QInputDialog.getText(
            self, "Create Folder", "Enter folder name:"
        )
        
        if ok and dirname:
            dir_path = parent_path / dirname
            
            # Create the directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                # Refresh the tree
                self.supplemental_refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create folder: {str(e)}"
                )

    def _on_create_sys_file(self, parent_item):
        """Create a new system prompt file."""
        # Determine parent directory
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            # Parent is a folder
            parent_path = self._get_item_path(parent_item, self.sys_list)
        else:
            # Root level
            parent_path = Path("system_prompts")
        
        # Get filename from user
        filename, ok = QtWidgets.QInputDialog.getText(
            self, "Create File", "Enter filename:"
        )
        
        if ok and filename:
            file_path = parent_path / filename
            
            # Create the file
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                # Refresh the tree
                self.system_refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create file: {str(e)}"
                )

    def _on_create_sys_directory(self, parent_item):
        """Create a new system prompt directory."""
        # Determine parent directory
        if parent_item and not parent_item.data(0, QtCore.Qt.UserRole):
            # Parent is a folder
            parent_path = self._get_item_path(parent_item, self.sys_list)
        else:
            # Root level
            parent_path = Path("system_prompts")
        
        # Get directory name from user
        dirname, ok = QtWidgets.QInputDialog.getText(
            self, "Create Folder", "Enter folder name:"
        )
        
        if ok and dirname:
            dir_path = parent_path / dirname
            
            # Create the directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Folder created: {dir_path}"
                )
                # Refresh the tree
                self.system_refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create folder: {str(e)}"
                )

    def _on_supplemental_delete_clicked(self, item):
        """Delete a supplemental file or folder."""
        if not item:
            return
        
        item_path = item.data(0, QtCore.Qt.UserRole)
        if item_path:
            # It's a file
            path = Path(item_path)
            item_type = "file"
        else:
            # It's a folder
            path = self._get_item_path(item, self.supp_list)
            item_type = "folder"
        
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete this {item_type}?\n{path}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                if item_type == "file":
                    path.unlink()
                else:
                    import shutil
                    shutil.rmtree(path)
                
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Deleted: {path}"
                )
                # Refresh the tree
                self.supplemental_refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to delete: {str(e)}"
                )

    def _on_system_delete_clicked(self, item):
        """Delete a system prompt file or folder."""
        if not item:
            return
        
        item_path = item.data(0, QtCore.Qt.UserRole)
        if item_path:
            # It's a file
            path = Path(item_path)
            item_type = "file"
        else:
            # It's a folder
            path = self._get_item_path(item, self.sys_list)
            item_type = "folder"
        
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete this {item_type}?\n{path}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                if item_type == "file":
                    path.unlink()
                else:
                    import shutil
                    shutil.rmtree(path)
                
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Deleted: {path}"
                )
                # Refresh the tree
                self.system_refresh_clicked.emit()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to delete: {str(e)}"
                )

    def _get_item_path(self, item, tree_widget):
        """Get the full path of a tree item by traversing up the tree."""
        parts = []
        current = item
        
        while current is not None:
            parts.insert(0, current.text(0))
            current = current.parent()
        
        # Determine the root path
        if tree_widget == self.supp_list:
            root = Path("supplemental")
        else:
            root = Path("system_prompts")
        
        # Combine parts
        for part in parts:
            root = root / part
        
        return root

