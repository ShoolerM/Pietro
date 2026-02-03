"""Controller for managing prompt files and operations."""
from pathlib import Path


class PromptController:
    """Handles operations related to supplemental and system prompts."""
    
    def __init__(self, prompt_model, view, settings_model):
        """Initialize the prompt controller.
        
        Args:
            prompt_model: PromptModel instance
            view: MainView instance
            settings_model: SettingsModel instance for saved selections
        """
        self.model = prompt_model
        self.view = view
        self.settings_model = settings_model
        
        # Connect view signals
        self.view.supplemental_refresh_clicked.connect(self.refresh_supplemental)
        self.view.supplemental_add_clicked.connect(self.add_supplemental)
        self.view.supplemental_file_opened.connect(self.open_supplemental_file)
        self.view.system_refresh_clicked.connect(self.refresh_system_prompts)
        self.view.system_add_clicked.connect(self.add_system_prompt)
        self.view.system_file_opened.connect(self.open_system_file)
        self.view.file_saved.connect(self.save_file)
        
        # Initial load with saved selections
        self.refresh_supplemental()
        self.refresh_system_prompts()
    
    def refresh_supplemental(self):
        """Refresh the supplemental prompts list."""
        try:
            files = self.model.get_supplemental_files()
            selected_files = self.settings_model.get_selected_supplemental_files()
            self.view.load_supplemental_files(files, selected_files)
        except Exception as e:
            print(f"Error refreshing supplemental prompts: {e}")
    
    def refresh_system_prompts(self):
        """Refresh the system prompts list."""
        try:
            files = self.model.get_system_prompt_files()
            selected_file = self.settings_model.get_selected_system_prompt()
            self.view.load_system_prompt_files(files, selected_file)
        except Exception as e:
            print(f"Error refreshing system prompts: {e}")
    
    def add_supplemental(self):
        """Add a new supplemental prompt file."""
        filename, ok = self.view.show_input_dialog(
            'New Supplemental Prompt',
            'Enter filename (e.g., my_prompt.txt):'
        )
        
        if not ok or not filename.strip():
            return
        
        success, result = self.model.create_supplemental_file(filename.strip())
        
        if not success:
            self.view.show_warning("Error", f"Could not create file:\n{result}")
            return
        
        # Refresh list and open the new file
        self.refresh_supplemental()
        self.view.open_file_tab(result)
    
    def add_system_prompt(self):
        """Add a new system prompt file."""
        filename, ok = self.view.show_input_dialog(
            'New System Prompt',
            'Enter filename (e.g., my_system_prompt.txt):'
        )
        
        if not ok or not filename.strip():
            return
        
        success, result = self.model.create_system_prompt_file(filename.strip())
        
        if not success:
            self.view.show_warning("Error", f"Could not create file:\n{result}")
            return
        
        # Refresh list and open the new file
        self.refresh_system_prompts()
        self.view.open_file_tab(result)
    
    def open_supplemental_file(self, file_path):
        """Open a supplemental file for editing."""
        self.view.open_file_tab(file_path)
    
    def open_system_file(self, file_path):
        """Open a system prompt file for editing."""
        self.view.open_file_tab(file_path)
    
    def save_file(self, file_path, content):
        """Save a file's content.
        
        Args:
            file_path: Path to the file
            content: Content to write
        """
        success = self.model.write_file(file_path, content)
        if not success:
            self.view.show_warning("Error", f"Could not save file:\n{file_path}")
