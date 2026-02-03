"""Model for managing supplemental and system prompts."""
import os
import pathlib
from base.observable import Observable


class PromptModel(Observable):
    """Manages supplemental and system prompt files."""
    
    def __init__(self):
        super().__init__()
        self._supplemental_dir = pathlib.Path(os.path.join(os.getcwd(), 'supplemental'))
        self._system_prompts_dir = pathlib.Path(os.path.join(os.getcwd(), 'system_prompts'))
    
    def get_supplemental_files(self):
        """Get all supplemental files recursively.
        
        Returns:
            list: List of (path, is_dir) tuples sorted by directories first, then files
        """
        if not self._supplemental_dir.exists() or not self._supplemental_dir.is_dir():
            return []
        
        def scan_directory(directory):
            """Recursively scan directory for files and subdirectories."""
            result = []
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            
            for entry in entries:
                result.append((entry, entry.is_dir()))
                if entry.is_dir():
                    result.extend(scan_directory(entry))
            
            return result
        
        return scan_directory(self._supplemental_dir)
    
    def get_system_prompt_files(self):
        """Get all system prompt files recursively.
        
        Returns:
            list: List of (path, is_dir) tuples sorted by directories first, then files
        """
        if not self._system_prompts_dir.exists() or not self._system_prompts_dir.is_dir():
            return []
        
        def scan_directory(directory):
            """Recursively scan directory for files and subdirectories."""
            result = []
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            
            for entry in entries:
                result.append((entry, entry.is_dir()))
                if entry.is_dir():
                    result.extend(scan_directory(entry))
            
            return result
        
        return scan_directory(self._system_prompts_dir)
    
    def read_file(self, file_path):
        """Read content from a prompt file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            str: File content, or empty string on error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def write_file(self, file_path, content):
        """Write content to a prompt file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")
            return False
    
    def create_supplemental_file(self, filename):
        """Create a new supplemental prompt file.
        
        Args:
            filename: Name of the file to create
            
        Returns:
            tuple: (success: bool, file_path or error_message)
        """
        try:
            self._supplemental_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._supplemental_dir / filename
            
            if file_path.exists():
                return False, f"File '{filename}' already exists."
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('')
            
            self.notify_observers('supplemental_file_created', str(file_path))
            return True, str(file_path)
        except Exception as e:
            return False, str(e)
    
    def create_system_prompt_file(self, filename):
        """Create a new system prompt file.
        
        Args:
            filename: Name of the file to create
            
        Returns:
            tuple: (success: bool, file_path or error_message)
        """
        try:
            self._system_prompts_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._system_prompts_dir / filename
            
            if file_path.exists():
                return False, f"File '{filename}' already exists."
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('')
            
            self.notify_observers('system_prompt_file_created', str(file_path))
            return True, str(file_path)
        except Exception as e:
            return False, str(e)
