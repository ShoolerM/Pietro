"""Model for managing application settings."""
import os
import json
from pathlib import Path
from base.observable import Observable


class SettingsModel(Observable):
    """Manages application settings and configuration."""
    
    # Default summary prompt template
    DEFAULT_SUMMARY_PROMPT = (
        "TASK: Create a detailed but condensed summary of this story.\n\n"
        "REQUIREMENTS:\n"
        "- List ALL main characters with their names, roles, and key personality traits\n"
        "- Describe character relationships and dynamics between them\n"
        "- Include ALL significant plot points in chronological order\n"
        "- Preserve important dialogue or quotes that define characters\n"
        "- Document world-building: locations, events, themes, etc.\n"
        "- Note any ongoing conflicts, or unresolved plot threads\n"
        "- Mention character motivations and goals\n"
        "- Include relevant backstory and historical context\n"
        "- Keep the summary detailed but aim for 30-40% of original length\n"
        "- Write in present tense, organized by topic (characters, plot, setting, etc.)\n\n"
    )
    
    def __init__(self):
        super().__init__()
        self._inference_ip = "192.168.0.1"
        self._inference_port = 1234
        self._base_url = f"http://{self._inference_ip}:{self._inference_port}/v1"
        self._temperature = 0.7
        self._context_limit = 4096
        self._base_font_size = 10
        self._current_font_size = 10
        self._markdown_enabled = True
        self._summarize_prompts = True
        self._build_with_rag = False
        self._summary_prompt_template = self.DEFAULT_SUMMARY_PROMPT
        
        # Prompt selections
        self._selected_supplemental_files = []
        self._selected_system_prompt = None
        self._supplemental_file_order = []  # Store the order of files
        
        self._load_inference_settings()
        self._load_summary_prompt()
        self._load_prompt_selections()
        self._load_build_with_rag()
    
    @property
    def base_url(self):
        """Get LLM API base URL."""
        return self._base_url
    
    @base_url.setter
    def base_url(self, value):
        """Set LLM API base URL."""
        self._base_url = value
        self.notify_observers('base_url_changed', value)
    
    @property
    def inference_ip(self):
        """Get inference server IP address."""
        return self._inference_ip
    
    @inference_ip.setter
    def inference_ip(self, value):
        """Set inference server IP address."""
        self._inference_ip = value
        self._update_base_url()
    
    @property
    def inference_port(self):
        """Get inference server port."""
        return self._inference_port
    
    @inference_port.setter
    def inference_port(self, value):
        """Set inference server port."""
        self._inference_port = value
        self._update_base_url()
    
    def _update_base_url(self):
        """Update base URL when IP or port changes."""
        self._base_url = f"http://{self._inference_ip}:{self._inference_port}/v1"
        self.notify_observers('base_url_changed', self._base_url)
    
    @property
    def temperature(self):
        """Get LLM temperature setting."""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        """Set LLM temperature setting."""
        self._temperature = value
        self.notify_observers('temperature_changed', value)
    
    @property
    def context_limit(self):
        """Get context limit in tokens."""
        return self._context_limit
    
    @context_limit.setter
    def context_limit(self, value):
        """Set context limit in tokens."""
        self._context_limit = value
        self.notify_observers('context_limit_changed', value)
    
    @property
    def base_font_size(self):
        """Get base font size."""
        return self._base_font_size
    
    @property
    def current_font_size(self):
        """Get current font size."""
        return self._current_font_size
    
    @current_font_size.setter
    def current_font_size(self, value):
        """Set current font size."""
        # Clamp to reasonable bounds
        self._current_font_size = max(6, min(32, value))
        self.notify_observers('font_size_changed', self._current_font_size)
    
    @property
    def markdown_enabled(self):
        """Check if markdown rendering is enabled."""
        return self._markdown_enabled
    
    @markdown_enabled.setter
    def markdown_enabled(self, value):
        """Set markdown rendering enabled state."""
        self._markdown_enabled = value
        self.notify_observers('markdown_enabled_changed', value)

    @property
    def summarize_prompts(self):
        """Check if prompt summarization (condensing supplemental/system/notes) is enabled."""
        return self._summarize_prompts

    @summarize_prompts.setter
    def summarize_prompts(self, value):
        """Enable or disable summarization of prompts."""
        self._summarize_prompts = bool(value)
        self.notify_observers('summarize_prompts_changed', self._summarize_prompts)
    
    @property
    def build_with_rag(self):
        """Check if auto-build story with RAG mode is enabled."""
        return self._build_with_rag

    @build_with_rag.setter
    def build_with_rag(self, value):
        """Enable or disable auto-build story with RAG mode."""
        self._build_with_rag = bool(value)
        self.notify_observers('build_with_rag_changed', self._build_with_rag)
        self._save_build_with_rag()
    
    @property
    def summary_prompt_template(self):
        """Get summary prompt template."""
        return self._summary_prompt_template
    
    @summary_prompt_template.setter
    def summary_prompt_template(self, value):
        """Set summary prompt template."""
        self._summary_prompt_template = value
        self.notify_observers('summary_prompt_changed', value)
    
    def _load_summary_prompt(self):
        """Load the summary prompt from settings/summary_prompt.txt or use default."""
        settings_dir = Path('settings')
        prompt_file = settings_dir / 'summary_prompt.txt'
        
        try:
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self._summary_prompt_template = f.read()
                print(f"✓ Loaded custom summary prompt from {prompt_file} ({len(self._summary_prompt_template)} chars)")
            else:
                print(f"Using default summary prompt ({len(self._summary_prompt_template)} chars)")
        except Exception as e:
            print(f"⚠ Error loading summary prompt: {e}. Using default.")
    
    def _load_inference_settings(self):
        """Load inference server settings from file."""
        settings_dir = Path('settings')
        settings_file = settings_dir / 'inference_settings.json'
        
        try:
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._inference_ip = data.get('ip', '192.168.0.1')
                self._inference_port = data.get('port', 1234)
                self._update_base_url()
                
                print(f"✓ Loaded inference settings: {self._inference_ip}:{self._inference_port}")
        except Exception as e:
            print(f"⚠ Error loading inference settings: {e}. Using defaults.")
    
    def save_inference_settings(self):
        """Save inference server settings to file."""
        settings_dir = Path('settings')
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / 'inference_settings.json'
        
        try:
            data = {
                'ip': self._inference_ip,
                'port': self._inference_port
            }
            
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved inference settings: {self._inference_ip}:{self._inference_port}")
            return True
        except Exception as e:
            print(f"⚠ Error saving inference settings: {e}")
            return False
    
    def save_summary_prompt(self, prompt_text: str):
        """Save the summary prompt to settings/summary_prompt.txt.
        
        Args:
            prompt_text: The prompt text to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        settings_dir = Path('settings')
        settings_dir.mkdir(exist_ok=True)
        prompt_file = settings_dir / 'summary_prompt.txt'
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            print(f"✓ Saved summary prompt to {prompt_file} ({len(prompt_text)} chars)")
            self._summary_prompt_template = prompt_text
            self.notify_observers('summary_prompt_saved', prompt_text)
            return True
        except Exception as e:
            print(f"⚠ Error saving summary prompt: {e}")
            return False

    def _load_prompt_selections(self):
        """Load saved prompt selections from file."""
        settings_dir = Path('settings')
        selections_file = settings_dir / 'prompt_selections.json'
        
        try:
            if selections_file.exists():
                with open(selections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._selected_supplemental_files = data.get('supplemental_files', [])
                self._selected_system_prompt = data.get('system_prompt', None)
                
                if self._selected_supplemental_files or self._selected_system_prompt:
                    print(f"✓ Loaded prompt selections: {len(self._selected_supplemental_files)} supplemental, system: {bool(self._selected_system_prompt)}")
        except Exception as e:
            print(f"⚠ Error loading prompt selections: {e}")
            self._selected_supplemental_files = []
            self._selected_system_prompt = None
    
    def save_prompt_selections(self, supplemental_files, system_prompt):
        """Save current prompt selections to persist across restarts.
        
        Args:
            supplemental_files: List of file paths for checked supplemental prompts
            system_prompt: File path of the selected system prompt (or None)
        """
        settings_dir = Path('settings')
        settings_dir.mkdir(exist_ok=True)
        selections_file = settings_dir / 'prompt_selections.json'
        
        try:
            self._selected_supplemental_files = supplemental_files
            self._selected_system_prompt = system_prompt
            
            data = {
                'supplemental_files': supplemental_files,
                'system_prompt': system_prompt
            }
            
            with open(selections_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved prompt selections: {len(supplemental_files)} supplemental, system: {bool(system_prompt)}")
        except Exception as e:
            print(f"⚠ Error saving prompt selections: {e}")
    
    def get_selected_supplemental_files(self):
        """Get list of selected supplemental file paths."""
        return self._selected_supplemental_files.copy()
    
    def get_selected_system_prompt(self):
        """Get path of selected system prompt file."""
        return self._selected_system_prompt

    def save_supplemental_file_order(self, ordered_files):
        """Save the order of supplemental files for drag and drop persistence.
        
        Args:
            ordered_files: List of file paths in their display order
        """
        settings_dir = Path('settings')
        settings_dir.mkdir(exist_ok=True)
        order_file = settings_dir / 'supplemental_file_order.json'
        
        try:
            self._supplemental_file_order = ordered_files
            
            data = {
                'file_order': ordered_files
            }
            
            with open(order_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved supplemental file order: {len(ordered_files)} files")
        except Exception as e:
            print(f"⚠ Error saving supplemental file order: {e}")
    
    def get_supplemental_file_order(self):
        """Get the saved order of supplemental files.
        
        Returns:
            List of file paths in their saved order, or empty list if none saved
        """
        if not self._supplemental_file_order:
            # Try to load from file if not in memory
            settings_dir = Path('settings')
            order_file = settings_dir / 'supplemental_file_order.json'
            
            if order_file.exists():
                try:
                    with open(order_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._supplemental_file_order = data.get('file_order', [])
                except Exception as e:
                    print(f"⚠ Error loading supplemental file order: {e}")
                    self._supplemental_file_order = []
        
        return self._supplemental_file_order.copy()

    def _load_build_with_rag(self):
        """Load build_with_rag setting from file."""
        settings_dir = Path('settings')
        settings_file = settings_dir / 'build_with_rag.json'
        
        try:
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._build_with_rag = data.get('enabled', False)
                print(f"✓ Loaded build_with_rag setting: {self._build_with_rag}")
        except Exception as e:
            print(f"⚠ Error loading build_with_rag setting: {e}. Using default.")
            self._build_with_rag = False
    
    def _save_build_with_rag(self):
        """Save build_with_rag setting to file."""
        settings_dir = Path('settings')
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / 'build_with_rag.json'
        
        try:
            data = {'enabled': self._build_with_rag}
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved build_with_rag setting: {self._build_with_rag}")
        except Exception as e:
            print(f"⚠ Error saving build_with_rag setting: {e}")

