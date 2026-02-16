"""Model for managing application settings."""

import json
from pathlib import Path
from base.observable import Observable
from models import base_prompts


class SettingsModel(Observable):
    """Manages application settings and configuration."""

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
        self._smart_mode = False
        self._auto_notes = True  # Auto-generate notes by default
        self._render_markdown = True  # Render story as markdown by default
        self._summary_prompt_template = base_prompts.DEFAULT_SUMMARY_PROMPT
        self._notes_prompt_template = base_prompts.DEFAULT_NOTES_PROMPT
        self._planning_prompt_template = "What story would you like to plan?"

        # Per-model profile settings
        self._model_profiles = {}

        # Last selected model
        self._last_model = None

        # Prompt selections
        self._selected_supplemental_files = []
        self._selected_system_prompt = None
        self._supplemental_file_order = []  # Store the order of files

        self._load_inference_settings()
        self._load_summary_prompt()
        self._load_notes_prompt()
        self._load_prompt_selections()
        self._load_smart_mode()
        self._load_render_markdown()
        self._load_model_profiles()
        self._load_last_model()
        self._load_planning_conversation()
        self._load_normal_conversation()

    @property
    def last_model(self):
        """Get last selected model name."""
        return self._last_model

    @last_model.setter
    def last_model(self, value):
        """Set last selected model name."""
        self._last_model = value
        self._save_last_model()

    def _model_profile_key(self, model_name, base_url):
        """Build a stable key for model profiles."""
        return f"{model_name}||{base_url}"

    def _load_model_profiles(self):
        """Load per-model settings profiles from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "model_profiles.json"

        try:
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._model_profiles = data.get("profiles", {})
                print(f"✓ Loaded {len(self._model_profiles)} model profile(s)")
        except Exception as e:
            print(f"⚠ Error loading model profiles: {e}. Using defaults.")

    def _save_model_profiles(self):
        """Persist per-model settings profiles to file."""
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "model_profiles.json"

        try:
            data = {"profiles": self._model_profiles}
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving model profiles: {e}")

    def _load_last_model(self):
        """Load last selected model from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "last_model.json"

        try:
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._last_model = data.get("model")
                if self._last_model:
                    print(f"✓ Loaded last model: {self._last_model}")
        except Exception as e:
            print(f"⚠ Error loading last model: {e}")

    def _save_last_model(self):
        """Persist last selected model to file."""
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "last_model.json"

        try:
            data = {"model": self._last_model}
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving last model: {e}")

    def save_model_profile(
        self, model_name, base_url, context_limit, inference_ip, inference_port
    ):
        """Save the current settings as a profile for a model/base_url."""
        if not model_name or not base_url:
            return

        key = self._model_profile_key(model_name, base_url)
        self._model_profiles[key] = {
            "model": model_name,
            "base_url": base_url,
            "inference_ip": inference_ip,
            "inference_port": inference_port,
            "context_limit": context_limit,
        }
        self._save_model_profiles()

    def get_model_profile(self, model_name, base_url):
        """Get a saved model profile, preferring exact base_url match."""
        if not model_name:
            return None

        # Exact match on model + base_url
        if base_url:
            key = self._model_profile_key(model_name, base_url)
            if key in self._model_profiles:
                return self._model_profiles[key]

        # Fallback: match by model name only (any base_url)
        for profile in self._model_profiles.values():
            if profile.get("model") == model_name:
                return profile

        return None

    @property
    def base_url(self):
        """Get LLM API base URL."""
        return self._base_url

    @base_url.setter
    def base_url(self, value):
        """Set LLM API base URL."""
        self._base_url = value
        self.notify_observers("base_url_changed", value)

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
        self.notify_observers("base_url_changed", self._base_url)

    @property
    def temperature(self):
        """Get LLM temperature setting."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """Set LLM temperature setting."""
        self._temperature = value
        self.notify_observers("temperature_changed", value)

    @property
    def context_limit(self):
        """Get context limit in tokens."""
        return self._context_limit

    @context_limit.setter
    def context_limit(self, value):
        """Set context limit in tokens."""
        self._context_limit = value
        self.notify_observers("context_limit_changed", value)

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
        self.notify_observers("font_size_changed", self._current_font_size)

    @property
    def markdown_enabled(self):
        """Check if markdown rendering is enabled."""
        return self._markdown_enabled

    @markdown_enabled.setter
    def markdown_enabled(self, value):
        """Set markdown rendering enabled state."""
        self._markdown_enabled = value
        self.notify_observers("markdown_enabled_changed", value)

    @property
    def summarize_prompts(self):
        """Check if prompt summarization (condensing supplemental/system/notes) is enabled."""
        return self._summarize_prompts

    @summarize_prompts.setter
    def summarize_prompts(self, value):
        """Enable or disable summarization of prompts."""
        self._summarize_prompts = bool(value)
        self.notify_observers("summarize_prompts_changed", self._summarize_prompts)

    @property
    def smart_mode(self):
        """Check if auto-build story with Smart Mode is enabled."""
        return self._smart_mode

    @smart_mode.setter
    def smart_mode(self, value):
        """Enable or disable auto-build story with Smart Mode."""
        self._smart_mode = bool(value)
        self.notify_observers("smart_mode_changed", self._smart_mode)
        self._save_smart_mode()

    @property
    def auto_notes(self):
        """Check if auto-generate notes is enabled."""
        return self._auto_notes

    @auto_notes.setter
    def auto_notes(self, value):
        """Enable or disable auto-generate notes."""
        self._auto_notes = bool(value)
        self.notify_observers("auto_notes_changed", self._auto_notes)

    @property
    def render_markdown(self):
        """Get render markdown enabled status."""
        return self._render_markdown

    @render_markdown.setter
    def render_markdown(self, value):
        """Set render markdown enabled status."""
        self._render_markdown = bool(value)
        self._save_render_markdown()
        self.notify_observers("render_markdown_changed", self._render_markdown)

    @property
    def summary_prompt_template(self):
        """Get summary prompt template."""
        return self._summary_prompt_template

    @summary_prompt_template.setter
    def summary_prompt_template(self, value):
        """Set summary prompt template."""
        self._summary_prompt_template = value
        self.notify_observers("summary_prompt_changed", value)

    @property
    def notes_prompt_template(self):
        """Get notes prompt template."""
        return self._notes_prompt_template

    @notes_prompt_template.setter
    def notes_prompt_template(self, value):
        """Set notes prompt template."""
        self._notes_prompt_template = value
        self.notify_observers("notes_prompt_changed", value)

    @property
    def planning_prompt_template(self):
        """Get planning prompt template."""
        return self._planning_prompt_template

    @planning_prompt_template.setter
    def planning_prompt_template(self, value):
        """Set planning prompt template."""
        self._planning_prompt_template = value
        self.notify_observers("planning_prompt_changed", value)

    def _load_summary_prompt(self):
        """Load the summary prompt from settings/summary_prompt.txt or use default."""
        settings_dir = Path("settings")
        prompt_file = settings_dir / "summary_prompt.txt"

        try:
            if prompt_file.exists():
                with open(prompt_file, "r", encoding="utf-8") as f:
                    self._summary_prompt_template = f.read()
                print(
                    f"✓ Loaded custom summary prompt from {prompt_file} ({len(self._summary_prompt_template)} chars)"
                )
            else:
                print(
                    f"Using default summary prompt ({len(self._summary_prompt_template)} chars)"
                )
        except Exception as e:
            print(f"⚠ Error loading summary prompt: {e}. Using default.")

    def _load_notes_prompt(self):
        """Load the notes prompt from settings/notes_prompt.txt or use default."""
        settings_dir = Path("settings")
        prompt_file = settings_dir / "notes_prompt.txt"

        try:
            if prompt_file.exists():
                with open(prompt_file, "r", encoding="utf-8") as f:
                    self._notes_prompt_template = f.read()
                print(
                    f"✓ Loaded custom notes prompt from {prompt_file} ({len(self._notes_prompt_template)} chars)"
                )
            else:
                print(
                    f"Using default notes prompt ({len(self._notes_prompt_template)} chars)"
                )
        except Exception as e:
            print(f"⚠ Error loading notes prompt: {e}. Using default.")

    def _load_inference_settings(self):
        """Load inference server settings from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "inference_settings.json"

        try:
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._inference_ip = data.get("ip", "192.168.0.1")
                self._inference_port = data.get("port", 1234)
                self._update_base_url()

                print(
                    f"✓ Loaded inference settings: {self._inference_ip}:{self._inference_port}"
                )
        except Exception as e:
            print(f"⚠ Error loading inference settings: {e}. Using defaults.")

    def save_inference_settings(self):
        """Save inference server settings to file."""
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "inference_settings.json"

        try:
            data = {"ip": self._inference_ip, "port": self._inference_port}

            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(
                f"✓ Saved inference settings: {self._inference_ip}:{self._inference_port}"
            )
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
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        prompt_file = settings_dir / "summary_prompt.txt"

        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            print(f"✓ Saved summary prompt to {prompt_file} ({len(prompt_text)} chars)")
            self._summary_prompt_template = prompt_text
            self.notify_observers("summary_prompt_saved", prompt_text)
            return True
        except Exception as e:
            print(f"⚠ Error saving summary prompt: {e}")
            return False

    def save_notes_prompt(self, prompt_text: str):
        """Save the notes prompt to settings/notes_prompt.txt.

        Args:
            prompt_text: The prompt text to save

        Returns:
            bool: True if save was successful, False otherwise
        """
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        prompt_file = settings_dir / "notes_prompt.txt"

        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            print(f"✓ Saved notes prompt to {prompt_file} ({len(prompt_text)} chars)")
            self._notes_prompt_template = prompt_text
            self.notify_observers("notes_prompt_saved", prompt_text)
            return True
        except Exception as e:
            print(f"⚠ Error saving notes prompt: {e}")
            return False

    def _load_prompt_selections(self):
        """Load saved prompt selections from file."""
        settings_dir = Path("settings")
        selections_file = settings_dir / "prompt_selections.json"

        try:
            if selections_file.exists():
                with open(selections_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._selected_supplemental_files = data.get("supplemental_files", [])
                self._selected_system_prompt = data.get("system_prompt", None)

                if self._selected_supplemental_files or self._selected_system_prompt:
                    print(
                        f"✓ Loaded prompt selections: {len(self._selected_supplemental_files)} supplemental, system: {bool(self._selected_system_prompt)}"
                    )
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
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        selections_file = settings_dir / "prompt_selections.json"

        try:
            self._selected_supplemental_files = supplemental_files
            self._selected_system_prompt = system_prompt

            data = {
                "supplemental_files": supplemental_files,
                "system_prompt": system_prompt,
            }

            with open(selections_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(
                f"✓ Saved prompt selections: {len(supplemental_files)} supplemental, system: {bool(system_prompt)}"
            )
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
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        order_file = settings_dir / "supplemental_file_order.json"

        try:
            self._supplemental_file_order = ordered_files

            data = {"file_order": ordered_files}

            with open(order_file, "w", encoding="utf-8") as f:
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
            settings_dir = Path("settings")
            order_file = settings_dir / "supplemental_file_order.json"

            if order_file.exists():
                try:
                    with open(order_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._supplemental_file_order = data.get("file_order", [])
                except Exception as e:
                    print(f"⚠ Error loading supplemental file order: {e}")
                    self._supplemental_file_order = []

        return self._supplemental_file_order.copy()

    def _load_smart_mode(self):
        """Load smart_mode setting from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "smart_mode.json"

        try:
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._smart_mode = data.get("enabled", False)
                print(f"✓ Loaded smart_mode setting: {self._smart_mode}")
        except Exception as e:
            print(f"⚠ Error loading smart_mode setting: {e}. Using default.")
            self._smart_mode = False

    def _load_render_markdown(self):
        """Load render markdown setting from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "inference_settings.json"

        try:
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._render_markdown = data.get("render_markdown", True)
                print(f"✓ Loaded render_markdown setting: {self._render_markdown}")
        except Exception as e:
            print(
                f"⚠ Error loading render_markdown setting: {e}. Using default (True)."
            )
            self._render_markdown = True

    def _save_render_markdown(self):
        """Save render markdown setting to file."""
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "inference_settings.json"

        try:
            # Load existing settings first
            data = {}
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # Update render_markdown
            data["render_markdown"] = self._render_markdown

            # Save back
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving render_markdown setting: {e}")

    def _save_smart_mode(self):
        """Save smart_mode setting to file."""
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "smart_mode.json"

        try:
            data = {"enabled": self._smart_mode}
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved smart_mode setting: {self._smart_mode}")
        except Exception as e:
            print(f"⚠ Error saving smart_mode setting: {e}")

    def _load_planning_conversation(self):
        """Load saved planning conversation from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "planning_conversation.json"
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._planning_conversation = data.get("conversation", [])
                    self._planning_outline = data.get("outline", "")
                    print(
                        f"✓ Loaded planning conversation ({len(self._planning_conversation)} messages)"
                    )
            except Exception as e:
                print(f"⚠ Error loading planning conversation: {e}")
                self._planning_conversation = []
                self._planning_outline = ""
        else:
            self._planning_conversation = []
            self._planning_outline = ""

    def _load_normal_conversation(self):
        """Load saved normal conversation from file."""
        settings_dir = Path("settings")
        settings_file = settings_dir / "normal_conversation.json"
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._normal_conversation = data.get("conversation", [])
                    print(
                        f"✓ Loaded normal conversation ({len(self._normal_conversation)} messages)"
                    )
            except Exception as e:
                print(f"⚠ Error loading normal conversation: {e}")
                self._normal_conversation = []
        else:
            self._normal_conversation = []

    def save_planning_conversation(self, conversation, outline=""):
        """Save planning conversation to file.

        Args:
            conversation: List of {"role": str, "content": str}
            outline: Current outline text
        """
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "planning_conversation.json"
        try:
            data = {"conversation": conversation, "outline": outline}
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving planning conversation: {e}")

    def get_planning_conversation(self):
        """Get saved planning conversation."""
        return getattr(self, "_planning_conversation", [])

    def get_planning_outline(self):
        """Get saved planning outline."""
        return getattr(self, "_planning_outline", "")

    def save_normal_conversation(self, conversation):
        """Save normal conversation to file.

        Args:
            conversation: List of {"role": str, "content": str}
        """
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "normal_conversation.json"
        try:
            data = {"conversation": conversation}
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving normal conversation: {e}")

    def get_normal_conversation(self):
        """Get saved normal conversation."""
        return getattr(self, "_normal_conversation", [])

    def append_normal_message(self, role, content):
        """Append a message to the normal conversation and persist it."""
        if not hasattr(self, "_normal_conversation"):
            self._normal_conversation = []
        self._normal_conversation.append({"role": role, "content": content})
        self.save_normal_conversation(self._normal_conversation)
