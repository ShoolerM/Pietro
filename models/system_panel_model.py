"""Model for system prompts panel state."""


class SystemPanelModel:
    """Stores selected system prompt."""

    def __init__(self):
        self.selected_file = None

    def set_selected_file(self, path):
        self.selected_file = path
