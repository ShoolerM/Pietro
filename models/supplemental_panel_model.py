"""Model for supplemental prompts panel state."""


class SupplementalPanelModel:
    """Stores supplemental prompt selections and ordering."""

    def __init__(self):
        self.selected_files = []
        self.file_order = []

    def set_selected_files(self, files):
        self.selected_files = list(files or [])

    def set_file_order(self, order):
        self.file_order = list(order or [])
