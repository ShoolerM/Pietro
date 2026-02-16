"""Controller for supplemental prompts panel."""


class SupplementalController:
    """Coordinates supplemental prompts view and model."""

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.selections_changed.connect(self._on_selections_changed)

    def _on_selections_changed(self, selected_files):
        self.model.set_selected_files(selected_files)
        self.model.set_file_order(self.view.get_file_order())

    def load_files(self, files, selected_files=None):
        self.view.load_files(files, selected_files)
        if selected_files is not None:
            self.model.set_selected_files(selected_files)

    def gather_supplemental_text(self):
        return self.view.gather_supplemental_text()

    def get_selected_files(self):
        return list(self.model.selected_files)
