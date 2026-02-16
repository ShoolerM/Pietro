"""Controller for system prompts panel."""


class SystemController:
    """Coordinates system prompts view and model."""

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.selection_changed.connect(self._on_selection_changed)

    def _on_selection_changed(self, selected_file):
        self.model.set_selected_file(selected_file)

    def load_files(self, files, selected_file=None):
        self.view.load_files(files, selected_file)
        if selected_file is not None:
            self.model.set_selected_file(selected_file)

    def get_system_prompt_text(self):
        return self.view.get_system_prompt_text()

    def get_selected_file(self):
        return self.model.selected_file
