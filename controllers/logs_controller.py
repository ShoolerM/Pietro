"""Controller for logs panel."""


class LogsController:
    """Coordinates logs panel view and model."""

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.clear_requested.connect(self.clear_logs)

    def append_logs(self, text: str):
        self.model.append(text)
        self.view.append_logs(text)

    def clear_logs(self):
        self.model.clear()
        self.view.clear_logs()
