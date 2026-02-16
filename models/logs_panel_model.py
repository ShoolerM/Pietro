"""Model for logs panel state."""


class LogsPanelModel:
    """Stores log text entries."""

    def __init__(self):
        self.entries = []

    def append(self, text):
        if text:
            self.entries.append(text)

    def clear(self):
        self.entries.clear()
