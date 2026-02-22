"""Model for RAG panel state."""


class RagPanelModel:
    """Stores RAG panel settings state."""

    def __init__(self):
        self.databases = []
        self.max_chunks = None
        self.summary_chunk_size = None
        self.score_threshold = None
        self.filename_boost_enabled = None
        self.max_filename_chunks = None
        self.levenshtein_threshold = None

    def set_databases(self, databases):
        self.databases = list(databases or [])

    def set_settings(
        self,
        max_chunks=None,
        summary_chunk_size=None,
        score_threshold=None,
        filename_boost_enabled=None,
        max_filename_chunks=None,
        levenshtein_threshold=None,
    ):
        if max_chunks is not None:
            self.max_chunks = max_chunks
        if summary_chunk_size is not None:
            self.summary_chunk_size = summary_chunk_size
        if score_threshold is not None:
            self.score_threshold = score_threshold
        if filename_boost_enabled is not None:
            self.filename_boost_enabled = filename_boost_enabled
        if max_filename_chunks is not None:
            self.max_filename_chunks = max_filename_chunks
        if levenshtein_threshold is not None:
            self.levenshtein_threshold = levenshtein_threshold
