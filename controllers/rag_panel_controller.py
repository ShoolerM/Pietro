"""Controller for RAG panel."""


class RagPanelController:
    """Coordinates RAG panel view and model."""

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.max_chunks_changed.connect(self._on_max_chunks_changed)
        self.view.summary_chunk_size_changed.connect(
            self._on_summary_chunk_size_changed
        )
        self.view.score_threshold_changed.connect(self._on_score_threshold_changed)
        self.view.filename_boost_enabled_changed.connect(
            self._on_filename_boost_enabled_changed
        )
        self.view.max_filename_chunks_changed.connect(
            self._on_max_filename_chunks_changed
        )
        self.view.levenshtein_threshold_changed.connect(
            self._on_levenshtein_threshold_changed
        )

    def _on_max_chunks_changed(self, value):
        self.model.set_settings(max_chunks=value)

    def _on_summary_chunk_size_changed(self, value):
        self.model.set_settings(summary_chunk_size=value)

    def _on_score_threshold_changed(self, value):
        self.model.set_settings(score_threshold=value)

    def _on_filename_boost_enabled_changed(self, value):
        self.model.set_settings(filename_boost_enabled=value)

    def _on_max_filename_chunks_changed(self, value):
        self.model.set_settings(max_filename_chunks=value)

    def _on_levenshtein_threshold_changed(self, value):
        self.model.set_settings(levenshtein_threshold=value)

    def load_databases(self, databases):
        self.model.set_databases(databases)
        self.view.load_databases(databases)

    def show_settings_dialog(
        self,
        current_max_chunks,
        current_summary_chunk_size,
        current_score_threshold,
        current_filename_boost_enabled,
        current_max_filename_chunks,
        current_levenshtein_threshold,
    ):
        self.view.show_settings_dialog(
            current_max_chunks=current_max_chunks,
            current_summary_chunk_size=current_summary_chunk_size,
            current_score_threshold=current_score_threshold,
            current_filename_boost_enabled=current_filename_boost_enabled,
            current_max_filename_chunks=current_max_filename_chunks,
            current_levenshtein_threshold=current_levenshtein_threshold,
        )
