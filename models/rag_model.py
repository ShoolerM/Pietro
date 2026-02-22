"""Smart Model for managing ChromaDB databases and document retrieval."""

import json
from pathlib import Path
from base.observable import Observable


class RAGModel(Observable):
    """Model for RAG database management.

    Manages ChromaDB databases, their metadata, and file associations.
    Actual vector operations are handled by RAGController to avoid
    heavy dependencies in the model layer.
    """

    def __init__(self, rag_dir="rag_databases"):
        """Initialize Smart Model.

        Args:
            rag_dir: Directory to store RAG databases
        """
        super().__init__()
        self.rag_dir = Path(rag_dir)
        self.rag_dir.mkdir(exist_ok=True)

        # Metadata file to track databases
        self.metadata_file = self.rag_dir / "databases.json"
        self._databases = {}
        self._selected_databases = set()

        # RAG query settings - will be loaded from file
        self.max_chunks = 10  # Default: auto-build chunks
        self.summary_chunk_size = 1500  # Default: max raw tokens for summarization
        self.score_variance_threshold = 0.05  # Default: 5% variance from top score

        # Filename matching/boosting settings
        self.filename_boost_enabled = True  # Default: enable filename boosting
        self.max_filename_chunks = 1  # Default: guarantee 1 chunk per matched file
        self.levenshtein_threshold = 2  # Default: allow 2 character edit distance

        # Load databases and settings
        self._load_data()

    def _load_data(self):
        """Load database metadata and settings from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check if this is the old format (databases at root) or new format
                if "databases" in data:
                    # New format with separate databases and settings keys
                    self._databases = data.get("databases", {})
                    settings = data.get("settings", {})
                    self.max_chunks = settings.get("max_chunks", 10)
                    self.score_variance_threshold = settings.get(
                        "score_variance_threshold", 0.05
                    )
                    self.summary_chunk_size = settings.get("summary_chunk_size", 1500)

                    # Load filename boost settings
                    self.filename_boost_enabled = settings.get(
                        "filename_boost_enabled", True
                    )
                    self.max_filename_chunks = settings.get("max_filename_chunks", 1)
                    self.levenshtein_threshold = settings.get(
                        "levenshtein_threshold", 2
                    )

                    # Persist missing settings keys for older files
                    if any(
                        key not in settings
                        for key in [
                            "max_chunks",
                            "summary_chunk_size",
                            "score_variance_threshold",
                            "filename_boost_enabled",
                            "max_filename_chunks",
                            "levenshtein_threshold",
                        ]
                    ):
                        self._save_data()

                    # Load selected databases
                    selected_list = data.get("selected_databases", [])
                    self._selected_databases = set(selected_list)
                else:
                    # Old format - databases at root level, no settings key
                    # Migrate to new format
                    print("Detected old database format, migrating...")
                    self._databases = data
                    self.max_chunks = 10
                    self.summary_chunk_size = 1500
                    # Save in new format immediately
                    self._save_data()
                    print("Migration complete!")

                print(
                    f"Loaded RAG settings: "
                    f"max_chunks={self.max_chunks}, "
                    f"summary_chunk_size={self.summary_chunk_size}, "
                    f"score_variance_threshold={self.score_variance_threshold:.1%}"
                )
                if self._selected_databases:
                    print(
                        f"Loaded selected databases: {list(self._selected_databases)}"
                    )
            except Exception as e:
                print(f"Error loading database metadata: {e}")
                self._databases = {}

    def _save_data(self):
        """Save database metadata and settings to file."""
        try:
            data = {
                "databases": self._databases,
                "settings": {
                    "max_chunks": self.max_chunks,
                    "summary_chunk_size": self.summary_chunk_size,
                    "score_variance_threshold": self.score_variance_threshold,
                    "filename_boost_enabled": self.filename_boost_enabled,
                    "max_filename_chunks": self.max_filename_chunks,
                    "levenshtein_threshold": self.levenshtein_threshold,
                },
                "selected_databases": list(self._selected_databases),
            }
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving database metadata: {e}")

    def get_databases(self, include_hidden=False):
        """Get list of available databases.

        Returns:
            list: List of (db_name, file_count, is_selected) tuples
        """
        result = []
        for db_name, metadata in self._databases.items():
            if metadata.get("hidden") and not include_hidden:
                continue
            file_count = len(metadata.get("files", []))
            is_selected = db_name in self._selected_databases
            result.append((db_name, file_count, is_selected))
        return result

    def ensure_database(self, db_name, hidden=False):
        """Ensure a database exists, creating it if necessary.

        Args:
            db_name: Name for the database
            hidden: Whether to mark as hidden

        Returns:
            tuple: (created: bool, db_name: str)
        """
        if db_name not in self._databases:
            self._databases[db_name] = {
                "files": [],
                "path": str(self.rag_dir / db_name),
                "hidden": bool(hidden),
                "meta": {},
            }

            db_path = self.rag_dir / db_name
            db_path.mkdir(exist_ok=True)

            self._save_data()
            self.notify_observers("database_created", db_name)
            return True, db_name

        if hidden and not self._databases[db_name].get("hidden"):
            self._databases[db_name]["hidden"] = True
            self._save_data()

        return False, db_name

    def get_database_meta(self, db_name, key, default=None):
        if db_name in self._databases:
            meta = self._databases[db_name].get("meta", {})
            return meta.get(key, default)
        return default

    def set_database_meta(self, db_name, key, value):
        if db_name not in self._databases:
            return False
        self._databases[db_name].setdefault("meta", {})[key] = value
        self._save_data()
        return True

    def create_database(self, db_name):
        """Create a new RAG database.

        Args:
            db_name: Name for the database

        Returns:
            tuple: (success: bool, message: str)
        """
        if not db_name or not db_name.strip():
            return False, "Database name cannot be empty"

        db_name = db_name.strip()

        if db_name in self._databases:
            return False, f"Database '{db_name}' already exists"

        # Create database entry
        self._databases[db_name] = {
            "files": [],
            "path": str(self.rag_dir / db_name),
            "hidden": False,
            "meta": {},
        }

        # Create directory for database
        db_path = self.rag_dir / db_name
        db_path.mkdir(exist_ok=True)

        self._save_data()
        self.notify_observers("database_created", db_name)

        return True, db_name

    def add_file_to_database(self, db_name, file_path, notify=True):
        """Add file metadata to database.

        Args:
            db_name: Name of the database
            file_path: Path to the file

        Returns:
            tuple: (success: bool, message: str)
        """
        if db_name not in self._databases:
            return False, f"Database '{db_name}' not found"

        file_path_str = str(Path(file_path).resolve())

        if file_path_str in self._databases[db_name]["files"]:
            return False, f"File already added to database"

        self._databases[db_name]["files"].append(file_path_str)
        self._save_data()
        if notify:
            self.notify_observers("file_added", (db_name, file_path_str))

        return True, file_path_str

    def get_database_path(self, db_name):
        """Get the filesystem path for a database.

        Args:
            db_name: Name of the database

        Returns:
            str: Path to database directory, or None if not found
        """
        if db_name in self._databases:
            return self._databases[db_name]["path"]
        return None

    def get_database_files(self, db_name):
        """Get list of files in a database.

        Args:
            db_name: Name of the database

        Returns:
            list: List of file paths
        """
        if db_name in self._databases:
            return self._databases[db_name]["files"].copy()
        return []

    def toggle_database_selection(self, db_name):
        """Toggle database selection state.

        Args:
            db_name: Name of the database
        """
        if db_name in self._selected_databases:
            self._selected_databases.remove(db_name)
        else:
            self._selected_databases.add(db_name)

        self._save_data()  # Persist selection changes
        self.notify_observers("selection_changed", list(self._selected_databases))

    def get_selected_databases(self):
        """Get list of selected database names.

        Returns:
            list: List of selected database names
        """
        return list(self._selected_databases)

    def delete_database(self, db_name):
        """Delete a database.

        Args:
            db_name: Name of the database

        Returns:
            tuple: (success: bool, message: str)
        """
        if db_name not in self._databases:
            return False, f"Database '{db_name}' not found"

        # Remove from selected if it was selected
        self._selected_databases.discard(db_name)

        # Remove from databases
        del self._databases[db_name]
        self._save_data()

        self.notify_observers("database_deleted", db_name)

        return True, f"Database '{db_name}' deleted"

    def set_max_chunks(self, max_chunks):
        """Set the maximum number of chunks to generate in auto-build mode.

        Args:
            max_chunks: Integer between 1 and 50
        """
        self.max_chunks = max(1, min(50, max_chunks))
        self._save_data()  # Save settings to persist
        print(f"RAG max chunks set to: {self.max_chunks}")

    def set_summary_chunk_size(self, chunk_size):
        """Set the max raw token chunk size used for story summarization.

        Args:
            chunk_size: Integer between 256 and 200000
        """
        self.summary_chunk_size = max(256, min(200000, int(chunk_size)))
        self._save_data()
        print(f"RAG summary chunk size set to: {self.summary_chunk_size}")

    def set_score_variance_threshold(self, percent):
        """Set the score variance threshold for filtering RAG results.

        Args:
            percent: Percentage value between 5 and 30 (displayed as 5%-30%)
        """
        # Convert percentage to decimal (5% -> 0.05, 30% -> 0.30)
        decimal_value = percent / 100.0
        self.score_variance_threshold = max(0.05, min(0.30, decimal_value))
        self._save_data()
        print(
            f"RAG score variance threshold set to: {self.score_variance_threshold:.2%}"
        )
        self.notify_observers(
            "score_variance_threshold_changed", self.score_variance_threshold
        )

    def set_filename_boost_enabled(self, enabled):
        """Enable or disable filename-based chunk boosting.

        Args:
            enabled: Boolean to enable/disable feature
        """
        self.filename_boost_enabled = bool(enabled)
        self._save_data()
        print(f"RAG filename boost enabled: {self.filename_boost_enabled}")
        self.notify_observers(
            "filename_boost_enabled_changed", self.filename_boost_enabled
        )

    def set_max_filename_chunks(self, count):
        """Set the maximum number of chunks to guarantee from filename-matched files.

        Args:
            count: Integer between 0 and 5
        """
        self.max_filename_chunks = max(0, min(5, int(count)))
        self._save_data()
        print(f"RAG max filename chunks set to: {self.max_filename_chunks}")
        self.notify_observers("max_filename_chunks_changed", self.max_filename_chunks)

    def set_levenshtein_threshold(self, threshold):
        """Set the Levenshtein distance threshold for fuzzy filename matching.

        Args:
            threshold: Integer between 0 and 3 (character edit distance)
        """
        self.levenshtein_threshold = max(0, min(3, int(threshold)))
        self._save_data()
        print(f"RAG Levenshtein threshold set to: {self.levenshtein_threshold}")
        self.notify_observers(
            "levenshtein_threshold_changed", self.levenshtein_threshold
        )
