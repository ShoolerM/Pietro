"""Controller for RAG operations using FAISS."""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
import numpy as np
from PyQt5 import QtWidgets
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from views.progress_dialog import ProgressDialog
from models.rag_model import RAGEvent
from models.story_model import StoryModel

if TYPE_CHECKING:
    from models.rag_model import RAGModel
    from views.main_view import MainView

# Supported text file extensions for RAG ingestion.
# Used both when filtering files for import and when reading them from disk.
TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".c",
        ".cfg",
        ".conf",
        ".cpp",
        ".css",
        ".csv",
        ".doc",
        ".docx",
        ".htm",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsonl",
        ".log",
        ".markdown",
        ".md",
        ".pdf",
        ".py",
        ".rtf",
        ".tex",
        ".toml",
        ".tsv",
        ".txt",
        ".xml",
    }
)

MINIMUM_LEVENSHTEIN_TOKEN_LENGTH: int = (
    3  # Minimum token length to consider for fuzzy filename matching
)
DEFAULT_TOKEN_BUDGET: int = 2000  # Default max tokens to allocate for RAG context

DB_INITIAL_K: int = 20  # Number of initial results to fetch from each database for greedy selection
FILENAME_BOOST_EXTRA_K: int = (
    500  # Candidates to fetch per database when boosting a matched filename
)
MIN_REMAINING_TOKENS: int = 100  # Minimum tokens left before attempting to fit a partial chunk
CHARS_PER_TOKEN: int = 4  # Approximate characters per token used when truncating a chunk


@dataclass
class ScoredChunk:
    """A retrieved document chunk paired with its similarity score and optional boost metadata."""

    doc: Document
    score: float
    # None = not boosted; a filename string = retrieved via filename boosting
    boosted_filename: str | None = None

    @property
    def is_boosted(self) -> bool:
        """True if this chunk was retrieved via filename boosting."""
        return self.boosted_filename is not None


@dataclass
class PackedChunk:
    """A chunk selected for inclusion in the RAG context after greedy token packing."""

    content: str
    score: float
    token_count: int
    truncated: bool
    # Propagated from the source ScoredChunk; None means not boosted
    boosted_filename: str | None

    @property
    def is_boosted(self) -> bool:
        """True if this chunk originated from a filename-boosted file."""
        return self.boosted_filename is not None


@dataclass
class OutlineCompletionStatus:
    """Result of checking how much of an outline has been addressed in story content."""

    completed_tasks: list[str]
    pending_tasks: list[str]
    task_similarities: dict[str, float]

    @property
    def all_completed(self) -> bool:
        """True if every task in the outline has been addressed."""
        return len(self.pending_tasks) == 0

    @property
    def completion_ratio(self) -> float:
        """Fraction of tasks addressed (0.0 – 1.0)."""
        total = len(self.completed_tasks) + len(self.pending_tasks)
        return len(self.completed_tasks) / total if total else 0.0

    @classmethod
    def empty(cls) -> OutlineCompletionStatus:
        """Return an empty status representing no tasks found or a hard failure."""
        return cls(completed_tasks=[], pending_tasks=[], task_similarities={})


class RAGController:
    """Handles RAG operations with FAISS.

    Manages document ingestion, vector storage, and retrieval.
    Lazy-loads FAISS dependencies to avoid import overhead.
    """

    def __init__(self, rag_model: RAGModel, view: MainView) -> None:
        """Initialize RAG controller.

        Args:
            rag_model: RAGModel instance
            view: MainView instance
        """
        self.rag_model: RAGModel = rag_model
        self.view: MainView = view
        self._ask_mode_db_name: str = "__ask_readme__"

        # Lazy-loaded components
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._text_splitter: RecursiveCharacterTextSplitter | None = None
        # Cache for loaded FAISS indexes {db_name: vectorstore}
        self._vectorstores: dict[str, FAISS] = {}

        # Register model observers
        self.rag_model.add_observer(self._on_rag_model_changed)

    def _init_components(self, quiet: bool = False) -> None:
        """Lazy-initialize FAISS components.

        Args:
            quiet: If True, suppress log messages during initialization
        """
        if self._embeddings is not None:
            return

        try:
            if not quiet:
                self.view.append_logs("Initializing RAG components...")

            # Get database path from model
            rag_dir = Path(self.rag_model.rag_dir)
            rag_dir.mkdir(exist_ok=True)

            if not quiet:
                self.view.append_logs(f"  RAG storage: {rag_dir}")
                self.view.append_logs("  Loading embedding model...")

            # Initialize embeddings (using a lightweight model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
            )

            if not quiet:
                self.view.append_logs("  ✓ Embedding model loaded")

            # Initialize text splitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )

            if not quiet:
                self.view.append_logs("✓ RAG components initialized\n")

        except Exception as e:
            if not quiet:
                self.view.append_logs(f"❌ Error initializing RAG components: {e}")
            traceback.print_exc()
            self.view.show_warning(
                "RAG Error",
                f"Failed to initialize RAG:\n{e}\n\nMake sure required packages are installed.",
            )

    def _get_vectorstore_path(self, db_name: str) -> Path:
        """Get the file path for a vectorstore.

        Args:
            db_name: Name of the database

        Returns:
            Path object for the vectorstore file
        """
        rag_dir = Path(self.rag_model.rag_dir)
        return rag_dir / f"{db_name}.faiss"

    def _load_vectorstore(self, db_name: str) -> FAISS | None:
        """Load a FAISS vectorstore from disk.

        Args:
            db_name: Name of the database

        Returns:
            FAISS vectorstore or None
        """
        self._init_components()

        if self._embeddings is None:  # Init components must have failed
            return None

        # Check cache first
        if db_name in self._vectorstores:
            return self._vectorstores[db_name]

        try:
            vectorstore_path: Path = self._get_vectorstore_path(db_name)

            # Load existing vectorstore if it exists
            if vectorstore_path.exists():
                vectorstore: FAISS = FAISS.load_local(
                    str(vectorstore_path.parent),
                    self._embeddings,
                    db_name,
                    allow_dangerous_deserialization=True,
                )
                self._vectorstores[db_name] = vectorstore
                return vectorstore

            # Create new empty vectorstore
            # We'll add documents to it when files are ingested
            return None

        except Exception as e:
            self.view.append_logs(f"Error loading vectorstore '{db_name}': {e}")
            traceback.print_exc()
            return None

    def _load_document(self, file_path: str | Path) -> str | None:
        """Load a document from file.

        Args:
            file_path: Path to the file

        Returns:
            str: Document text or None
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return None

            # Check if file is empty
            if path.stat().st_size == 0:
                return None

            # Read text file with multiple encoding attempts
            if path.suffix.lower() in TEXT_EXTENSIONS:
                # Try multiple encodings
                encodings: list[str] = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

                for encoding in encodings:
                    try:
                        with open(path, "r", encoding=encoding, errors="strict") as f:
                            content = f.read()
                            if content and content.strip():
                                return content
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception:
                        continue

                # If all encodings fail, try with error replacement
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                        return content if content and content.strip() else None
                except Exception:
                    return None

            # For unknown file types, try to read as text
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    return content if content and content.strip() else None
            except Exception:
                return None

        except Exception as e:
            self.view.append_logs(f"⚠️ Error loading document '{file_path}': {e}")
            return None

    def _on_rag_model_changed(self, event_type: RAGEvent, data: Any) -> None:
        """Handle Smart Model changes."""
        try:
            if event_type != RAGEvent.SELECTION_CHANGED:
                self.view.append_logs(f"\n🔔 Smart Model event: {event_type.value}")

            if event_type == RAGEvent.DATABASE_CREATED:
                self.refresh_databases()
            elif event_type == RAGEvent.FILE_ADDED:
                db_name: str
                file_path: str
                db_name, file_path = data
                self.view.append_logs(f"   Database: {db_name}")
                self.view.append_logs(f"   File: {file_path}")
                self._ingest_file_with_progress(db_name, file_path)
            elif event_type == RAGEvent.DATABASE_DELETED:
                self.refresh_databases()
                # Remove from cache
                if data in self._vectorstores:
                    del self._vectorstores[data]
            elif event_type == RAGEvent.SELECTION_CHANGED:
                self.refresh_databases()
                self.view.set_rag_selection(data)
        except Exception as e:
            self.view.append_logs(f"\n❌ EXCEPTION in _on_rag_model_changed:")
            self.view.append_logs(f"   Event type: {event_type}")
            self.view.append_logs(f"   Data: {data}")
            self.view.append_logs(f"   Error: {e}")
            traceback.print_exc()

    def _ingest_file_with_progress(
        self,
        db_name: str,
        file_path: str,
        progress: ProgressDialog | None = None,
        quiet: bool = False,
    ) -> None:
        """Ingest a file into the database with optional progress reporting.

        Args:
            db_name: Name of the database
            file_path: Path to the file
            progress: Optional ProgressDialog instance for reporting progress
            quiet: If True, suppresses logging
        """

        def log(msg: str) -> None:
            """Helper to log to both console and progress dialog."""
            if not quiet:
                self.view.append_logs(msg)
                if progress:
                    progress.append_detail(msg)

        try:
            log(f"\n{'=' * 80}")
            log("STARTING FILE INGESTION (FAISS)")
            log(f"Database: {db_name}")
            log(f"File: {file_path}")
            log(f"{'=' * 80}")

            log("Step 1: Initializing components...")
            self._init_components(quiet=quiet)

            if self._embeddings is None:
                log("❌ Embeddings model is None after initialization!")
                raise Exception("Failed to initialize embeddings model")

            log("✓ Components initialized")

            # Load document
            file_name = Path(file_path).name
            log(f"\nStep 2: Loading document '{file_name}'...")

            # Check file size first
            file_size = Path(file_path).stat().st_size
            if file_size == 0:
                log("⚠️ File is empty (0 bytes) - skipping")
                raise Exception(f"File is empty: {file_path}")

            text = self._load_document(file_path)
            if not text:
                log("❌ Could not read file (may be empty or unsupported encoding)")
                raise Exception(f"Could not read file: {file_path}")

            log(f"✓ Loaded {len(text)} characters")

            # Split into chunks
            log("\nStep 3: Splitting text into chunks...")
            if not self._text_splitter:
                log("❌ Text splitter is None after initialization!")
                raise Exception("Failed to initialize text splitter")
            chunks: list[str] = self._text_splitter.split_text(text)

            if not chunks:
                log("❌ No content extracted")
                raise Exception(f"No content extracted from file: {file_path}")

            log(f"✓ Split into {len(chunks)} chunks")

            # Create LangChain Documents with metadata
            log("\nStep 4: Creating documents with metadata...")
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_name,
                        "chunk_index": i,
                    },
                )
                documents.append(doc)

            log(f"✓ Created {len(documents)} documents")

            # Load existing vectorstore or create new one
            log(f"\nStep 5: Loading/creating vectorstore for '{db_name}'...")
            vectorstore = self._load_vectorstore(db_name)

            if vectorstore is None:
                # Create new vectorstore from documents
                log("  Creating new FAISS vectorstore...")
                log("  (This may take a moment for embedding generation...)")
                vectorstore = FAISS.from_documents(documents, self._embeddings)
                log("  ✓ New vectorstore created")
            else:
                # Add documents to existing vectorstore
                log("  Adding documents to existing vectorstore...")
                log("  (Generating embeddings for {len(documents)} chunks...)")
                vectorstore.add_documents(documents)
                log("  ✓ Documents added")

            # Save vectorstore to disk
            log("\nStep 6: Saving vectorstore to disk...")
            vectorstore_path = self._get_vectorstore_path(db_name)
            vectorstore.save_local(str(vectorstore_path.parent), db_name)
            log(f"  ✓ Saved to {vectorstore_path}")

            # Update cache
            self._vectorstores[db_name] = vectorstore

            log(f"\n✓ Successfully added {len(chunks)} chunks to '{db_name}'")
            log(f"{'=' * 80}\n")

        except Exception as e:
            error_msg = f"❌ FATAL ERROR ingesting file '{file_path}'"
            log(f"\n{error_msg}")
            log(f"Error type: {type(e).__name__}")
            log(f"Error message: {e}")

            if not progress:
                traceback.print_exc()
                log(f"{'=' * 80}\n")

            # Re-raise so caller can handle
            raise

    def create_database(self) -> None:
        """Create a new RAG database."""
        db_name: str
        ok: bool
        db_name, ok = self.view.show_input_dialog("New RAG Database", "Enter database name:")

        if not ok or not db_name.strip():
            return

        success: bool
        result: str
        success, result = self.rag_model.create_database(db_name.strip())

        if not success:
            self.view.show_warning("Error", result)

    def _collect_file_paths(self, selected: list[str]) -> list[str]:
        """Resolve a list of user-selected paths into individual file paths.

        Directories are walked recursively; hidden entries are skipped.
        Files are included only if their extension is in TEXT_EXTENSIONS.
        Explicitly selected files are always included regardless of extension.

        Args:
            selected: Paths returned by the file chooser (files or directories)

        Returns:
            Flat list of absolute file path strings
        """
        file_paths: list[str] = []

        for selection_str in selected:
            selection: Path = Path(selection_str)
            if selection.is_dir():
                for file_path in selection.rglob("*"):
                    # Skip hidden files and directories (starting with .)
                    if any(
                        part.startswith(".") for part in file_path.parts[len(selection.parts) :]
                    ):
                        continue
                    if file_path.is_file() and file_path.suffix.lower() in TEXT_EXTENSIONS:
                        file_paths.append(str(file_path))
            elif selection.is_file():
                file_paths.append(str(selection))

        return file_paths

    def _process_files_with_progress(
        self, db_name: str, file_paths: list[str], progress: ProgressDialog
    ) -> tuple[int, int]:
        """Ingest each file into the database, updating the progress dialog.

        Args:
            db_name: Name of the target database
            file_paths: Ordered list of file paths to ingest
            progress: ProgressDialog to report status to

        Returns:
            Tuple of (success_count, error_count)
        """
        success_count: int = 0
        error_count: int = 0

        for idx, file_path in enumerate(file_paths, 1):
            file_name = Path(file_path).name
            progress.set_message(f"Processing {idx}/{len(file_paths)}: {file_name}")
            progress.append_detail(f"\n[{idx}/{len(file_paths)}] Processing: {file_name}")

            # Register file metadata in the model
            success, result = self.rag_model.add_file_to_database(db_name, file_path)
            if not success:
                progress.append_detail(f"  ❌ Failed to add metadata: {result}")
                error_count += 1
                progress.set_progress(idx)
                continue

            progress.append_detail("  ✓ Added to database metadata")

            # Vectorise and store
            progress.append_detail("  → Ingesting file...")
            try:
                self._ingest_file_with_progress(db_name, file_path, progress)
                progress.append_detail("  ✓ Successfully ingested")
                success_count += 1
            except Exception as e:
                progress.append_detail(f"  ❌ {type(e).__name__}: {e}")
                self.view.append_logs(f"\n{'=' * 80}")
                self.view.append_logs(f"ERROR ingesting {file_name}:")
                traceback.print_exc()
                self.view.append_logs(f"{'=' * 80}\n")
                error_count += 1

            progress.set_progress(idx)

        return success_count, error_count

    def _finish_progress_dialog(
        self, progress: ProgressDialog, success_count: int, error_count: int
    ) -> None:
        """Display a completion summary on the progress dialog.

        Args:
            progress: The ProgressDialog to update
            success_count: Number of files successfully ingested
            error_count: Number of files that failed
        """
        if error_count == 0:
            progress.finish_success(f"Successfully added {success_count} file(s)")
        else:
            msg = f"Added {success_count} file(s), {error_count} failed"
            progress.set_message(msg)
            progress.append_detail(f"\n⚠️  {msg}")
            progress.show_cancel_button()
            progress.cancel_button.setText("Close")
            progress.exec_()

    def add_files_to_database(self, db_name: str) -> None:
        """Add files to a database.

        Prompts the user to choose files or a directory, then ingests every
        qualifying file into the named FAISS database with a progress dialog.

        Args:
            db_name: Name of the database
        """
        selected: list[str] | None = self.view.show_file_chooser(
            "Add Files to Database", allow_directory=True
        )
        if not selected:
            return

        file_paths: list[str] = self._collect_file_paths(selected)

        if not file_paths:
            QtWidgets.QMessageBox.warning(
                self.view,
                "No Text Files Found",
                f"No text files with recognized extensions were found in the selected items.\n\nSupported: {' '.join(sorted(TEXT_EXTENSIONS))}",
            )
            return

        progress: ProgressDialog = ProgressDialog(f"Adding Files to {db_name}", self.view)
        progress.set_message(f"Adding {len(file_paths)} file(s) to database...")
        progress.set_progress(0, len(file_paths))
        progress.show()

        success_count, error_count = self._process_files_with_progress(
            db_name, file_paths, progress
        )

        self._finish_progress_dialog(progress, success_count, error_count)
        self.refresh_databases()

    def refresh_databases(self) -> None:
        """Refresh the RAG databases list in view."""
        databases: list[str] = self.rag_model.get_databases()
        self.view.load_rag_databases(databases)
        self.view.set_rag_selection(self.rag_model.get_selected_databases())

    def _has_ask_docs_changed(self, existing_paths: list[Path]) -> bool:
        """Check whether any ask-docs paths are new or modified since last ingestion.

        Newly discovered paths are registered in the model as a side effect.

        Args:
            existing_paths: Paths to check against stored metadata

        Returns:
            True if any path is new or has a newer mtime than what is stored
        """
        files: list[str] = self.rag_model.get_database_files(self._ask_mode_db_name)
        changed: bool = False

        for path in existing_paths:
            path_str: str = str(path.resolve())
            mtime_val: float = path.stat().st_mtime
            stored_mtime: float | None = self.rag_model.get_database_meta(
                self._ask_mode_db_name, f"mtime:{path.name}"
            )

            if path_str not in files:
                self.rag_model.add_file_to_database(self._ask_mode_db_name, path_str, notify=False)
                changed = True

            if stored_mtime is None or mtime_val > stored_mtime:
                changed = True

        return changed

    def _clear_vectorstore_files(self, db_name: str) -> None:
        """Delete vectorstore files from disk and evict from the in-memory cache.

        Args:
            db_name: Name of the database whose vectorstore should be cleared
        """
        try:
            vectorstore_path: Path = self._get_vectorstore_path(db_name)
            pkl_file: Path = vectorstore_path.parent / f"{db_name}.pkl"

            if vectorstore_path.exists():
                vectorstore_path.unlink()
            if pkl_file.exists():
                pkl_file.unlink()
            if db_name in self._vectorstores:
                del self._vectorstores[db_name]
        except Exception:
            pass

    def _reingest_paths(self, db_name: str, paths: list[Path]) -> None:
        """Ingest a list of paths into a database and update their stored mtimes.

        Args:
            db_name: Name of the target database
            paths: Paths to ingest
        """
        for path in paths:
            path_str: str = str(path.resolve())
            self._ingest_file_with_progress(db_name, path_str, progress=None, quiet=True)
            self.rag_model.set_database_meta(db_name, f"mtime:{path.name}", path.stat().st_mtime)

    def ensure_ask_mode_readme_database(
        self, readme_path: Path, extra_paths: list[Path] | None = None
    ) -> bool:
        """Ensure the hidden ask-docs database is created and up-to-date.

        Creates the database if it does not exist, checks whether any source
        files are new or modified, and if so rebuilds the vectorstore from
        scratch.

        Args:
            readme_path: Primary document to ingest (README / guide)
            extra_paths: Additional documents to include alongside readme_path

        Returns:
            True if the vectorstore was rebuilt, False if already current
        """
        try:
            paths: list[Path] = [readme_path]
            if extra_paths:
                paths.extend(extra_paths)

            existing_paths: list[Path] = [p for p in paths if p and p.exists()]
            if not existing_paths:
                return False

            self.rag_model.ensure_database(self._ask_mode_db_name, hidden=True)

            if not self._has_ask_docs_changed(existing_paths):
                return False

            self._clear_vectorstore_files(self._ask_mode_db_name)
            self._reingest_paths(self._ask_mode_db_name, existing_paths)

            return True
        except Exception:
            return False

    def toggle_database(self, db_name: str) -> None:
        """Toggle database selection. In the RAG selection panel.

        Selection is triggered by clicking on a database name,
        which calls this method to toggle its selection state in the model.
        The view then updates to reflect the new selection and triggers a
        re-query of the databases for context.

        Args:
            db_name: Name of the database
        """
        try:
            self.rag_model.toggle_database_selection(db_name)
        except Exception as e:
            self.view.append_logs(f"Error toggling database '{db_name}': {e}")
            traceback.print_exc()

    def delete_database(self, db_name: str) -> None:
        """Delete a database and its FAISS vectorstore.

        Args:
            db_name: Name of the database to delete
        """
        try:
            self.view.append_logs(f"\n🗑️  Deleting database: {db_name}")

            # Delete from model first
            success, message = self.rag_model.delete_database(db_name)

            if not success:
                self.view.append_logs(f"  ❌ {message}")
                self.view.show_warning("Delete Failed", message)
                return

            self.view.append_logs("  ✓ Removed from database list")

            # Try to delete the FAISS vectorstore files
            try:
                vectorstore_path = self._get_vectorstore_path(db_name)
                index_file = vectorstore_path
                pkl_file = vectorstore_path.parent / f"{db_name}.pkl"

                if index_file.exists():
                    index_file.unlink()
                    self.view.append_logs(f"  ✓ Deleted {index_file.name}")

                if pkl_file.exists():
                    pkl_file.unlink()
                    self.view.append_logs(f"  ✓ Deleted {pkl_file.name}")

            except Exception as file_error:
                self.view.append_logs(f"  ⚠️  Could not delete vectorstore files: {file_error}")

            # Remove from cache
            if db_name in self._vectorstores:
                del self._vectorstores[db_name]
                self.view.append_logs("  ✓ Removed from vectorstore cache")

            self.view.append_logs(f"✓ Database '{db_name}' deleted successfully\n")

        except Exception as e:
            self.view.append_logs(f"❌ Error deleting database '{db_name}': {e}")
            traceback.print_exc()
            self.view.show_warning("Delete Error", f"Failed to delete database:\n{e}")

    def _calculate_levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            int: Levenshtein distance
        """
        s1: str = s1.lower()
        s2: str = s2.lower()

        # Quick length check - if difference is > threshold, skip expensive calculation
        if abs(len(s1) - len(s2)) > self.rag_model.levenshtein_threshold:
            return 999  # Return high value to indicate no match

        # Initialize distance matrix
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Calculate distances
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        return matrix[len1][len2]

    def _extract_matching_filenames(
        self,
        query: str,
        selected_dbs: list[str],
    ) -> list[tuple[str, str]]:
        """Extract filenames from query that fuzzy match files in selected databases.

        Args:
            query: User query string
            selected_dbs: List of selected database names

        Returns:
            list: List of (db_name, filename) tuples for matched files
        """
        if not self.rag_model.filename_boost_enabled:
            return []

        # Tokenize query using regex to split on whitespace and punctuation
        tokens = [t for t in re.split(r"[\s\W]+", query.lower()) if t]

        if not tokens:
            return []

        matches: list[tuple[str, str]] = []
        threshold: int = self.rag_model.levenshtein_threshold

        # Iterate through all files in selected databases
        for db_name in selected_dbs:
            file_paths = self.rag_model.get_database_files(db_name)

            for file_path in file_paths:
                # Extract filename (remove extension, keep rest as-is)
                filename = Path(file_path).name
                stem = Path(file_path).stem.lower()

                # Check if any query token fuzzy matches the filename stem
                for token in tokens:
                    # Skip very short tokens (< 3 chars) to avoid false positives
                    if len(token) < MINIMUM_LEVENSHTEIN_TOKEN_LENGTH:
                        continue

                    # Check for exact match first (case-insensitive)
                    if token == stem:
                        matches.append((db_name, filename))
                        break  # Only add each file once

                    # For tokens >= MINIMUM_LEVENSHTEIN_TOKEN_LENGTH characters, allow fuzzy matching
                    # Skip if token and stem differ too much in length
                    # Allow at most threshold difference in length
                    if abs(len(token) - len(stem)) > threshold:
                        continue

                    distance = self._calculate_levenshtein_distance(token, stem)

                    if distance <= threshold:
                        matches.append((db_name, filename))
                        break  # Only add each file once

        return matches

    def _fetch_raw_results(
        self, query: str, selected_dbs: list[str], quiet: bool = False
    ) -> list[ScoredChunk]:
        """Query each selected vectorstore and return a flat list of ScoredChunks.

        Args:
            query: Query string
            selected_dbs: Database names to query
            quiet: If True, suppresses logging

        Returns:
            Flat list of unboosted ScoredChunks from all databases
        """
        results: list[ScoredChunk] = []

        for db_name in selected_dbs:
            vectorstore = self._load_vectorstore(db_name)
            if not vectorstore:
                if db_name != self._ask_mode_db_name:
                    self.view.append_logs(f"  ⚠️  Database '{db_name}' not found or empty")
                continue

            try:
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=DB_INITIAL_K)
                for doc, score in docs_with_scores:
                    results.append(ScoredChunk(doc=doc, score=score))
            except Exception as e:
                if db_name != self._ask_mode_db_name:
                    self.view.append_logs(f"Error querying database '{db_name}': {e}")
                    traceback.print_exc()

        return results

    def _fetch_boosted_chunks(
        self,
        query: str,
        matched_files: list[tuple[str, str]],
        quiet: bool = False,
    ) -> list[ScoredChunk]:
        """Retrieve extra chunks from files whose names fuzzy-matched the query.

        Re-queries each matched file's vectorstore with a filename-augmented query
        and filters down to only chunks belonging to that file.

        Args:
            query: Original query string
            matched_files: (db_name, filename) pairs from filename matching
            quiet: If True, suppresses logging

        Returns:
            List of boosted ScoredChunks with boosted_filename set
        """
        boosted: list[ScoredChunk] = []

        if not matched_files:
            return boosted

        if not quiet:
            self.view.append_logs(
                f"Filename matches: {', '.join(fname for _, fname in matched_files)}"
            )

        for db_name, target_filename in matched_files:
            vectorstore = self._load_vectorstore(db_name)
            if not vectorstore:
                continue

            try:
                boosted_query = f"{query} {Path(target_filename).stem}"
                docs_with_scores = vectorstore.similarity_search_with_score(
                    boosted_query, k=FILENAME_BOOST_EXTRA_K
                )

                file_chunks = [
                    (doc, score)
                    for doc, score in docs_with_scores
                    if doc.metadata.get("file_name") == target_filename
                ]

                if not quiet:
                    if not file_chunks:
                        self.view.append_logs(
                            f"  Warning: No chunks found for {target_filename} "
                            f"in top {FILENAME_BOOST_EXTRA_K} results"
                        )
                    else:
                        self.view.append_logs(
                            f"  Found {len(file_chunks)} chunks from {target_filename}"
                        )

                file_chunks.sort(key=lambda x: x[1])
                for doc, score in file_chunks[: self.rag_model.max_filename_chunks]:
                    boosted.append(
                        ScoredChunk(doc=doc, score=score, boosted_filename=target_filename)
                    )

            except Exception as e:
                if not quiet:
                    self.view.append_logs(
                        f"Error retrieving boosted chunks from '{target_filename}': {e}"
                    )

        return boosted

    def _deduplicate_and_merge(
        self,
        raw_results: list[ScoredChunk],
        boosted_chunks: list[ScoredChunk],
        quiet: bool = False,
    ) -> list[ScoredChunk]:
        """Merge base results with boosted chunks, deduplicating by content hash.

        Non-boosted entries that also appear in boosted_chunks are upgraded to
        boosted in-place so they sort to the front during ranking.

        Args:
            raw_results: Unboosted ScoredChunks from plain vectorstore queries
            boosted_chunks: Boosted ScoredChunks from filename boost queries
            quiet: If True, suppresses logging

        Returns:
            Deduplicated list of ScoredChunks with boosts applied
        """
        seen: dict[int, int] = {}  # content_hash -> index in merged
        merged: list[ScoredChunk] = []

        for chunk in raw_results:
            h = hash(chunk.doc.page_content)
            if h not in seen:
                seen[h] = len(merged)
                merged.append(chunk)

        for chunk in boosted_chunks:
            h = hash(chunk.doc.page_content)
            if h in seen:
                # Upgrade the existing unboosted entry to boosted in-place
                existing = merged[seen[h]]
                if not existing.is_boosted:
                    existing.boosted_filename = chunk.boosted_filename
            else:
                seen[h] = len(merged)
                merged.append(chunk)

        duplicates_removed = len(raw_results) - len(merged)
        if duplicates_removed > 0 and not quiet:
            self.view.append_logs(f"Deduplication: removed {duplicates_removed} duplicate chunks")

        return merged

    def _apply_adaptive_threshold(
        self,
        results: list[ScoredChunk],
        matched_files: list[tuple[str, str]],
        quiet: bool = False,
    ) -> list[ScoredChunk]:
        """Filter results to within an adaptive score threshold.

        Keeps chunks within a configurable percentage of the best score.
        Guarantees at least one chunk per boosted file is retained, and
        always returns at least 3 results. Re-sorts boosted-first by score.

        Args:
            results: ScoredChunks, pre-sorted boosted-first
            matched_files: (db_name, filename) pairs used for boosting
            quiet: If True, suppresses logging

        Returns:
            Filtered and re-sorted list
        """
        if not results:
            return results

        sort_key = lambda c: (not c.is_boosted, c.score)

        best_score: float = results[0].score
        adaptive_threshold: float = best_score * (1 + self.rag_model.score_variance_threshold)

        filtered: list[ScoredChunk] = []
        boosted_files_kept: set[str] = set()

        # First pass: keep everything within the score threshold
        for chunk in results:
            if chunk.score <= adaptive_threshold:
                filtered.append(chunk)
                if chunk.boosted_filename is not None:
                    boosted_files_kept.add(chunk.boosted_filename)

        # Second pass: ensure at least one chunk per boosted file survives
        for chunk in results:
            if (
                chunk.boosted_filename is not None
                and chunk.boosted_filename not in boosted_files_kept
            ):
                filtered.append(chunk)
                boosted_files_kept.add(chunk.boosted_filename)
                if not quiet:
                    self.view.append_logs(
                        f"  Preserving boosted chunk from {chunk.boosted_filename} "
                        f"(score: {chunk.score:.4f}, threshold: {adaptive_threshold:.4f})"
                    )

        filtered.sort(key=sort_key)

        # Always guarantee at least 3 results regardless of threshold
        if len(filtered) < 3:
            return sorted(results, key=sort_key)[: max(3, len(filtered))]

        return filtered

    def _greedy_pack_chunks(
        self,
        results: list[ScoredChunk],
        max_tokens: int,
    ) -> list[PackedChunk]:
        """Select chunks greedily until the token budget is exhausted.

        If a chunk would overflow the budget, a truncated version is appended
        if meaningful space remains, then packing stops.

        Args:
            results: Pre-sorted ScoredChunks
            max_tokens: Maximum token budget

        Returns:
            List of PackedChunks ready for context assembly
        """
        selected: list[PackedChunk] = []
        total_tokens: int = 0

        for chunk in results:
            chunk_tokens: int = StoryModel.estimate_token_count(chunk.doc.page_content)

            if total_tokens + chunk_tokens > max_tokens:
                remaining: int = max_tokens - total_tokens
                if remaining > MIN_REMAINING_TOKENS:
                    truncated_content = (
                        chunk.doc.page_content[: remaining * CHARS_PER_TOKEN] + "..."
                    )
                    selected.append(
                        PackedChunk(
                            content=truncated_content,
                            score=chunk.score,
                            token_count=chunk_tokens,
                            truncated=True,
                            boosted_filename=chunk.boosted_filename,
                        )
                    )
                break

            selected.append(
                PackedChunk(
                    content=chunk.doc.page_content,
                    score=chunk.score,
                    token_count=chunk_tokens,
                    truncated=False,
                    boosted_filename=chunk.boosted_filename,
                )
            )
            total_tokens += chunk_tokens

        return selected

    def _build_rag_display_items(
        self,
        selected_chunks: list[PackedChunk],
        all_results: list[ScoredChunk],
    ) -> list[str]:
        """Build display strings for the RAG panel from the selected chunks.

        Looks up source file and chunk index from document metadata.

        Args:
            selected_chunks: PackedChunks selected for context
            all_results: Full ScoredChunk list used for metadata lookup

        Returns:
            Human-readable label strings, one per selected chunk
        """
        rag_items: list[str] = []

        for packed in selected_chunks:
            trunc_marker: str = " [TRUNCATED]" if packed.truncated else ""
            boost_marker: str = " ★" if packed.is_boosted else ""
            source_file: str = "unknown"
            chunk_idx: int | str = "?"

            for scored in all_results:
                if scored.doc.page_content == packed.content or (
                    packed.truncated and packed.content.startswith(scored.doc.page_content[:50])
                ):
                    source_file = scored.doc.metadata.get("file_name", "unknown")
                    chunk_idx = scored.doc.metadata.get("chunk_index", "?")
                    break

            rag_items.append(
                f"{source_file} (chunk {chunk_idx}) — score: {packed.score:.4f}{trunc_marker}{boost_marker}"
            )

        return rag_items

    def query_databases(
        self,
        query: str,
        max_tokens: int | None = None,
        selected_dbs_override: list[str] | None = None,
        quiet: bool = False,
    ) -> str:
        """Query selected databases for relevant context using dynamic K.

        Uses greedy packing: retrieves chunks sorted by relevance and adds them
        until the token budget is exhausted.

        Args:
            query: Query string
            max_tokens: Maximum tokens to allocate for RAG context (defaults to DEFAULT_TOKEN_BUDGET)
            selected_dbs_override: Optional list of database names to query instead of current selection
            quiet: If True, suppresses logging output

        Returns:
            str: Combined context from all selected databases
        """
        self._init_components(quiet=quiet)

        if self._embeddings is None:
            return ""

        if max_tokens is None:
            max_tokens = DEFAULT_TOKEN_BUDGET

        selected_dbs: list[str] = (
            list(selected_dbs_override)
            if selected_dbs_override is not None
            else self.rag_model.get_selected_databases()
        )

        if not selected_dbs:
            self.view.set_rag_items([])
            return ""

        try:
            raw_results: list[ScoredChunk] = self._fetch_raw_results(query, selected_dbs, quiet)
            if not raw_results:
                self.view.set_rag_items([])
                return ""

            matched_files: list[tuple[str, str]] = self._extract_matching_filenames(
                query, selected_dbs
            )
            boosted_chunks: list[ScoredChunk] = self._fetch_boosted_chunks(
                query, matched_files, quiet
            )

            all_results: list[ScoredChunk] = self._deduplicate_and_merge(
                raw_results, boosted_chunks, quiet
            )
            all_results.sort(key=lambda c: (not c.is_boosted, c.score))
            all_results = self._apply_adaptive_threshold(all_results, matched_files, quiet)

            selected_chunks: list[PackedChunk] = self._greedy_pack_chunks(all_results, max_tokens)
            if not selected_chunks:
                self.view.set_rag_items([])
                return ""

            self.view.set_rag_items(self._build_rag_display_items(selected_chunks, all_results))
            return "\n\n---\n\n".join(c.content for c in selected_chunks)

        except Exception as e:
            self.view.append_logs(f"Error querying databases: {e}")
            traceback.print_exc()
            return ""

    # ------------------------------------------------------------------
    # Outline completion helpers
    # ------------------------------------------------------------------

    OUTLINE_TASK_PATTERN: str = r"- \[[x ]\]\s*(.+?)(?=\n- \[|$)"

    def _extract_outline_tasks(self, outline_text: str) -> list[str]:
        """Parse a markdown checklist and return the task strings.

        Args:
            outline_text: Markdown checklist (e.g. "- [ ] Task 1\n- [x] Task 2")

        Returns:
            Ordered list of task strings stripped of leading/trailing whitespace
        """
        matches = re.findall(self.OUTLINE_TASK_PATTERN, outline_text, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]

    def _score_tasks_semantically(
        self, tasks: list[str], story_content: str, similarity_threshold: float
    ) -> OutlineCompletionStatus:
        """Score each task against story content using embedding cosine similarity.

        Args:
            tasks: Task strings extracted from the outline
            story_content: Full story text to compare against
            similarity_threshold: Minimum cosine similarity to consider a task addressed

        Returns:
            OutlineCompletionStatus populated from semantic scoring
        """
        task_embeddings: list[list[float]] = self._embeddings.embed_documents(tasks)
        chunk_embeddings: list[list[float]] = self._embeddings.embed_documents(
            story_content.split("\n\n")
        )

        completed: list[str] = []
        pending: list[str] = []
        similarities: dict[str, float] = {}

        for task, task_emb in zip(tasks, task_embeddings):
            best_similarity: float = 0.0
            if chunk_embeddings:
                best_similarity = float(np.max(cosine_similarity([task_emb], chunk_embeddings)[0]))

            similarities[task] = best_similarity
            if best_similarity >= similarity_threshold:
                completed.append(task)
            else:
                pending.append(task)

        return OutlineCompletionStatus(
            completed_tasks=completed,
            pending_tasks=pending,
            task_similarities=similarities,
        )

    def _score_tasks_by_substring(
        self, tasks: list[str], story_content: str
    ) -> OutlineCompletionStatus:
        """Score tasks against story content using simple case-insensitive substring matching.

        Used as a fallback when scikit-learn is not available.

        Args:
            tasks: Task strings extracted from the outline
            story_content: Full story text to compare against

        Returns:
            OutlineCompletionStatus populated from substring matching
        """
        story_lower = story_content.lower()
        completed: list[str] = []
        pending: list[str] = []
        similarities: dict[str, float] = {}

        for task in tasks:
            found = task.lower() in story_lower
            similarities[task] = 1.0 if found else 0.0
            if found:
                completed.append(task)
            else:
                pending.append(task)

        return OutlineCompletionStatus(
            completed_tasks=completed,
            pending_tasks=pending,
            task_similarities=similarities,
        )

    def get_outline_completion_status(
        self, outline_text: str, story_content: str, similarity_threshold: float = 0.75
    ) -> OutlineCompletionStatus:
        """Determine which outline tasks have been addressed in the story.

        Uses semantic similarity (embeddings) to match outline tasks against story
        content. Falls back to substring matching if scikit-learn is unavailable.

        Args:
            outline_text: Markdown checklist outline (e.g. "- [ ] Task 1\n- [ ] Task 2")
            story_content: The story text to check for task completion
            similarity_threshold: Minimum cosine similarity (0.0-1.0) to consider a
                task addressed (default 0.75)

        Returns:
            OutlineCompletionStatus with completed/pending tasks and similarity scores
        """
        try:
            self._init_components()

            tasks = self._extract_outline_tasks(outline_text)
            if not tasks:
                return OutlineCompletionStatus.empty()

            if not story_content.strip():
                return OutlineCompletionStatus(
                    completed_tasks=[],
                    pending_tasks=tasks,
                    task_similarities={task: 0.0 for task in tasks},
                )

            return self._score_tasks_semantically(tasks, story_content, similarity_threshold)

        except ImportError:
            self.view.append_logs(
                "Warning: scikit-learn not available for semantic similarity. "
                "Falling back to substring matching."
            )
            tasks = self._extract_outline_tasks(outline_text)
            return self._score_tasks_by_substring(tasks, story_content)

        except Exception as e:
            self.view.append_logs(f"Error checking outline completion: {e}")
            traceback.print_exc()
            return OutlineCompletionStatus.empty()
