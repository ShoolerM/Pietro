"""Controller for RAG operations using FAISS."""

import re
import traceback
from pathlib import Path
import numpy as np
from PyQt5 import QtWidgets
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from views.progress_dialog import ProgressDialog


class RAGController:
    """Handles RAG operations with FAISS.

    Manages document ingestion, vector storage, and retrieval.
    Lazy-loads FAISS dependencies to avoid import overhead.
    """

    def __init__(self, rag_model, view):
        """Initialize RAG controller.

        Args:
            rag_model: RAGModel instance
            view: MainView instance
        """
        self.model = rag_model
        self.view = view

        # Lazy-loaded components
        self._embeddings = None
        self._text_splitter = None

        # Cache for loaded FAISS indexes {db_name: vectorstore}
        self._vectorstores = {}

        # Register model observers
        self.model.add_observer(self._on_rag_model_changed)

    def _init_components(self):
        """Lazy-initialize FAISS components."""
        if self._embeddings is not None:
            return

        try:
            self.view.append_logs("Initializing RAG components...")

            # Get database path from model
            rag_dir = Path(self.model.rag_dir)
            rag_dir.mkdir(exist_ok=True)

            self.view.append_logs(f"  RAG storage: {rag_dir}")
            self.view.append_logs("  Loading embedding model...")

            # Initialize embeddings (using a lightweight model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
            )

            self.view.append_logs("  ✓ Embedding model loaded")

            # Initialize text splitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )

            self.view.append_logs("✓ RAG components initialized\n")

        except Exception as e:
            self.view.append_logs(f"❌ Error initializing RAG components: {e}")
            traceback.print_exc()
            self.view.show_warning(
                "RAG Error",
                f"Failed to initialize RAG:\n{e}\n\nMake sure required packages are installed.",
            )

    def _get_vectorstore_path(self, db_name):
        """Get the file path for a vectorstore.

        Args:
            db_name: Name of the database

        Returns:
            Path object for the vectorstore file
        """
        rag_dir = Path(self.model.rag_dir)
        return rag_dir / f"{db_name}.faiss"

    def _load_vectorstore(self, db_name):
        """Load a FAISS vectorstore from disk.

        Args:
            db_name: Name of the database

        Returns:
            FAISS vectorstore or None
        """
        self._init_components()

        if self._embeddings is None:
            return None

        # Check cache first
        if db_name in self._vectorstores:
            return self._vectorstores[db_name]

        try:
            vectorstore_path = self._get_vectorstore_path(db_name)

            # Load existing vectorstore if it exists
            if vectorstore_path.exists():
                vectorstore = FAISS.load_local(
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

    def _load_document(self, file_path):
        """Load a document from file.

        Args:
            file_path: Path to the file

        Returns:
            str: Document text or None
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return None

            # Read text file
            if file_path.suffix.lower() in [
                ".txt",
                ".md",
                ".py",
                ".json",
                ".xml",
                ".html",
                ".css",
                ".js",
            ]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

            # Add more file types as needed
            # For now, treat unknown types as text
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except:
                return None

        except Exception as e:
            self.view.append_logs(f"Error loading document '{file_path}': {e}")
            return None

    def _on_rag_model_changed(self, event_type, data):
        """Handle Smart Model changes."""
        try:
            self.view.append_logs(f"\n🔔 Smart Model event: {event_type}")

            if event_type == "database_created":
                self.refresh_databases()
            elif event_type == "file_added":
                db_name, file_path = data
                self.view.append_logs(f"   Database: {db_name}")
                self.view.append_logs(f"   File: {file_path}")
                self._ingest_file(db_name, file_path)
            elif event_type == "database_deleted":
                self.refresh_databases()
                # Remove from cache
                if data in self._vectorstores:
                    del self._vectorstores[data]
            elif event_type == "selection_changed":
                self.refresh_databases()
        except Exception as e:
            self.view.append_logs(f"\n❌ EXCEPTION in _on_rag_model_changed:")
            self.view.append_logs(f"   Event type: {event_type}")
            self.view.append_logs(f"   Data: {data}")
            self.view.append_logs(f"   Error: {e}")
            traceback.print_exc()

    def _ingest_file(self, db_name, file_path):
        """Ingest a file into the database (legacy method for backward compatibility).

        Args:
            db_name: Name of the database
            file_path: Path to the file
        """
        # Call the new method with no progress dialog
        self._ingest_file_with_progress(db_name, file_path, None)

    def _ingest_file_with_progress(self, db_name, file_path, progress=None):
        """Ingest a file into the database with optional progress reporting.

        Args:
            db_name: Name of the database
            file_path: Path to the file
            progress: Optional ProgressDialog instance for reporting progress
        """

        def log(msg):
            """Helper to log to both console and progress dialog."""
            self.view.append_logs(msg)
            if progress:
                progress.append_detail(msg)

        try:
            log(f"\n{'=' * 80}")
            log(f"STARTING FILE INGESTION (FAISS)")
            log(f"Database: {db_name}")
            log(f"File: {file_path}")
            log(f"{'=' * 80}")

            log("Step 1: Initializing components...")
            self._init_components()

            if self._embeddings is None:
                log("❌ Embeddings model is None after initialization!")
                raise Exception("Failed to initialize embeddings model")

            log("✓ Components initialized")

            # Load document
            file_name = Path(file_path).name
            log(f"\nStep 2: Loading document '{file_name}'...")

            text = self._load_document(file_path)
            if not text:
                log(f"❌ Could not read file")
                raise Exception(f"Could not read file: {file_path}")

            log(f"✓ Loaded {len(text)} characters")

            # Split into chunks
            log(f"\nStep 3: Splitting text into chunks...")
            chunks = self._text_splitter.split_text(text)

            if not chunks:
                log(f"❌ No content extracted")
                raise Exception(f"No content extracted from file: {file_path}")

            log(f"✓ Split into {len(chunks)} chunks")

            # Create LangChain Documents with metadata
            log(f"\nStep 4: Creating documents with metadata...")
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
                log(f"  Creating new FAISS vectorstore...")
                log(f"  (This may take a moment for embedding generation...)")
                vectorstore = FAISS.from_documents(documents, self._embeddings)
                log(f"  ✓ New vectorstore created")
            else:
                # Add documents to existing vectorstore
                log(f"  Adding documents to existing vectorstore...")
                log(f"  (Generating embeddings for {len(documents)} chunks...)")
                vectorstore.add_documents(documents)
                log(f"  ✓ Documents added")

            # Save vectorstore to disk
            log(f"\nStep 6: Saving vectorstore to disk...")
            vectorstore_path = self._get_vectorstore_path(db_name)
            vectorstore.save_local(str(vectorstore_path.parent), db_name)
            log(f"  ✓ Saved to {vectorstore_path}")

            # Update cache
            self._vectorstores[db_name] = vectorstore

            log(f"\n✓ Successfully added {len(chunks)} chunks to '{db_name}'")
            log(f"{'=' * 80}\n")

            # Show success message in LLM Panel if visible
            if not progress and self.view.thinking_panel.isVisible():
                self.view.append_logs(
                    f"✓ Added '{file_name}' to '{db_name}' ({len(chunks)} chunks)\n"
                )

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

    def create_database(self):
        """Create a new RAG database."""
        db_name, ok = self.view.show_input_dialog(
            "New RAG Database", "Enter database name:"
        )

        if not ok or not db_name.strip():
            return

        success, result = self.model.create_database(db_name.strip())

        if not success:
            self.view.show_warning("Error", result)

    def add_files_to_database(self, db_name):
        """Add files to a database.

        Args:
            db_name: Name of the database
        """
        # Get files or directory from user
        selected = self.view.show_file_chooser(
            "Add Files to Database", allow_directory=True
        )

        if not selected:
            return

        # Process selected items (can be files or directories)
        text_extensions = {
            ".txt",
            ".csv",
            ".html",
            ".htm",
            ".md",
            ".markdown",
            ".json",
            ".jsonl",
            ".pdf",
            ".doc",
            ".docx",
            ".tex",
            ".rtf",
            ".xml",
        }
        file_paths = []

        for selection_str in selected:
            selection = Path(selection_str)
            if selection.is_dir():
                # Recursively find all text files in directory
                for file_path in selection.rglob("*"):
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in text_extensions
                    ):
                        file_paths.append(str(file_path))
            elif selection.is_file():
                # Add file directly
                file_paths.append(str(selection))

        if not file_paths:
            QtWidgets.QMessageBox.warning(
                self.view,
                "No Text Files Found",
                f"No text files with recognized extensions were found in the selected items.\n\nSupported: {' '.join(sorted(text_extensions))}",
            )
            return

        # Show progress dialog
        progress = ProgressDialog(f"Adding Files to {db_name}", self.view)
        progress.set_message(f"Adding {len(file_paths)} file(s) to database...")
        progress.set_progress(0, len(file_paths))
        progress.show()

        success_count = 0
        error_count = 0

        for idx, file_path in enumerate(file_paths, 1):
            file_name = Path(file_path).name
            progress.set_message(f"Processing {idx}/{len(file_paths)}: {file_name}")
            progress.append_detail(
                f"\n[{idx}/{len(file_paths)}] Processing: {file_name}"
            )

            # Add file metadata to model
            success, result = self.model.add_file_to_database(db_name, file_path)
            if not success:
                progress.append_detail(f"  ❌ Failed to add metadata: {result}")
                error_count += 1
                continue

            progress.append_detail(f"  ✓ Added to database metadata")

            # Ingest file (vectorize and store)
            progress.append_detail(f"  → Ingesting file...")
            try:
                self._ingest_file_with_progress(db_name, file_path, progress)
                progress.append_detail(f"  ✓ Successfully ingested")
                success_count += 1
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                progress.append_detail(f"  ❌ {error_type}: {error_msg}")
                # Also log the full traceback to console for debugging
                self.view.append_logs(f"\n{'=' * 80}")
                self.view.append_logs(f"ERROR ingesting {file_name}:")
                traceback.print_exc()
                self.view.append_logs(f"{'=' * 80}\n")
                error_count += 1

            progress.set_progress(idx)

        # Show summary
        if error_count == 0:
            progress.finish_success(f"Successfully added {success_count} file(s)")
        else:
            msg = f"Added {success_count} file(s), {error_count} failed"
            progress.set_message(msg)
            progress.append_detail(f"\n⚠️  {msg}")
            progress.show_cancel_button()
            progress.cancel_button.setText("Close")
            progress.exec_()

        # Refresh the database list
        self.refresh_databases()

    def refresh_databases(self):
        """Refresh the RAG databases list in view."""
        databases = self.model.get_databases()
        self.view.load_rag_databases(databases)

    def toggle_database(self, db_name):
        """Toggle database selection.

        Args:
            db_name: Name of the database
        """
        try:
            self.view.append_logs(f"Toggling database: {db_name}")
            self.model.toggle_database_selection(db_name)
            self.view.append_logs(
                f"  Success - Selected databases: {self.model.get_selected_databases()}"
            )
        except Exception as e:
            self.view.append_logs(f"Error toggling database '{db_name}': {e}")
            traceback.print_exc()

    def delete_database(self, db_name):
        """Delete a database and its FAISS vectorstore.

        Args:
            db_name: Name of the database to delete
        """
        try:
            self.view.append_logs(f"\n🗑️  Deleting database: {db_name}")

            # Delete from model first
            success, message = self.model.delete_database(db_name)

            if not success:
                self.view.append_logs(f"  ❌ {message}")
                self.view.show_warning("Delete Failed", message)
                return

            self.view.append_logs(f"  ✓ Removed from database list")

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
                self.view.append_logs(
                    f"  ⚠️  Could not delete vectorstore files: {file_error}"
                )

            # Remove from cache
            if db_name in self._vectorstores:
                del self._vectorstores[db_name]
                self.view.append_logs(f"  ✓ Removed from vectorstore cache")

            self.view.append_logs(f"✓ Database '{db_name}' deleted successfully\n")

        except Exception as e:
            self.view.append_logs(f"❌ Error deleting database '{db_name}': {e}")
            traceback.print_exc()
            self.view.show_warning("Delete Error", f"Failed to delete database:\n{e}")

    def query_databases(self, query, max_tokens=None):
        """Query selected databases for relevant context using dynamic K.

        Uses greedy packing: retrieves chunks sorted by relevance and adds them
        until the token budget is exhausted (~60-70% of available context).

        Args:
            query: Query string
            max_tokens: Maximum tokens to allocate for RAG context (defaults to 2000)

        Returns:
            str: Combined context from all selected databases
        """
        from models.story_model import StoryModel

        self._init_components()

        if self._embeddings is None:
            return ""

        # Default token budget if not specified
        if max_tokens is None:
            max_tokens = 2000

        selected_dbs = self.model.get_selected_databases()

        if not selected_dbs:
            return ""

        self.view.append_logs(f"\n{'=' * 80}")
        self.view.append_logs(
            f"RAG QUERY: {query[:100]}{'...' if len(query) > 100 else ''}"
        )
        self.view.append_logs(f"Selected Databases: {', '.join(selected_dbs)}")
        self.view.append_logs(f"Token Budget: {max_tokens:,} tokens")
        self.view.append_logs(f"Adaptive Threshold: Enabled (2.5x best score)")
        self.view.append_logs(f"{'=' * 80}")

        try:
            all_results = []

            # Fetch initial batch (20 per database) to have pool for greedy selection
            initial_k = 20

            # Query each selected database
            for db_name in selected_dbs:
                vectorstore = self._load_vectorstore(db_name)
                if not vectorstore:
                    self.view.append_logs(
                        f"  ⚠️  Database '{db_name}' not found or empty"
                    )
                    continue

                try:
                    # Query vectorstore with similarity search
                    docs_with_scores = vectorstore.similarity_search_with_score(
                        query, k=initial_k
                    )

                    if docs_with_scores:
                        for doc, score in docs_with_scores:
                            all_results.append((doc, score))

                except Exception as e:
                    self.view.append_logs(f"Error querying database '{db_name}': {e}")
                    traceback.print_exc()

            if not all_results:
                self.view.append_logs(f"\n⚠️  No results found in any database")
                self.view.append_logs(f"{'=' * 80}\n")
                return ""

            # Sort by relevance (lower score = more relevant)
            all_results.sort(key=lambda x: x[1])

            # Adaptive threshold: filter out chunks significantly worse than best result
            if all_results:
                best_score = all_results[0][1]
                # Use configurable percentage threshold (default 5%)
                threshold_multiplier = 1 + self.model.score_variance_threshold
                adaptive_threshold = best_score * threshold_multiplier

                filtered_results = [
                    (doc, score)
                    for doc, score in all_results
                    if score <= adaptive_threshold
                ]

                if len(filtered_results) < len(all_results):
                    self.view.append_logs(
                        f"Adaptive filter: kept {len(filtered_results)}/{len(all_results)} chunks "
                        f"(threshold: {adaptive_threshold:.4f}, variance: {self.model.score_variance_threshold:.1%})"
                    )

                # Always keep at least top 3 results regardless of threshold
                all_results = filtered_results if filtered_results else all_results[:3]
                if len(all_results) < 3 and len(all_results) < len(filtered_results):
                    all_results = all_results[:3]

            # Greedy packing: add chunks until token budget exhausted
            selected_chunks = []
            total_tokens = 0

            for doc, score in all_results:
                chunk_tokens = StoryModel.estimate_token_count(doc.page_content)

                # Check if adding this chunk would exceed budget
                if total_tokens + chunk_tokens > max_tokens:
                    # Try to fit partial chunk if there's meaningful space left
                    remaining = max_tokens - total_tokens
                    if (
                        remaining > 100
                    ):  # Only bother if we have room for meaningful content
                        # Truncate chunk to fit
                        chars_to_keep = remaining * 4  # ~4 chars per token
                        truncated_content = doc.page_content[:chars_to_keep] + "..."
                        selected_chunks.append(
                            (truncated_content, score, chunk_tokens, True)
                        )
                        total_tokens += remaining
                    break

                selected_chunks.append((doc.page_content, score, chunk_tokens, False))
                total_tokens += chunk_tokens

            if not selected_chunks:
                self.view.append_logs(f"\n⚠️  No chunks fit within token budget")
                self.view.append_logs(f"{'=' * 80}\n")
                return ""

            self.view.append_logs(f"\n{'=' * 80}")
            self.view.append_logs(
                f"✓ Selected {len(selected_chunks)} chunks ({total_tokens:,}/{max_tokens:,} tokens)"
            )
            self.view.append_logs(
                f"  Dynamic K: packed {len(selected_chunks)} from {len(all_results)} candidates"
            )

            # Log selected chunks
            for idx, (content, score, tokens, truncated) in enumerate(
                selected_chunks, 1
            ):
                trunc_marker = " [TRUNCATED]" if truncated else ""
                source_file = "unknown"
                chunk_idx = "?"
                # Try to get metadata from original doc
                for doc, s in all_results:
                    if doc.page_content == content or (
                        truncated and content.startswith(doc.page_content[:50])
                    ):
                        source_file = doc.metadata.get("file_name", "unknown")
                        chunk_idx = doc.metadata.get("chunk_index", "?")
                        break
                self.view.append_logs(
                    f"  [{idx}] {source_file} (chunk {chunk_idx}): {tokens:,} tokens, score: {score:.4f}{trunc_marker}"
                )

            self.view.append_logs(f"{'=' * 80}\n")

            # Combine results
            context = "\n\n---\n\n".join(
                [content for content, _, _, _ in selected_chunks]
            )

            return context

        except Exception as e:
            self.view.append_logs(f"Error querying databases: {e}")
            traceback.print_exc()
            return ""

    def get_outline_completion_status(
        self, outline_text, story_content, similarity_threshold=0.75
    ):
        """Determine which outline tasks have been addressed in the story.

        Uses semantic similarity (embeddings) to match outline tasks against story content.
        Each task in the outline (markdown checklist format) is compared to the story
        to see if semantically similar content exists.

        Args:
            outline_text: Markdown checklist outline (e.g., "- [ ] Task 1\n- [ ] Task 2")
            story_content: The story text to check for task completion
            similarity_threshold: Minimum cosine similarity (0.0-1.0) to consider task addressed (default 0.75)

        Returns:
            dict: {
                'all_completed': bool - True if all tasks are addressed,
                'completed_tasks': list[str] - Tasks found in story,
                'pending_tasks': list[str] - Tasks not yet in story,
                'completion_ratio': float - Fraction of tasks completed (0.0-1.0),
                'task_similarities': dict[str, float] - Each task's best match similarity score
            }
        """
        try:
            self._init_components()

            # Extract tasks from markdown checklist
            # Format: - [ ] Task text or - [x] Task text
            task_pattern = r"- \[[x ]\]\s*(.+?)(?=\n- \[|$)"
            matches = re.findall(task_pattern, outline_text, re.DOTALL | re.IGNORECASE)
            tasks = [m.strip() for m in matches if m.strip()]

            if not tasks:
                return {
                    "all_completed": False,
                    "completed_tasks": [],
                    "pending_tasks": [],
                    "completion_ratio": 0.0,
                    "task_similarities": {},
                }

            if not story_content.strip():
                return {
                    "all_completed": False,
                    "completed_tasks": [],
                    "pending_tasks": tasks,
                    "completion_ratio": 0.0,
                    "task_similarities": {task: 0.0 for task in tasks},
                }

            # Get embeddings for tasks
            task_embeddings = self._embeddings.embed_documents(tasks)

            # Split story into chunks for comparison
            chunks = story_content.split("\n\n")
            chunk_embeddings = self._embeddings.embed_documents(chunks)

            # Compute similarity between each task and all story chunks
            completed_tasks = []
            task_similarities = {}

            for task, task_emb in zip(tasks, task_embeddings):
                # Find best match with any chunk
                best_similarity = 0.0

                if chunk_embeddings:
                    similarities = cosine_similarity([task_emb], chunk_embeddings)[0]
                    best_similarity = float(np.max(similarities))

                task_similarities[task] = best_similarity

                # Task is complete if similarity exceeds threshold
                if best_similarity >= similarity_threshold:
                    completed_tasks.append(task)

            all_completed = len(completed_tasks) == len(tasks)
            pending_tasks = [t for t in tasks if t not in completed_tasks]

            return {
                "all_completed": all_completed,
                "completed_tasks": completed_tasks,
                "pending_tasks": pending_tasks,
                "completion_ratio": len(completed_tasks) / len(tasks) if tasks else 0.0,
                "task_similarities": task_similarities,
            }

        except ImportError:
            self.view.append_logs(
                "Warning: scikit-learn not available for semantic similarity. Falling back to basic completion."
            )
            # Fallback: simple substring matching
            completed_tasks = []
            task_pattern = r"- \[[x ]\]\s*(.+?)(?=\n- \[|$)"
            matches = re.findall(task_pattern, outline_text, re.DOTALL | re.IGNORECASE)
            tasks = [m.strip() for m in matches if m.strip()]

            for task in tasks:
                if task.lower() in story_content.lower():
                    completed_tasks.append(task)

            return {
                "all_completed": len(completed_tasks) == len(tasks),
                "completed_tasks": completed_tasks,
                "pending_tasks": [t for t in tasks if t not in completed_tasks],
                "completion_ratio": len(completed_tasks) / len(tasks) if tasks else 0.0,
                "task_similarities": {
                    task: 1.0 if task.lower() in story_content.lower() else 0.0
                    for task in tasks
                },
            }
        except Exception as e:
            self.view.append_logs(f"Error checking outline completion: {e}")
        except Exception as e:
            print(f"Error checking outline completion: {e}")
            traceback.print_exc()
            return {
                "all_completed": False,
                "completed_tasks": [],
                "pending_tasks": [],
                "completion_ratio": 0.0,
                "task_similarities": {},
            }
