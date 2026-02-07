"""RAG Model for managing ChromaDB databases and document retrieval."""
import os
import json
from pathlib import Path
from base.observable import Observable


class RAGModel(Observable):
    """Model for RAG database management.
    
    Manages ChromaDB databases, their metadata, and file associations.
    Actual vector operations are handled by RAGController to avoid
    heavy dependencies in the model layer.
    """
    
    def __init__(self, rag_dir='rag_databases'):
        """Initialize RAG model.
        
        Args:
            rag_dir: Directory to store RAG databases
        """
        super().__init__()
        self.rag_dir = Path(rag_dir)
        self.rag_dir.mkdir(exist_ok=True)
        
        # Metadata file to track databases
        self.metadata_file = self.rag_dir / 'databases.json'
        self._databases = {}
        self._selected_databases = set()
        
        # RAG query settings - will be loaded from file
        self.similarity_threshold = 0.0  # Default: no filtering
        self.max_docs = 3  # Default: retrieve 3 chunks per database
        self.max_chunks = 10  # Default: auto-build chunks
        self.summary_chunk_size = 1500  # Default: max raw tokens for summarization
        
        # Load databases and settings
        self._load_data()
    
    def _load_data(self):
        """Load database metadata and settings from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if this is the old format (databases at root) or new format
                if 'databases' in data:
                    # New format with separate databases and settings keys
                    self._databases = data.get('databases', {})
                    settings = data.get('settings', {})
                    self.similarity_threshold = settings.get('similarity_threshold', 0.0)
                    self.max_docs = settings.get('max_docs', 3)
                    self.max_chunks = settings.get('max_chunks', 10)
                    self.summary_chunk_size = settings.get('summary_chunk_size', 1500)

                    # Persist missing settings keys for older files
                    if any(
                        key not in settings
                        for key in ['similarity_threshold', 'max_docs', 'max_chunks', 'summary_chunk_size']
                    ):
                        self._save_data()
                    
                    # Load selected databases
                    selected_list = data.get('selected_databases', [])
                    self._selected_databases = set(selected_list)
                else:
                    # Old format - databases at root level, no settings key
                    # Migrate to new format
                    print("Detected old database format, migrating...")
                    self._databases = data
                    self.similarity_threshold = 0.0
                    self.max_docs = 3
                    self.max_chunks = 10
                    self.summary_chunk_size = 1500
                    # Save in new format immediately
                    self._save_data()
                    print("Migration complete!")
                
                print(
                    f"Loaded RAG settings: threshold={self.similarity_threshold}, "
                    f"max_docs={self.max_docs}, max_chunks={self.max_chunks}, "
                    f"summary_chunk_size={self.summary_chunk_size}"
                )
                if self._selected_databases:
                    print(f"Loaded selected databases: {list(self._selected_databases)}")
            except Exception as e:
                print(f"Error loading database metadata: {e}")
                self._databases = {}
    
    def _save_data(self):
        """Save database metadata and settings to file."""
        try:
            data = {
                'databases': self._databases,
                'settings': {
                    'similarity_threshold': self.similarity_threshold,
                    'max_docs': self.max_docs,
                    'max_chunks': self.max_chunks,
                    'summary_chunk_size': self.summary_chunk_size
                },
                'selected_databases': list(self._selected_databases)
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving database metadata: {e}")
    
    def get_databases(self):
        """Get list of available databases.
        
        Returns:
            list: List of (db_name, file_count, is_selected) tuples
        """
        result = []
        for db_name, metadata in self._databases.items():
            file_count = len(metadata.get('files', []))
            is_selected = db_name in self._selected_databases
            result.append((db_name, file_count, is_selected))
        return result
    
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
            'files': [],
            'path': str(self.rag_dir / db_name)
        }
        
        # Create directory for database
        db_path = self.rag_dir / db_name
        db_path.mkdir(exist_ok=True)
        
        self._save_data()
        self.notify_observers('database_created', db_name)
        
        return True, db_name
    
    def add_file_to_database(self, db_name, file_path):
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
        
        if file_path_str in self._databases[db_name]['files']:
            return False, f"File already added to database"
        
        self._databases[db_name]['files'].append(file_path_str)
        self._save_data()
        self.notify_observers('file_added', (db_name, file_path_str))
        
        return True, file_path_str
    
    def get_database_path(self, db_name):
        """Get the filesystem path for a database.
        
        Args:
            db_name: Name of the database
            
        Returns:
            str: Path to database directory, or None if not found
        """
        if db_name in self._databases:
            return self._databases[db_name]['path']
        return None
    
    def get_database_files(self, db_name):
        """Get list of files in a database.
        
        Args:
            db_name: Name of the database
            
        Returns:
            list: List of file paths
        """
        if db_name in self._databases:
            return self._databases[db_name]['files'].copy()
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
        self.notify_observers('selection_changed', list(self._selected_databases))
    
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
        
        self.notify_observers('database_deleted', db_name)
        
        return True, f"Database '{db_name}' deleted"
    
    def set_similarity_threshold(self, threshold):
        """Set the similarity threshold for RAG queries.
        
        Args:
            threshold: Float between 0.0 and 1.0
        """
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        self._save_data()  # Save settings to persist
        print(f"RAG similarity threshold set to: {self.similarity_threshold}")
    
    def set_max_docs(self, max_docs):
        """Set the maximum number of documents to retrieve per database.
        
        Args:
            max_docs: Integer between 1 and 20
        """
        self.max_docs = max(1, min(20, max_docs))
        self._save_data()  # Save settings to persist
        print(f"RAG max documents set to: {self.max_docs}")

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

