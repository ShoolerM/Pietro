# Pietro - AI-Powered Story Builder

An intelligent story writing application that combines **LangChain**, **PyQt5**, and **RAG (Retrieval-Augmented Generation)** to help writers create long-form narratives with AI assistance.

## Features

### Core Functionality
- **Interactive Story Generation**: Collaborate with LLMs to write stories with real-time streaming
- **Planning Mode**: Conversational interface to develop story outlines with the AI, then generate content following structured plot points
- **Auto-Build Mode**: Automatically generate complete stories with iterative RAG context refresh and intelligent summarization
- **Markdown Rendering**: View stories in formatted markdown or edit in plain text
- **Multi-Model Support**: Compatible with OpenAI and OpenAI-compatible APIs (LM Studio, Ollama, etc.)

### Knowledge Management
- **RAG Integration**: Query multiple FAISS-based knowledge bases to ground story generation in specific contexts
- **Document Ingestion**: Import text files, PDFs, and folders to create searchable knowledge bases
- **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **Dynamic Context**: Automatically queries relevant information based on story content

### Smart Context Management
- **Hierarchical Summarization**: Automatically compresses older content to maintain context windows
- **Rolling Context Window**: Keeps recent content while summarizing older sections
- **Auto-Generated Notes**: AI creates scene notes from story content to maintain continuity
- **Configurable Prompts**: Customize system prompts, notes templates, and summarization instructions

### Writing Tools
- **Supplemental Prompts**: Define multiple reusable prompt sections (character profiles, world-building, style guides)
- **System Prompt Editor**: Full control over LLM behavior and instructions
- **Author's Notes**: Maintain story-specific notes that inform generation
- **History Management**: Track story versions and revisions

### Advanced Features
- **Model Profiles**: Save per-model configurations (context limits, temperature, system prompts)
- **Structured Output**: Uses JSON schema for reliable outline generation
- **Thinking Panel**: View LLM reasoning and generation metadata
- **Token Tracking**: Monitor token usage and context limits in real-time
- **Font Scaling**: Adjustable UI text size with zoom controls

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Option 1: Using Poetry (Recommended)

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Pietro
   ```

3. **Install dependencies and activate environment**
   ```bash
   poetry install
   poetry shell
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

### Option 2: Using pip

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShoolerM/Pietro
   cd Pietro
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Quick Start

### 1. Configure LLM Connection
- **File → Settings → General Settings**
- Enter your API base URL (e.g., `http://localhost:1234/v1` for Your inference server)
- Select your model from the dropdown
- Adjust temperature and context limit as needed

### 2. Basic Story Writing
1. Enter your story prompt in the **User Input** field
2. Add optional notes or supplemental prompts
3. Click **Send** or press **Ctrl+Enter**
4. Story generates with real-time streaming

### 3. Planning Mode
1. **Tools → Planning Mode** (or **Ctrl+P**)
2. Chat with the AI to develop your story outline
3. Request an outline: *"Create a story outline"*
4. Click **Start Writing** to generate story following the outline
5. AI generates content for each plot point sequentially

### 4. Auto-Build with RAG
1. Create a knowledge base (**Tools → RAG: Manage Databases**)
2. Enable **Build with RAG** (right-click story panel)
3. Click **Send** with your initial prompt
4. AI automatically generates story chunks with continuous RAG queries

### 5. Create Knowledge Bases
1. **Tools → RAG: Manage Databases**
2. Click **Create New Database**
3. Add documents (text files, PDFs, or folders)
4. Knowledge base is indexed with FAISS for semantic search

## Configuration

### Model Profiles
Automatically saves per-model settings:
- Context limit
- Temperature
- System prompt
- Notes template
- Summarization prompts

## Architecture

Pietro follows an **MVC (Model-View-Controller)** architecture:

```
Pietro/
├── base/              # Observable pattern base classes
├── models/            # Data models (Story, Settings, RAG, LLM, etc.)
├── views/             # PyQt5 UI components
├── controllers/       # Business logic and coordination
│   ├── main_controller.py
│   ├── planning_controller.py
│   ├── llm_controller.py
│   ├── rag_controller.py
│   └── prompt_controller.py
├── settings/          # Configuration files
├── rag_databases/     # FAISS knowledge bases
└── main.py           # Application entry point
```

## Dependencies

- **PyQt5** - Desktop GUI framework
- **LangChain** - LLM orchestration and chains
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **Markdown** - Markdown to HTML conversion

See `requirements.txt` or `pyproject.toml` for complete list.

## Troubleshooting

### LLM Connection Issues
- Verify API base URL is correct
- Check that your LLM server is running
- Ensure API endpoint is OpenAI-compatible
- Test with curl: `curl http://localhost:1234/v1/models`

### RAG Not Finding Documents
- Verify documents were ingested successfully
- Check that knowledge base is activated
- Ensure sufficient `max_docs` setting in RAG model
- Try different search queries

### Memory Issues
- Reduce context limit in model settings
- Enable prompt summarization
- Clear story history periodically
- Reduce RAG `max_docs` parameter

### Generation Stops Unexpectedly
- Check LLM server logs for errors
- Verify context limit isn't exceeded
- Ensure model supports streaming
- Try reducing temperature


### Export Requirements
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

**Note**: This application requires a running LLM server (OpenAI API, LM Studio, Ollama, etc.) to function.
