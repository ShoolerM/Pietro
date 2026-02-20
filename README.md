# Pietro - AI-Powered Story Builder

An intelligent story writing application that combines **LangChain**, **PyQt5**, and **RAG (Retrieval-Augmented Generation)** to help writers create long-form narratives with AI assistance.

## Features

### Core Functionality
- **Interactive Story Generation**: Collaborate with LLMs to write stories with real-time streaming
- **Inline Editing**: Highlight selected text, and with Ctrl + R, or via right click menu, ask the LLM to re-word the selection in a context-aware manner.
- **Ask Mode**: Chat/Q&A mode for questions about Pietro and your selected RAG Database(s)
- **Write Mode**: Standard story continuation mode. LLM will use RAG, Notes, Story Summarization and write as much as it sees fit.
- **Planning Mode**: Conversational interface to develop story outlines with the AI, then generate content following structured plot points
- **Story Mode**: Automatically generate stories or chunks of stories with iterative RAG context refresh and intelligent summarization.
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
   git clone https://github.com/ShoolerM/Pietro
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
- **Inference → Server Settings...**
- Enter your API base URL components (IP + port)
- Optional: set an API key if your server requires it
- Select your model from the dropdown
- Adjust temperature and context limit as needed

### 2. Choose a Mode
- **Ask**: Q&A / chat (no story writes)
- **Write**: Continue the story in the Story panel
- **Planning**: Build outlines and then generate from them
- **Story Mode**: Auto-build story with continuous RAG

### 3. Basic Story Writing (Write Mode)
1. Enter your story prompt in the **User Input** field
2. Add optional notes or supplemental prompts
3. Click **Send** or press **Ctrl+Enter**
4. Story generates with real-time streaming

### 4. Planning Mode
1. **Tools → Planning Mode** (or **Ctrl+P**)
2. Chat with the AI to develop your story outline
3. Request an outline: *"Create a story outline"*
4. Click **Start Writing** to generate story following the outline
5. AI generates content for each plot point sequentially

### 5. Story Mode (Auto-Build with RAG)
1. Create a knowledge base (**RAG → Manage Databases**)
2. Enable **Story Mode** in the LLM panel
3. Click **Send** with your initial prompt
4. AI automatically generates story chunks with continuous RAG queries

### 6. Create Knowledge Bases
1. **RAG → Manage Databases**
2. Click **Create New Database**
3. Add documents (text files, PDFs, or folders)
4. Knowledge base is indexed with FAISS for semantic search

### 7. Ask Mode Details
- Ask mode uses a hidden README knowledge base by default to answer questions about the app.
- Ask mode also includes any RAG databases you have selected in the RAG panel.
- Ask mode responses render markdown in the LLM panel.
- Edit the Ask prompt in **Prompts → Ask Prompt** to change assistant behavior.

### 8. Attachments and Drag & Drop
- Use the **+** button to attach files or images.
- Drag & drop images onto the LLM panel to attach them.
- Large images are skipped with a warning.

## Configuration

### Model Profiles
Automatically saves per-model settings:
- Context limit
- Temperature
- System prompt
- Summarization prompts
- Ask prompt

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
- Try different search queries

### Memory Issues
- Reduce context limit in model settings
- Clear story history periodically

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
