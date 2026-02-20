# Pietro Feature Guide (Ask Mode Reference)

This guide is indexed by Ask mode to answer questions about how to use the app.

## Layout Overview

- **Left pane (top): Story panel**
  - Main story editor with tabs for Story and opened files.
- **Left pane (bottom): Utilities**
  - Tabs: Notes, Supplemental, System, RAG, Logs.
- **Right pane: LLM Panel**
  - Chat history, RAG items used, prompt box, and model/mode controls.

## Menu Bar

### File
- **Load...**: Open a story file.
- **Save** / **Save As...**: Save the current story tab.

### Inference
- **Server Settings...**: Set inference IP/port and optional API key.

### Prompts
- **Summarization Prompt**: Edit the summary template used for long‑story summarization.
- **Notes Prompt**: Edit the notes template used before story generation (Write/Story Mode).
- **Ask Prompt**: Edit the system prompt used for Ask mode responses.

### RAG
- **RAG Settings...**: Configure chunk count, summarization chunk size, and score threshold.

### Settings
- **General...**: Toggle Auto Notes and Render Markdown.
- **Model Settings...**: Set context limit.

## Modes (LLM Panel dropdown)

- **Ask**: Chat/Q&A only. Does not write to Story panel. Uses hidden README + guide RAG and any selected RAG databases.
- **Write**: Standard story continuation. Writes to Story panel.
- **Planning**: Outline‑first workflow. Chat to build outline, then “Start Writing.”
- **Story Mode**: Auto‑build story with continuous RAG and chunk generation. The number of chunks written at each iteration of Story Mode is editable in the RAG settings menu.

## LLM Panel

- **Chat history**: Shows user and AI messages. Ask mode responses render Markdown.
- **RAG Items**: Collapsible list of RAG chunks used for the most recent query.
- **Prompt box**: Type message; Enter to send, Shift+Enter for newline.
- **Buttons**: “+” attaches files, “>” sends.
- **Drag & drop**: Drop images anywhere on the LLM panel to attach.
- **Model controls**: Mode dropdown, model dropdown, and refresh button.

## Story Panel (Right‑click context menu)

- **Send Prompt**: Sends the current prompt.
- **Undo**: Undo last change.
- **Stop Generation**: Stop streaming.
- **Clear Story**: Clears story content.
- **Update Selected Text**: Rewrite a selected block with instructions.
- **Update Summary**: Regenerate summary.
- **Summarize Prompts: ON/OFF**: Toggle prompt summarization.
- **Story Mode: ON/OFF**: Toggle Story Mode (auto‑build behavior).
- **Show/Hide LLM Panel**: Toggle the right panel.

## Utilities Panel

### Notes tab
- Stores author notes used in story generation.
- When Auto Notes is enabled, notes are regenerated automatically.

### Supplemental tab
- Manage reusable prompt fragments (characters, style, world lore). Appended to user prompts when sent to the LLM.
- **Refresh** to reload, **Add** to create a new file.

### System tab
- Manage system prompt files that affect model behavior. System prompts are sent in all modes.

### RAG tab
- Create/delete databases and add files/folders to a database.
- Checkboxes select active databases for RAG queries.

### Logs tab
- Shows diagnostic logs and status messages.

## Attachments

- Attach files via **+** in the LLM Panel, or drag and drop.
- Images are supported (png/jpg/jpeg). Large images may be skipped with a warning.
- Attachments are appended to the request context.
- Will alert the user if the model does not support vision.

## Ask Mode RAG Behavior
- Ask mode uses a hidden RAG database built from README + this guide.
- Ask mode also uses any user‑selected RAG databases.
- If README or this guide changes, the hidden Ask database re‑indexes automatically.
