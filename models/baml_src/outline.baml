// PietroLLM is overridden at call time via ClientRegistry.
// This stub is required so the BAML runtime can validate references at init.
client<llm> PietroLLM {
  provider openai-generic
  options {
    base_url "http://localhost:1234/v1"
    api_key "lm-studio"
    model "placeholder"
  }
}

// ── Data classes ─────────────────────────────────────────────────────────────

class OutlinePlotPoint {
  description string @alias("event")
  details string?
  completed bool
}

// Returned by Plan. action is either "chat" or "outline".
// "chat"    → message is populated, plot_points is null/empty.
// "outline" → plot_points is populated, message is null/empty.
class PlanningResponse {
  action string
  message string?
  plot_points OutlinePlotPoint[]?
}

// ── System prompt ─────────────────────────────────────────────────────────────

template_string PlanningSystemPrompt(context: string) #"
You are a creative writing assistant helping plan a story outline.

Your role is to:
1. Help the user develop their story idea through conversation
2. Generate or refine a structured outline when the user asks for one or when you have enough information
3. Mark plot points as completed when they describe events already written in existing story content

IMPORTANT RULES:
- Have discussions and ask clarifying questions when you need more information
- Generate an outline when explicitly asked, or when you have a clear story concept
- "completed": true means the event is already written in the existing story text
- "completed": false means it still needs to be written, or is a new addition
- If NO story content exists, all plot points must be completed: false
- Plot points must describe specific narrative events — not themes, metadata, or abstractions
- When refining an existing outline, build on the existing plot points rather than replacing them

{{ context }}
"#

// ── Functions ─────────────────────────────────────────────────────────────────

/// Agentic planning turn: the LLM decides whether to have a conversation or
/// generate/update an outline. Returns a PlanningResponse with action="chat"
/// or action="outline" accordingly.
function Plan(context: string, conversation_history: string, user_message: string) -> PlanningResponse {
  client PietroLLM
  prompt #"
    {{ _.role("system") }}
    {{ PlanningSystemPrompt(context) }}

    {{ _.role("user") }}
    {% if conversation_history %}
    Previous conversation:
    {{ conversation_history }}

    {% endif %}
    {{ user_message }}

    Respond with a single JSON object. No text outside the JSON. No markdown fences.

    If you want to discuss or need more information:
    {"action": "chat", "message": "your response here"}

    If you are generating or updating an outline:
    {"action": "outline", "plot_points": [{"event": "short title for this section", "details": "A few sentences describing what specifically happens: which characters are involved, what they do, and how it moves the story forward.", "completed": false}]}

    Rules when action is "outline":
    - If the user specified a count (e.g. "3 sections", "5 plot points"), the array MUST have EXACTLY that many items.
    - If an existing outline appears in the context, select/trim/merge from those items. Do not invent replacements.
    - "completed": true only for events already written in the story.
  "#
}

/// Check whether a conversational exchange warrants an outline update.
/// Returns an updated list of plot points, or null if no change is needed.
function MaybeUpdateOutline(current_outline: string, user_message: string, assistant_reply: string) -> OutlinePlotPoint[]? {
  client PietroLLM
  prompt #"
    {{ _.role("system") }}
    You are a story outline manager. Given the current outline and a recent
    conversational exchange, decide whether the outline needs to be updated.

    Rules:
    - Return an updated list of plot points ONLY if something meaningful changed
      (e.g. new plot point added, completion status changed, a point rewritten).
    - If nothing changed, return null.
    - "completed": true means the event is already written in the story.
    - "completed": false means not yet written.
    - Do NOT add speculative points; only record changes explicitly discussed.

    CURRENT OUTLINE:
    {{ current_outline }}

    {{ _.role("user") }}
    The user just said:
    {{ user_message }}

    The assistant replied:
    {{ assistant_reply }}

    Should the outline be updated? If yes, output the full updated plot point list
    as a raw JSON array — no markdown fences, no extra text:
    [{"event": "short title", "details": "A few sentences describing what happens in this section.", "completed": false}]
    If no changes are needed, return null.
  "#
}