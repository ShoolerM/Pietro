# Planning mode system prompt is defined in models/baml_src/outline.baml
# (PlanningSystemPrompt template string) and is no longer stored here.


# Default summary prompt template
DEFAULT_SUMMARY_PROMPT = (
    "TASK: Create a detailed but condensed summary of this story.\n\n"
    "REQUIREMENTS:\n"
    "- List ALL main characters with their names, roles, and key personality traits\n"
    "- Describe character relationships and dynamics between them\n"
    "- Include ALL significant plot points in chronological order\n"
    "- Preserve important dialogue or quotes that define characters\n"
    "- Document world-building: locations, events, themes, etc.\n"
    "- Note any ongoing conflicts, or unresolved plot threads\n"
    "- Mention character motivations and goals\n"
    "- Include relevant backstory and historical context\n"
    "- Keep the summary detailed but aim for 30-40% of original length\n"
    "- Write in present tense, organized by topic (characters, plot, setting, etc.)\n\n"
)

# Default notes prompt template
DEFAULT_NOTES_PROMPT = (
    "TASK: Generate a structured notes section for the current scene.\n\n"
    "REQUIREMENTS:\n"
    "- List ALL current characters in the scene with their names and roles\n"
    "- For each character, include: motivations, goals, current emotional state\n"
    "- Document each character's clothing or appearance details\n"
    "- Describe relationships and dynamics between characters in the scene\n"
    "- State what each character was doing last (e.g., sitting, talking, fighting, etc.)\n"
    "- Include any relevant physical location details or environmental context\n"
    "- Note any props, objects, or items relevant to the scene\n"
    "- Keep it concise but comprehensive - aim for 200-400 words\n"
    "- Organize by character for clarity\n\n"
)

DEFAULT_ASK_PROMPT = """You are a helpful assistant. Be succinct. Use the information available to answer the user's questions clearly and accurately.
If you are unsure or the answer is not available, say you don't know and ask a clarifying question. Do not invent details."""
