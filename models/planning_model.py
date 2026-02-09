"""Planning Mode Data Model

Manages story planning state, outline structure, and conversation history.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from base.observable import Observable


class OutlinePlotPoint(BaseModel):
    """A single plot point in a story outline.

    Attributes:
        description: Specific narrative event or action
        completed: Whether this plot point has been written in the story
    """

    description: str = Field(
        ...,
        description="A specific narrative event or action that happens in the story",
    )
    completed: bool = Field(
        default=False,
        description="True if this plot point has already been written, False if it still needs to be written",
    )


class StoryOutline(BaseModel):
    """Complete story outline with plot points and optional discussion.

    Attributes:
        plot_points: List of narrative events in sequence
        discussion: Optional explanatory text or questions about the outline
    """

    plot_points: List[OutlinePlotPoint] = Field(
        default_factory=list,
        description="Sequential list of plot points that make up the story outline",
    )
    discussion: Optional[str] = Field(
        None,
        description="Optional discussion, questions, or explanatory text about the outline (but not the outline itself)",
    )


class PlanningModel(Observable):
    """Model for planning mode state and data.

    Manages:
    - Conversation history with LLM
    - Current outline state
    - Build state for outline-driven generation
    - Planning configuration
    """

    def __init__(self):
        """Initialize the planning model."""
        super().__init__()
        self._conversation_history: List[Dict[str, str]] = []
        self._conversation_markdown: str = ""
        self._current_outline: Optional[str] = None
        self._build_state: Optional[Dict[str, Any]] = None
        self._is_active: bool = False

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of messages with 'role' and 'content' keys
        """
        return self._conversation_history

    def add_message(self, role: str, content: str):
        """Add a message to conversation history.

        Args:
            role: Either 'user' or 'assistant'
            content: Message content
        """
        self._conversation_history.append({"role": role, "content": content})
        self.notify_observers("message_added", {"role": role, "content": content})

    def clear_conversation(self):
        """Clear conversation history."""
        self._conversation_history = []
        self._conversation_markdown = ""
        self.notify_observers("conversation_cleared")

    @property
    def conversation_markdown(self) -> str:
        """Get full conversation as markdown.

        Returns:
            Formatted conversation markdown
        """
        return self._conversation_markdown

    @conversation_markdown.setter
    def conversation_markdown(self, value: str):
        """Set conversation markdown.

        Args:
            value: Markdown text
        """
        self._conversation_markdown = value
        self.notify_observers("conversation_markdown_changed", value)

    @property
    def current_outline(self) -> Optional[str]:
        """Get current outline text.

        Returns:
            Outline as markdown checklist, or None
        """
        return self._current_outline

    @current_outline.setter
    def current_outline(self, value: Optional[str]):
        """Set current outline.

        Args:
            value: Outline markdown text
        """
        self._current_outline = value
        self.notify_observers("outline_changed", value)

    @property
    def build_state(self) -> Optional[Dict[str, Any]]:
        """Get current build state for outline-driven generation.

        Returns:
            Build state dictionary with task queue and progress
        """
        return self._build_state

    @build_state.setter
    def build_state(self, value: Optional[Dict[str, Any]]):
        """Set build state.

        Args:
            value: Build state dictionary
        """
        self._build_state = value
        self.notify_observers("build_state_changed", value)

    @property
    def is_active(self) -> bool:
        """Check if planning mode is currently active.

        Returns:
            True if planning mode is running
        """
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        """Set planning mode active state.

        Args:
            value: Active state
        """
        self._is_active = value
        self.notify_observers("active_state_changed", value)

    def parse_outline_tasks(self, outline_text: str) -> List[str]:
        """Parse outline markdown into list of task descriptions.

        Extracts unchecked [ ] items from the outline.

        Args:
            outline_text: Markdown checklist text

        Returns:
            List of task descriptions (unchecked items only)
        """
        tasks = []
        for line in outline_text.strip().split("\n"):
            line = line.strip()
            # Look for unchecked markdown checkboxes
            if line.startswith("- [ ]") or line.startswith("* [ ]"):
                # Extract task text after checkbox
                task = line[5:].strip()
                if task:
                    tasks.append(task)
        return tasks

    def reset_build_state(self):
        """Reset build state to None."""
        self._build_state = None
        self.notify_observers("build_state_reset")

    def get_build_progress(self) -> Dict[str, Any]:
        """Get current build progress information.

        Returns:
            Dictionary with current_task, total_tasks, and progress info
        """
        if not self._build_state:
            return {
                "current_task": 0,
                "total_tasks": 0,
                "completed": True,
            }

        remaining = self._build_state.get("remaining_tasks", [])
        original = self._build_state.get("original_tasks", [])

        return {
            "current_task": len(original) - len(remaining),
            "total_tasks": len(original),
            "completed": len(remaining) == 0,
        }
