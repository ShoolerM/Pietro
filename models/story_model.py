"""Model for managing story content and state."""
from base.observable import Observable


class StoryModel(Observable):
    """Manages story content, history, and markdown state."""
    
    def __init__(self):
        super().__init__()
        self._content = ""
        self._history = []
        self._notes = ""
        self._planning_outline = ""
        self._planning_active = False
    
    @property
    def content(self):
        """Get current story content."""
        return self._content
    
    @content.setter
    def content(self, value):
        """Set story content and notify observers."""
        self._content = value
        self.notify_observers('content_changed', value)
    
    @property
    def notes(self):
        """Get current notes content."""
        return self._notes
    
    @notes.setter
    def notes(self, value):
        """Set notes content and notify observers."""
        self._notes = value
        self.notify_observers('notes_changed', value)
    
    @property
    def planning_outline(self):
        """Get current planning outline (markdown checklist)."""
        return self._planning_outline
    
    @planning_outline.setter
    def planning_outline(self, value):
        """Set planning outline and notify observers."""
        self._planning_outline = value
        self.notify_observers('planning_outline_changed', value)
    
    @property
    def planning_active(self):
        """Check if planning mode is active."""
        return self._planning_active
    
    @planning_active.setter
    def planning_active(self, value):
        """Set planning mode active flag and notify observers."""
        self._planning_active = value
        self.notify_observers('planning_active_changed', value)
    
    def append_content(self, text):
        """Append text to story content.
        
        Args:
            text: Text to append
        """
        self._content += text
        self.notify_observers('content_appended', text)
    
    def clear_content(self):
        """Clear story content."""
        self._content = ""
        self.notify_observers('content_cleared', None)
    
    def save_to_history(self):
        """Save current content state to history for undo."""
        self._history.append(self._content)
        self.notify_observers('history_saved', None)
    
    def undo(self):
        """Restore previous content state from history.
        
        Returns:
            bool: True if undo was successful, False if no history
        """
        if not self._history:
            return False
        
        self._content = self._history.pop()
        self.notify_observers('content_restored', self._content)
        return True
    
    def has_history(self):
        """Check if undo history exists.
        
        Returns:
            bool: True if history exists
        """
        return len(self._history) > 0
    
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Estimate the number of tokens in a text string.
        Uses a rough approximation of 4 characters per token.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        if not text:
            return 0
        return len(text) // 4
    
    @staticmethod
    def extract_last_paragraphs(text: str, max_tokens: int = 500) -> str:
        """Extract the last 2 paragraphs from the text.
        This provides immediate context for the LLM after a summary.
        
        Args:
            text: The full text to extract from
            max_tokens: Maximum tokens to extract (approximately)
            
        Returns:
            The last 2 paragraphs of the text
        """
        if not text:
            return ""
        
        # Split by paragraph breaks (double newlines)
        paragraphs = text.split('\n\n')
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return text
        
        # Get the last 2 paragraphs
        if len(paragraphs) >= 2:
            last_two = '\n\n'.join(paragraphs[-2:])
        else:
            # If there's only one paragraph, return it
            last_two = paragraphs[-1]
        
        # If the result is too long, truncate to max_tokens
        max_chars = max_tokens * 4
        if len(last_two) > max_chars:
            last_two = last_two[-max_chars:]
            # Try to start at a sentence boundary
            period_pos = last_two.find('. ')
            if period_pos > 0 and period_pos < len(last_two) * 0.3:
                last_two = last_two[period_pos + 2:].strip()
        
        return last_two
    
    @staticmethod
    def extract_recent_content(text: str, max_tokens: int) -> tuple[str, int]:
        """Extract recent content from end of text up to max_tokens.
        
        Args:
            text: Full text to extract from
            max_tokens: Maximum tokens to extract
            
        Returns:
            tuple: (extracted_text, start_position_in_original)
        """
        if not text:
            return "", 0
        
        tokens = StoryModel.estimate_token_count(text)
        if tokens <= max_tokens:
            return text, 0
        
        # Estimate character position to start from
        max_chars = max_tokens * 4
        start_pos = max(0, len(text) - max_chars)
        
        # Try to find a good boundary (paragraph break)
        remaining = text[start_pos:]
        paragraph_break = remaining.find('\n\n')
        if paragraph_break > 0 and paragraph_break < len(remaining) * 0.3:
            start_pos += paragraph_break + 2
        else:
            # Try sentence boundary
            for delimiter in ['. ', '! ', '? ']:
                sentence_break = remaining.find(delimiter)
                if sentence_break > 0 and sentence_break < len(remaining) * 0.3:
                    start_pos += sentence_break + len(delimiter)
                    break
        
        return text[start_pos:], start_pos
    
    @staticmethod
    def find_chunk_boundary(text: str, target_position: int) -> int:
        """Find a good boundary position for chunking near target position.
        Prioritizes paragraph breaks, then sentence endings.
        
        Args:
            text: Text to find boundary in
            target_position: Approximate position to split at
            
        Returns:
            int: Actual boundary position
        """
        if target_position >= len(text):
            return len(text)
        
        if target_position <= 0:
            return 0
        
        # Search window: 20% before and after target
        window_size = int(len(text) * 0.1)
        search_start = max(0, target_position - window_size)
        search_end = min(len(text), target_position + window_size)
        search_text = text[search_start:search_end]
        
        # Priority 1: Look for paragraph breaks (double newline)
        # Search backwards from target first
        rel_target = target_position - search_start
        before_text = search_text[:rel_target]
        after_text = search_text[rel_target:]
        
        # Look for paragraph break near target (within 30% of window)
        para_before = before_text.rfind('\n\n')
        para_after = after_text.find('\n\n')
        
        # Choose closest paragraph break
        if para_before >= len(before_text) * 0.7:  # Found near end of before section
            return search_start + para_before + 2
        if para_after >= 0 and para_after < len(after_text) * 0.3:  # Found near start of after section
            return search_start + rel_target + para_after + 2
        
        # Priority 2: Look for sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        best_pos = None
        best_distance = float('inf')
        
        for ending in sentence_endings:
            # Search backwards
            pos = before_text.rfind(ending)
            if pos >= 0:
                distance = len(before_text) - pos
                if distance < best_distance:
                    best_distance = distance
                    best_pos = search_start + pos + len(ending)
            
            # Search forwards
            pos = after_text.find(ending)
            if pos >= 0:
                distance = pos
                if distance < best_distance:
                    best_distance = distance
                    best_pos = search_start + rel_target + pos + len(ending)
        
        if best_pos is not None:
            return best_pos
        
        # Fallback: Use target position
        return target_position
    
    @staticmethod
    def chunk_text(text: str, target_chunk_tokens: int) -> list[tuple[str, int, int]]:
        """Split text into chunks at natural boundaries.
        
        Args:
            text: Text to chunk
            target_chunk_tokens: Target size for each chunk in tokens
            
        Returns:
            list: List of (chunk_text, start_pos, end_pos) tuples
        """
        if not text:
            return []
        
        total_tokens = StoryModel.estimate_token_count(text)
        if total_tokens <= target_chunk_tokens:
            return [(text, 0, len(text))]
        
        chunks = []
        current_pos = 0
        target_chars = target_chunk_tokens * 4
        
        while current_pos < len(text):
            # Calculate target end position
            target_end = current_pos + target_chars
            
            if target_end >= len(text):
                # Last chunk
                chunk_text = text[current_pos:]
                chunks.append((chunk_text, current_pos, len(text)))
                break
            
            # Find good boundary near target
            actual_end = StoryModel.find_chunk_boundary(text, target_end)
            
            # Ensure we make progress (avoid infinite loop)
            if actual_end <= current_pos:
                actual_end = min(current_pos + target_chars, len(text))
            
            chunk_text = text[current_pos:actual_end]
            chunks.append((chunk_text, current_pos, actual_end))
            current_pos = actual_end
        
        return chunks
