"""Unit tests for the _title_from_body() helper in views.custom_widgets.

These tests exercise every branch of the title-derivation logic without
requiring any Qt infrastructure.
"""

import pytest
from views.custom_widgets import _title_from_body, _TITLE_MAX_CHARS


class TestTitleFromBody:
    """Pure unit tests for _title_from_body()."""

    def test_empty_string_returns_empty(self):
        # Empty body text should yield an empty string
        assert _title_from_body("") == ""

    def test_whitespace_only_returns_empty(self):
        # A body that is all whitespace produces an empty first sentence
        result = _title_from_body("   ")
        assert result == ""

    def test_short_body_returned_unchanged(self):
        # Body shorter than the limit and no dot → returned as-is
        body = "A short section"
        assert _title_from_body(body) == body

    def test_body_exactly_at_limit(self):
        # Body exactly _TITLE_MAX_CHARS long with no dot → returned unchanged
        body = "A" * _TITLE_MAX_CHARS
        assert _title_from_body(body) == body

    def test_first_sentence_used_when_dot_present(self):
        # Only the text before the first dot is used as the title
        body = "First sentence. Second sentence."
        assert _title_from_body(body) == "First sentence"

    def test_first_sentence_short_enough_returned_fully(self):
        # When the first sentence is within the limit it is returned in full
        body = "Short intro. " + "x" * 200
        assert _title_from_body(body) == "Short intro"

    def test_long_first_sentence_truncated_at_word_boundary(self):
        # A first sentence longer than _TITLE_MAX_CHARS should be cut at the
        # last space within the limit so words are not split
        body = "The quick brown fox jumped over the lazy dog and kept going on and on."
        result = _title_from_body(body)
        assert len(result) <= _TITLE_MAX_CHARS
        # The result must end at a word boundary, not mid-word
        assert not result.endswith(" ")
        # Verify containment: the result is a prefix of the original sentence
        assert body.startswith(result)

    def test_long_first_sentence_no_space_truncates_at_char_limit(self):
        # A single unbroken word longer than the limit is truncated at the limit
        body = "X" * (_TITLE_MAX_CHARS * 2)
        result = _title_from_body(body)
        assert len(result) == _TITLE_MAX_CHARS

    def test_multiple_dots_uses_only_first_sentence(self):
        # Only the portion before the very first dot is used
        body = "First. Second. Third."
        assert _title_from_body(body) == "First"

    def test_body_with_no_space_at_all_returns_full_body_up_to_limit(self):
        # No spaces → falls back to raw char truncation
        body = "abcdefghij" * 10  # 100 chars, no dots
        result = _title_from_body(body)
        assert len(result) == _TITLE_MAX_CHARS
        assert result == body[:_TITLE_MAX_CHARS]

    def test_leading_spaces_stripped_from_first_sentence(self):
        # Leading whitespace is stripped via .strip() inside _title_from_body
        body = "  Leading space sentence"
        assert _title_from_body(body) == "Leading space sentence"

    def test_returns_string_type(self):
        # Return value is always a str
        assert isinstance(_title_from_body("Hello"), str)
        assert isinstance(_title_from_body(""), str)
