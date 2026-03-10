"""Widget-level tests for OutlineSectionRow.

These tests verify:
- Row construction (no edit button, no separate title label, in-place QPlainTextEdit).
- Status icon rendering and click signals.
- update_content() does not emit section_edited.
- _emit_section_edited() emits with auto-derived title.
- Debounce timer is started on text change.
- delete_clicked signal fires.
"""

import pytest
from PyQt5 import QtCore, QtWidgets

from views.custom_widgets import OutlineSectionRow, _EDIT_DEBOUNCE_MS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_signal(signal) -> list:
    """Wire *signal* to a list and return that list for later inspection."""
    received: list = []
    signal.connect(lambda *args: received.append(args))
    return received


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestOutlineSectionRowConstruction:
    """Tests that verify the widget is constructed correctly."""

    def test_details_edit_shows_provided_text(self, qapp):
        row = OutlineSectionRow(0, "My Title", "Some details")
        assert row._details_edit.toPlainText() == "Some details"

    def test_title_stored_internally(self, qapp):
        row = OutlineSectionRow(0, "My Title", "details")
        assert row._title == "My Title"

    def test_no_edit_button_attribute(self, qapp):
        # The old popup-based edit button must not exist on the new row
        row = OutlineSectionRow(0, "Title", "details")
        assert not hasattr(row, "_edit_button"), (
            "_edit_button should have been removed; in-place editing is now via QPlainTextEdit"
        )

    def test_no_title_label_attribute(self, qapp):
        # The old bold title QLabel must not exist on the new row
        row = OutlineSectionRow(0, "Title", "details")
        assert not hasattr(row, "_title_label"), (
            "_title_label should have been removed; the title is no longer displayed separately"
        )

    def test_details_edit_is_plain_text_edit(self, qapp):
        row = OutlineSectionRow(0, "T", "body")
        assert isinstance(row._details_edit, QtWidgets.QPlainTextEdit)

    def test_empty_details_creates_empty_editor(self, qapp):
        row = OutlineSectionRow(0, "T", "")
        assert row._details_edit.toPlainText() == ""

    def test_default_status_is_pending(self, qapp):
        row = OutlineSectionRow(0, "T", "d")
        assert row._status == "pending"

    def test_explicit_status_stored(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="done")
        assert row._status == "done"

    def test_delete_button_exists(self, qapp):
        row = OutlineSectionRow(0, "T", "d")
        assert hasattr(row, "_delete_button")


# ---------------------------------------------------------------------------
# Status icon
# ---------------------------------------------------------------------------


class TestOutlineSectionRowStatusIcon:
    """Tests for status icon display and status transitions."""

    def test_pending_icon(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="pending")
        assert row._status_label.text() == "[ ]"

    def test_active_icon(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="active")
        assert row._status_label.text() == "[▶]"

    def test_done_icon(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="done")
        assert row._status_label.text() == "[✓]"

    def test_set_status_updates_icon(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="pending")
        row.set_status("done")
        assert row._status_label.text() == "[✓]"
        assert row._status == "done"

    def test_set_status_active(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="pending")
        row.set_status("active")
        assert row._status_label.text() == "[▶]"


# ---------------------------------------------------------------------------
# Status click signals
# ---------------------------------------------------------------------------


class TestOutlineSectionRowStatusClickSignals:
    """Tests for check_clicked / uncheck_clicked signals."""

    def test_check_clicked_emitted_for_pending(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="pending")
        received = _collect_signal(row.check_clicked)
        row._status_label.click()
        assert len(received) == 1

    def test_check_clicked_emitted_for_active(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="active")
        received = _collect_signal(row.check_clicked)
        row._status_label.click()
        assert len(received) == 1

    def test_uncheck_clicked_emitted_for_done(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="done")
        received = _collect_signal(row.uncheck_clicked)
        row._status_label.click()
        assert len(received) == 1

    def test_check_not_emitted_for_done(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="done")
        received = _collect_signal(row.check_clicked)
        row._status_label.click()
        assert len(received) == 0

    def test_uncheck_not_emitted_for_pending(self, qapp):
        row = OutlineSectionRow(0, "T", "d", status="pending")
        received = _collect_signal(row.uncheck_clicked)
        row._status_label.click()
        assert len(received) == 0


# ---------------------------------------------------------------------------
# Delete signal
# ---------------------------------------------------------------------------


class TestOutlineSectionRowDeleteSignal:
    def test_delete_clicked_emitted(self, qapp):
        row = OutlineSectionRow(0, "T", "d")
        received = _collect_signal(row.delete_clicked)
        row._delete_button.click()
        assert len(received) == 1


# ---------------------------------------------------------------------------
# update_content
# ---------------------------------------------------------------------------


class TestOutlineSectionRowUpdateContent:
    """Tests for update_content()."""

    def test_update_content_changes_stored_title(self, qapp):
        row = OutlineSectionRow(0, "Old Title", "old details")
        row.update_content("New Title", "new details")
        assert row._title == "New Title"

    def test_update_content_changes_editor_text(self, qapp):
        row = OutlineSectionRow(0, "T", "old")
        row.update_content("T", "new details here")
        assert row._details_edit.toPlainText() == "new details here"

    def test_update_content_does_not_emit_section_edited(self, qapp):
        # Updating programmatically must NOT fire section_edited (would cause
        # unwanted outline-changed events when the controller writes back state)
        row = OutlineSectionRow(0, "T", "old")
        received = _collect_signal(row.section_edited)
        row.update_content("New Title", "new details")
        # Process any pending Qt events to catch accidental deferred signals
        QtWidgets.QApplication.processEvents()
        assert received == [], "update_content must not emit section_edited"

    def test_update_content_with_empty_details(self, qapp):
        row = OutlineSectionRow(0, "T", "existing")
        row.update_content("T", "")
        assert row._details_edit.toPlainText() == ""


# ---------------------------------------------------------------------------
# _emit_section_edited
# ---------------------------------------------------------------------------


class TestOutlineSectionRowEmitSectionEdited:
    """Tests for the _emit_section_edited method invoked after debounce."""

    def test_emit_derives_title_from_body(self, qapp):
        row = OutlineSectionRow(0, "original", "")
        row._details_edit.setPlainText("A new section body")
        received = _collect_signal(row.section_edited)
        row._emit_section_edited()
        assert len(received) == 1
        emitted_title, emitted_details = received[0]
        # Title is derived from body, not the original stored title
        assert emitted_title == "A new section body"
        assert emitted_details == "A new section body"

    def test_emit_updates_internal_title(self, qapp):
        row = OutlineSectionRow(0, "old", "")
        row._details_edit.setPlainText("Brand new content")
        row._emit_section_edited()
        assert row._title == "Brand new content"

    def test_emit_with_empty_text_sends_empty_title(self, qapp):
        row = OutlineSectionRow(0, "old", "some text")
        row._details_edit.setPlainText("")
        received = _collect_signal(row.section_edited)
        row._emit_section_edited()
        emitted_title, emitted_details = received[0]
        assert emitted_title == ""
        assert emitted_details == ""

    def test_emit_strips_whitespace(self, qapp):
        row = OutlineSectionRow(0, "", "")
        row._details_edit.setPlainText("  trimmed  ")
        received = _collect_signal(row.section_edited)
        row._emit_section_edited()
        _, emitted_details = received[0]
        assert emitted_details == "trimmed"

    def test_long_body_title_truncated(self, qapp):
        from views.custom_widgets import _TITLE_MAX_CHARS

        long_body = "word " * 20  # well over the limit
        row = OutlineSectionRow(0, "", "")
        row._details_edit.setPlainText(long_body)
        received = _collect_signal(row.section_edited)
        row._emit_section_edited()
        emitted_title, _ = received[0]
        assert len(emitted_title) <= _TITLE_MAX_CHARS


# ---------------------------------------------------------------------------
# Debounce timer
# ---------------------------------------------------------------------------


class TestOutlineSectionRowDebounceTimer:
    """Tests for the debounce timer started on text change."""

    def test_text_change_starts_debounce_timer(self, qapp):
        row = OutlineSectionRow(0, "T", "")
        # Timer should not be active before any change
        assert not row._edit_debounce_timer.isActive()
        # Directly call _on_text_changed (simulates a text change event)
        row._on_text_changed()
        assert row._edit_debounce_timer.isActive()

    def test_timer_is_single_shot(self, qapp):
        row = OutlineSectionRow(0, "T", "")
        assert row._edit_debounce_timer.isSingleShot()

    def test_timer_interval_matches_constant(self, qapp):
        row = OutlineSectionRow(0, "T", "")
        assert row._edit_debounce_timer.interval() == _EDIT_DEBOUNCE_MS

    def test_second_text_change_restarts_timer(self, qapp):
        """Each change resets the timer so rapid typing does not fire early."""
        row = OutlineSectionRow(0, "T", "")
        row._on_text_changed()
        # Record remaining time after first trigger
        remaining_after_first = row._edit_debounce_timer.remainingTime()
        # Simulate immediate second change — timer must restart to full interval
        row._on_text_changed()
        remaining_after_second = row._edit_debounce_timer.remainingTime()
        # After restart, remaining should be >= first remaining (reset to full)
        assert (
            remaining_after_second >= remaining_after_first
            or remaining_after_second == _EDIT_DEBOUNCE_MS
        )
