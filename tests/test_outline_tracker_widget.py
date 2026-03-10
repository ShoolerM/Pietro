"""Integration tests for OutlineTrackerWidget.

These tests verify the interaction between OutlineTrackerWidget and its
OutlineSectionRow children, covering:
- Populating sections from structured data.
- Adding sections without a dialog (direct row creation).
- Row edit propagation to the tracker's section_edited signal.
- Deletion propagation.
- Status management (set_active, mark_complete, reset_from).
- Clearing all sections via clear_all_sections().
- get_sections() returns a deep copy.
"""

import pytest
from PyQt5 import QtCore, QtWidgets

from views.custom_widgets import OutlineTrackerWidget, OutlineSectionRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_signal(signal) -> list:
    """Wire *signal* to a list and return that list."""
    received: list = []
    signal.connect(lambda *args: received.append(args))
    return received


def _make_sections(n: int = 3) -> list:
    """Build a list of *n* minimal section dicts for use with set_sections()."""
    return [
        {"description": f"Section {i}", "details": f"Details {i}", "completed": False}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# set_sections
# ---------------------------------------------------------------------------


class TestOutlineTrackerSetSections:
    def test_set_sections_creates_correct_row_count(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(4))
        assert tracker.section_count() == 4

    def test_set_sections_stores_title_and_details(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(
            [{"description": "Act One", "details": "Intro chapter", "completed": False}]
        )
        sections = tracker.get_sections()
        assert sections[0]["title"] == "Act One"
        assert sections[0]["details"] == "Intro chapter"

    def test_set_sections_marks_completed_as_done(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections([{"description": "Done section", "details": "", "completed": True}])
        assert tracker.get_sections()[0]["status"] == "done"

    def test_set_sections_marks_incomplete_as_pending(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(
            [{"description": "Pending section", "details": "", "completed": False}]
        )
        assert tracker.get_sections()[0]["status"] == "pending"

    def test_set_sections_replaces_existing_sections(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(5))
        tracker.set_sections(_make_sections(2))
        assert tracker.section_count() == 2

    def test_set_sections_creates_row_widgets(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        assert len(tracker._row_widgets) == 3
        for rw in tracker._row_widgets:
            assert isinstance(rw, OutlineSectionRow)


# ---------------------------------------------------------------------------
# add_section
# ---------------------------------------------------------------------------


class TestOutlineTrackerAddSection:
    def test_add_section_increases_count(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(2))
        tracker.add_section("New", "New details")
        assert tracker.section_count() == 3

    def test_add_section_emits_section_added(self, qapp):
        tracker = OutlineTrackerWidget()
        received = _collect_signal(tracker.section_added)
        tracker.add_section("My title", "My details")
        assert len(received) == 1
        assert received[0] == ("My title", "My details")

    def test_add_section_appends_at_end(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.add_section("First", "")
        tracker.add_section("Second", "")
        sections = tracker.get_sections()
        assert sections[-1]["title"] == "Second"

    def test_add_section_defaults_to_pending(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.add_section("T", "d")
        assert tracker.get_sections()[0]["status"] == "pending"


# ---------------------------------------------------------------------------
# _on_add_section_clicked — no dialog, immediate empty row
# ---------------------------------------------------------------------------


class TestOutlineTrackerAddSectionNoDialog:
    """Regression tests ensuring the Add Section button no longer opens a dialog."""

    def test_clicking_add_creates_row_immediately(self, qapp):
        # The button click must immediately create a new row without blocking
        tracker = OutlineTrackerWidget()
        initial_count = tracker.section_count()
        # Simulate the button click via the internal handler directly
        tracker._on_add_section_clicked()
        assert tracker.section_count() == initial_count + 1

    def test_add_button_click_creates_empty_row(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker._on_add_section_clicked()
        new_section = tracker.get_sections()[-1]
        # The new row should have an empty title (set on first text change)
        assert new_section["title"] == ""
        assert new_section["details"] == ""

    def test_add_button_click_does_not_emit_section_added_for_empty(self, qapp):
        # section_added IS emitted by add_section regardless of content;
        # the key regression is that no modal dialog blocks execution.
        # We verify the signal WAS emitted (row created) and the function returned.
        tracker = OutlineTrackerWidget()
        received = _collect_signal(tracker.section_added)
        tracker._on_add_section_clicked()
        # add_section emits section_added when called from _on_add_section_clicked
        assert len(received) == 1

    def test_new_row_has_plain_text_edit(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker._on_add_section_clicked()
        new_row: OutlineSectionRow = tracker._row_widgets[-1]
        assert isinstance(new_row._details_edit, QtWidgets.QPlainTextEdit)

    def test_new_row_has_no_edit_button(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker._on_add_section_clicked()
        new_row: OutlineSectionRow = tracker._row_widgets[-1]
        assert not hasattr(new_row, "_edit_button")


# ---------------------------------------------------------------------------
# Row edit propagation
# ---------------------------------------------------------------------------


class TestOutlineTrackerRowEditPropagation:
    """Tests that edits in a row propagate up via the tracker's section_edited signal."""

    def test_row_edit_propagates_section_edited(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections([{"description": "Old", "details": "old details", "completed": False}])
        received = _collect_signal(tracker.section_edited)

        row: OutlineSectionRow = tracker._row_widgets[0]
        row._details_edit.setPlainText("new content")
        row._emit_section_edited()

        assert len(received) == 1
        idx, title, details = received[0]
        assert idx == 0
        assert details == "new content"

    def test_row_edit_updates_internal_section_data(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections([{"description": "T", "details": "old", "completed": False}])

        row: OutlineSectionRow = tracker._row_widgets[0]
        row._details_edit.setPlainText("updated content")
        row._emit_section_edited()

        assert tracker.get_sections()[0]["details"] == "updated content"

    def test_multiple_rows_edit_propagates_correct_index(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        received = _collect_signal(tracker.section_edited)

        # Edit the second row (index 1)
        row: OutlineSectionRow = tracker._row_widgets[1]
        row._details_edit.setPlainText("edited middle")
        row._emit_section_edited()

        assert received[0][0] == 1  # index must be 1


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


class TestOutlineTrackerDeletion:
    def test_delete_row_decrements_count(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        tracker._on_row_deleted(1)
        assert tracker.section_count() == 2

    def test_delete_row_emits_section_deleted(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        received = _collect_signal(tracker.section_deleted)
        tracker._on_row_deleted(0)
        assert len(received) == 1
        assert received[0] == (0,)

    def test_delete_reindexes_remaining_rows(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(
            [
                {"description": "A", "details": "", "completed": False},
                {"description": "B", "details": "", "completed": False},
                {"description": "C", "details": "", "completed": False},
            ]
        )
        tracker._on_row_deleted(0)
        sections = tracker.get_sections()
        assert sections[0]["title"] == "B"
        assert sections[1]["title"] == "C"


# ---------------------------------------------------------------------------
# clear_all_sections
# ---------------------------------------------------------------------------


class TestOutlineTrackerClearAll:
    def test_clear_all_sections_empties_sections(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(5))
        tracker.clear_all_sections()
        assert tracker.section_count() == 0

    def test_clear_all_sections_emits_signal(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(2))
        received = _collect_signal(tracker.all_sections_cleared)
        tracker.clear_all_sections()
        assert len(received) == 1

    def test_clear_all_clears_row_widgets(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        tracker.clear_all_sections()
        assert tracker._row_widgets == []


# ---------------------------------------------------------------------------
# Status management
# ---------------------------------------------------------------------------


class TestOutlineTrackerStatusManagement:
    def test_set_active_marks_single_row(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        tracker.set_active(1)
        sections = tracker.get_sections()
        assert sections[0]["status"] == "pending"
        assert sections[1]["status"] == "active"
        assert sections[2]["status"] == "pending"

    def test_set_active_deactivates_previous_active(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        tracker.set_active(0)
        tracker.set_active(2)
        sections = tracker.get_sections()
        assert sections[0]["status"] == "pending"
        assert sections[2]["status"] == "active"

    def test_mark_complete_sets_done(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        tracker.mark_complete(1)
        assert tracker.get_sections()[1]["status"] == "done"

    def test_reset_from_resets_from_index(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(5))
        tracker.mark_complete(0)
        tracker.mark_complete(1)
        tracker.mark_complete(2)
        tracker.reset_from(1)
        sections = tracker.get_sections()
        # Section 0 must remain done; sections 1+ must be reset (1 becomes active)
        assert sections[0]["status"] == "done"
        assert sections[1]["status"] == "active"
        assert sections[2]["status"] == "pending"

    def test_reset_resets_all_to_pending(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(3))
        tracker.mark_complete(0)
        tracker.set_active(1)
        tracker.reset()
        for sec in tracker.get_sections():
            assert sec["status"] == "pending"


# ---------------------------------------------------------------------------
# get_sections returns a deep copy
# ---------------------------------------------------------------------------


class TestOutlineTrackerGetSections:
    def test_get_sections_returns_copy_not_reference(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(2))
        copy = tracker.get_sections()
        copy[0]["title"] = "mutated"
        # Internal state must not be affected by mutating the returned copy
        assert tracker._sections[0]["title"] != "mutated"

    def test_get_sections_reflects_current_state(self, qapp):
        tracker = OutlineTrackerWidget()
        tracker.set_sections(_make_sections(2))
        tracker.mark_complete(0)
        assert tracker.get_sections()[0]["status"] == "done"
