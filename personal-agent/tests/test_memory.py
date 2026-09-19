"""Memory layers: provenance, decay, precedence and budgeted retrieval."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from personalos.database.models import MemorySource, Preference
from personalos.memory.memory_manager import MemoryManager, classify_intent
from personalos.utils.timeutil import utcnow


# ---- intent classification -------------------------------------------------
def test_intent_classification() -> None:
    assert classify_intent("summarize the PDFs in Downloads") == "summarise"
    assert classify_intent("find the invoice from March") == "search"
    assert classify_intent("organize my research folder") == "organise"
    assert classify_intent("hello there") == "general"


@pytest.mark.parametrize(
    "request_text",
    [
        "list /tmp/pytest-of-runner/pytest-0/test_x0/workspace",
        "list /home/runner/work/_temp/workspace",
        "list C:/Users/runner/AppData/Local/Temp/workspace",
    ],
)
def test_a_username_in_a_path_does_not_change_the_intent(request_text: str) -> None:
    """Substring matching read "run" inside "runner" and called this a run request.

    It passed on a developer machine (/tmp/pytest-of-root/) and failed on CI,
    where the account is called `runner` — so the whole plan came out empty and
    six integration tests failed with nothing recorded.
    """
    assert classify_intent(request_text) == "inspect"


@pytest.mark.parametrize(
    "request_text,expected",
    [
        ("prune the logs", "general"),        # "run" inside "prune"
        ("run the deploy script", "run"),     # the real verb still works
        ("read the report", "inspect"),
        ("find the file", "search"),          # leading verb beats a later keyword
        ("schedule a recurring cleanup", "automate"),
    ],
)
def test_keywords_match_whole_words(request_text: str, expected: str) -> None:
    assert classify_intent(request_text) == expected


# ---- preferences -----------------------------------------------------------
def test_explicit_preference_has_full_confidence(memory: MemoryManager) -> None:
    stored = memory.learn_preference("preferred_report_format", "markdown", explicit=True)
    assert stored.explicit
    assert stored.confidence == 1.0


def test_inferred_cannot_overwrite_explicit(memory: MemoryManager) -> None:
    memory.learn_preference("report_format", "markdown", explicit=True)
    memory.learn_preference("report_format", "pdf", explicit=False, confidence=0.9)
    assert memory.preferences.get("report_format") == "markdown"


def test_explicit_overwrites_inferred(memory: MemoryManager) -> None:
    memory.learn_preference("report_format", "pdf", explicit=False)
    memory.learn_preference("report_format", "markdown", explicit=True)
    resolved = memory.preferences.get_resolved("report_format")
    assert resolved is not None and resolved.value == "markdown" and resolved.explicit


def test_inferred_confidence_decays_but_explicit_does_not(memory: MemoryManager) -> None:
    memory.learn_preference("guessed", "a", explicit=False, confidence=0.9)
    memory.learn_preference("stated", "b", explicit=True)

    # Age both items well past the configured half-life.
    long_ago = utcnow() - timedelta(days=400)
    with memory.database.session() as session:
        for row in session.scalars(select(Preference)):
            row.updated_at = long_ago
            row.created_at = long_ago
            row.last_used = long_ago

    assert memory.preferences.get_resolved("stated") is not None
    assert memory.preferences.get_resolved("guessed") is None, "a stale inference stops being used"


def test_reinforce_and_weaken(memory: MemoryManager) -> None:
    memory.learn_preference("guessed", "a", explicit=False, confidence=0.5)
    memory.preferences.reinforce("guessed")
    assert (memory.preferences.get_resolved("guessed")).confidence > 0.5

    memory.preferences.weaken("guessed", amount=0.9)
    assert memory.preferences.get_resolved("guessed") is None


def test_reinforce_leaves_explicit_preferences_alone(memory: MemoryManager) -> None:
    memory.learn_preference("stated", "x", explicit=True)
    memory.preferences.reinforce("stated")
    resolved = memory.preferences.get_resolved("stated")
    assert resolved is not None and resolved.confidence == 1.0


def test_project_namespace_shadows_global(memory: MemoryManager) -> None:
    memory.learn_preference("format", "pdf", explicit=True)
    memory.learn_preference("format", "markdown", explicit=True, namespace="research")
    assert memory.preferences.get("format") == "pdf"
    assert memory.preferences.get("format", namespace="research") == "markdown"


# ---- semantic memory -------------------------------------------------------
def test_semantic_facts_are_searchable(memory: MemoryManager) -> None:
    memory.semantic.remember("research_folder", "~/Documents/threat-research", category="path")
    matches = memory.semantic.search("where is my research folder")
    assert matches and matches[0].fact.key == "research_folder"


def test_observed_fact_cannot_overwrite_a_stated_one(memory: MemoryManager) -> None:
    memory.semantic.remember("inbox", "~/A", source=MemorySource.EXPLICIT)
    memory.semantic.remember("inbox", "~/B", source=MemorySource.OBSERVED)
    fact = memory.semantic.get("inbox")
    assert fact is not None and fact.value == "~/A"


# ---- episodic memory -------------------------------------------------------
def test_episode_signature_ignores_arguments(memory: MemoryManager) -> None:
    signature = memory.episodic.build_signature(
        [{"tool": "filesystem", "operation": "list"}, {"tool": "documents", "operation": "extract_text"}]
    )
    assert signature == "filesystem.list > documents.extract_text"


def test_episodic_search_prefers_successes(memory: MemoryManager) -> None:
    memory.episodic.save(
        task_id="t1",
        task_text="summarise the quarterly research PDFs",
        actions=[{"tool": "documents", "operation": "extract_text"}],
        result="done",
        success=True,
    )
    memory.episodic.save(
        task_id="t2",
        task_text="summarise the quarterly research PDFs",
        actions=[{"tool": "documents", "operation": "extract_text"}],
        result="failed",
        success=False,
    )
    matches = memory.episodic.search("summarise research PDFs")
    assert matches
    assert matches[0].episode.success


def test_feedback_attaches_to_the_episode(memory: MemoryManager) -> None:
    memory.episodic.save(
        task_id="t1", task_text="rename files", actions=[], result="ok", success=True
    )
    assert memory.episodic.add_feedback("t1", "use YYYY-MM-DD next time")
    assert memory.episodic.recent(1)[0].user_feedback == "use YYYY-MM-DD next time"


def test_pruning_respects_retention(memory: MemoryManager) -> None:
    from personalos.database.models import Episode

    memory.episodic.save(task_id="old", task_text="x", actions=[], result=None, success=True)
    with memory.database.session() as session:
        session.scalars(select(Episode)).one().created_at = utcnow() - timedelta(days=9999)
    assert memory.episodic.prune() == 1


# ---- retrieval -------------------------------------------------------------
def test_retrieval_respects_the_token_budget(memory: MemoryManager) -> None:
    for index in range(40):
        memory.semantic.remember(f"fact_{index}", "research " + "x" * 300, confidence=0.9)
    retrieved = memory.retrieve("research", budget_tokens=200)
    assert retrieved.used_tokens <= 200
    assert retrieved.items, "some memory should still make it in"


def test_explicit_preferences_outrank_everything_in_retrieval(memory: MemoryManager) -> None:
    memory.learn_preference("report_format", "markdown", explicit=True)
    memory.semantic.remember("noise", "unrelated trivia", confidence=0.9)
    retrieved = memory.retrieve("write me a report", budget_tokens=60)
    assert retrieved.items[0].kind == "preference"


def test_retrieval_renders_grouped_context(memory: MemoryManager) -> None:
    memory.learn_preference("report_format", "markdown", explicit=True)
    rendered = memory.retrieve("write a report").render()
    assert "Your stated preferences" in rendered
    assert "report_format" in rendered


def test_statistics_counts_every_layer(memory: MemoryManager) -> None:
    memory.learn_preference("a", 1, explicit=True)
    memory.semantic.remember("b", "c")
    memory.episodic.save(task_id="t", task_text="x", actions=[], result=None, success=True)
    memory.workflows.record(name="w", signature="a > b", steps=[])
    stats = memory.statistics()
    assert stats == {"episodes": 1, "facts": 1, "preferences": 1, "workflows": 1}


def test_working_memory_is_not_persisted(memory: MemoryManager) -> None:
    working = memory.open_working("task-1", "tidy my downloads")
    working.remember("files_found", 12)
    working.note("three were duplicates")
    assert "files_found" in working.render()
    assert memory.close_working("task-1") is working
    assert memory.working("task-1") is None
