"""Skills, workflow detection, suggestions and feedback learning."""

from __future__ import annotations

from pathlib import Path

import pytest

from personalos.agent.reasoning import FeedbackInterpretation, parse_feedback_offline
from personalos.database.models import Workflow
from personalos.errors import SkillError
from personalos.learning.feedback_learner import FeedbackLearner
from personalos.learning.pattern_detector import find_repeated
from personalos.learning.skill_generator import generate_skill
from personalos.learning.suggestions import SuggestionDraft, SuggestionStore
from personalos.learning.workflow_detector import WorkflowDetector
from personalos.security.risk import RiskLevel
from personalos.skills.schema import SkillDefinition, SkillInput, SkillStep, load_skill_file
from personalos.skills.skill_manager import SkillManager
from personalos.skills.skill_registry import SkillRegistry


def sample_definition(name: str = "tidy_reports") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description="Tidy up a reports folder.",
        inputs={"folder": SkillInput(type="string", required=True, description="Where.")},
        steps=[
            SkillStep(id="scan", description="list", tool="filesystem", operation="list",
                      arguments={"path": "{{ inputs.folder }}"}),
        ],
    )


# ---- skill schema ----------------------------------------------------------
def test_skill_requires_at_least_one_step() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        SkillDefinition(name="empty", steps=[])


def test_skill_rejects_duplicate_step_ids() -> None:
    step = SkillStep(id="a", description="d", tool="filesystem", operation="list")
    with pytest.raises(ValueError, match="unique"):
        SkillDefinition(name="dupes", steps=[step, step])


def test_skill_name_must_be_an_identifier() -> None:
    with pytest.raises(ValueError):
        SkillDefinition(name="Not A Skill", steps=[
            SkillStep(id="a", description="d", tool="filesystem", operation="list")
        ])


def test_missing_required_input_is_reported() -> None:
    with pytest.raises(SkillError, match="needs the input"):
        sample_definition().to_plan({})


def test_unknown_input_is_rejected() -> None:
    with pytest.raises(SkillError, match="does not take"):
        sample_definition().to_plan({"folder": "~/x", "surprise": 1})


def test_skill_becomes_a_plan_with_inputs_bound() -> None:
    plan = sample_definition().to_plan({"folder": "~/Reports"})
    assert plan.source == "skill"
    assert plan.inputs == {"folder": "~/Reports"}
    assert plan.steps[0].arguments["path"] == "{{ inputs.folder }}"


def test_skill_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(sample_definition().to_yaml(), encoding="utf-8")
    assert load_skill_file(path).name == "tidy_reports"


def test_malformed_skill_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: broken\nsteps: []\n", encoding="utf-8")
    with pytest.raises(SkillError, match="not a valid skill"):
        load_skill_file(path)


# ---- skill manager ---------------------------------------------------------
def test_builtin_skills_load(settings, database) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    names = {item.definition.name for item in manager.list_skills()}
    assert {"organize_downloads", "daily_summary"} <= names


def test_create_inspect_and_version(settings, database, tmp_path: Path) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    manager.create(sample_definition())
    assert manager.inspect("tidy_reports").definition.version == 1

    updated = sample_definition()
    updated.description = "Now with more tidying."
    manager.version("tidy_reports", updated)
    assert manager.inspect("tidy_reports").definition.version == 2
    archived = settings.skills_dir / "versions" / "tidy_reports.v1.yaml"
    assert archived.exists(), "the previous version is kept"


def test_creating_a_duplicate_is_refused(settings, database) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    manager.create(sample_definition())
    with pytest.raises(SkillError, match="already exists"):
        manager.create(sample_definition())


def test_disabled_skills_refuse_to_build_a_plan(settings, database) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    manager.create(sample_definition())
    manager.set_enabled("tidy_reports", False)
    with pytest.raises(SkillError, match="disabled"):
        manager.build_plan("tidy_reports", {"folder": "~/x"})


def test_builtin_skills_cannot_be_deleted(settings, database) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    with pytest.raises(SkillError, match="cannot be deleted"):
        manager.delete("organize_downloads")


def test_deleting_keeps_a_backup(settings, database) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    manager.create(sample_definition())
    manager.delete("tidy_reports")
    assert (settings.skills_dir / "deleted" / "tidy_reports.yaml").exists()
    assert not manager.registry.exists("tidy_reports")


def test_run_statistics_are_tracked(settings, database) -> None:
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    manager.create(sample_definition())
    manager.record_run("tidy_reports", success=True)
    manager.record_run("tidy_reports", success=False)
    summary = next(s for s in manager.list_skills() if s.definition.name == "tidy_reports")
    assert summary.run_count == 2
    assert summary.success_rate == 0.5


def test_broken_skill_files_are_reported_not_hidden(settings, database) -> None:
    (settings.skills_dir / "broken.yaml").write_text("name: broken\nsteps: []\n", encoding="utf-8")
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    manager.list_skills()
    assert any("broken.yaml" in str(path) for path in manager.registry.errors)


# ---- pattern detection -----------------------------------------------------
def test_repeated_signatures_are_detected(memory) -> None:
    actions = [
        {"tool": "filesystem", "operation": "rename"},
        {"tool": "documents", "operation": "extract_text"},
        {"tool": "filesystem", "operation": "move"},
    ]
    for index in range(3):
        memory.episodic.save(
            task_id=f"t{index}",
            task_text="rename, summarise and file the new PDFs",
            actions=actions,
            result="done",
            success=True,
        )
    patterns = find_repeated(memory.episodic.recent(50), minimum_occurrences=3)
    assert patterns
    assert patterns[0].occurrences == 3
    assert "filesystem.rename" in patterns[0].signature


def test_failures_do_not_count_towards_a_pattern(memory) -> None:
    actions = [{"tool": "filesystem", "operation": "move"}]
    for index in range(4):
        memory.episodic.save(
            task_id=f"t{index}", task_text="move things", actions=actions,
            result="nope", success=False,
        )
    assert find_repeated(memory.episodic.recent(50), minimum_occurrences=2) == []


# ---- suggestions -----------------------------------------------------------
def test_suggestions_deduplicate_by_key(database) -> None:
    store = SuggestionStore(database, cooldown_hours=24, max_per_session=10)
    draft = SuggestionDraft(kind="workflow", key="abc", title="Repeat noticed")
    assert store.offer(draft) is not None
    assert store.offer(draft) is None, "the same pending suggestion is not raised twice"


def test_suggestions_respect_the_session_cap(database) -> None:
    store = SuggestionStore(database, cooldown_hours=0, max_per_session=2)
    for index in range(5):
        store.offer(SuggestionDraft(kind="workflow", key=f"k{index}", title="t"))
    assert len(store.pending()) == 2


def test_dismissed_suggestions_respect_the_cooldown(database) -> None:
    store = SuggestionStore(database, cooldown_hours=24, max_per_session=10)
    suggestion = store.offer(SuggestionDraft(kind="workflow", key="abc", title="t"))
    assert suggestion is not None
    store.resolve(suggestion.id, "dismissed")
    assert store.offer(SuggestionDraft(kind="workflow", key="abc", title="t")) is None


# ---- workflow detection ----------------------------------------------------
def test_detector_suggests_but_does_not_create_a_skill(settings, database, memory) -> None:
    store = SuggestionStore(database, cooldown_hours=0, max_per_session=5)
    detector = WorkflowDetector(memory, store, settings.learning)
    actions = [
        {"tool": "filesystem", "operation": "rename", "description": "rename", "arguments": {"path": "/tmp/a"}},
        {"tool": "filesystem", "operation": "move", "description": "move", "arguments": {"source": "/tmp/a", "destination": "/tmp/b"}},
    ]
    for index in range(3):
        memory.episodic.save(
            task_id=f"t{index}", task_text="rename then file the download",
            actions=actions, result="ok", success=True,
        )

    observations = detector.scan()
    assert observations
    assert observations[0].suggestion is not None
    assert "reusable skill" in observations[0].suggestion.title
    manager = SkillManager(SkillRegistry(settings.skills_dir), database)
    generated = [s for s in manager.list_skills() if s.definition.created_by == "agent"]
    assert generated == [], "nothing is automated until the user accepts"


def test_detector_is_silent_below_the_threshold(settings, database, memory) -> None:
    settings.learning.min_occurrences_for_suggestion = 5
    store = SuggestionStore(database, cooldown_hours=0)
    detector = WorkflowDetector(memory, store, settings.learning)
    for index in range(2):
        memory.episodic.save(
            task_id=f"t{index}", task_text="x",
            actions=[{"tool": "filesystem", "operation": "move"}], result="ok", success=True,
        )
    assert all(observation.suggestion is None for observation in detector.scan())


def test_learning_can_be_switched_off(settings, database, memory) -> None:
    settings.learning.enabled = False
    detector = WorkflowDetector(memory, SuggestionStore(database), settings.learning)
    assert detector.scan() == []


# ---- skill generation ------------------------------------------------------
def test_generated_skill_parameterises_paths() -> None:
    workflow = Workflow(
        name="wf",
        signature="filesystem.list > filesystem.write",
        description="",
        success_count=3,
        example_requests=["tidy the downloads folder"],
        steps=[
            {"id": "scan", "tool": "filesystem", "operation": "list",
             "description": "list", "arguments": {"path": "/home/me/Downloads"}, "risk_level": 0},
            {"id": "write", "tool": "filesystem", "operation": "write",
             "description": "write", "arguments": {"path": "/home/me/Downloads/out.md", "content": "x"},
             "risk_level": 2},
        ],
    )
    definition = generate_skill(workflow, name="tidy_downloads")
    assert definition.created_by == "agent"
    assert definition.derived_from_workflow == "wf"
    assert definition.inputs, "recurring paths become inputs"
    first = definition.steps[0].arguments["path"]
    assert first.startswith("{{ inputs."), first
    assert definition.max_risk_level == int(RiskLevel.MODERATE)


def test_generating_from_an_empty_workflow_is_refused() -> None:
    with pytest.raises(SkillError, match="no recorded steps"):
        generate_skill(Workflow(name="wf", signature="", steps=[]))


# ---- feedback learning -----------------------------------------------------
def test_offline_feedback_parsing() -> None:
    assert parse_feedback_offline("always save reports as markdown").key == "preferred_report_format"
    assert parse_feedback_offline("use YYYY-MM-DD filenames").key == "filename_date_format"
    assert parse_feedback_offline("never rename files in this folder").kind == "rule"
    assert parse_feedback_offline("the sky is blue").kind == "none"


def test_feedback_creates_an_explicit_preference(memory, policy) -> None:
    learner = FeedbackLearner(memory, policy)
    item = learner.apply(
        FeedbackInterpretation(kind="preference", key="preferred_report_format",
                               value="markdown", raw="always markdown")
    )
    assert item.kind == "preference"
    resolved = memory.preferences.get_resolved("preferred_report_format")
    assert resolved is not None and resolved.explicit


def test_forbidding_feedback_creates_a_deny_rule(memory, policy, tmp_path: Path) -> None:
    learner = FeedbackLearner(memory, policy)
    item = learner.apply(
        FeedbackInterpretation(kind="rule", key="forbid_rename", value=True, raw="never rename here"),
        path_scope=str(tmp_path),
    )
    assert item.kind == "rule"
    rule = next(r for r in policy.list_rules() if r.name.startswith("forbid_rename"))
    assert rule.effect == "deny"
    assert rule.tool == "filesystem" and rule.operation == "rename"


def test_permissive_feedback_is_clamped_to_level_two(memory, policy, tmp_path: Path) -> None:
    learner = FeedbackLearner(memory, policy)
    learner.apply(
        FeedbackInterpretation(kind="rule", key="reduce_prompts_scope", value=True,
                               raw="don't ask before moving files here"),
        path_scope=str(tmp_path),
    )
    rule = next(r for r in policy.list_rules() if r.name.startswith("reduce_prompts_scope"))
    assert rule.effect == "allow"
    assert rule.max_risk_level == int(RiskLevel.MODERATE)


def test_unparseable_feedback_is_still_kept(memory, policy) -> None:
    learner = FeedbackLearner(memory, policy)
    item = learner.apply(parse_feedback_offline("something idiosyncratic about my setup"))
    assert item.kind == "note"
    assert item.stored
