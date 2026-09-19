"""The agent loop.

    OBSERVE → UNDERSTAND → RETRIEVE MEMORY → PLAN → CHECK PERMISSIONS
            → EXECUTE → VERIFY → LOG → LEARN

:class:`Agent` is the object everything else drives: the CLI, the scheduler,
the folder watcher and the dashboard all call :meth:`Agent.ask` or
:meth:`Agent.run_skill` and get the same behaviour, including the same approval
path. There is no second, quieter route to execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from personalos.agent.executor import ExecutionReport, Executor
from personalos.agent.plan import Plan
from personalos.agent.planner import HeuristicPlanner, LLMPlanner
from personalos.agent.reasoning import FeedbackInterpretation, Reasoner
from personalos.agent.state import AgentState, StateMachine
from personalos.agent.task_manager import TaskManager
from personalos.audit.audit_log import AuditLog
from personalos.audit.rollback_journal import RollbackJournal, RollbackResult
from personalos.database.database import Database, build_database
from personalos.database.models import Suggestion, TaskStatus
from personalos.errors import PersonalOSError, SkillError
from personalos.learning.feedback_learner import FeedbackLearner, LearnedItem
from personalos.learning.skill_generator import generate_skill
from personalos.learning.suggestions import SuggestionStore
from personalos.learning.workflow_detector import WorkflowDetector
from personalos.llm.base import LLMProvider
from personalos.llm.factory import build_provider
from personalos.memory.memory_manager import MemoryManager, RetrievedContext
from personalos.projects.project_manager import ProjectContext, ProjectManager
from personalos.security.permissions import ApprovalHandler, DenyAllHandler
from personalos.security.policy_engine import PolicyEngine
from personalos.security.risk import RiskLevel
from personalos.security.sandbox import SubprocessSandbox
from personalos.settings import Settings
from personalos.skills.schema import SkillDefinition
from personalos.skills.skill_manager import SkillManager
from personalos.skills.skill_registry import SkillRegistry
from personalos.tools import build_default_registry
from personalos.tools.base import ToolContext
from personalos.tools.registry import ToolRegistry
from personalos.utils.paths import PathResolver
from personalos.utils.textutil import truncate


@dataclass
class TaskOutcome:
    """Everything one task produced."""

    task_id: str
    request: str
    plan: Plan
    report: ExecutionReport | None
    answer: str
    success: bool
    retrieved: RetrievedContext | None = None
    suggestions: list[Suggestion] = field(default_factory=list)
    error: str | None = None

    @property
    def max_risk_level(self) -> RiskLevel:
        return self.report.max_risk_level if self.report else RiskLevel.READ_ONLY


class Agent:
    """The assembled agent.

    Construct it with :meth:`build`, which wires every subsystem from settings.
    The constructor takes the pieces directly so tests can substitute any of
    them — an echo provider, an in-memory database, a recording approval
    handler — without monkey-patching.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        memory: MemoryManager,
        registry: ToolRegistry,
        policy: PolicyEngine,
        audit: AuditLog,
        journal: RollbackJournal,
        provider: LLMProvider,
        skills: SkillManager,
        projects: ProjectManager,
        suggestions: SuggestionStore,
        tasks: TaskManager,
        state: StateMachine | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.memory = memory
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.journal = journal
        self.provider = provider
        self.skills = skills
        self.projects = projects
        self.suggestions = suggestions
        self.tasks = tasks
        self.state = state or StateMachine(AgentState.IDLE)

        self.sandbox = SubprocessSandbox()
        self.executor = Executor(
            registry, policy, audit, max_retries=settings.security.max_retries
        )
        self.reasoner = Reasoner(
            provider,
            max_output_tokens=settings.llm.max_output_tokens,
            context_budget=settings.llm.context_budget_tokens,
        )
        self.planner = LLMPlanner(
            provider,
            registry,
            memory,
            max_output_tokens=settings.llm.max_output_tokens,
            context_budget=settings.llm.context_budget_tokens,
        )
        self.offline_planner = HeuristicPlanner(registry)
        self.learner = FeedbackLearner(memory, policy)
        self.detector = WorkflowDetector(memory, suggestions, settings.learning)
        self.active_project: ProjectContext | None = None
        self._path_resolver = self._build_resolver()

    # ---- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        approval_handler: ApprovalHandler | None = None,
        provider: LLMProvider | None = None,
        database: Database | None = None,
        include_optional_tools: bool = False,
    ) -> Agent:
        """Wire an agent from settings.

        ``approval_handler`` defaults to :class:`DenyAllHandler`, which refuses
        anything needing confirmation. That is the right default for an
        unattended caller; the CLI passes an interactive handler instead.
        """
        settings.ensure_directories()
        database = database or build_database(settings.database_path)
        memory = MemoryManager(settings, database)
        registry = build_default_registry(include_optional=include_optional_tools)
        audit = AuditLog(
            settings.logs_dir, database, redact_enabled=settings.security.redact_logs
        )
        journal = RollbackJournal(database, settings.home / "backups")
        policy = PolicyEngine(settings, database, approval_handler or DenyAllHandler())
        provider = provider or build_provider(settings, fallback_to_echo=True)
        skill_registry = SkillRegistry(settings.skills_dir)
        agent = cls(
            settings=settings,
            database=database,
            memory=memory,
            registry=registry,
            policy=policy,
            audit=audit,
            journal=journal,
            provider=provider,
            skills=SkillManager(skill_registry, database),
            projects=ProjectManager(settings.projects_dir),
            suggestions=SuggestionStore(
                database,
                cooldown_hours=settings.learning.suggestion_cooldown_hours,
                max_per_session=settings.learning.max_suggestions_per_session,
            ),
            tasks=TaskManager(database),
        )
        if settings.active_project:
            try:
                agent.switch_project(settings.active_project)
            except PersonalOSError:
                # A stale project name in config must not stop the agent starting.
                pass
        return agent

    def _build_resolver(self) -> PathResolver:
        roots = list(self.settings.workspace.allowed_roots)
        # The agent's own home is always reachable: reports, trash and scratch
        # files live there, and a task that cannot write its own output is useless.
        roots.append(str(self.settings.home))
        if self.active_project:
            roots.extend(str(path) for path in self.active_project.extra_roots)
        return PathResolver(
            roots,
            self.settings.workspace.denied_paths,
            follow_symlinks=self.settings.workspace.follow_symlinks,
        )

    @property
    def paths(self) -> PathResolver:
        return self._path_resolver

    def tool_context(self, task_id: str, *, dry_run: bool = False) -> ToolContext:
        return ToolContext(
            settings=self.settings,
            paths=self._path_resolver,
            journal=self.journal,
            audit=self.audit,
            sandbox=self.sandbox,
            task_id=task_id,
            project=self.project_name,
            dry_run=dry_run,
        )

    # ---- projects ----------------------------------------------------------
    @property
    def project_name(self) -> str | None:
        return self.active_project.definition.name if self.active_project else None

    @property
    def namespace(self) -> str:
        return self.active_project.definition.namespace if self.active_project else "global"

    def switch_project(self, name: str | None) -> ProjectContext | None:
        """Activate a project, widening the path allow-list to its directories."""
        self.active_project = self.projects.load(name) if name else None
        self._path_resolver = self._build_resolver()
        return self.active_project

    # ---- the loop ----------------------------------------------------------
    async def ask(
        self,
        request: str,
        *,
        origin: str = "cli",
        dry_run: bool = False,
        plan_only: bool = False,
        extra_context: str | None = None,
    ) -> TaskOutcome:
        """Run one full cycle of the agent loop for ``request``."""
        if self.state.paused():
            return TaskOutcome(
                task_id="",
                request=request,
                plan=Plan(goal=request),
                report=None,
                answer="I am paused. Run `agent resume` when you want me working again.",
                success=False,
                error="paused",
            )

        # OBSERVE — open a task record and a scratchpad.
        task = self.tasks.create(request, project=self.project_name, origin=origin)
        working = self.memory.open_working(task.task_id, request, project=self.project_name)

        try:
            # UNDERSTAND + RETRIEVE — classify, then pull only relevant memory.
            self.state.transition(AgentState.PLANNING)
            retrieved = self.memory.retrieve(request, project=self.namespace)
            working.retrieved_memory = [item.text for item in retrieved.items]

            # PLAN
            project_block = self.active_project.render() if self.active_project else None
            combined_context = "\n\n".join(part for part in (project_block, extra_context) if part)
            plan = await self.planner.plan(
                request,
                project=self.project_name,
                extra_context=combined_context or None,
                default_path=self._default_path(),
            )
            working.plan_id = plan.plan_id
            self.tasks.update(
                task.task_id,
                status=TaskStatus.PLANNING,
                plan=plan.model_dump(mode="json"),
            )

            if plan_only:
                self.state.transition(AgentState.IDLE)
                self.tasks.update(
                    task.task_id, status=TaskStatus.CANCELLED, result_summary="Plan only.",
                    finished=True,
                )
                return TaskOutcome(
                    task_id=task.task_id,
                    request=request,
                    plan=plan,
                    report=None,
                    answer=plan.describe(),
                    success=True,
                    retrieved=retrieved,
                )

            # CHECK PERMISSIONS + EXECUTE — per step, inside the executor.
            self.state.transition(AgentState.EXECUTING)
            self.tasks.update(task.task_id, status=TaskStatus.EXECUTING)
            context = self.tool_context(task.task_id, dry_run=dry_run)
            report = await self.executor.execute(plan, context)

            # VERIFY happened per step; summarise the whole thing here.
            self.state.transition(AgentState.VERIFYING)
            answer = await self.reasoner.answer(request, report, retrieved=retrieved)

            # LOG
            self.tasks.update(
                task.task_id,
                status=TaskStatus.COMPLETED if report.success else TaskStatus.FAILED,
                agent_state=AgentState.LEARNING,
                result_summary=truncate(answer, 4000),
                success=report.success,
                error=report.halted_reason,
                max_risk_level=int(report.max_risk_level),
                finished=True,
            )

            # LEARN
            self.state.transition(AgentState.LEARNING)
            suggestions = self._learn(task.task_id, request, report, answer)

            self.state.transition(AgentState.IDLE)
            return TaskOutcome(
                task_id=task.task_id,
                request=request,
                plan=plan,
                report=report,
                answer=answer,
                success=report.success,
                retrieved=retrieved,
                suggestions=suggestions,
                error=report.halted_reason,
            )

        except PersonalOSError as exc:
            self.state.transition(AgentState.ERROR)
            self.tasks.update(
                task.task_id,
                status=TaskStatus.FAILED,
                success=False,
                error=str(exc),
                result_summary=str(exc),
                finished=True,
            )
            self.audit.record(
                task_id=task.task_id,
                action="task failed",
                tool="agent",
                parameters={"request": truncate(request, 500)},
                success=False,
                error=str(exc),
                project=self.project_name,
            )
            self.state.transition(AgentState.IDLE)
            return TaskOutcome(
                task_id=task.task_id,
                request=request,
                plan=Plan(goal=request),
                report=None,
                answer=str(exc),
                success=False,
                error=str(exc),
            )
        finally:
            self.memory.close_working(task.task_id)

    async def run_skill(
        self,
        name: str,
        inputs: dict[str, Any] | None = None,
        *,
        origin: str = "cli",
        dry_run: bool = False,
    ) -> TaskOutcome:
        """Run a saved skill. Goes through the same approval path as anything else."""
        task = self.tasks.create(f"run skill: {name}", project=self.project_name, origin=origin)
        try:
            plan = self.skills.build_plan(name, inputs)
        except SkillError as exc:
            self.tasks.update(
                task.task_id, status=TaskStatus.FAILED, success=False, error=str(exc), finished=True
            )
            return TaskOutcome(
                task_id=task.task_id,
                request=f"run skill: {name}",
                plan=Plan(goal=name),
                report=None,
                answer=str(exc),
                success=False,
                error=str(exc),
            )

        self.state.transition(AgentState.EXECUTING)
        self.tasks.update(
            task.task_id,
            status=TaskStatus.EXECUTING,
            plan=plan.model_dump(mode="json"),
        )
        context = self.tool_context(task.task_id, dry_run=dry_run)
        report = await self.executor.execute(plan, context)
        self.skills.record_run(name, success=report.success)

        answer = report.summary()
        self.tasks.update(
            task.task_id,
            status=TaskStatus.COMPLETED if report.success else TaskStatus.FAILED,
            result_summary=truncate(answer, 4000),
            success=report.success,
            error=report.halted_reason,
            max_risk_level=int(report.max_risk_level),
            finished=True,
        )
        self.state.transition(AgentState.LEARNING)
        suggestions = self._learn(task.task_id, f"run skill: {name}", report, answer)
        self.state.transition(AgentState.IDLE)
        return TaskOutcome(
            task_id=task.task_id,
            request=f"run skill: {name}",
            plan=plan,
            report=report,
            answer=answer,
            success=report.success,
            suggestions=suggestions,
            error=report.halted_reason,
        )

    # ---- learning ----------------------------------------------------------
    def _learn(
        self, task_id: str, request: str, report: ExecutionReport, answer: str
    ) -> list[Suggestion]:
        """Save the episode and look for patterns. Never changes permissions."""
        self.memory.episodic.save(
            task_id=task_id,
            task_text=request,
            actions=report.action_trace(),
            result=truncate(answer, 2000),
            success=report.success,
            project=self.project_name,
        )
        if not self.settings.learning.enabled:
            return []
        observations = self.detector.scan()
        return [obs.suggestion for obs in observations if obs.suggestion is not None]

    async def record_feedback(
        self, feedback: str, *, task_id: str | None = None, path_scope: str | None = None
    ) -> LearnedItem:
        """Interpret a correction and store it as a preference, rule or fact."""
        interpretation: FeedbackInterpretation = await self.reasoner.interpret_feedback(feedback)
        if task_id:
            self.memory.episodic.add_feedback(task_id, feedback)
        item = self.learner.apply(
            interpretation, project=self.namespace if self.active_project else None,
            path_scope=path_scope,
        )
        self.audit.record(
            task_id=task_id or "feedback",
            action="learned from feedback",
            tool="agent",
            parameters={"feedback": truncate(feedback, 500), "kind": item.kind, "key": item.key},
            success=True,
            project=self.project_name,
        )
        return item

    def accept_suggestion(
        self, suggestion_id: int, *, skill_name: str | None = None
    ) -> SkillDefinition:
        """Turn an accepted workflow suggestion into a skill file.

        This is the only path from an observation to a reusable automation, and
        it runs exclusively from an explicit user action.
        """
        suggestion = self.suggestions.get(suggestion_id)
        if suggestion is None:
            raise SkillError(f"There is no suggestion #{suggestion_id}.")
        if suggestion.kind != "workflow":
            raise SkillError(
                f"Suggestion #{suggestion_id} is not a workflow suggestion.",
                remediation="Only workflow suggestions can become skills.",
            )
        workflow_name = str((suggestion.payload or {}).get("workflow", ""))
        workflow = self.memory.workflows.get(workflow_name)
        if workflow is None:
            raise SkillError(f"The workflow {workflow_name!r} is no longer recorded.")

        definition = generate_skill(workflow, name=skill_name)
        self.skills.create(definition, generated=True)
        self.memory.workflows.mark_promoted(workflow_name, definition.name)
        self.suggestions.resolve(suggestion_id, "accepted")
        self.audit.record(
            task_id=f"suggestion-{suggestion_id}",
            action="created a skill from an approved suggestion",
            tool="skills",
            parameters={"skill": definition.name, "workflow": workflow_name},
            success=True,
            project=self.project_name,
        )
        return definition

    def dismiss_suggestion(self, suggestion_id: int) -> bool:
        return self.suggestions.resolve(suggestion_id, "dismissed")

    # ---- rollback ----------------------------------------------------------
    def rollback(self, task_id: str, *, dry_run: bool = False) -> list[RollbackResult]:
        """Undo a task's recorded changes, newest first."""
        results = self.journal.rollback_task(task_id, dry_run=dry_run)
        if not dry_run:
            self.audit.record(
                task_id=task_id,
                action="rolled back",
                tool="agent",
                parameters={"entries": len(results)},
                risk_level=RiskLevel.MODERATE,
                success=all(result.applied or "not reversible" in result.detail.lower() for result in results),
                project=self.project_name,
            )
        return results

    # ---- helpers -----------------------------------------------------------
    def _default_path(self) -> str | None:
        """The folder a vague request most likely means."""
        if self.active_project and self.active_project.extra_roots:
            return str(self.active_project.extra_roots[0])
        roots = self.settings.workspace.allowed_roots
        return roots[0] if roots else None

    def status(self) -> dict[str, Any]:
        """A snapshot for `agent status` and the dashboard."""
        recent = self.tasks.recent(limit=5)
        return {
            "state": str(self.state.state),
            "state_description": self.state.state.description,
            "project": self.project_name,
            "provider": f"{self.provider.name}/{self.provider.model}",
            "approval_mode": str(self.settings.security.approval_mode),
            "home": str(self.settings.home),
            "tools": self.registry.available(),
            "memory": self.memory.statistics(),
            "skills": len(self.skills.list_skills()),
            "pending_suggestions": len(self.suggestions.pending()),
            "recent_tasks": [
                {
                    "task_id": task.task_id,
                    "request": truncate(task.request, 80),
                    "status": task.status,
                    "success": task.success,
                }
                for task in recent
            ],
        }

    def close(self) -> None:
        self.database.dispose()
        self.state.transition(AgentState.OFFLINE)


def workspace_roots(settings: Settings) -> list[Path]:
    """The directories the agent may touch, for display."""
    return [Path(root) for root in settings.workspace.allowed_roots]
