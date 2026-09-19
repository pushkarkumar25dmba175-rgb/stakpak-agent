"""Finding skill files on disk.

Skills come from three places, searched in this order so a user copy always
shadows a shipped one of the same name:

1. ``~/.personalos/skills/`` — what you wrote yourself
2. ``~/.personalos/skills/generated/`` — what the agent wrote and you approved
3. the package's ``builtin/`` directory — the examples that ship with it
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from personalos.errors import SkillError
from personalos.skills.schema import SkillDefinition, load_skill_file

BUILTIN_DIR = Path(__file__).parent / "builtin"


@dataclass
class SkillFile:
    """A skill definition together with where it came from."""

    definition: SkillDefinition
    path: Path
    origin: str
    """user | generated | builtin"""


class SkillRegistry:
    """Loads and caches skill files, reporting the broken ones instead of hiding them."""

    def __init__(self, skills_dir: Path, *, include_builtin: bool = True) -> None:
        self.skills_dir = skills_dir
        self.generated_dir = skills_dir / "generated"
        self.include_builtin = include_builtin
        self.errors: dict[Path, str] = {}
        self._cache: dict[str, SkillFile] = {}

    def search_paths(self) -> list[tuple[Path, str]]:
        paths = [(self.skills_dir, "user"), (self.generated_dir, "generated")]
        if self.include_builtin:
            paths.append((BUILTIN_DIR, "builtin"))
        return paths

    def reload(self) -> dict[str, SkillFile]:
        """Re-scan every search path. Later directories never shadow earlier ones."""
        found: dict[str, SkillFile] = {}
        self.errors.clear()
        for directory, origin in self.search_paths():
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.y*ml")):
                if origin == "user" and path.parent != self.skills_dir:
                    continue
                try:
                    definition = load_skill_file(path)
                except SkillError as exc:
                    self.errors[path] = str(exc)
                    continue
                found.setdefault(definition.name, SkillFile(definition, path, origin))
        self._cache = found
        return found

    def all(self, *, reload: bool = True) -> list[SkillFile]:
        if reload or not self._cache:
            self.reload()
        return sorted(self._cache.values(), key=lambda item: item.definition.name)

    def get(self, name: str, *, reload: bool = True) -> SkillFile:
        if reload or not self._cache:
            self.reload()
        try:
            return self._cache[name]
        except KeyError:
            known = ", ".join(sorted(self._cache)) or "none yet"
            raise SkillError(
                f"There is no skill called {name!r}.",
                remediation=f"Available skills: {known}. List them with `agent skills`.",
            ) from None

    def exists(self, name: str) -> bool:
        if not self._cache:
            self.reload()
        return name in self._cache

    def path_for(self, name: str, *, generated: bool) -> Path:
        directory = self.generated_dir if generated else self.skills_dir
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.yaml"
