import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

log = logging.getLogger('nmagmake')

@dataclass
class InferenceEntity:
    name: str
    depends_on: List[str] = field(default_factory=list)
    how_to_make: List[Callable[..., Any]] = field(default_factory=list)
    also_updates: List[str] = field(default_factory=list)
    _is_prerequisite: List[str] = field(default_factory=list, init=False)
    _is_uptodate: bool = field(default=False, init=False)

class InferenceEngine:
    def __init__(self, entities: Optional[List[Dict[str, Any]]] = None):
        self.entities: Dict[str, InferenceEntity] = {}
        if entities:
            for e_desc in entities:
                ie = InferenceEntity(**e_desc)
                self.entities[ie.name] = ie
        
        self._build_backlinks()
        self._check_for_cycles()

    def _build_backlinks(self) -> None:
        """Connects dependencies to their dependents and validates existence."""
        for name, ie in self.entities.items():
            for dep in ie.depends_on:
                if (ie_dep := self.entities.get(dep)) is None:
                    raise KeyError(f"Dependency '{dep}' required by '{name}' not found.")
                ie_dep._is_prerequisite.append(name)

    def _check_for_cycles(self) -> None:
        """Detects circular dependencies using DFS to prevent infinite recursion."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)

            for neighbor in self.entities[name].depends_on:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(name)
            return False

        for node in self.entities:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(f"Circular dependency detected involving '{node}'")

    def invalidate(self, name: str) -> None:
        """Recursively marks an entity and its dependents as dirty."""
        if (ie := self.entities.get(name)) is None:
            raise KeyError(f"Entity '{name}' is unknown.")

        if ie._is_uptodate:
            ie._is_uptodate = False
            log.debug(f"Invalidated: {name}")
            for dependent_name in ie._is_prerequisite:
                self.invalidate(dependent_name)

    def make(self, name: str, **make_args: Any) -> None:
        """Builds target and dependencies recursively."""
        if (ie := self.entities.get(name)) is None:
            raise KeyError(f"Cannot build unknown entity '{name}'.")

        if ie._is_uptodate:
            log.debug(f"'{name}' is up-to-date.")
            return

        for dep_name in ie.depends_on:
            self.make(dep_name, **make_args)

        log.info(f"Building: {name}")
        for step in ie.how_to_make:
            step(**make_args)

        for also_name in ie.also_updates:
            if (other := self.entities.get(also_name)):
                other._is_uptodate = True
        
        ie._is_uptodate = True
