"""Oracle Engine — static and patch-coherence verification for structured claims."""

from __future__ import annotations

import ast
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import libcst as cst
    import libcst.matchers as m
    import libcst.metadata as meta
    _LIBCST_AVAILABLE = True
except ImportError:
    cst = None  # type: ignore[assignment]
    m = None  # type: ignore[assignment]
    meta = None  # type: ignore[assignment]
    _LIBCST_AVAILABLE = False

_CST_VISITOR_BASE = cst.CSTVisitor if cst is not None else object

try:
    from models import PatchHunk, StructuredThink, ThinkClaim
except ImportError:
    from ..models import PatchHunk, StructuredThink, ThinkClaim

logger = logging.getLogger(__name__)

_SYNC_PRIMITIVES = {
    "asyncio.Lock",
    "asyncio.Semaphore",
    "asyncio.Event",
    "threading.Lock",
    "threading.RLock",
    "threading.Semaphore",
    "anyio.Lock",
    "trio.Lock",
}

_SYNC_ATTRS = {"Lock", "RLock", "Semaphore", "Event", "Condition"}
_MUTATING_METHODS = {"append", "extend", "pop", "clear", "update", "add", "remove", "discard", "setdefault"}
_BLOCKING_SYNC_CALLS = {"time.sleep", "requests.get", "requests.post", "urllib.request.urlopen"}


@dataclass
class OracleEvidence:
    entity_resolved: bool = False
    pre_condition_met: bool = False
    post_condition_met: bool = False
    patch_addresses_claim: bool = False
    dynamic_confirmed: bool = False
    notes: List[str] = field(default_factory=list)


class _SourcePair:
    """Pre-patch and post-patch source text for a single file."""

    def __init__(self, pre: str, post: str) -> None:
        self.pre_src = pre
        self.post_src = post
        self._pre_tree: Optional[ast.AST] = None
        self._post_tree: Optional[ast.AST] = None

    def pre_ast(self) -> Optional[ast.AST]:
        if self._pre_tree is None and self.pre_src:
            try:
                self._pre_tree = ast.parse(self.pre_src)
            except SyntaxError:
                pass
        return self._pre_tree

    def post_ast(self) -> Optional[ast.AST]:
        if self._post_tree is None and self.post_src:
            try:
                self._post_tree = ast.parse(self.post_src)
            except SyntaxError:
                pass
        return self._post_tree


def _build_source_map(
    pre_sources: Dict[str, str],
    post_sources: Dict[str, str],
) -> Dict[str, _SourcePair]:
    all_keys = {_normalise_rel(k) for k in set(pre_sources) | set(post_sources)}
    return {
        k: _SourcePair(
            pre_sources.get(k, pre_sources.get(k.replace("/", "\\"), "")),
            post_sources.get(k, post_sources.get(k.replace("/", "\\"), "")),
        )
        for k in all_keys
    }


def _normalise_rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _get_pair(path: str, source_map: Dict[str, _SourcePair]) -> Optional[_SourcePair]:
    path = _normalise_rel(path)
    return source_map.get(path)


def _resolve_location(location: str) -> Tuple[str, str, str]:
    """Parse 'path/to/file.py::ClassName.method' into file/class/function."""
    if "::" not in location:
        return _normalise_rel(location), "", ""
    file_part, qual = location.split("::", 1)
    parts = qual.rsplit(".", 1)
    if len(parts) == 2:
        return _normalise_rel(file_part), parts[0], parts[1]
    return _normalise_rel(file_part), "", qual


def _get_function_src(tree: ast.AST, func_name: str, class_name: str = "") -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and class_name and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == func_name:
                    return ast.unparse(item)
        if not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.unparse(node)
    return ""


def _libcst_parse_safe(source: str) -> Optional[cst.Module]:
    if not _LIBCST_AVAILABLE or cst is None or not source:
        return None
    try:
        return cst.parse_module(source)
    except Exception:
        return None


def _expr_name(node: Any) -> str:
    if cst is None:
        return ""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        left = _expr_name(node.value)
        return f"{left}.{node.attr.value}" if left else node.attr.value
    if isinstance(node, cst.Call):
        return _expr_name(node.func)
    if isinstance(node, cst.Subscript):
        return _expr_name(node.value)
    return ""


def _entity_leaf(entity: str) -> str:
    return entity.split(".")[-1].strip()


def _targets_entity(expr: Any, entity: str) -> bool:
    name = _expr_name(expr)
    leaf = _entity_leaf(entity)
    if not name:
        return False
    return name == entity or name.endswith(f".{leaf}") or name == leaf


def _iter_hunks_for_claim(claim: ThinkClaim, patch_hunks: Sequence[PatchHunk]) -> List[PatchHunk]:
    if not patch_hunks:
        return []
    exact = [h for h in patch_hunks if h.addresses_claim and h.addresses_claim == claim.claim_id]
    if exact:
        return exact
    file_path, _, _ = _resolve_location(claim.location)
    return [h for h in patch_hunks if _normalise_rel(h.file) == file_path]


def _hunk_added_lines(hunk: PatchHunk) -> List[str]:
    search_counts: Dict[str, int] = {}
    for line in hunk.search.splitlines():
        search_counts[line] = search_counts.get(line, 0) + 1
    added: List[str] = []
    for line in hunk.replace.splitlines():
        count = search_counts.get(line, 0)
        if count:
            search_counts[line] = count - 1
        else:
            added.append(line)
    return added


def _contains_sync_primitive(text: str) -> bool:
    return bool(re.search(r"\b(?:Lock|RLock|Semaphore|Event|Condition)\s*\(", text))


def _contains_sync_usage(text: str) -> bool:
    return (
        bool(re.search(r"\bwith\s+.*(?:lock|Lock|RLock|Semaphore|Event|Condition)", text))
        or ".acquire(" in text
        or ".release(" in text
        or "async with" in text
    )


class OraclePlugin(ABC):
    """Base class for a single-category claim verifier."""

    category: str = ""

    @abstractmethod
    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        """Return (verdict, reason_note, evidence)."""


def _verify_patch_addresses_claim(
    claim: ThinkClaim,
    patch_hunks: Sequence[PatchHunk],
) -> Tuple[bool, str]:
    relevant_hunks = _iter_hunks_for_claim(claim, patch_hunks)
    if not relevant_hunks:
        return False, "no hunk claims or targets this claim"

    file_path, _, _ = _resolve_location(claim.location)
    for hunk in relevant_hunks:
        if _normalise_rel(hunk.file) != file_path:
            return False, f"hunk targets {hunk.file}, not {file_path}"

    if claim.category in {"concurrency", "async_wait"}:
        replacement = "\n".join(h.replace for h in relevant_hunks)
        search = "\n".join(h.search for h in relevant_hunks)
        entity_mentioned = bool(claim.entity and (claim.entity in replacement or claim.entity in search))
        lock_added = any(_contains_sync_primitive("\n".join(_hunk_added_lines(h))) for h in relevant_hunks)
        sync_used = _contains_sync_usage(replacement)
        if "lock" in claim.reason.lower() or claim.category == "concurrency":
            if not (lock_added or sync_used):
                return False, "patch does not introduce or use synchronization"
        if claim.entity and not entity_mentioned and not lock_added and not sync_used:
            return False, "patch does not touch claimed entity or synchronization"

    return True, "patch structurally addresses claim"


class _FunctionAccessCollector(_CST_VISITOR_BASE):
    METADATA_DEPENDENCIES = (meta.ParentNodeProvider, meta.QualifiedNameProvider) if meta else ()

    def __init__(self, entity: str, class_name: str, func_name: str) -> None:
        self.entity = entity
        self.class_name = class_name
        self.func_name = func_name
        self.accesses: List[cst.CSTNode] = []
        self.entity_defined = False
        self._class_depth = 0
        self._function_depth = 0
        self._in_target_class = not class_name
        self._in_target_function = False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._class_depth += 1
        if self.class_name and node.name.value == self.class_name:
            self._in_target_class = True
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        if self.class_name and node.name.value == self.class_name:
            self._in_target_class = False
        self._class_depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self._function_depth += 1
        if self._in_target_class and node.name.value == self.func_name:
            self._in_target_function = True
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self._in_target_class and node.name.value == self.func_name:
            self._in_target_function = False
        self._function_depth -= 1

    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
        if _targets_entity(node.target, self.entity):
            self.entity_defined = True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if _targets_entity(node.target, self.entity):
            self.entity_defined = True

    def visit_Attribute(self, node: cst.Attribute) -> None:
        if self._in_target_function and _targets_entity(node, self.entity):
            self.accesses.append(node)

    def visit_Name(self, node: cst.Name) -> None:
        if self._in_target_function and _targets_entity(node, self.entity):
            self.accesses.append(node)


class _SyncGuardAnalyzer:
    def __init__(self, wrapper: meta.MetadataWrapper, module: cst.Module) -> None:
        self.wrapper = wrapper
        self.module = module
        self.parents = wrapper.resolve(meta.ParentNodeProvider)
        self.qnames = wrapper.resolve(meta.QualifiedNameProvider)

    def is_sync_primitive(self, node: cst.BaseExpression) -> bool:
        qualified = self.qnames.get(node, set())
        if any(q.name in _SYNC_PRIMITIVES for q in qualified):
            return True
        name = _expr_name(node)
        return name.split(".")[-1] in _SYNC_ATTRS

    def guarding_locks(self, node: cst.CSTNode) -> List[str]:
        guards: List[str] = []
        current: Optional[cst.CSTNode] = node
        while current is not None and current in self.parents:
            parent = self.parents[current]
            if isinstance(parent, (cst.With, cst.WithItem)):
                items = parent.items if isinstance(parent, cst.With) else [parent]
                for item in items:
                    expr = item.item
                    if isinstance(expr, cst.Call):
                        expr = expr.func
                    if self.is_sync_primitive(expr):
                        guards.append(_expr_name(expr) or "sync_primitive")
            current = parent
        return guards


class RaceConditionOracle(OraclePlugin):
    """Verify concurrency claims using entity access and synchronization guards."""

    category = "concurrency"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        file_path, class_name, func_name = _resolve_location(claim.location)
        evidence = OracleEvidence()
        patch_ok, patch_note = _verify_patch_addresses_claim(claim, patch_hunks)
        evidence.patch_addresses_claim = patch_ok
        evidence.notes.append(patch_note)

        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", f"file {file_path!r} not in source map", evidence

        dynamic_pairs = (dynamic_evidence or {}).get("detected_race_pairs", [])
        if any(claim.entity and claim.entity in str(item) for item in dynamic_pairs):
            evidence.dynamic_confirmed = True
            evidence.entity_resolved = True
            evidence.pre_condition_met = True

        pre_unprotected, pre_note = self._find_unprotected_accesses(pair.pre_src, claim, class_name, func_name)
        evidence.notes.append(pre_note)
        evidence.entity_resolved = evidence.entity_resolved or "entity resolved" in pre_note
        evidence.pre_condition_met = evidence.pre_condition_met or bool(pre_unprotected)

        if not evidence.pre_condition_met:
            if claim.polarity == "present":
                return "refuted", pre_note or "no unprotected entity access found", evidence
            return "inconclusive", "no unprotected entity access in pre-patch", evidence

        post_unprotected, post_note = self._find_unprotected_accesses(pair.post_src, claim, class_name, func_name)
        evidence.notes.append(post_note)
        evidence.post_condition_met = not post_unprotected and bool(pair.post_src)

        if claim.polarity == "absent":
            if evidence.post_condition_met:
                return "confirmed", "unprotected access absent post-patch", evidence
            return "refuted", "unprotected access still present post-patch", evidence

        if pair.post_src and not post_unprotected:
            if not patch_ok:
                return "inconclusive", patch_note, evidence
            return "confirmed", "race existed and patch protects the claimed entity", evidence

        if not patch_ok:
            return "inconclusive", patch_note, evidence
        return "confirmed", "race exists but remains partially unprotected", evidence

    def _find_unprotected_accesses(
        self,
        source: str,
        claim: ThinkClaim,
        class_name: str,
        func_name: str,
    ) -> Tuple[List[cst.CSTNode], str]:
        module = _libcst_parse_safe(source)
        if module is None or cst is None or meta is None:
            return [], "libcst unavailable"
        if not func_name:
            return [], "claim location does not name a function"

        wrapper = meta.MetadataWrapper(module)
        collector = _FunctionAccessCollector(claim.entity, class_name, func_name)
        wrapper.visit(collector)

        if not collector.entity_defined and "." in claim.entity:
            return [], f"entity {claim.entity!r} not resolved in {claim.location}"
        if not collector.accesses:
            return [], f"no access to {claim.entity!r} in {claim.location}"

        analyzer = _SyncGuardAnalyzer(wrapper, module)
        unprotected = [access for access in collector.accesses if not analyzer.guarding_locks(access)]
        return unprotected, f"entity resolved; {len(unprotected)} unprotected access(es)"


class AsyncWaitOracle(OraclePlugin):
    """Verify async wait/timeout symptoms without treating all sleeps as races."""

    category = "async_wait"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        del dynamic_evidence
        file_path, class_name, func_name = _resolve_location(claim.location)
        evidence = OracleEvidence()
        patch_ok, patch_note = _verify_patch_addresses_claim(claim, patch_hunks)
        evidence.patch_addresses_claim = patch_ok
        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", f"file {file_path!r} not in source map", evidence

        pre_tree = pair.pre_ast()
        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable", evidence
        pre_fn = _get_function_src(pre_tree, func_name or claim.entity, class_name)
        evidence.entity_resolved = bool(pre_fn)
        if not pre_fn:
            return "inconclusive", f"function {claim.entity!r} not found", evidence

        pre_issue = self._has_async_wait_issue(pre_fn)
        evidence.pre_condition_met = pre_issue
        if not pre_issue:
            return ("refuted" if claim.polarity == "present" else "inconclusive"), "no async wait smell found", evidence

        post_tree = pair.post_ast()
        if post_tree is None:
            return "confirmed", "async wait smell present pre-patch", evidence
        post_fn = _get_function_src(post_tree, func_name or claim.entity, class_name)
        post_issue = self._has_async_wait_issue(post_fn)
        evidence.post_condition_met = not post_issue
        if not post_issue and patch_ok:
            return "confirmed", "async wait issue fixed", evidence
        if not patch_ok:
            return "inconclusive", patch_note, evidence
        return "confirmed", "async wait issue still present", evidence

    @staticmethod
    def _has_async_wait_issue(func_src: str) -> bool:
        try:
            tree = ast.parse(func_src)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call.endswith("wait_for"):
                    has_timeout = any(k.arg == "timeout" for k in node.keywords)
                    return not has_timeout or any(
                        isinstance(k.value, ast.Constant) and isinstance(k.value.value, (int, float)) and k.value.value < 0.1
                        for k in node.keywords
                        if k.arg == "timeout"
                    )
                if call in _BLOCKING_SYNC_CALLS:
                    return True
        return False


class LRUCacheOracle(OraclePlugin):
    category = "module_cache_pollution"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        del patch_hunks, dynamic_evidence
        file_path, class_name, func_name = _resolve_location(claim.location)
        evidence = OracleEvidence()
        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", "file not in source map", evidence
        pre_tree = pair.pre_ast()
        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable", evidence

        target = func_name or claim.entity
        evidence.entity_resolved = bool(target)
        pre_has_cache = self._has_lru_cache(pre_tree, target)
        evidence.pre_condition_met = pre_has_cache
        if claim.polarity == "present" and not pre_has_cache:
            return "refuted", f"no lru_cache on {target!r} in pre-patch", evidence

        post_tree = pair.post_ast()
        post_has_cache = self._has_lru_cache(post_tree, target) if post_tree else pre_has_cache
        evidence.post_condition_met = not post_has_cache
        if claim.polarity == "absent":
            return ("confirmed" if not post_has_cache else "refuted"), "cache absent post-patch" if not post_has_cache else "cache still present", evidence
        return "confirmed", "cache removed post-patch" if not post_has_cache else "cache still present", evidence

    @staticmethod
    def _has_lru_cache(tree: Optional[ast.AST], entity: str) -> bool:
        if tree is None:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name == entity or not entity):
                for dec in node.decorator_list:
                    dec_src = ast.unparse(dec).lower()
                    if "lru_cache" in dec_src or "cache" in dec_src:
                        return True
        return False


class MockLeakOracle(OraclePlugin):
    category = "mock_residue"
    _MOCK_PATTERNS = ("mock.patch(", "unittest.mock.patch(", "@patch(", "@mock.patch(")

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        del patch_hunks, dynamic_evidence
        file_path, _, _ = _resolve_location(claim.location)
        evidence = OracleEvidence(entity_resolved=True)
        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", "file not in source map", evidence

        pre_leak = self._has_unscoped_mock(pair.pre_src)
        evidence.pre_condition_met = pre_leak
        if claim.polarity == "present" and not pre_leak:
            return "refuted", "no unscoped mock pattern in pre-patch", evidence
        post_leak = self._has_unscoped_mock(pair.post_src)
        evidence.post_condition_met = not post_leak
        if claim.polarity == "absent":
            return ("confirmed" if not post_leak else "refuted"), "mock is scoped post-patch" if not post_leak else "unscoped mock remains", evidence
        return "confirmed", "fixed post-patch" if not post_leak else "leak persists", evidence

    def _has_unscoped_mock(self, source: str) -> bool:
        if not source or not any(p in source for p in self._MOCK_PATTERNS):
            return False
        return not ("with mock.patch" in source or "with patch" in source or ".stop()" in source or "addCleanup" in source)


class _MutationFinder(_CST_VISITOR_BASE):
    def __init__(self, entity: str) -> None:
        self.entity = entity
        self.mutations: List[cst.CSTNode] = []

    def visit_Assign(self, node: cst.Assign) -> None:
        for target in node.targets:
            if _targets_entity(target.target, self.entity):
                self.mutations.append(node)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if _targets_entity(node.target, self.entity):
            self.mutations.append(node)

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        if _targets_entity(node.target, self.entity):
            self.mutations.append(node)

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Attribute) and node.func.attr.value in _MUTATING_METHODS:
            if _targets_entity(node.func.value, self.entity):
                self.mutations.append(node)


class SharedStateOracle(OraclePlugin):
    category = "shared_state"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        del patch_hunks, dynamic_evidence
        file_path, _, _ = _resolve_location(claim.location)
        evidence = OracleEvidence()
        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", "file not in source map", evidence

        pre_mutations = self._find_mutations(pair.pre_src, claim.entity)
        evidence.entity_resolved = bool(claim.entity)
        evidence.pre_condition_met = bool(pre_mutations)
        if claim.polarity == "present" and not pre_mutations:
            return "refuted", f"no mutation of {claim.entity!r} found", evidence

        post_mutations = self._find_mutations(pair.post_src, claim.entity)
        evidence.post_condition_met = not post_mutations
        if claim.polarity == "absent":
            return ("confirmed" if not post_mutations else "refuted"), "mutation absent post-patch" if not post_mutations else "mutation remains", evidence
        return "confirmed", "mutation removed post-patch" if not post_mutations else "mutation persists", evidence

    @staticmethod
    def _find_mutations(source: str, entity: str) -> List[Any]:
        module = _libcst_parse_safe(source)
        if module is not None and cst is not None:
            visitor = _MutationFinder(entity)
            module.visit(visitor)
            return visitor.mutations
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        mutations: List[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = getattr(node, "targets", [getattr(node, "target", None)])
                if any(t is not None and entity in ast.unparse(t) for t in targets):
                    mutations.append(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _MUTATING_METHODS and entity in ast.unparse(node.func.value):
                    mutations.append(node)
        return mutations


class TestOrderOracle(SharedStateOracle):
    category = "test_order_dependency"


class FixtureScopeOracle(OraclePlugin):
    category = "fixture_scope_leak"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        del patch_hunks, dynamic_evidence
        file_path, _, func_name = _resolve_location(claim.location)
        evidence = OracleEvidence()
        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", "file not in source map", evidence
        pre_tree = pair.pre_ast()
        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable", evidence

        target = func_name or claim.entity
        evidence.entity_resolved = bool(target)
        pre_leak = self._has_scope_leak(pre_tree, target)
        evidence.pre_condition_met = pre_leak
        if claim.polarity == "present" and not pre_leak:
            return "refuted", f"no fixture scope leak found for {target!r}", evidence
        post_tree = pair.post_ast()
        post_leak = self._has_scope_leak(post_tree, target) if post_tree else pre_leak
        evidence.post_condition_met = not post_leak
        if claim.polarity == "absent":
            return ("confirmed" if not post_leak else "refuted"), "fixture scope clean post-patch" if not post_leak else "fixture leak remains", evidence
        return "confirmed", "fixed post-patch" if not post_leak else "leak persists", evidence

    @staticmethod
    def _has_scope_leak(tree: Optional[ast.AST], entity: str) -> bool:
        if tree is None:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if entity and node.name != entity:
                    continue
                for dec in node.decorator_list:
                    dec_src = ast.unparse(dec)
                    if "fixture" in dec_src and ("session" in dec_src or "module" in dec_src):
                        if not any(isinstance(n, ast.Yield) for n in ast.walk(node)):
                            return True
        return False


class PatternOracle(OraclePlugin):
    """Small static oracle for categories without a full semantic verifier."""

    category = "unknown"
    patterns: Tuple[str, ...] = ()

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
        *,
        patch_hunks: Sequence[PatchHunk] = (),
        dynamic_evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, OracleEvidence]:
        del patch_hunks, dynamic_evidence
        file_path, _, _ = _resolve_location(claim.location)
        evidence = OracleEvidence(entity_resolved=bool(claim.location))
        pair = _get_pair(file_path, source_map)
        if pair is None:
            return "inconclusive", "file not in source map", evidence
        pre_hit = any(re.search(p, pair.pre_src) for p in self.patterns)
        post_hit = any(re.search(p, pair.post_src) for p in self.patterns) if pair.post_src else pre_hit
        evidence.pre_condition_met = pre_hit
        evidence.post_condition_met = not post_hit
        if not pre_hit:
            return ("refuted" if claim.polarity == "present" else "inconclusive"), f"no static pattern for {self.category}", evidence
        if claim.polarity == "absent":
            return ("confirmed" if not post_hit else "refuted"), "pattern absent post-patch" if not post_hit else "pattern remains", evidence
        return "confirmed", "pattern removed post-patch" if not post_hit else "static pattern present", evidence


class ResourceLeakOracle(PatternOracle):
    category = "resource_leak"
    patterns = (r"\bopen\s*\(", r"\bsocket\s*\(", r"\bconnect\s*\(", r"\.close\s*\(")


class NetworkOracle(PatternOracle):
    category = "network"
    patterns = (r"\brequests\.(get|post|put|delete)\s*\(", r"urllib\.request", r"httpx\.")


class ImportSideEffectOracle(PatternOracle):
    category = "import_side_effect"
    patterns = (r"(?m)^[A-Za-z_]\w*\s*=\s*(?![\"'\d\[\]\{\}\(\)]+$).+", r"(?m)^[A-Za-z_][\w\.]+\s*\(")


class PlatformDependencyOracle(PatternOracle):
    category = "platform_dependency"
    patterns = (r"sys\.platform", r"os\.name", r"platform\.system", r"[A-Za-z]:\\\\", r"/tmp/")


class NondeterminismOracle(PatternOracle):
    category = "nondeterminism"
    patterns = (r"random\.", r"uuid\.uuid", r"datetime\.now", r"time\.time\(", r"secrets\.")


class PatchCoherenceOracle:
    """Cross-cutting check that patch mechanics match the claim mechanism."""

    def verify(self, claim: ThinkClaim, post_sources: Dict[str, str], patch_hunks: Sequence[PatchHunk]) -> Tuple[bool, str]:
        relevant = _iter_hunks_for_claim(claim, patch_hunks)
        if not relevant:
            return False, "no hunk linked to claim"
        file_path, _, _ = _resolve_location(claim.location)
        post_src = post_sources.get(file_path, "")
        replacement = "\n".join(h.replace for h in relevant)
        reason = claim.reason.lower()

        if claim.category in {"concurrency", "async_wait"} or "lock" in reason:
            has_instantiation = _contains_sync_primitive(post_src) or _contains_sync_primitive(replacement)
            has_usage = _contains_sync_usage(post_src) or _contains_sync_usage(replacement)
            if has_instantiation and not has_usage:
                return False, "lock instantiated but never acquired or used as context manager"

        deleted_function = bool(re.search(r"^\s*(def|async def)\s+", "\n".join(h.search for h in relevant), re.MULTILINE)) and not replacement.strip()
        if deleted_function:
            return False, "patch deletes the claimed function instead of fixing it"

        return True, "patch coherence accepted"


_PLUGIN_REGISTRY: Dict[str, OraclePlugin] = {}


def _register(*plugins: OraclePlugin) -> None:
    for plugin in plugins:
        _PLUGIN_REGISTRY[plugin.category] = plugin


_register(
    RaceConditionOracle(),
    AsyncWaitOracle(),
    LRUCacheOracle(),
    MockLeakOracle(),
    SharedStateOracle(),
    TestOrderOracle(),
    FixtureScopeOracle(),
    ResourceLeakOracle(),
    NetworkOracle(),
    ImportSideEffectOracle(),
    PlatformDependencyOracle(),
    NondeterminismOracle(),
)


def _score_claim(verdict: str, evidence: OracleEvidence) -> float:
    if verdict == "refuted":
        return -1.0
    if verdict == "unverified":
        return 0.0

    score = 0.0
    if evidence.entity_resolved:
        score += 0.2
    if evidence.pre_condition_met:
        score += 0.3
    if evidence.post_condition_met:
        score += 0.3
    if evidence.patch_addresses_claim:
        score += 0.2
    if evidence.dynamic_confirmed:
        score = max(score, 0.95)

    if verdict == "inconclusive":
        score = min(score, 0.55)
    return round(min(1.0, score), 4)


def verify_structured_think(
    structured: StructuredThink,
    pre_sources: Dict[str, str],
    post_sources: Dict[str, str],
    *,
    patch_hunks: Sequence[PatchHunk] = (),
    dynamic_evidence: Optional[Dict[str, Any]] = None,
) -> Tuple[StructuredThink, float]:
    """Verify all claims against pre/post source and patch coherence."""
    if not structured.claims:
        raw_score = structured.format_penalty
        return structured, float(max(-1.0, min(1.0, raw_score)))

    source_map = _build_source_map(pre_sources, post_sources)
    coherence = PatchCoherenceOracle()
    annotated_claims: List[ThinkClaim] = []
    claim_scores: List[float] = []

    for claim in structured.claims:
        plugin = _PLUGIN_REGISTRY.get(claim.category)
        if plugin is None:
            verdict = "inconclusive"
            note = f"category {claim.category!r} has no static oracle"
            evidence = OracleEvidence(entity_resolved=bool(claim.location))
        else:
            try:
                verdict, note, evidence = plugin.verify(
                    claim,
                    source_map,
                    patch_hunks=patch_hunks,
                    dynamic_evidence=dynamic_evidence,
                )
            except Exception as exc:
                verdict, note, evidence = "unverified", f"oracle error: {exc}", OracleEvidence()
                logger.warning("[ORACLE] %s claim=%s error: %s", claim.category, claim.claim_id, exc)

        coherent, coherence_note = coherence.verify(claim, post_sources, patch_hunks)
        if patch_hunks:
            evidence.patch_addresses_claim = evidence.patch_addresses_claim and coherent
            if not coherent and verdict == "confirmed":
                verdict = "inconclusive"
                note = coherence_note

        score = _score_claim(verdict, evidence)
        claim_scores.append(score)
        annotated_claims.append(claim.model_copy(update={"verdict": verdict, "oracle_score": score}))

        logger.debug(
            "[ORACLE] claim=%s cat=%s verdict=%s score=%.2f note=%s coherence=%s",
            claim.claim_id,
            claim.category,
            verdict,
            score,
            note,
            coherence_note,
        )

    mean_claim_score = sum(claim_scores) / len(claim_scores)
    raw_score = mean_claim_score + structured.format_penalty
    oracle_score = float(max(-1.0, min(1.0, raw_score)))
    return structured.model_copy(update={"claims": annotated_claims}), oracle_score
