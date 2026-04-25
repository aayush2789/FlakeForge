"""Oracle Engine — libcst-based differential claim verification."""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

try:
    import libcst as cst
    import libcst.metadata as meta
    _LIBCST_AVAILABLE = True
except ImportError:
    _LIBCST_AVAILABLE = False

try:
    from models import StructuredThink, ThinkClaim
except ImportError:
    from ..models import StructuredThink, ThinkClaim

logger = logging.getLogger(__name__)


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
    """Merge pre/post dicts (keyed by relative file path) into _SourcePairs."""
    all_keys = set(pre_sources) | set(post_sources)
    return {
        k: _SourcePair(pre_sources.get(k, ""), post_sources.get(k, ""))
        for k in all_keys
    }


class OraclePlugin(ABC):
    """Base class for a single-category claim verifier."""

    category: str = ""  # Must be overridden.

    @abstractmethod
    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
    ) -> Tuple[str, str]:
        """Return (verdict, reason_note).

        verdict  ∈ {"confirmed", "inconclusive", "refuted", "unverified"}
        reason_note: short string that will be logged.
        """


def _resolve_location(location: str) -> Tuple[str, str, str]:
    """Parse 'path/to/file.py::ClassName.method' → (file, class_name, func_name)."""
    if "::" in location:
        file_part, qual = location.split("::", 1)
        parts = qual.rsplit(".", 1)
        if len(parts) == 2:
            return file_part, parts[0], parts[1]
        return file_part, "", qual
    return location, "", ""


def _get_function_src(tree: ast.AST, func_name: str, class_name: str = "") -> str:
    """Extract the source of a specific function using the stdlib ast module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and class_name:
            if node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == func_name:
                            return ast.unparse(item)
        if not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                return ast.unparse(node)
    return ""


def _count_occurrences(src: str, pattern: str) -> int:
    return src.count(pattern)


def _libcst_parse_safe(source: str) -> Optional[cst.Module]:
    if not _LIBCST_AVAILABLE or not source:
        return None
    try:
        return cst.parse_module(source)
    except Exception:
        return None


def _qualified_names_in_function(
    source: str,
    func_name: str,
) -> List[str]:
    """Return qualified names referenced inside *func_name* using libcst metadata."""
    module = _libcst_parse_safe(source)
    if module is None:
        return []
    try:
        wrapper = meta.MetadataWrapper(module)

        class _Collector(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (meta.QualifiedNameProvider,)

            def __init__(self) -> None:
                self.names: List[str] = []
                self._in_func = False

            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                if node.name.value == func_name:
                    self._in_func = True

            def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
                self._in_func = False

            def visit_Name(self, node: cst.Name) -> None:
                if self._in_func:
                    try:
                        qnames = self.get_metadata(meta.QualifiedNameProvider, node, set())
                        self.names.extend(q.name for q in qnames)
                    except Exception:
                        pass

        collector = _Collector()
        wrapper.visit(collector)
        return collector.names
    except Exception:
        return []


_SYNC_PRIMITIVES = {
    "asyncio.Lock", "asyncio.Semaphore", "asyncio.Event",
    "threading.Lock", "threading.RLock", "threading.Semaphore",
    "anyio.Lock", "trio.Lock",
}


class RaceConditionOracle(OraclePlugin):
    """Verify async_wait / concurrency claims.

    Checks whether the named entity uses asyncio.wait_for / sleep without
    adequate timeout and whether the post-patch version corrects it.
    """

    category = "async_wait"

    _RACE_PATTERNS = (
        "wait_for",
        "sleep(",
        "asyncio.wait_for",
        "timeout=",
        "random.random",
        "queue_full",
        "QUEUE_CAPACITY",
    )
    _SYNC_PATTERNS = tuple(s.split(".")[-1].lower() for s in _SYNC_PRIMITIVES)

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
    ) -> Tuple[str, str]:
        file_path, class_name, func_name = _resolve_location(claim.location)

        pair = source_map.get(file_path) or self._fuzzy_file(file_path, source_map)
        if pair is None:
            return "inconclusive", f"file {file_path!r} not in source map"

        pre_tree = pair.pre_ast()
        post_tree = pair.post_ast()

        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable"

        pre_fn = _get_function_src(pre_tree, func_name or claim.entity, class_name)
        if not pre_fn:
            return "inconclusive", f"function {claim.entity!r} not found in pre-patch"

        pre_has_race = any(p in pre_fn for p in self._RACE_PATTERNS)

        if not pre_has_race:
            if claim.polarity == "present":
                return "refuted", "no timing/wait pattern found in pre-patch function"
            return "inconclusive", "no race pattern present — claim polarity=absent may be valid"

        # Race pattern confirmed in pre-patch.
        if claim.polarity == "present":
            if post_tree is None:
                return "confirmed", "race pattern present in pre-patch (no post-patch to diff)"
            post_fn = _get_function_src(post_tree, func_name or claim.entity, class_name)
            if not post_fn:
                return "confirmed", "race pattern present; function removed in post-patch"
            post_has_race = any(p in post_fn for p in self._RACE_PATTERNS)
            note = "fixed in post-patch" if not post_has_race else "still present in post-patch"
            return "confirmed", note

        # polarity == "absent": race should be gone post-patch.
        if post_tree is None:
            return "inconclusive", "race present pre-patch; no post-patch to verify removal"
        post_fn = _get_function_src(post_tree, func_name or claim.entity, class_name)
        post_has_race = any(p in post_fn for p in self._RACE_PATTERNS)
        if not post_has_race:
            return "confirmed", "race pattern removed in post-patch as claimed"
        return "refuted", "race pattern still present in post-patch despite polarity=absent"

    @staticmethod
    def _fuzzy_file(
        path: str, source_map: Dict[str, _SourcePair]
    ) -> Optional[_SourcePair]:
        base = Path(path).name
        for k, v in source_map.items():
            if Path(k).name == base:
                return v
        return None


class LRUCacheOracle(OraclePlugin):
    """Verify module_cache_pollution claims.

    Detects @functools.lru_cache / @lru_cache decorators and checks whether
    the post-patch removes or bounds them correctly.
    """

    category = "module_cache_pollution"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
    ) -> Tuple[str, str]:
        file_path, class_name, func_name = _resolve_location(claim.location)
        pair = source_map.get(file_path) or RaceConditionOracle._fuzzy_file(
            file_path, source_map
        )
        if pair is None:
            return "inconclusive", "file not in source map"

        pre_tree = pair.pre_ast()
        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable"

        target = func_name or claim.entity
        pre_has_cache = self._has_lru_cache(pre_tree, target)

        if claim.polarity == "present":
            if not pre_has_cache:
                return "refuted", f"no lru_cache on {target!r} in pre-patch"
            # Verify post removes / bounds it.
            post_tree = pair.post_ast()
            if post_tree is None:
                return "confirmed", "lru_cache present in pre-patch"
            post_has_cache = self._has_lru_cache(post_tree, target)
            note = "cache removed post-patch" if not post_has_cache else "cache still present"
            return "confirmed", note

        # polarity == "absent"
        post_tree = pair.post_ast()
        if post_tree is None:
            return "inconclusive", "no post-patch source"
        if not self._has_lru_cache(post_tree, target):
            return "confirmed", "lru_cache absent in post-patch as claimed"
        return "refuted", "lru_cache still present post-patch"

    @staticmethod
    def _has_lru_cache(tree: ast.AST, entity: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == entity or not entity:
                    for dec in node.decorator_list:
                        dec_src = ast.unparse(dec).lower()
                        if "lru_cache" in dec_src or "cache" in dec_src:
                            return True
        return False


class MockLeakOracle(OraclePlugin):
    """Verify mock_residue claims.

    Checks whether mock.patch() calls are properly scoped via context managers
    or .stop() calls.  Uses libcst if available for precise scope analysis.
    """

    category = "mock_residue"

    _MOCK_PATTERNS = ("mock.patch(", "unittest.mock.patch(", "@patch(", "@mock.patch(")

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
    ) -> Tuple[str, str]:
        file_path, _, _ = _resolve_location(claim.location)
        pair = source_map.get(file_path) or RaceConditionOracle._fuzzy_file(
            file_path, source_map
        )
        if pair is None:
            return "inconclusive", "file not in source map"

        pre_leak = self._has_unscoped_mock(pair.pre_src)

        if claim.polarity == "present":
            if not pre_leak:
                return "refuted", "no unscoped mock pattern in pre-patch"
            post_leak = self._has_unscoped_mock(pair.post_src)
            note = "fixed post-patch" if not post_leak else "leak persists post-patch"
            return "confirmed", note

        post_leak = self._has_unscoped_mock(pair.post_src)
        if not post_leak:
            return "confirmed", "no unscoped mocks in post-patch"
        return "refuted", "unscoped mock still present post-patch"

    def _has_unscoped_mock(self, source: str) -> bool:
        if not source:
            return False
        has_patch = any(p in source for p in self._MOCK_PATTERNS)
        if not has_patch:
            return False
        scoped = ("with mock.patch" in source or "with patch" in source or
                  ".stop()" in source or "addCleanup" in source)
        return not scoped


class SharedStateOracle(OraclePlugin):
    """Verify shared_state / test_order_dependency claims.

    Checks for global variable mutations and class-level state that survive
    between tests.
    """

    category = "shared_state"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
    ) -> Tuple[str, str]:
        file_path, class_name, func_name = _resolve_location(claim.location)
        pair = source_map.get(file_path) or RaceConditionOracle._fuzzy_file(
            file_path, source_map
        )
        if pair is None:
            return "inconclusive", "file not in source map"

        pre_tree = pair.pre_ast()
        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable"

        entity = claim.entity
        pre_global = self._has_global_mutation(pre_tree, entity)

        if claim.polarity == "present":
            if not pre_global:
                return "refuted", f"no global/class-level mutation of {entity!r} found"
            post_tree = pair.post_ast()
            if post_tree is None:
                return "confirmed", "global mutation present in pre-patch"
            post_global = self._has_global_mutation(post_tree, entity)
            note = "mutation removed post-patch" if not post_global else "mutation persists"
            return "confirmed", note

        post_tree = pair.post_ast()
        if post_tree is None:
            return "inconclusive", "no post-patch source"
        post_global = self._has_global_mutation(post_tree, entity)
        if not post_global:
            return "confirmed", "no shared state mutation in post-patch"
        return "refuted", "shared state mutation still present post-patch"

    @staticmethod
    def _has_global_mutation(tree: ast.AST, entity: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                if not entity or entity in node.names:
                    return True
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Store):
                        if not entity or entity in ast.unparse(t):
                            return True
        return False


class FixtureScopeOracle(OraclePlugin):
    """Verify fixture_scope_leak claims.

    Looks for session/module-scoped fixtures that return mutable objects
    without a yield + teardown pattern.
    """

    category = "fixture_scope_leak"

    def verify(
        self,
        claim: ThinkClaim,
        source_map: Dict[str, _SourcePair],
    ) -> Tuple[str, str]:
        file_path, _, func_name = _resolve_location(claim.location)
        pair = source_map.get(file_path) or RaceConditionOracle._fuzzy_file(
            file_path, source_map
        )
        if pair is None:
            return "inconclusive", "file not in source map"

        pre_tree = pair.pre_ast()
        if pre_tree is None:
            return "inconclusive", "pre-patch AST unavailable"

        target = func_name or claim.entity
        pre_leak = self._has_scope_leak(pre_tree, target)

        if claim.polarity == "present":
            if not pre_leak:
                return "refuted", f"no fixture scope leak found for {target!r}"
            post_tree = pair.post_ast()
            if post_tree is None:
                return "confirmed", "fixture scope leak present in pre-patch"
            post_leak = self._has_scope_leak(post_tree, target)
            return "confirmed", "fixed post-patch" if not post_leak else "leak persists"

        post_tree = pair.post_ast()
        if post_tree is None:
            return "inconclusive", "no post-patch source"
        if not self._has_scope_leak(post_tree, target):
            return "confirmed", "fixture scope clean in post-patch"
        return "refuted", "fixture scope leak still present post-patch"

    @staticmethod
    def _has_scope_leak(tree: ast.AST, entity: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if entity and node.name != entity:
                    continue
                for dec in node.decorator_list:
                    dec_src = ast.unparse(dec)
                    if "fixture" in dec_src and ("session" in dec_src or "module" in dec_src):
                        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))
                        if not has_yield:
                            return True
        return False


_PLUGIN_REGISTRY: Dict[str, OraclePlugin] = {}


def _register(*plugins: OraclePlugin) -> None:
    for p in plugins:
        _PLUGIN_REGISTRY[p.category] = p
        _PLUGIN_REGISTRY.setdefault("concurrency", RaceConditionOracle())
    _PLUGIN_REGISTRY.setdefault("test_order_dependency", SharedStateOracle())


_register(
    RaceConditionOracle(),
    LRUCacheOracle(),
    MockLeakOracle(),
    SharedStateOracle(),
    FixtureScopeOracle(),
)


_VERDICT_SCORES: Dict[str, float] = {
    "confirmed": 1.0,
    "inconclusive": 0.2,
    "refuted": -1.0,
    "unverified": 0.0,
}


def verify_structured_think(
    structured: StructuredThink,
    pre_sources: Dict[str, str],
    post_sources: Dict[str, str],
) -> Tuple[StructuredThink, float]:
    """Verify all claims in *structured* against pre/post patch sources.

    Returns (annotated_structured_think, oracle_score_in_[-1,1]).

    oracle_score is the mean claim score, offset by the format penalty.
    An all-inconclusive result gives 0.2; confirmed+removed gives 1.0.
    If structured.claims is empty or format_penalty is heavily negative,
    the score will be in [-1, 0].
    """
    if not structured.claims:
        # Format-penalised but no claims → base score from format only.
        raw_score = structured.format_penalty  # ≤ 0
        return structured, float(max(-1.0, min(1.0, raw_score)))

    source_map = _build_source_map(pre_sources, post_sources)

    annotated_claims: List[ThinkClaim] = []
    claim_scores: List[float] = []

    for claim in structured.claims:
        plugin = _PLUGIN_REGISTRY.get(claim.category)
        if plugin is None:
            verdict, note = "unverified", f"no oracle for category {claim.category!r}"
        else:
            try:
                verdict, note = plugin.verify(claim, source_map)
            except Exception as exc:
                verdict, note = "unverified", f"oracle error: {exc}"
                logger.warning("[ORACLE] %s claim=%s error: %s", claim.category, claim.claim_id, exc)

        score = _VERDICT_SCORES[verdict]
        claim_scores.append(score)

        updated = claim.model_copy(update={"verdict": verdict, "oracle_score": score})
        annotated_claims.append(updated)

        logger.debug(
            "[ORACLE] claim=%s cat=%s verdict=%s score=%.1f note=%s",
            claim.claim_id, claim.category, verdict, score, note,
        )

    mean_claim_score = sum(claim_scores) / len(claim_scores)

    raw_score = mean_claim_score + structured.format_penalty
    oracle_score = float(max(-1.0, min(1.0, raw_score)))

    new_struct = structured.model_copy(update={"claims": annotated_claims})
    return new_struct, oracle_score
