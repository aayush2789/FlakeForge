"""Patch Validation Engine — code-side verification (not reasoning).

OracleEngine answers: "Is the structured thinking consistent with the code?"
PatchValidator answers: "Is the patch well-formed, applicable, safe, and meaningful?"

Pipeline (stages):
  1. Format — SEARCH/REPLACE blocks, file headers
  2. Apply simulation — SEARCH must match source (via simulate_search_replace_patch)
  3. Syntax — ast.parse on post-patch text
  4. Compile — compile() on each modified module
  5. Structure — e.g. empty function bodies, broken control flow heuristics
  6. Causal proximity — optional warning if patch files far from failure frontier

Invalid patches must be rejected *before* disk writes; see FlakeForgeEnvironment.step.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import libcst as cst
    _LIBCST_AVAILABLE = True
except ImportError:
    cst = None
    _LIBCST_AVAILABLE = False

try:
    from server.patch_applier import parse_search_replace_hunks, simulate_search_replace_patch
except ImportError:
    try:
        from ..server.patch_applier import parse_search_replace_hunks, simulate_search_replace_patch
    except ImportError:
        from FlakeForge.server.patch_applier import parse_search_replace_hunks, simulate_search_replace_patch


@dataclass
class ValidationResult:
    """Outcome of patch validation (action / code path, not oracle)."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0  # 0..1 for reward shaping when is_valid
    simulate_result: Dict[str, Any] = field(default_factory=dict)


_SLEEP_PATTERNS = ("time.sleep(", "await asyncio.sleep(", "sleep(")
_SKIP_PATTERNS = ("@pytest.mark.skip", "@unittest.skip", "pytest.skip(")
_FLAKINESS_PATTERNS = (
    ("flaky_time_sleep", "time.sleep("),
    ("flaky_asyncio_sleep", "asyncio.sleep("),
    ("flaky_random_random", "random.random("),
    ("flaky_datetime_now", "datetime.now("),
    ("flaky_datetime_utcnow", "datetime.utcnow("),
)


def _normalise_rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _resolve_claim_location(location: str) -> Tuple[str, str, str]:
    """Return (rel_path, class_name, function_or_entity) from claim.location."""
    if "::" not in location:
        return _normalise_rel(location), "", ""
    file_part, qual = location.split("::", 1)
    bits = qual.rsplit(".", 1)
    if len(bits) == 2:
        return _normalise_rel(file_part), bits[0], bits[1]
    return _normalise_rel(file_part), "", qual


def _claim_value(claim: Any, name: str, default: str = "") -> str:
    if isinstance(claim, dict):
        value = claim.get(name, default)
    else:
        value = getattr(claim, name, default)
    return str(value or "")


def _find_source_key(path: str, source_map: Dict[str, str]) -> Optional[str]:
    path = _normalise_rel(path)
    if path in source_map:
        return path
    base = Path(path).name
    for key in source_map:
        if Path(key).name == base:
            return key
    return None


def _node_text(source: str, node: ast.AST) -> str:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    lines = source.splitlines()
    start = max(int(node.lineno) - 1, 0)
    end = int(node.end_lineno)
    return "\n".join(lines[start:end])


def _find_entity_node(tree: ast.AST, entity: str, class_name: str = "") -> Optional[ast.AST]:
    if not entity:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and class_name and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == entity:
                    return child
        if not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == entity:
                return node
    return None


def _decorator_contains(node: ast.AST, token: str) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    for dec in getattr(node, "decorator_list", []):
        if token in ast.unparse(dec):
            return True
    return False


def _added_lines_for_hunks(hunks: Sequence[Any]) -> List[str]:
    """Return approximate added lines, enough for anti-hack/flakiness checks."""
    added: List[str] = []
    for hunk in hunks:
        search_counts: Dict[str, int] = {}
        for line in hunk.search_text.splitlines():
            search_counts[line] = search_counts.get(line, 0) + 1
        for line in hunk.replace_text.splitlines():
            count = search_counts.get(line, 0)
            if count:
                search_counts[line] = count - 1
            else:
                added.append(line)
    return added


def _added_lines_from_diff(diff_text: str) -> List[str]:
    """Return real added lines from a unified diff, ignoring file/hunk headers."""
    added: List[str] = []
    for line in (diff_text or "").splitlines():
        if not line.startswith("+"):
            continue
        if line.startswith("+++") or line.startswith("+@@"):
            continue
        added.append(line[1:])
    return added


def _hard_anti_hack_errors(hunks: Sequence[Any]) -> List[str]:
    errors: List[str] = []
    for hunk in hunks:
        search_lower = hunk.search_text.lower()
        replace_lower = hunk.replace_text.lower()

        if replace_lower.count("assert") < search_lower.count("assert"):
            errors.append("anti_hack_assertion_deletion")

        for pattern in _SLEEP_PATTERNS:
            if replace_lower.count(pattern) > search_lower.count(pattern):
                errors.append("anti_hack_sleep_injection")
                break

        for pattern in _SKIP_PATTERNS:
            if replace_lower.count(pattern) > search_lower.count(pattern):
                errors.append("anti_hack_test_skip")
                break

        introduced = "\n".join(_added_lines_for_hunks([hunk]))
        if re.search(r"except\s*(?:Exception)?\s*:\s*(?:\n\s*)?pass\b", introduced):
            errors.append("anti_hack_exception_swallowing")

    return sorted(set(errors))


def _structural_issues(tree: ast.AST, rel_path: str) -> List[str]:
    """Detect obviously broken structure (empty bodies, etc.)."""
    issues: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.body) == 0:
                issues.append(f"{rel_path}: empty body for function {node.name!r}")
        if isinstance(node, ast.ClassDef):
            if len(node.body) == 0:
                issues.append(f"{rel_path}: empty class body for {node.name!r}")
    return issues


def _causal_proximity_warnings(
    modified_rel_paths: List[str],
    failure_frontier: str,
    call_chain: Optional[List[str]],
) -> List[str]:
    """Warn when patched files are far from the failure frontier (mirrors reward signal)."""
    if not modified_rel_paths or not failure_frontier:
        return []

    frontier_file = failure_frontier.split(":")[0] if ":" in failure_frontier else failure_frontier
    frontier_name = Path(frontier_file.replace("\\", "/")).name

    hit = False
    for rel in modified_rel_paths:
        if Path(rel.replace("\\", "/")).name == frontier_name:
            hit = True
            break
        pf_name = Path(rel.replace("\\", "/")).name.replace(".py", "")
        if call_chain:
            for frame in call_chain:
                if pf_name in frame:
                    hit = True
                    break
        if hit:
            break

    if not hit:
        return [
            f"patch targets {modified_rel_paths} but failure frontier is {failure_frontier!r} "
            "(may be a workaround, not a localised fix)",
        ]
    return []


def _defined_names(tree: ast.AST) -> Set[str]:
    names: Set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    names.add(arg.arg)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_target_names(node.target))
        elif isinstance(node, ast.For):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
    return names


def _target_names(target: ast.AST) -> Set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: Set[str] = set()
        for elt in target.elts:
            out.update(_target_names(elt))
        return out
    return set()


def _introduced_name_errors(src: str, added_lines: Sequence[str], rel: str) -> List[str]:
    """Catch common undefined names introduced by flaky-fix patches."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    available = _defined_names(tree)
    text = "\n".join(added_lines)

    candidates = {
        "threading", "asyncio", "pytest", "random", "datetime", "time",
        "Lock", "RLock", "Semaphore", "Event",
    }
    errors: List[str] = []
    for name in sorted(candidates):
        if not re.search(rf"\b{re.escape(name)}\b", text):
            continue
        if name in available:
            continue
        # Attribute access like threading.Lock requires the module name. Bare
        # RLock requires a direct import or definition.
        errors.append(f"undefined_name: {name} in {rel}")
    return errors


def _libcst_errors(src: str, rel: str) -> List[str]:
    if not _LIBCST_AVAILABLE or cst is None:
        return []
    try:
        module = cst.parse_module(src)
    except Exception as exc:
        return [f"libcst_parse_error in {rel}: {exc}"]
    try:
        if module.code != src:
            return [f"libcst_roundtrip_mismatch in {rel}"]
    except Exception as exc:
        return [f"libcst_roundtrip_error in {rel}: {exc}"]
    return []


def _reasoning_alignment_errors(
    claims: Optional[Sequence[Any]],
    original_sources: Dict[str, str],
    modified_sources: Dict[str, str],
) -> List[str]:
    if not claims:
        return []

    errors: List[str] = []
    for claim in claims:
        location = _claim_value(claim, "location")
        category = _claim_value(claim, "category")
        entity = _claim_value(claim, "entity")
        reason = _claim_value(claim, "reason").lower()
        file_path, class_name, func_name = _resolve_claim_location(location)
        target_entity = func_name or entity

        key = _find_source_key(file_path, modified_sources)
        if key is None:
            errors.append(
                f"reasoning_action_misalignment: claim targets {file_path or '<unknown>'} "
                "but patch modifies different files"
            )
            continue

        pre = original_sources.get(key, "")
        post = modified_sources.get(key, "")
        try:
            pre_tree = ast.parse(pre) if pre else None
            post_tree = ast.parse(post) if post else None
        except SyntaxError:
            continue

        if target_entity and pre_tree is not None and post_tree is not None:
            pre_node = _find_entity_node(pre_tree, target_entity, class_name)
            post_node = _find_entity_node(post_tree, target_entity, class_name)
            if pre_node is not None and post_node is not None:
                if _node_text(pre, pre_node) == _node_text(post, post_node):
                    errors.append(
                        f"reasoning_action_misalignment: claim targets {key}::{target_entity} "
                        "but that entity was not changed"
                    )

                if category == "module_cache_pollution":
                    if _decorator_contains(post_node, "lru_cache") or _decorator_contains(post_node, "cache"):
                        errors.append(
                            f"reasoning_action_misalignment: cache decorator still present on {key}::{target_entity}"
                        )

                post_node_text = _node_text(post, post_node)
                uses_sync_primitive = bool(
                    re.search(r"\b(Lock|RLock|Semaphore|Event)\b", post_node_text)
                    or re.search(r"\bwith\s+[\w.]*_(?:lock|rlock|semaphore|event)\s*:", post_node_text)
                    or re.search(r"\bwith\s+[\w.]*\.(?:lock|rlock|semaphore|event)\s*:", post_node_text)
                    or re.search(r"\b(?:acquire|release)\s*\(", post_node_text)
                )
                if ("lock" in reason or "semaphore" in reason) and not uses_sync_primitive:
                    errors.append(
                        f"reasoning_action_misalignment: claim mentions synchronization "
                        f"but {key}::{target_entity} does not use a sync primitive"
                    )
            elif pre_node is not None and post_node is None:
                errors.append(
                    f"reasoning_action_misalignment: claim target {key}::{target_entity} was removed"
                )

    return sorted(set(errors))


def _flakiness_smell_errors(added_lines: Sequence[str]) -> Tuple[List[str], List[str]]:
    text = "\n".join(added_lines)
    errors: List[str] = []
    warnings: List[str] = []

    for code, pattern in _FLAKINESS_PATTERNS:
        if pattern in text:
            errors.append(code)

    # New module-level mutable/global-ish assignments are a flaky-test smell.
    for line in added_lines:
        stripped = line.strip()
        if not stripped or line[:1].isspace():
            continue
        if re.match(r"^[A-Z_a-z]\w*\s*=\s*(\[\]|\{\}|set\(\)|dict\(\)|list\(\))", stripped):
            errors.append("flaky_global_mutable_assignment")

    if re.search(r"@\s*(?:functools\.)?lru_cache\b", text) or "lru_cache(" in text:
        warnings.append("potential_new_cache_pollution")

    return sorted(set(errors)), sorted(set(warnings))


def _idempotency_issues(
    repo_path: Path,
    patch_text: str,
    default_target: Optional[str],
    modified_sources: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    second = simulate_search_replace_patch(
        repo_path,
        patch_text,
        default_target=default_target,
        pre_sources=modified_sources,
    )
    if not second.get("success"):
        return [], [f"non_idempotent_patch: {second.get('error') or 'second_apply_failed'}"]

    second_modified = second.get("modified_sources") or {}
    for rel, src in second_modified.items():
        if modified_sources.get(rel) != src:
            return ["non_idempotent_patch"], []
    return [], []


class PatchValidator:
    """Validate model-produced patches before they touch the repo on disk."""

    def validate(
        self,
        patch_text: str,
        *,
        repo_path: Path,
        pre_sources: Optional[Dict[str, str]] = None,
        claims: Optional[Sequence[Any]] = None,
        default_target: Optional[str] = None,
        failure_frontier: str = "",
        call_chain: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Run all validation stages. Does not write files.

        Args:
            patch_text: Raw model patch (SEARCH/REPLACE hunks).
            pre_sources: Optional snapshot rel path -> text; overrides disk for simulation.
            claims: Optional structured think claims; used for reasoning-action alignment.
            repo_path: Repository root.
            default_target: File path when hunks omit ``---`` header.
            failure_frontier: From observation (for proximity warnings).
            call_chain: Call chain strings (for proximity warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # ── Stage 1: format ─────────────────────────────────────────────────
        text = (patch_text or "").strip()
        if not text:
            return ValidationResult(
                is_valid=False,
                errors=["empty_patch"],
                score=0.0,
            )

        if not (
            ("<<<<<<<" in patch_text or "SEARCH" in patch_text)
            and "=======" in patch_text
            and ">>>>>>>" in patch_text
        ):
            errors.append("invalid_patch_format: missing SEARCH/=======/REPLACE markers")

        hunks = parse_search_replace_hunks(patch_text)
        if not hunks:
            errors.append("no_valid_hunks_found")

        if hunks:
            errors.extend(_hard_anti_hack_errors(hunks))

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                score=0.0,
            )

        # ── Stage 2: apply simulation (SEARCH must exist in source) ─────────
        sim = simulate_search_replace_patch(
            repo_path,
            patch_text,
            default_target=default_target,
            pre_sources=pre_sources,
        )

        if not sim.get("success"):
            err = sim.get("error") or "simulate_failed"
            errors.append(f"apply_simulation_failed: {err}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                score=0.0,
                simulate_result=sim,
            )

        if sim.get("fuzzy_applied"):
            warnings.append(
                "fuzzy_indent_match_used: SEARCH was not an exact substring; "
                "indentation-normalised match was used",
            )

        modified_sources: Dict[str, str] = sim.get("modified_sources") or {}
        original_sources: Dict[str, str] = sim.get("original_sources") or sim.get("rollback_snapshots") or {}
        lines_changed = int(sim.get("lines_changed") or 0)
        # Prefer the simulated diff for smell checks. Semantic/fuzzy fallbacks
        # may clean up malformed model hunk text before producing final code.
        added_lines = _added_lines_from_diff(sim.get("diff") or "")
        if not added_lines:
            added_lines = _added_lines_for_hunks(hunks)

        # ── Stage 5 (partial): minimal destructiveness ─────────────────────
        if lines_changed > 120:
            errors.append(f"patch_too_large: {lines_changed} lines changed (max 120)")
        elif lines_changed > 80:
            warnings.append(f"large_patch: {lines_changed} lines changed")

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                score=0.0,
                simulate_result=sim,
            )

        # ── Stage 2b: reasoning-to-action semantic bridge ──────────────────
        errors.extend(
            _reasoning_alignment_errors(
                claims=claims,
                original_sources=original_sources,
                modified_sources=modified_sources,
            )
        )

        smell_errors, smell_warnings = _flakiness_smell_errors(added_lines)
        errors.extend(smell_errors)
        warnings.extend(smell_warnings)

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=sorted(set(errors)),
                warnings=warnings,
                score=0.0,
                simulate_result=sim,
            )

        # ── Stages 3–4–5: syntax, compile, structure ───────────────────────
        for rel, src in modified_sources.items():
            if not rel.endswith(".py"):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                errors.append(f"syntax_error in {rel}: {exc.msg} (line {exc.lineno})")
                continue

            try:
                compile(src, rel, "exec")
            except SyntaxError as exc:
                errors.append(f"compile_error in {rel}: {exc.msg} (line {exc.lineno})")

            issues = _structural_issues(tree, rel)
            for msg in issues:
                errors.append(f"structure: {msg}")

            errors.extend(_libcst_errors(src, rel))
            errors.extend(_introduced_name_errors(src, added_lines, rel))

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=sorted(set(errors)),
                warnings=warnings,
                score=0.0,
                simulate_result=sim,
            )

        idempotency_errors, idempotency_warnings = _idempotency_issues(
            repo_path=repo_path,
            patch_text=patch_text,
            default_target=default_target,
            modified_sources=modified_sources,
        )
        errors.extend(idempotency_errors)
        warnings.extend(idempotency_warnings)

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=sorted(set(errors)),
                warnings=warnings,
                score=0.0,
                simulate_result=sim,
            )

        # ── Stage 6: causal proximity (warnings only) ───────────────────────
        modified_rels = list(modified_sources.keys())
        warnings.extend(
            _causal_proximity_warnings(modified_rels, failure_frontier, call_chain or [])
        )

        # ── Score 0..1 for reward shaping ─────────────────────────────────
        score = 1.0
        if sim.get("noop"):
            score -= 0.35
            warnings.append("noop_patch: no effective line change")
        if sim.get("fuzzy_applied"):
            score -= 0.1
        if lines_changed > 40:
            score -= 0.05
        score = max(0.0, min(1.0, score))

        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            score=round(score, 3),
            simulate_result=sim,
        )
