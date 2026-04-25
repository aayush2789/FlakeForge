"""V3 Patch Applier — Search/Replace hunk parser and applier.

Replaces the old action_executor.py dispatch logic.
The model outputs standard search/replace blocks, and this module
applies them atomically with rollback on failure.

Format:
    --- path/to/file.py
    <<<<<<< SEARCH
    exact lines to find
    =======
    replacement lines
    >>>>>>> REPLACE
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROTECTED_EXACT_NAMES = {
    "conftest.py",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "pyproject.toml",
    "requirements.txt",
}

PROTECTED_DIR_PREFIXES = {
    "agent/",
    "server/",
    "training/",
    ".github/",
    ".cursor/",
}

PROTECTED_SUFFIXES = {
    ".yaml",
    ".yml",
    ".toml",
}


@dataclass
class PatchHunk:
    """A single search/replace hunk."""
    file_path: str
    search_text: str
    replace_text: str


def parse_search_replace_hunks(patch_text: str) -> List[PatchHunk]:
    """Parse <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks.

    Supports two formats:
    1. With file header: --- path/to/file.py
    2. Without file header (applies to default target)
    """
    hunks: List[PatchHunk] = []
    if not patch_text or not patch_text.strip():
        return hunks

    current_file = ""

    # Split into sections by file headers
    lines = patch_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect file header
        if line.startswith("--- "):
            current_file = line[4:].strip()
            i += 1
            continue

        # Detect search block start
        if line.strip().startswith("<<<<<<< SEARCH") or line.strip() == "<<<<<<<":
            search_lines: List[str] = []
            replace_lines: List[str] = []
            i += 1

            # Collect search text until =======
            while i < len(lines) and not lines[i].strip().startswith("======="):
                search_lines.append(lines[i])
                i += 1

            if i < len(lines):
                i += 1  # skip =======

            # Collect replace text until >>>>>>> REPLACE
            while i < len(lines) and not lines[i].strip().startswith(">>>>>>> REPLACE") and not lines[i].strip() == ">>>>>>>":
                replace_lines.append(lines[i])
                i += 1

            if i < len(lines):
                i += 1  # skip >>>>>>>

            search = "\n".join(search_lines)
            replace = "\n".join(replace_lines)

            if search.strip():  # Only add if search text is non-empty
                hunks.append(PatchHunk(
                    file_path=current_file,
                    search_text=search,
                    replace_text=replace,
                ))
            continue

        i += 1

    return hunks


def simulate_search_replace_patch(
    repo_path: Path,
    patch_text: str,
    default_target: Optional[str] = None,
    pre_sources: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Dry-run: resolve and apply all hunks in memory only (no disk writes).

    Returns the same shape as ``apply_search_replace_patch`` plus:
      - ``modified_sources``: rel POSIX path -> post-patch full file text
      - ``original_sources``: rel POSIX path -> original file text before any hunk
      - ``rollback_snapshots``: alias of original_sources, kept for existing callers

    If *pre_sources* is provided (relative POSIX path -> file text), the simulator
    uses that text instead of reading from disk for the initial snapshot — keeps
    validation aligned with the environment's view when snapshots are taken before
    the step.

    Used by :class:`server.patch_validator.PatchValidator` so invalid patches never
    touch the working tree.
    """
    hunks = parse_search_replace_hunks(patch_text)

    if not hunks:
        return {
            "success": False,
            "error": "no_valid_hunks_found",
            "files_modified": [],
            "lines_changed": 0,
            "hunks_applied": 0,
            "diff": "",
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": False,
            "modified_sources": {},
            "original_sources": {},
            "rollback_snapshots": {},
        }

    files_modified: List[str] = []
    total_lines_changed = 0
    hunks_applied = 0
    all_diffs: List[str] = []
    fuzzy_applied = False
    # rel_path -> current in-memory content (starts as disk snapshot once loaded)
    memory_files: Dict[str, str] = {}
    rollback_snapshots: Dict[str, str] = {}

    def _rel_key(target: Path) -> str:
        return str(target.resolve().relative_to(repo_path.resolve())).replace("\\", "/")

    for hunk in hunks:
        file_path = hunk.file_path or default_target or ""
        if not file_path:
            continue
        if _is_protected_path(file_path):
            return {
                "success": False,
                "error": f"protected_file_{file_path}",
                "files_modified": files_modified,
                "lines_changed": total_lines_changed,
                "hunks_applied": hunks_applied,
                "diff": "\n".join(all_diffs),
                "noop": False,
                "protected_file": True,
                "fuzzy_applied": fuzzy_applied,
                "modified_sources": {},
                "original_sources": dict(rollback_snapshots),
                "rollback_snapshots": dict(rollback_snapshots),
            }

        target = repo_path / file_path
        if not target.exists():
            candidates = [
                candidate
                for candidate in repo_path.rglob(Path(file_path).name)
                if not _is_protected_path(str(candidate.relative_to(repo_path)))
            ]
            if candidates:
                target = candidates[0]
            else:
                return {
                    "success": False,
                    "error": f"target_file_not_found_{file_path}",
                    "files_modified": files_modified,
                    "lines_changed": total_lines_changed,
                    "hunks_applied": hunks_applied,
                    "diff": "\n".join(all_diffs),
                    "noop": False,
                    "protected_file": False,
                    "fuzzy_applied": fuzzy_applied,
                    "modified_sources": {},
                    "original_sources": dict(rollback_snapshots),
                    "rollback_snapshots": dict(rollback_snapshots),
                }
        else:
            try:
                resolved_relative = str(target.resolve().relative_to(repo_path.resolve())).replace("\\", "/")
            except ValueError:
                return {
                    "success": False,
                    "error": f"target_outside_repo_{file_path}",
                    "files_modified": files_modified,
                    "lines_changed": total_lines_changed,
                    "hunks_applied": hunks_applied,
                    "diff": "\n".join(all_diffs),
                    "noop": False,
                    "protected_file": True,
                    "fuzzy_applied": fuzzy_applied,
                    "modified_sources": {},
                    "original_sources": dict(rollback_snapshots),
                    "rollback_snapshots": dict(rollback_snapshots),
                }
            if _is_protected_path(resolved_relative):
                return {
                    "success": False,
                    "error": f"protected_file_{resolved_relative}",
                    "files_modified": files_modified,
                    "lines_changed": total_lines_changed,
                    "hunks_applied": hunks_applied,
                    "diff": "\n".join(all_diffs),
                    "noop": False,
                    "protected_file": True,
                    "fuzzy_applied": fuzzy_applied,
                    "modified_sources": {},
                    "original_sources": dict(rollback_snapshots),
                    "rollback_snapshots": dict(rollback_snapshots),
                }

        rel = _rel_key(target)
        if rel not in memory_files:
            if pre_sources is not None and rel in pre_sources:
                original = pre_sources[rel]
            elif pre_sources is not None and file_path.replace("\\", "/") in pre_sources:
                original = pre_sources[file_path.replace("\\", "/")]
            else:
                original = target.read_text(encoding="utf-8", errors="ignore")
            memory_files[rel] = original
            rollback_snapshots[rel] = original

        original = memory_files[rel]
        modified = _apply_single_hunk(original, hunk.search_text, hunk.replace_text)
        if modified is None:
            modified = _apply_fuzzy_hunk(original, hunk.search_text, hunk.replace_text)
            if modified is None:
                return {
                    "success": False,
                    "error": f"search_text_not_found_in_{target.name}",
                    "files_modified": files_modified,
                    "lines_changed": total_lines_changed,
                    "hunks_applied": hunks_applied,
                    "diff": "\n".join(all_diffs),
                    "noop": False,
                    "protected_file": False,
                    "fuzzy_applied": fuzzy_applied,
                    "modified_sources": {},
                    "original_sources": dict(rollback_snapshots),
                    "rollback_snapshots": dict(rollback_snapshots),
                }
            fuzzy_applied = True

        memory_files[rel] = modified
        lines_changed = _count_lines_changed(original, modified)
        total_lines_changed += lines_changed
        hunks_applied += 1
        abs_path = str(target)
        if abs_path not in files_modified:
            files_modified.append(abs_path)
        diff_text = _make_unified_diff(original, modified, rel)
        if diff_text:
            all_diffs.append(diff_text)

    if hunks_applied != len(hunks):
        return {
            "success": False,
            "error": "not_all_hunks_applied",
            "files_modified": files_modified,
            "lines_changed": total_lines_changed,
            "hunks_applied": hunks_applied,
            "diff": "\n".join(all_diffs),
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": fuzzy_applied,
            "modified_sources": {},
            "original_sources": dict(rollback_snapshots),
            "rollback_snapshots": dict(rollback_snapshots),
        }

    noop = total_lines_changed == 0 or not "\n".join(all_diffs).strip()
    modified_sources = {k: v for k, v in memory_files.items() if k in rollback_snapshots}

    return {
        "success": True,
        "files_modified": files_modified,
        "lines_changed": total_lines_changed,
        "hunks_applied": hunks_applied,
        "diff": "\n".join(all_diffs),
        "error": None,
        "noop": noop,
        "protected_file": False,
        "fuzzy_applied": fuzzy_applied,
        "modified_sources": modified_sources,
        "original_sources": dict(rollback_snapshots),
        "rollback_snapshots": dict(rollback_snapshots),
    }


def restore_repo_files(repo_path: Path, rollback_snapshots: Dict[str, str]) -> None:
    """Write *rollback_snapshots* (relative path -> text) back under *repo_path*."""
    for rel, content in rollback_snapshots.items():
        path = repo_path / rel.replace("/", os.sep)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_validated_sources(repo_path: Path, modified_sources: Dict[str, str]) -> None:
    """Write validator-approved ``modified_sources`` to disk exactly once."""
    for rel, content in modified_sources.items():
        path = repo_path / rel.replace("/", os.sep)
        try:
            path.resolve().relative_to(repo_path.resolve())
        except ValueError as exc:
            raise ValueError(f"target_outside_repo_{rel}") from exc
        if _is_protected_path(rel):
            raise ValueError(f"protected_file_{rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def apply_search_replace_patch(
    repo_path: Path,
    patch_text: str,
    default_target: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply search/replace patches atomically. Rolls back on failure.

    Args:
        repo_path: Root of the repository
        patch_text: Raw patch text with search/replace hunks
        default_target: Default file to target if no file header in patch

    Returns:
        Dict with keys: success, files_modified, lines_changed, hunks_applied,
                        error (if failed), diff
    """
    hunks = parse_search_replace_hunks(patch_text)

    if not hunks:
        return {
            "success": False,
            "error": "no_valid_hunks_found",
            "files_modified": [],
            "lines_changed": 0,
            "hunks_applied": 0,
            "diff": "",
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": False,
        }

    files_modified: List[str] = []
    total_lines_changed = 0
    hunks_applied = 0
    all_diffs: List[str] = []
    originals: Dict[Path, str] = {}
    fuzzy_applied = False

    def rollback() -> None:
        for path, content in originals.items():
            path.write_text(content, encoding="utf-8")

    try:
        for hunk in hunks:
            # Resolve file path
            file_path = hunk.file_path or default_target or ""
            if not file_path:
                continue
            if _is_protected_path(file_path):
                rollback()
                return {
                    "success": False,
                    "error": f"protected_file_{file_path}",
                    "files_modified": files_modified,
                    "lines_changed": total_lines_changed,
                    "hunks_applied": hunks_applied,
                    "diff": "\n".join(all_diffs),
                    "noop": False,
                    "protected_file": True,
                    "fuzzy_applied": fuzzy_applied,
                }

            target = repo_path / file_path
            if not target.exists():
                # Try to find the file
                candidates = [
                    candidate
                    for candidate in repo_path.rglob(Path(file_path).name)
                    if not _is_protected_path(str(candidate.relative_to(repo_path)))
                ]
                if candidates:
                    target = candidates[0]
                else:
                    rollback()
                    return {
                        "success": False,
                        "error": f"target_file_not_found_{file_path}",
                        "files_modified": files_modified,
                        "lines_changed": total_lines_changed,
                        "hunks_applied": hunks_applied,
                        "diff": "\n".join(all_diffs),
                        "noop": False,
                        "protected_file": False,
                        "fuzzy_applied": fuzzy_applied,
                    }
            else:
                try:
                    resolved_relative = str(target.resolve().relative_to(repo_path.resolve())).replace("\\", "/")
                except ValueError:
                    rollback()
                    return {
                        "success": False,
                        "error": f"target_outside_repo_{file_path}",
                        "files_modified": files_modified,
                        "lines_changed": total_lines_changed,
                        "hunks_applied": hunks_applied,
                        "diff": "\n".join(all_diffs),
                        "noop": False,
                        "protected_file": True,
                        "fuzzy_applied": fuzzy_applied,
                    }
                if _is_protected_path(resolved_relative):
                    rollback()
                    return {
                        "success": False,
                        "error": f"protected_file_{resolved_relative}",
                        "files_modified": files_modified,
                        "lines_changed": total_lines_changed,
                        "hunks_applied": hunks_applied,
                        "diff": "\n".join(all_diffs),
                        "noop": False,
                        "protected_file": True,
                        "fuzzy_applied": fuzzy_applied,
                    }

            original = target.read_text(encoding="utf-8", errors="ignore")
            originals.setdefault(target, original)

            # Apply the hunk
            modified = _apply_single_hunk(original, hunk.search_text, hunk.replace_text)
            if modified is None:
                # Search text not found — try conservative indentation-only matching.
                modified = _apply_fuzzy_hunk(original, hunk.search_text, hunk.replace_text)
                if modified is None:
                    rollback()
                    return {
                        "success": False,
                        "error": f"search_text_not_found_in_{target.name}",
                        "files_modified": files_modified,
                        "lines_changed": total_lines_changed,
                        "hunks_applied": hunks_applied,
                        "diff": "\n".join(all_diffs),
                        "noop": False,
                        "protected_file": False,
                        "fuzzy_applied": fuzzy_applied,
                    }
                fuzzy_applied = True

            # Write modified content
            target.write_text(modified, encoding="utf-8")

            # Track changes
            lines_changed = _count_lines_changed(original, modified)
            total_lines_changed += lines_changed
            hunks_applied += 1
            if str(target) not in files_modified:
                files_modified.append(str(target))

            # Generate diff
            diff_text = _make_unified_diff(original, modified, str(target.relative_to(repo_path)))
            if diff_text:
                all_diffs.append(diff_text)

        if hunks_applied != len(hunks):
            rollback()
            return {
                "success": False,
                "error": "not_all_hunks_applied",
                "files_modified": files_modified,
                "lines_changed": total_lines_changed,
                "hunks_applied": hunks_applied,
                "diff": "\n".join(all_diffs),
                "noop": False,
                "protected_file": False,
                "fuzzy_applied": fuzzy_applied,
            }

        noop = total_lines_changed == 0 or not "\n".join(all_diffs).strip()
        return {
            "success": True,
            "files_modified": files_modified,
            "lines_changed": total_lines_changed,
            "hunks_applied": hunks_applied,
            "diff": "\n".join(all_diffs),
            "error": None,
            "noop": noop,
            "protected_file": False,
            "fuzzy_applied": fuzzy_applied,
        }

    except Exception as exc:
        # Rollback on any error
        rollback()
        return {
            "success": False,
            "error": str(exc),
            "files_modified": [],
            "lines_changed": 0,
            "hunks_applied": 0,
            "diff": "",
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": fuzzy_applied,
        }


def apply_structured_patch(
    repo_path: Path,
    structured_patch: Any,  # models.StructuredPatch
    default_target: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a StructuredPatch (typed Pydantic model) atomically.

    This is the preferred entry point when the agent emits valid JSON.
    Each hunk's `file`, `search`, and `replace` fields are used directly —
    no text parsing required. Falls back to ``apply_search_replace_patch``
    when ``structured_patch`` has no hunks or is None.

    Args:
        repo_path: Root of the repository (Path).
        structured_patch: ``models.StructuredPatch`` with a ``hunks`` list.
        default_target: Fallback file if a hunk has no ``file`` set.

    Returns:
        Same dict shape as ``apply_search_replace_patch``.
    """
    if structured_patch is None or not structured_patch.hunks:
        return {
            "success": False,
            "error": "no_structured_hunks",
            "files_modified": [],
            "lines_changed": 0,
            "hunks_applied": 0,
            "diff": "",
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": False,
        }

    # Convert StructuredPatch.hunks → internal PatchHunk dataclass list.
    internal_hunks: List[PatchHunk] = []
    for h in structured_patch.hunks:
        file_path = h.file or default_target or ""
        if not file_path or not h.search:
            continue  # skip mal-formed hunks
        internal_hunks.append(PatchHunk(
            file_path=file_path,
            search_text=h.search,
            replace_text=h.replace,
        ))

    if not internal_hunks:
        return {
            "success": False,
            "error": "no_valid_structured_hunks_after_filter",
            "files_modified": [],
            "lines_changed": 0,
            "hunks_applied": 0,
            "diff": "",
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": False,
        }

    # ---- Core apply logic (mirrors apply_search_replace_patch) ----
    files_modified: List[str] = []
    total_lines_changed = 0
    hunks_applied = 0
    all_diffs: List[str] = []
    originals: Dict[Path, str] = {}
    fuzzy_applied = False

    def rollback() -> None:
        for path, content in originals.items():
            path.write_text(content, encoding="utf-8")

    try:
        for hunk in internal_hunks:
            file_path = hunk.file_path
            if _is_protected_path(file_path):
                rollback()
                return {
                    "success": False,
                    "error": f"protected_file_{file_path}",
                    "files_modified": files_modified,
                    "lines_changed": total_lines_changed,
                    "hunks_applied": hunks_applied,
                    "diff": "\n".join(all_diffs),
                    "noop": False,
                    "protected_file": True,
                    "fuzzy_applied": fuzzy_applied,
                }

            target = repo_path / file_path
            if not target.exists():
                candidates = [
                    c for c in repo_path.rglob(Path(file_path).name)
                    if not _is_protected_path(str(c.relative_to(repo_path)))
                ]
                if candidates:
                    target = candidates[0]
                else:
                    rollback()
                    return {
                        "success": False,
                        "error": f"target_file_not_found_{file_path}",
                        "files_modified": files_modified,
                        "lines_changed": total_lines_changed,
                        "hunks_applied": hunks_applied,
                        "diff": "\n".join(all_diffs),
                        "noop": False,
                        "protected_file": False,
                        "fuzzy_applied": fuzzy_applied,
                    }

            original = target.read_text(encoding="utf-8", errors="ignore")
            originals.setdefault(target, original)

            modified = _apply_single_hunk(original, hunk.search_text, hunk.replace_text)
            if modified is None:
                modified = _apply_fuzzy_hunk(original, hunk.search_text, hunk.replace_text)
                if modified is None:
                    rollback()
                    return {
                        "success": False,
                        "error": f"search_text_not_found_in_{target.name}",
                        "files_modified": files_modified,
                        "lines_changed": total_lines_changed,
                        "hunks_applied": hunks_applied,
                        "diff": "\n".join(all_diffs),
                        "noop": False,
                        "protected_file": False,
                        "fuzzy_applied": fuzzy_applied,
                    }
                fuzzy_applied = True

            target.write_text(modified, encoding="utf-8")
            lines_changed = _count_lines_changed(original, modified)
            total_lines_changed += lines_changed
            hunks_applied += 1
            if str(target) not in files_modified:
                files_modified.append(str(target))
            diff_text = _make_unified_diff(original, modified, str(target.relative_to(repo_path)))
            if diff_text:
                all_diffs.append(diff_text)

        if hunks_applied != len(internal_hunks):
            rollback()
            return {
                "success": False,
                "error": "not_all_structured_hunks_applied",
                "files_modified": files_modified,
                "lines_changed": total_lines_changed,
                "hunks_applied": hunks_applied,
                "diff": "\n".join(all_diffs),
                "noop": False,
                "protected_file": False,
                "fuzzy_applied": fuzzy_applied,
            }

        noop = total_lines_changed == 0 or not "\n".join(all_diffs).strip()
        return {
            "success": True,
            "files_modified": files_modified,
            "lines_changed": total_lines_changed,
            "hunks_applied": hunks_applied,
            "diff": "\n".join(all_diffs),
            "error": None,
            "noop": noop,
            "protected_file": False,
            "fuzzy_applied": fuzzy_applied,
        }

    except Exception as exc:
        rollback()
        return {
            "success": False,
            "error": str(exc),
            "files_modified": [],
            "lines_changed": 0,
            "hunks_applied": 0,
            "diff": "",
            "noop": False,
            "protected_file": False,
            "fuzzy_applied": fuzzy_applied,
        }


def _apply_single_hunk(original: str, search: str, replace: str) -> Optional[str]:
    """Apply a single search/replace hunk. Returns None if search text not found."""
    # Exact match first
    if search in original:
        return original.replace(search, replace, 1)

    return None


def _normalise_line(line: str) -> str:
    """Strip whitespace and trailing Python line-continuation backslashes."""
    return line.strip().rstrip("\\").rstrip()


def _apply_fuzzy_hunk(original: str, search: str, replace: str) -> Optional[str]:
    """Try conservative indentation-only matching when exact match fails."""
    search_lines = [_normalise_line(line) for line in search.strip().split("\n")]
    original_lines = original.split("\n")

    # Find the starting line
    for start_idx in range(len(original_lines)):
        if _normalise_line(original_lines[start_idx]) == search_lines[0]:
            # Check if all search lines match
            match = True
            for j, search_line in enumerate(search_lines):
                if start_idx + j >= len(original_lines):
                    match = False
                    break
                if _normalise_line(original_lines[start_idx + j]) != search_line:
                    match = False
                    break

            if match:
                # Determine indentation from original
                first_orig = original_lines[start_idx]
                indent = first_orig[:len(first_orig) - len(first_orig.lstrip())]

                # Build replacement
                replace_lines = replace.strip().split("\n")
                indented_replace = [indent + line if line.strip() else line for line in replace_lines]

                # Replace the matched block
                result_lines = (
                    original_lines[:start_idx]
                    + indented_replace
                    + original_lines[start_idx + len(search_lines):]
                )
                return "\n".join(result_lines)

    return None


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for fuzzy matching."""
    return re.sub(r'\s+', ' ', text.strip())


def _is_protected_path(path: str) -> bool:
    """Return True when a model patch targets infrastructure/config files."""
    normalized = path.replace("\\", "/").lstrip("./")
    name = Path(normalized).name
    if name in PROTECTED_EXACT_NAMES:
        return True
    if any(normalized.startswith(prefix) for prefix in PROTECTED_DIR_PREFIXES):
        return True
    return any(normalized.endswith(suffix) for suffix in PROTECTED_SUFFIXES)


def _count_lines_changed(original: str, modified: str) -> int:
    """Count the number of lines that differ."""
    orig_lines = original.split("\n")
    mod_lines = modified.split("\n")
    changed = abs(len(orig_lines) - len(mod_lines))
    for a, b in zip(orig_lines, mod_lines):
        if a != b:
            changed += 1
    return changed


def _make_unified_diff(before: str, after: str, path: str) -> str:
    """Generate unified diff."""
    import difflib
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )


def _create_git_stash(repo_path: Path) -> bool:
    """Create a git stash for rollback. Returns True if stash was created."""
    try:
        result = subprocess.run(
            ["git", "stash", "push", "-m", "flakeforge_rollback"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        return "No local changes" not in result.stdout
    except Exception:
        return False


def _git_stash_pop(repo_path: Path) -> None:
    """Pop the git stash to rollback changes."""
    try:
        subprocess.run(
            ["git", "stash", "pop"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        pass
