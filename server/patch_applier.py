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

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        }

    files_modified: List[str] = []
    total_lines_changed = 0
    hunks_applied = 0
    all_diffs: List[str] = []
    originals: Dict[Path, str] = {}

    def rollback() -> None:
        for path, content in originals.items():
            path.write_text(content, encoding="utf-8")

    try:
        for hunk in hunks:
            # Resolve file path
            file_path = hunk.file_path or default_target or ""
            if not file_path:
                continue

            target = repo_path / file_path
            if not target.exists():
                # Try to find the file
                candidates = list(repo_path.rglob(Path(file_path).name))
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
                    }

            original = target.read_text(encoding="utf-8", errors="ignore")
            originals.setdefault(target, original)

            # Apply the hunk
            modified = _apply_single_hunk(original, hunk.search_text, hunk.replace_text)
            if modified is None:
                # Search text not found — try fuzzy match
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
                    }

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
            }

        return {
            "success": True,
            "files_modified": files_modified,
            "lines_changed": total_lines_changed,
            "hunks_applied": hunks_applied,
            "diff": "\n".join(all_diffs),
            "error": None,
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
        }


def _apply_single_hunk(original: str, search: str, replace: str) -> Optional[str]:
    """Apply a single search/replace hunk. Returns None if search text not found."""
    # Exact match first
    if search in original:
        return original.replace(search, replace, 1)
    
    # Try with normalized whitespace
    search_stripped = _normalize_whitespace(search)
    lines = original.split("\n")
    original_stripped_lines = [_normalize_whitespace(line) for line in lines]
    original_stripped = "\n".join(original_stripped_lines)
    
    if search_stripped in original_stripped:
        return original.replace(search.strip(), replace.strip(), 1)
    
    return None


def _apply_fuzzy_hunk(original: str, search: str, replace: str) -> Optional[str]:
    """Try fuzzy matching when exact match fails.
    
    Attempts to find the search text with minor whitespace differences.
    """
    # Strip leading/trailing whitespace from each line
    search_lines = [line.strip() for line in search.strip().split("\n")]
    original_lines = original.split("\n")

    # Find the starting line
    for start_idx in range(len(original_lines)):
        if original_lines[start_idx].strip() == search_lines[0]:
            # Check if all search lines match
            match = True
            for j, search_line in enumerate(search_lines):
                if start_idx + j >= len(original_lines):
                    match = False
                    break
                if original_lines[start_idx + j].strip() != search_line:
                    match = False
                    break

            if match:
                # Determine indentation from original
                indent = ""
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
