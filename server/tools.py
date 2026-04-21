from __future__ import annotations

import ast
import difflib
import itertools
import re
from pathlib import Path
from typing import Any, Dict, List

import libcst as cst

from models import ASTSummary


def list_repo_structure(root_path: str) -> List[dict]:
    root = Path(root_path)
    ignored_dirs = {"__pycache__", ".git", "node_modules", "venv", ".venv"}
    entries: List[dict] = []

    for path in root.rglob("*.py"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        name = path.name
        entries.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "is_test": name.startswith("test_") or name.endswith("_test.py"),
                "size_bytes": path.stat().st_size,
                "has_async": ("async def" in content) or ("await " in content),
            }
        )
    return entries


def read_file_excerpt(path: str, start_line: int, end_line: int) -> str:
    if end_line - start_line > 100:
        raise ValueError("Cannot read more than 100 lines per call")
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        lines = itertools.islice(f, max(start_line - 1, 0), end_line)
        return "".join(lines)


def parse_ast_summary(path: str) -> ASTSummary:
    p = Path(path)
    if p.suffix in {".js", ".ts"}:
        return _parse_js_ts_summary(p)

    source = p.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source)

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[str] = []
    global_vars: List[str] = []
    threading_primitives: List[str] = []
    external_calls: List[str] = []

    external_modules = {"requests", "boto3", "httpx", "redis", "psycopg2"}
    primitive_markers = {
        "threading.Lock",
        "threading.Event",
        "asyncio.Event",
        "asyncio.Lock",
        "asyncio.Barrier",
        "asyncio.sleep",
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "line_range": [getattr(node, "lineno", 0), getattr(node, "end_lineno", 0)],
                }
            )
        elif isinstance(node, ast.ClassDef):
            classes.append(
                {
                    "name": node.name,
                    "bases": [ast.unparse(b) for b in node.bases],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "line_range": [getattr(node, "lineno", 0), getattr(node, "end_lineno", 0)],
                }
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    global_vars.append(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _node_to_name(node.func)
            if call_name in primitive_markers:
                threading_primitives.append(call_name)
            for module in external_modules:
                if call_name.startswith(f"{module}."):
                    external_calls.append(call_name)

    return ASTSummary(
        functions=functions,
        classes=classes,
        imports=sorted(set(imports)),
        global_vars=sorted(set(global_vars)),
        threading_primitives=sorted(set(threading_primitives)),
        external_calls=sorted(set(external_calls)),
    )


class _LoggingInjector(cst.CSTTransformer):
    def __init__(self, targets: List[Dict[str, str]]) -> None:
        self.targets = targets

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        target = next((t for t in self.targets if t.get("function_name") == original_node.name.value), None)
        if not target:
            return updated_node

        payload = {
            "timestamp": "{time.time()}",
            "thread_id": "{threading.get_ident()}",
            "task_id": "{_task_name()}",
            "location": f"{original_node.name.value}:{target.get('position', 'entry')}",
        }
        log_stmt = cst.parse_statement(
            "print(json.dumps({'timestamp': time.time(), 'thread_id': threading.get_ident(), "
            "'task_id': _task_name(), 'location': '%s'}), file=sys.stderr)\n" % payload["location"]
        )

        body_stmts = list(updated_node.body.body)
        if target.get("position") == "exit":
            body_stmts.append(log_stmt)
        else:
            body_stmts.insert(0, log_stmt)

        return updated_node.with_changes(body=updated_node.body.with_changes(body=body_stmts))


def inject_logging(path: str, injection_points: List[Dict[str, str]]) -> str:
    source = Path(path).read_text(encoding="utf-8", errors="ignore")
    module = cst.parse_module(source)

    header = (
        "import asyncio\n"
        "import json\n"
        "import sys\n"
        "import threading\n"
        "import time\n\n"
        "def _task_name():\n"
        "    try:\n"
        "        task = asyncio.current_task()\n"
        "        return task.get_name() if task else None\n"
        "    except Exception:\n"
        "        return None\n\n"
    )
    transformed = module.visit(_LoggingInjector(injection_points)).code
    if "def _task_name():" not in transformed:
        transformed = header + transformed
    return transformed


def apply_ast_patch(path: str, patch_spec: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(path)
    original = p.read_text(encoding="utf-8", errors="ignore")

    try:
        target = patch_spec.get("target", {})
        identifier = str(target.get("identifier", ""))
        operation = patch_spec.get("operation")
        template = str(patch_spec.get("code_template", ""))
        params = patch_spec.get("parameters", {})
        rendered = template.format(**params)

        updated = _apply_textual_operation(original, operation, identifier, rendered)
        p.write_text(updated, encoding="utf-8")
        diff = _make_unified_diff(original, updated, str(p))
        lines_changed = _count_diff_lines(diff)

        return {
            "success": True,
            "diff": diff,
            "lines_changed": lines_changed,
        }
    except Exception as exc:
        return {
            "success": False,
            "diff": "",
            "lines_changed": 0,
            "error": str(exc),
        }


def compute_diff(original_path: str, patched_source: str) -> Dict[str, Any]:
    original = Path(original_path).read_text(encoding="utf-8", errors="ignore")
    unified_diff = _make_unified_diff(original, patched_source, original_path)
    lines_changed = _count_diff_lines(unified_diff)

    before = _safe_summary_from_source(original)
    after = _safe_summary_from_source(patched_source)

    before_funcs = {f["name"] for f in before.functions}
    after_funcs = {f["name"] for f in after.functions}

    ast_diff = {
        "functions_added": sorted(after_funcs - before_funcs),
        "functions_removed": sorted(before_funcs - after_funcs),
        "functions_modified": sorted(before_funcs & after_funcs),
    }

    return {
        "unified_diff": unified_diff,
        "lines_changed": lines_changed,
        "ast_diff": ast_diff,
    }


def get_similar_fixes(root_cause_category: str, test_source: str, embedding_model: Any) -> List[Dict[str, str]]:
    # Placeholder retrieval implementation. This keeps the server interface stable while
    # allowing pluggable Chroma/FAISS integration in training infrastructure.
    _ = test_source
    _ = embedding_model
    return [
        {
            "original": "",
            "fixed": "",
            "diff": "",
            "repo": f"seed_example::{root_cause_category.lower()}",
        }
    ][:3]


def _safe_summary_from_source(source: str) -> ASTSummary:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ASTSummary([], [], [], [], [], [])

    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(
                {
                    "name": node.name,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "args": [a.arg for a in node.args.args],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "line_range": [getattr(node, "lineno", 0), getattr(node, "end_lineno", 0)],
                }
            )
    return ASTSummary(funcs, [], [], [], [], [])


def _make_unified_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )


def _count_diff_lines(diff: str) -> int:
    count = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _node_to_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _node_to_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    return ""


def _apply_textual_operation(source: str, operation: str, identifier: str, rendered: str) -> str:
    if operation in {"insert_before", "insert_after"} and identifier:
        pattern = re.escape(identifier)
        match = re.search(pattern, source)
        if not match:
            raise ValueError(f"Target identifier not found: {identifier}")

        insert = rendered + "\n"
        if operation == "insert_before":
            return source[: match.start()] + insert + source[match.start() :]
        return source[: match.end()] + "\n" + rendered + source[match.end() :]

    if operation == "replace_call" and identifier:
        return source.replace(identifier, rendered, 1)

    if operation in {"wrap_with", "add_decorator"}:
        if operation == "add_decorator":
            return f"{rendered}\n{source}"
        return rendered.replace("{body}", source)

    raise ValueError(f"Unsupported operation: {operation}")


def _parse_js_ts_summary(path: Path) -> ASTSummary:
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_javascript as ts_js
    except Exception:
        return ASTSummary([], [], [], [], [], [])

    parser = Parser()
    language = Language(ts_js.language())
    parser.language = language
    tree = parser.parse(bytes(source, "utf-8"))

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[str] = []
    external_calls: List[str] = []

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if node.type in {"function_declaration", "method_definition", "arrow_function"}:
            name = "<anonymous>"
            for child in node.children:
                if child.type in {"identifier", "property_identifier"}:
                    name = source[child.start_byte : child.end_byte]
                    break
            functions.append(
                {
                    "name": name,
                    "is_async": "async" in source[node.start_byte : node.end_byte].split("(", 1)[0],
                    "args": [],
                    "decorators": [],
                    "line_range": [node.start_point[0] + 1, node.end_point[0] + 1],
                }
            )

        if node.type == "class_declaration":
            class_name = "<anonymous_class>"
            for child in node.children:
                if child.type == "identifier":
                    class_name = source[child.start_byte : child.end_byte]
                    break
            classes.append(
                {
                    "name": class_name,
                    "bases": [],
                    "decorators": [],
                    "line_range": [node.start_point[0] + 1, node.end_point[0] + 1],
                }
            )

        if node.type == "import_statement":
            imports.append(source[node.start_byte : node.end_byte].strip())

        if node.type == "call_expression":
            snippet = source[node.start_byte : node.end_byte]
            if any(mod in snippet for mod in ["axios", "fetch", "redis", "pg."]):
                external_calls.append(snippet[:120])

    return ASTSummary(
        functions=functions,
        classes=classes,
        imports=sorted(set(imports)),
        global_vars=[],
        threading_primitives=[],
        external_calls=sorted(set(external_calls)),
    )
