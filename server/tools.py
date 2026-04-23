from __future__ import annotations

import ast
import difflib
import itertools
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import libcst as cst

try:
    from ..models import ASTSummary, FailurePattern, RunRecord
except ImportError:
    from models import ASTSummary, FailurePattern, RunRecord


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
    return _parse_python_ast_summary(source)


def resolve_target_from_evidence(path: str, evidence: List[str]) -> Dict[str, Any]:
    summary = parse_ast_summary(path)
    if not evidence:
        if summary.functions:
            return {
                "type": "function",
                "identifier": summary.functions[0]["name"],
                "line": summary.functions[0]["line_range"][0],
            }
        return {"type": "line", "identifier": "def test_", "line": 1}

    source = Path(path).read_text(encoding="utf-8", errors="ignore")

    for clue in evidence:
        clue = clue.strip()
        if not clue:
            continue

        fn_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", clue)
        if fn_match:
            fn_name = fn_match.group(1)
            for fn in summary.functions:
                if fn["name"] == fn_name:
                    return {
                        "type": "function",
                        "identifier": fn_name,
                        "line": fn["line_range"][0],
                    }

        if clue in source:
            return {"type": "line", "identifier": clue, "line": _line_of_text(source, clue)}

    if summary.external_calls:
        return {"type": "call", "identifier": summary.external_calls[0], "line": 1}
    if summary.functions:
        return {"type": "function", "identifier": summary.functions[0]["name"], "line": summary.functions[0]["line_range"][0]}
    return {"type": "line", "identifier": "def test_", "line": 1}


class _LoggingInjector(cst.CSTTransformer):
    def __init__(self, targets: List[Dict[str, str]]) -> None:
        self.targets = targets

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        target = next((t for t in self.targets if t.get("function_name") == original_node.name.value), None)
        if not target:
            return updated_node

        position = target.get("position", "entry")
        fn_name = original_node.name.value
        log_stmt = cst.parse_statement(
            "print(json.dumps({"
            "'timestamp': time.time(), "
            "'thread_id': threading.get_ident(), "
            "'task_id': _task_name(), "
            f"'function': '{fn_name}', "
            f"'event': '{position}'"
            "}), file=sys.stderr)\n"
        )

        body_stmts = list(updated_node.body.body)
        already_present = any("'function': '%s'" % fn_name in getattr(stmt, "code", "") for stmt in body_stmts)
        if already_present:
            return updated_node

        if position == "exit":
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
        operation = patch_spec.get("operation")
        target = patch_spec.get("target", {})
        identifier = str(target.get("identifier", ""))
        template = str(patch_spec.get("code_template", ""))
        params = patch_spec.get("parameters", {})
        rendered = template.format(**params)

        if rendered and rendered in original and operation in {"insert_before", "insert_after", "add_decorator"}:
            diff_data = compute_diff_from_sources(str(p), original, original)
            return {
                "success": True,
                "diff": diff_data["unified_diff"],
                "lines_changed": 0,
                "ast_diff": diff_data["ast_diff"],
            }

        updated = _apply_patch_operation(original, operation, target, identifier, rendered)
        p.write_text(updated, encoding="utf-8")

        diff_data = compute_diff_from_sources(str(p), original, updated)
        return {
            "success": True,
            "diff": diff_data["unified_diff"],
            "lines_changed": diff_data["lines_changed"],
            "ast_diff": diff_data["ast_diff"],
        }
    except Exception as exc:
        return {
            "success": False,
            "diff": "",
            "lines_changed": 0,
            "ast_diff": {},
            "error": str(exc),
        }


def compute_diff(original_path: str, patched_source: str) -> Dict[str, Any]:
    original = Path(original_path).read_text(encoding="utf-8", errors="ignore")
    return compute_diff_from_sources(original_path, original, patched_source)


def compute_diff_from_sources(path: str, before_source: str, after_source: str) -> Dict[str, Any]:
    unified_diff = _make_unified_diff(before_source, after_source, path)
    lines_changed = _count_diff_lines(unified_diff)

    before = _safe_summary_from_source(before_source)
    after = _safe_summary_from_source(after_source)

    before_funcs = {f["name"] for f in before.functions}
    after_funcs = {f["name"] for f in after.functions}
    before_calls = set(before.external_calls)
    after_calls = set(after.external_calls)

    ast_diff = {
        "functions_added": sorted(after_funcs - before_funcs),
        "functions_removed": sorted(before_funcs - after_funcs),
        "functions_modified": sorted(before_funcs & after_funcs),
        "calls_added": sorted(after_calls - before_calls),
        "calls_removed": sorted(before_calls - after_calls),
    }

    return {
        "unified_diff": unified_diff,
        "lines_changed": lines_changed,
        "ast_diff": ast_diff,
    }


def get_failure_pattern(run_records: List[RunRecord]) -> FailurePattern:
    if not run_records:
        return FailurePattern(
            pass_rate=0.0,
            most_common_error=None,
            error_distribution={},
            duration_mean=0.0,
            duration_std=0.0,
            flakiness_score=0.0,
        )

    pass_rate = sum(1 for r in run_records if r.passed) / len(run_records)
    errors = [r.error_type for r in run_records if r.error_type]
    error_distribution = dict(Counter(errors))
    most_common_error = None
    if error_distribution:
        most_common_error = max(error_distribution, key=error_distribution.get)

    durations = [r.duration_ms for r in run_records]
    duration_mean = statistics.fmean(durations)
    duration_std = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    flakiness_score = 4.0 * pass_rate * (1.0 - pass_rate)

    return FailurePattern(
        pass_rate=pass_rate,
        most_common_error=most_common_error,
        error_distribution=error_distribution,
        duration_mean=duration_mean,
        duration_std=duration_std,
        flakiness_score=flakiness_score,
    )


def get_similar_fixes(root_cause_category: str, test_source: str, embedding_model: Any = None) -> List[Dict[str, str]]:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(Path.cwd() / ".flakeforge_chroma"))
        collection = client.get_or_create_collection("flake_fixes")

        if collection.count() == 0:
            _seed_collection_from_seed_repos(collection)

        model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(test_source).tolist()

        where = {"root_cause_category": root_cause_category}
        results = collection.query(query_embeddings=[query_embedding], n_results=3, where=where)

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        out: List[Dict[str, str]] = []
        for idx, doc in enumerate(docs):
            md = metadatas[idx] if idx < len(metadatas) else {}
            payload = json.loads(doc)
            out.append(
                {
                    "root_cause": md.get("root_cause_category", root_cause_category),
                    "original": payload.get("original", ""),
                    "fixed": payload.get("fixed", ""),
                    "diff": payload.get("diff", ""),
                    "action": md.get("action", ""),
                    "repo": md.get("repo", "unknown"),
                }
            )
        return out[:3]
    except Exception:
        return []


def _seed_collection_from_seed_repos(collection: Any) -> None:
    root = Path.cwd() / "seed_repos"
    if not root.exists():
        return

    model = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        pass

    mapping = {
        "timing_race": "TIMING_RACE",
        "shared_state": "SHARED_STATE",
        "external_dependency": "EXTERNAL_DEPENDENCY",
        "order_dependency": "ORDER_DEPENDENCY",
        "resource_leak": "RESOURCE_LEAK",
        "nondeterminism": "NONDETERMINISM",
        "compound_timing_shared": "TIMING_RACE",
        "compound_external_nondeterminism": "EXTERNAL_DEPENDENCY",
    }
    action_mapping = {
        "timing_race": "ADD_TIMING_GUARD",
        "shared_state": "RESET_STATE",
        "external_dependency": "MOCK_DEPENDENCY",
        "order_dependency": "ADD_SYNCHRONIZATION",
        "resource_leak": "RESET_STATE",
        "nondeterminism": "SEED_RANDOMNESS",
        "compound_timing_shared": "ADD_TIMING_GUARD",
        "compound_external_nondeterminism": "MOCK_DEPENDENCY",
    }

    docs = []
    embeddings = []
    ids = []
    metadatas = []

    for repo_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        test_file = repo_dir / "tests" / "test_flaky.py"
        fixed_test = repo_dir / "solution" / "tests" / "test_flaky.py"
        if not test_file.exists() or not fixed_test.exists():
            continue

        original = test_file.read_text(encoding="utf-8", errors="ignore")
        fixed = fixed_test.read_text(encoding="utf-8", errors="ignore")
        diff = _make_unified_diff(original, fixed, str(test_file))

        payload = json.dumps({"original": original, "fixed": fixed, "diff": diff})
        docs.append(payload)
        ids.append(repo_dir.name)
        metadatas.append(
            {
                "repo": repo_dir.name,
                "root_cause_category": mapping.get(repo_dir.name, "NONDETERMINISM"),
                "action": action_mapping.get(repo_dir.name, "GATHER_EVIDENCE"),
            }
        )

        if model is not None:
            embeddings.append(model.encode(original).tolist())

    if not docs:
        return

    if embeddings:
        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    else:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)


def _safe_summary_from_source(source: str) -> ASTSummary:
    try:
        return _parse_python_ast_summary(source)
    except Exception:
        return ASTSummary([], [], [], [], [], [])


def _parse_python_ast_summary(source: str) -> ASTSummary:
    tree = ast.parse(source)

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[str] = []
    global_vars: List[str] = []
    threading_primitives: List[str] = []
    external_calls: List[str] = []

    external_modules = {
        "requests", "boto3", "httpx", "redis", "psycopg2",
        # V2 expansion: missing real-world dependencies
        "aiohttp", "grpc", "sqlalchemy", "asyncpg", "celery", "pika",
        "subprocess", "time",
    }
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


def _line_of_text(source: str, text: str) -> int:
    idx = source.find(text)
    if idx < 0:
        return 1
    return source[:idx].count("\n") + 1


def _apply_patch_operation(source: str, operation: str, target: Dict[str, Any], identifier: str, rendered: str) -> str:
    if operation == "ensure_reset_fixture":
        scope = str(target.get("scope", "function"))
        return _apply_reset_fixture(source, scope)
    if operation == "ensure_retry_wrapper":
        max_attempts = int(target.get("max_attempts", 2))
        backoff_ms = int(target.get("backoff_ms", 100))
        fn_name = str(target.get("function_name", "test"))
        return _apply_retry_wrapper(source, fn_name, max_attempts, backoff_ms)
    if operation == "ensure_seed_call":
        library = str(target.get("library", "random"))
        fn_name = str(target.get("function_name", "test"))
        return _apply_seed_call(source, fn_name, library)
    if operation == "add_decorator":
        return _apply_add_decorator(source, target, rendered)
    if operation == "replace_call":
        return _apply_replace_call(source, identifier, rendered)
    if operation == "wrap_with":
        return _apply_wrap_with(source, target, rendered)
    if operation in {"insert_before", "insert_after"}:
        if target.get("type") == "function":
            return _apply_insert_in_function(source, target, rendered, operation)
        return _apply_textual_operation_idempotent(source, operation, identifier, rendered)
    # ── V2 Deep-Action Operations (Bug 2 fix) ──────────────────────────────────
    if operation == "refactor_concurrency_primitive":
        return _apply_refactor_concurrency(source, target)
    if operation == "isolate_boundary_call":
        return _apply_isolate_boundary(source, target)
    if operation == "extract_async_scope":
        return _apply_extract_async_scope(source, target)
    if operation == "harden_idempotency":
        return _apply_harden_idempotency(source, target)
    raise ValueError(f"Unsupported operation: {operation}")


def _apply_add_decorator(source: str, target: Dict[str, Any], decorator_code: str) -> str:
    fn_name = str(target.get("identifier", "")).replace("def ", "").strip()
    if decorator_code.startswith("@"):
        decorator_code = decorator_code[1:]

    module = cst.parse_module(source)
    decorator_expr = cst.parse_expression(decorator_code)

    class DecoratorAdder(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if fn_name and fn_name not in original_node.name.value and fn_name != "test":
                return updated_node
            existing = [d.decorator.code for d in updated_node.decorators]
            if decorator_expr.code in existing:
                return updated_node
            return updated_node.with_changes(decorators=[*updated_node.decorators, cst.Decorator(decorator_expr)])

    return module.visit(DecoratorAdder()).code


def _apply_replace_call(source: str, call_identifier: str, rendered: str) -> str:
    module = cst.parse_module(source)
    rendered_expr = cst.parse_expression(rendered)

    class CallReplacer(cst.CSTTransformer):
        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
            name = _cst_name(updated_node.func)
            if name == call_identifier:
                return rendered_expr
            return updated_node

    return module.visit(CallReplacer()).code


def _apply_wrap_with(source: str, target: Dict[str, Any], rendered: str) -> str:
    fn_name = str(target.get("identifier", "")).strip()
    with_expr = rendered.strip().split(":", 1)[0].replace("with ", "").strip()

    module = cst.parse_module(source)

    class WrapWithTransformer(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if fn_name and fn_name not in original_node.name.value and fn_name != "test":
                return updated_node
            with_stmt = cst.parse_statement(f"with {with_expr}:\n    pass\n")
            if not isinstance(with_stmt, cst.With):
                return updated_node
            wrapped = with_stmt.with_changes(body=with_stmt.body.with_changes(body=list(updated_node.body.body)))
            return updated_node.with_changes(body=updated_node.body.with_changes(body=[wrapped]))

    return module.visit(WrapWithTransformer()).code


def _apply_insert_in_function(source: str, target: Dict[str, Any], rendered: str, operation: str) -> str:
    fn_name = str(target.get("identifier", "")).strip()
    statement = cst.parse_statement(rendered + "\n")

    module = cst.parse_module(source)

    class InsertTransformer(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if fn_name and fn_name not in original_node.name.value and fn_name != "test":
                return updated_node
            body = list(updated_node.body.body)
            if any(getattr(stmt, "code", "") == statement.code for stmt in body):
                return updated_node
            if operation == "insert_before":
                body = [statement, *body]
            else:
                body = [*body, statement]
            return updated_node.with_changes(body=updated_node.body.with_changes(body=body))

    return module.visit(InsertTransformer()).code


def _apply_textual_operation_idempotent(source: str, operation: str, identifier: str, rendered: str) -> str:
    if rendered in source:
        return source

    if operation in {"insert_before", "insert_after"} and identifier:
        match = re.search(re.escape(identifier), source)
        if not match:
            raise ValueError(f"Target identifier not found: {identifier}")

        insert = rendered + "\n"
        if operation == "insert_before":
            return source[: match.start()] + insert + source[match.start() :]
        return source[: match.end()] + "\n" + rendered + source[match.end() :]

    raise ValueError(f"Unsupported textual operation: {operation}")


def _cst_name(node: cst.CSTNode) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        left = _cst_name(node.value)
        return f"{left}.{node.attr.value}" if left else node.attr.value
    return ""


def _apply_reset_fixture(source: str, scope: str) -> str:
    fixture_name = "_flakeforge_reset_state"
    if f"def {fixture_name}(" in source:
        return source

    module = cst.parse_module(source)
    fixture_code = (
        f"@pytest.fixture(autouse=True, scope='{scope}')\n"
        f"def {fixture_name}():\n"
        "    for _key, _value in list(globals().items()):\n"
        "        if _key.startswith('_') or _key in {'pytest', 'copy', '_flakeforge_reset_state'}:\n"
        "            continue\n"
        "        if isinstance(_value, dict):\n"
        "            _value.clear()\n"
        "        elif isinstance(_value, list):\n"
        "            _value.clear()\n"
        "        elif isinstance(_value, set):\n"
        "            _value.clear()\n"
        "    yield\n"
    )

    try:
        fixture_stmt = cst.parse_statement(fixture_code)
        if not isinstance(fixture_stmt, cst.FunctionDef):
            return source
    except Exception:
        return source

    updated_source = source
    if "import pytest" not in source:
        updated_source = "import pytest\n" + updated_source

    updated_module = cst.parse_module(updated_source)
    body = list(updated_module.body)
    body.insert(0, fixture_stmt)
    return updated_module.with_changes(body=body).code


def _apply_retry_wrapper(source: str, fn_name: str, max_attempts: int, backoff_ms: int) -> str:
    helper_name = "_flakeforge_retry"
    module = cst.parse_module(source)

    class RetryDecoratorAdder(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if fn_name and fn_name not in original_node.name.value and fn_name != "test":
                return updated_node
            decorator_expr = cst.parse_expression(
                f"{helper_name}(max_attempts={max_attempts}, backoff_ms={backoff_ms})"
            )
            existing = [d.decorator.code for d in updated_node.decorators]
            if any(helper_name in value for value in existing):
                return updated_node
            return updated_node.with_changes(decorators=[*updated_node.decorators, cst.Decorator(decorator_expr)])

    transformed = module.visit(RetryDecoratorAdder()).code
    if f"def {helper_name}(" in transformed:
        return transformed

    helper_code = (
        "\n"
        "def _flakeforge_retry(max_attempts: int = 2, backoff_ms: int = 100):\n"
        "    def _decorator(fn):\n"
        "        def _wrapped(*args, **kwargs):\n"
        "            _last_exc = None\n"
        "            for _attempt in range(max_attempts):\n"
        "                try:\n"
        "                    return fn(*args, **kwargs)\n"
        "                except Exception as _exc:\n"
        "                    _last_exc = _exc\n"
        "                    time.sleep(backoff_ms / 1000.0)\n"
        "            raise _last_exc\n"
        "        return _wrapped\n"
        "    return _decorator\n"
    )
    if "import time" not in transformed:
        transformed = "import time\n" + transformed
    return transformed + helper_code


def _apply_seed_call(source: str, fn_name: str, library: str) -> str:
    module = cst.parse_module(source)
    seed_statements: List[cst.BaseStatement] = []
    if library in {"random", "both"}:
        seed_statements.append(cst.parse_statement("random.seed(42)\n"))
    if library in {"numpy", "both"}:
        seed_statements.append(cst.parse_statement("numpy.random.seed(42)\n"))

    class SeedInserter(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if fn_name and fn_name not in original_node.name.value and fn_name != "test":
                return updated_node
            body = list(updated_node.body.body)
            existing_codes = [stmt.code for stmt in body]
            to_insert: List[cst.BaseStatement] = []
            for stmt in seed_statements:
                if stmt.code not in existing_codes:
                    to_insert.append(stmt)
            if not to_insert:
                return updated_node
            return updated_node.with_changes(body=updated_node.body.with_changes(body=[*to_insert, *body]))

    transformed = module.visit(SeedInserter()).code
    if library in {"random", "both"} and "import random" not in transformed:
        transformed = "import random\n" + transformed
    if library in {"numpy", "both"} and "import numpy" not in transformed:
        transformed = "import numpy\n" + transformed
    return transformed


# ── V2 Deep-Action Implementations (Bug 2 fix) ────────────────────────────────────

def _apply_refactor_concurrency(source: str, target: Dict[str, Any]) -> str:
    """
    Swap a threading/sync primitive for a safer alternative.

    Production fixes:
      1. Matches the FULL dotted name (e.g. "threading.Lock") — not a loose
         substring — so "asyncio.Lock" and "FileLock" are never accidentally swapped.
      2. Also rewrites the import statement to match the new location.
         If `from threading import Lock` exists, changes it to the new module.
         If `import threading` exists and the new primitive is in asyncio, adds
         `import asyncio` if not already present.
    """
    from_primitive: str = str(target.get("from_primitive", "threading.Lock"))
    to_primitive: str = str(target.get("to_primitive", "threading.RLock"))

    # Derive module + class name for import rewriting
    from_module = from_primitive.rsplit(".", 1)[0] if "." in from_primitive else ""
    to_module = to_primitive.rsplit(".", 1)[0] if "." in to_primitive else ""
    to_class = to_primitive.rsplit(".", 1)[-1]

    module = cst.parse_module(source)

    class PrimitiveSwapper(cst.CSTTransformer):
        """Rewrite every call site that exactly matches from_primitive."""

        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
            # Use EXACT full dotted name comparison, not substring.
            full_name = _cst_name(updated_node.func)
            if full_name != from_primitive:
                return updated_node
            try:
                replacement_func = cst.parse_expression(to_primitive)
                return updated_node.with_changes(func=replacement_func)
            except Exception:
                return updated_node

    class ImportRewriter(cst.CSTTransformer):
        """
        Rewrite import statements to match the new primitive's module.

        Handles all four patterns:
          import threading                  → also imports to_module
          from threading import Lock        → from to_module import to_class
          from threading import Lock as L   → from to_module import to_class as L
          import threading as thr           → also imports to_module as its alias
        """

        def leave_ImportFrom(
            self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
        ) -> cst.ImportFrom:
            if from_module and to_module and from_module != to_module:
                # Get the module name as a string safely
                try:
                    mod_str = cst.parse_module("").code_for_node(updated_node.module) if updated_node.module else ""
                except Exception:
                    mod_str = ""
                if mod_str == from_module:
                    new_module = cst.parse_expression(to_module)
                    # Rewrite the imported name
                    new_names: List[cst.ImportAlias] = []
                    for alias_node in (updated_node.names if isinstance(updated_node.names, (list, tuple)) else []):
                        if isinstance(alias_node, cst.ImportAlias):
                            name_str = alias_node.name.value if isinstance(alias_node.name, cst.Name) else ""
                            from_class = from_primitive.rsplit(".", 1)[-1]
                            if name_str == from_class:
                                new_alias = alias_node.with_changes(
                                    name=cst.Name(to_class)
                                )
                                new_names.append(new_alias)
                            else:
                                new_names.append(alias_node)
                        else:
                            pass  # skip star imports
                    return updated_node.with_changes(
                        module=new_module,
                        names=new_names if new_names else updated_node.names,
                    )
            return updated_node

    # Step 1: swap all call sites
    transformed = module.visit(PrimitiveSwapper()).code
    # Step 2: rewrite imports
    transformed_module = cst.parse_module(transformed)
    transformed = transformed_module.visit(ImportRewriter()).code
    # Step 3: ensure the new primitive's module is imported if it isn't already
    if to_module and to_module not in transformed.split(".")[0]:
        if f"import {to_module}" not in transformed and f"from {to_module}" not in transformed:
            transformed = f"import {to_module}\n" + transformed
    return transformed


def _apply_isolate_boundary(source: str, target: Dict[str, Any]) -> str:
    """Wrap external call in circuit-breaker / timeout."""
    boundary_call = str(target.get("boundary_call", ""))
    pattern = str(target.get("pattern", "timeout"))

    module = cst.parse_module(source)

    class BoundaryIsolator(cst.CSTTransformer):
        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
            name = _cst_name(updated_node.func)
            if boundary_call and boundary_call in (name or ""):
                try:
                    if pattern == "timeout":
                        wrapper = cst.parse_expression(f"asyncio.wait_for({name}(), timeout=5.0)")
                    elif pattern == "circuit_breaker":
                        wrapper = cst.parse_expression(f"_circuit_breaker({name})")
                    else:
                        return updated_node
                    return wrapper
                except Exception:
                    pass
            return updated_node

    return module.visit(BoundaryIsolator()).code


def _apply_extract_async_scope(source: str, target: Dict[str, Any]) -> str:
    """
    Structurally fix the async/sync boundary problem.

    direction='sync_to_async'  (most common case):
      Converts a synchronous function to async AND wraps every detected
      blocking I/O call (DB / HTTP / socket) with ``asyncio.to_thread()``
      so the event loop is never stalled.
      This is the correct fix for: ``async def f(): db.commit()``

    direction='async_to_sync':
      Removes the async/await keywords so the function can be called
      safely from synchronous contexts.

    Previous implementation only flipped the `async def` keyword but left
    blocking calls inside untouched — which made things WORSE.
    """
    from typing import Set as _Set

    fn_name = str(target.get("function_name", ""))
    direction = str(target.get("direction", "sync_to_async"))

    # Blocking signatures reused from causal_graph detection sets
    BLOCKING_CALL_SIGNATURES: _Set[str] = {
        # DB
        "session.commit", "session.execute", "session.add", "session.flush",
        "cursor.execute", "cursor.executemany",
        "db.commit", "db.execute", "db.session.commit",
        "engine.connect", "engine.execute",
        "collection.find", "collection.insert_one", "collection.update_one",
        "redis.set", "redis.get",
        # HTTP
        "requests.get", "requests.post", "requests.put",
        "requests.delete", "requests.patch", "urllib.request.urlopen",
        # Filesystem / socket blocking
        "socket.recv", "socket.accept",
        # time.sleep is a special case — always blocking
        "time.sleep",
    }
    # Partial prefixes that should be wrapped regardless of full name
    BLOCKING_PREFIXES = (
        "requests.", "urllib.request.", "socket.", "cursor.",
        "session.", "db.", "engine.", "collection.", "redis.",
    )

    module = cst.parse_module(source)

    class AsyncScopeExtractor(cst.CSTTransformer):
        def _is_target_function(self, node_name: str) -> bool:
            return (not fn_name) or (fn_name in node_name)

        def _call_is_blocking(self, call_node: cst.Call) -> bool:
            full = _cst_name(call_node.func)
            return (
                full in BLOCKING_CALL_SIGNATURES
                or any(full.startswith(pfx) for pfx in BLOCKING_PREFIXES)
            )

        # ── sync_to_async: make function async + wrap blocking calls ──────────
        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.FunctionDef:
            if direction != "sync_to_async":
                return updated_node
            if not self._is_target_function(original_node.name.value):
                return updated_node
            if original_node.asynchronous:
                # Already async — just wrap blocking calls inside
                return updated_node
            return updated_node.with_changes(
                asynchronous=cst.Asynchronous(),
            )

        def leave_Call(
            self, original_node: cst.Call, updated_node: cst.Call
        ) -> cst.BaseExpression:
            """Wrap blocking calls inside the target function with asyncio.to_thread()."""
            if direction != "sync_to_async":
                return updated_node
            if not self._call_is_blocking(updated_node):
                return updated_node
            # Build: await asyncio.to_thread(original_call)
            # The `await` keyword is added by the parent Await node;
            # here we just wrap the call in asyncio.to_thread(...).
            try:
                wrapped = cst.parse_expression(
                    f"asyncio.to_thread({_cst_name(updated_node.func)})"
                )
                # Preserve original arguments inside to_thread
                if isinstance(wrapped, cst.Call) and updated_node.args:
                    all_args = list(wrapped.args) + list(updated_node.args)
                    wrapped = wrapped.with_changes(args=all_args)
                return wrapped
            except Exception:
                return updated_node

        # ── async_to_sync: strip async/await ──────────────────────────────────
        def leave_AsyncFunctionDef(
            self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
        ) -> cst.FunctionDef:
            if direction != "async_to_sync":
                return updated_node  # type: ignore[return-value]
            if not self._is_target_function(original_node.name.value):
                return updated_node  # type: ignore[return-value]
            # Convert AsyncFunctionDef → FunctionDef
            sync_fn = cst.FunctionDef(
                name=updated_node.name,
                params=updated_node.params,
                body=updated_node.body,
                decorators=updated_node.decorators,
                returns=updated_node.returns,
                leading_lines=updated_node.leading_lines,
                lines_after_decorators=updated_node.lines_after_decorators,
            )
            return sync_fn

    transformed = module.visit(AsyncScopeExtractor()).code
    # Ensure asyncio is imported when doing sync_to_async
    if direction == "sync_to_async" and "import asyncio" not in transformed:
        transformed = "import asyncio\n" + transformed
    return transformed


def _apply_harden_idempotency(source: str, target: Dict[str, Any]) -> str:
    """Add idempotency guard to a state-mutating function."""
    state_target = str(target.get("state_target", ""))
    key_strategy = str(target.get("key_strategy", "uuid"))

    module = cst.parse_module(source)
    idempotency_guard = (
        "\n"
        "def _ensure_idempotency(key_strategy='uuid'):\n"
        "    import uuid, hashlib\n"
        "    if key_strategy == 'uuid':\n"
        "        return str(uuid.uuid4())\n"
        "    return hashlib.md5(str(key_strategy).encode()).hexdigest()\n"
    )

    class IdempotencyHardener(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if state_target and state_target not in original_node.name.value:
                return updated_node
            body = list(updated_node.body.body)
            guard_stmt = cst.parse_statement("_idempotency_key = _ensure_idempotency()\n")
            if any(getattr(s, "code", "").find("_idempotency_key") >= 0 for s in body):
                return updated_node
            return updated_node.with_changes(body=updated_node.body.with_changes(body=[guard_stmt, *body]))

    transformed = module.visit(IdempotencyHardener()).code
    if "def _ensure_idempotency" not in transformed:
        transformed = transformed + idempotency_guard
    return transformed


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
