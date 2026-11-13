"""Prove every project source file still runs on Python 3.9, without a 3.9 interpreter.

    python3 scripts/check_python39_syntax.py

Three checks, because no single one catches everything:
  1. Grammar: each file is parsed with the 3.9 grammar (match statements,
     parenthesised context managers and similar newer syntax are refused).
  2. Union annotations: `int | None` parses in every version but fails at run time
     on 3.9 inside an annotation, so every annotation subtree is walked for `|`.
  3. Newer standard library names: names added after 3.9 parse cleanly and fail at
     run time, so they are searched for by name.
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import config  # noqa: E402

# Attribute or import names that do not exist on Python 3.9.
FORBIDDEN_NAMES = {
    ("datetime", "UTC"), ("itertools", "pairwise"), ("itertools", "batched"),
    ("typing", "Self"), ("typing", "TypeAlias"), ("typing", "ParamSpec"),
    ("typing", "Concatenate"), ("typing", "TypeGuard"), ("typing", "Never"),
    ("typing", "LiteralString"), ("typing", "Required"), ("typing", "NotRequired"),
    ("typing", "Unpack"), ("typing", "assert_never"), ("typing", "reveal_type"),
    ("typing", "override"), ("enum", "StrEnum"), ("enum", "verify"), ("math", "cbrt"),
    ("math", "exp2"), ("asyncio", "TaskGroup"), ("asyncio", "timeout"),
    ("contextlib", "chdir"), ("hashlib", "file_digest"), ("tomllib", "*"),
    ("operator", "call"),
}
FORBIDDEN_MODULES = {"tomllib", "wsgiref.types"}


def _annotation_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for argument in arguments.args + arguments.kwonlyargs + arguments.posonlyargs + \
                    [a for a in (arguments.vararg, arguments.kwarg) if a]:
                if argument.annotation is not None:
                    yield argument.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def check_file(path):
    findings = []
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source, filename=path, feature_version=(3, 9))
    except SyntaxError as error:
        return [path + ":" + str(error.lineno) + " grammar: " + str(error.msg)]
    for annotation in _annotation_nodes(tree):
        for node in ast.walk(annotation):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                findings.append(path + ":" + str(node.lineno)
                                + " union annotation with | (use typing.Optional/Union)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in FORBIDDEN_NAMES:
                findings.append(path + ":" + str(node.lineno) + " " + node.value.id + "."
                                + node.attr + " does not exist on 3.9")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in FORBIDDEN_MODULES:
                findings.append(path + ":" + str(node.lineno) + " module " + node.module
                                + " does not exist on 3.9")
            for alias in node.names:
                if (node.module, alias.name) in FORBIDDEN_NAMES:
                    findings.append(path + ":" + str(node.lineno) + " " + node.module + "."
                                    + alias.name + " does not exist on 3.9")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_MODULES:
                    findings.append(path + ":" + str(node.lineno) + " module " + alias.name
                                    + " does not exist on 3.9")
    return findings


def check_project(root_path):
    """Return (files checked, findings). An empty findings list means all clear."""
    skipped = set(config.SCRIPTS_SKIPPED_DIRECTORIES)
    checked, findings = [], []
    for folder, directories, files in os.walk(root_path):
        directories[:] = sorted(d for d in directories if d not in skipped
                                and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                checked.append(path)
                findings.extend(check_file(path))
    return checked, findings


def main():
    root = os.path.dirname(HERE)
    checked, findings = check_project(root)
    print("Checking {0} source files against Python 3.9".format(len(checked)))
    if findings:
        for finding in findings:
            print("    " + os.path.relpath(finding, root) if os.path.isabs(finding) else
                  "    " + finding)
        print(str(len(findings)) + " finding(s).")
        return 1
    print("All clear. Every file parses and every name exists on 3.9.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
