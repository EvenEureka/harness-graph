"""Size of a Commit0 library at its stubbed commit: source files, functions, and how many bodies are stubs."""
import ast
import sys
from pathlib import Path


def is_stub(fn) -> bool:
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def main(src):
    files = stubs = funcs = lines = with_stubs = 0
    per = []
    for p in sorted(Path(src).rglob("*.py")):
        if "test" in p.parts or p.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        fs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        s = sum(is_stub(f) for f in fs)
        files += 1; funcs += len(fs); stubs += s; with_stubs += s > 0
        lines += len(p.read_text(errors="replace").splitlines())
        if s:
            per.append((s, str(p.relative_to(src))))
    print(f"files={files} lines={lines} functions={funcs} stubs={stubs} files_with_stubs={with_stubs}")
    print("  top:", ", ".join(f"{n}:{s}" for s, n in sorted(per, reverse=True)[:8]))


if __name__ == "__main__":
    main(sys.argv[1])
