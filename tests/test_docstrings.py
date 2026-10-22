"""Tests for the docstrings: every public function documents its arguments, return value and errors.

The HTML documentation (pdoc, Google style) is built from the docstrings, so a docstring that leaves out an
argument, or still names one the function no longer has, puts the documentation out of step with the code.
The checks read the source and import nothing, so they need none of the optional dependencies.
"""

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "treeskel"
MODULES = sorted(PACKAGE.glob("*.py"))

SECTION = re.compile(r"^(\s*)(Args|Arguments|Returns|Yields|Raises|Attributes|Example|Examples|Note|Notes"
                     r"|Warning|Warnings|See Also|References|Todo):\s*$")
ENTRY = re.compile(r"^\s*(\**[A-Za-z_]\w*)\s*(\([^)]*\))?:")


def section_names(doc: str) -> dict[str, list[str]]:
    """The names each section of a Google-style docstring lists: ``{"Args": [...], "Attributes": [...], ...}``.

    Only an entry's first line counts (the lines indented under it continue it). An entry naming several
    things at once (``a, b: ...``) gives no name, so the check below reports both as undocumented.
    """
    out: dict[str, list[str]] = {}
    current, indent = None, None
    for line in doc.splitlines():
        m = SECTION.match(line)
        if m:
            current, indent = m.group(2).replace("Arguments", "Args"), None
            out.setdefault(current, [])
            continue
        if current is None or not line.strip():
            continue
        level = len(line) - len(line.lstrip())
        if indent is None:
            indent = level
        if level == indent and (e := ENTRY.match(line)):
            out[current].append(e.group(1))
    return out


def parameters(fn: ast.FunctionDef) -> list[str]:
    """A function's parameters as a docstring names them (``*args``, ``**kwargs``), without ``self``/``cls``."""
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return [n for n in names if n not in ("self", "cls")]


def own_nodes(fn: ast.FunctionDef):
    """The nodes of a function's body, not descending into nested functions or classes."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def is_property(fn: ast.FunctionDef) -> bool:
    """A property's docstring describes the attribute, so it needs no Returns section."""
    return any(isinstance(d, ast.Name) and d.id in ("property", "cached_property") for d in fn.decorator_list)


def is_dataclass(cls: ast.ClassDef) -> bool:
    """Decorated with ``@dataclass`` or ``@dataclass(...)``."""
    return any((isinstance(d, ast.Name) and d.id == "dataclass")
               or (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass") for d in cls.decorator_list)


def function_problems(fn: ast.FunctionDef, name: str, class_args: list[str] | None = None) -> list[str]:
    """What a public function's docstring lacks or gets wrong; ``class_args`` are a class docstring's Args,
    which may document ``__init__`` instead (Google style allows either place)."""
    doc = ast.get_docstring(fn)
    if not doc:
        return [f"{name}: no docstring"]
    found = section_names(doc)
    out = []
    params = parameters(fn)
    if fn.name == "__init__" and "Args" not in found and class_args is not None:
        found["Args"] = class_args
    if params and "Args" not in found:
        out.append(f"{name}: no Args section for {params}")
    elif "Args" in found and found["Args"] != params:
        out.append(f"{name}: Args lists {found['Args']}, the signature has {params}")
    returns = any(isinstance(n, ast.Return) and n.value is not None
                  and not (isinstance(n.value, ast.Constant) and n.value.value is None) for n in own_nodes(fn))
    yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in own_nodes(fn))
    if (returns or yields) and not is_property(fn) and not {"Returns", "Yields"} & found.keys():
        out.append(f"{name}: returns a value but has no Returns section")
    if any(isinstance(n, ast.Raise) and n.exc is not None for n in own_nodes(fn)) and "Raises" not in found:
        out.append(f"{name}: raises but has no Raises section")
    return out


def problems(source: str) -> list[str]:
    """Every docstring problem in a module's source: the module, its public functions, and its classes with
    their public methods (private helpers and functions nested in others are not part of the API)."""
    tree = ast.parse(source)
    out = [] if ast.get_docstring(tree) else ["module: no docstring"]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            out += function_problems(node, node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            if not doc:
                out.append(f"{node.name}: no docstring")
                continue
            found = section_names(doc)
            if is_dataclass(node):
                fields = [x.target.id for x in node.body
                          if isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name)]
                if fields and found.get("Attributes") != fields:
                    out.append(f"{node.name}: Attributes lists {found.get('Attributes')}, the fields are {fields}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                        not sub.name.startswith("_") or sub.name == "__init__"):
                    if sub.name == "__init__" and not ast.get_docstring(sub):
                        sub_problems = [] if found.get("Args") == parameters(sub) else [
                            f"{node.name}.__init__: the class's Args list {found.get('Args')}, the signature has "
                            f"{parameters(sub)}"]
                    else:
                        sub_problems = function_problems(sub, f"{node.name}.{sub.name}", found.get("Args"))
                    out += sub_problems
    return out


# --- the checker itself, on small sources with a known answer ---------------------------------------------

GOOD = '''"""A module."""


def area(width: float, height: float = 1.0) -> float:
    """A rectangle's area.

    Args:
        width: Its width (m).
        height: Its height (m).

    Returns:
        The area (m²).

    Raises:
        ValueError: If a side is negative.
    """
    if width < 0 or height < 0:
        raise ValueError("negative side")
    return width * height
'''


def test_a_complete_docstring_passes():
    assert problems(GOOD) == []


def test_a_missing_or_stale_argument_is_found():
    missing = GOOD.replace("        height: Its height (m).\n", "")
    assert problems(missing) == ["area: Args lists ['width'], the signature has ['width', 'height']"]
    renamed = GOOD.replace("height: float = 1.0", "depth: float = 1.0").replace("or height < 0", "or depth < 0")
    assert problems(renamed) == ["area: Args lists ['width', 'height'], the signature has ['width', 'depth']"]


def test_grouped_arguments_count_as_undocumented():
    grouped = GOOD.replace("        width: Its width (m).\n        height: Its height (m).\n",
                           "        width, height: Its sides (m).\n")
    assert problems(grouped) == ["area: Args lists [], the signature has ['width', 'height']"]


def test_missing_returns_and_raises_are_found():
    no_returns = GOOD.replace("    Returns:\n        The area (m²).\n\n", "")
    assert problems(no_returns) == ["area: returns a value but has no Returns section"]
    no_raises = GOOD.replace("\n    Raises:\n        ValueError: If a side is negative.\n", "")
    assert problems(no_raises) == ["area: raises but has no Raises section"]


def test_private_nested_and_property_functions_are_exempt():
    source = '''"""A module."""


def _helper(x):
    return x


class Box:
    """A box.

    Args:
        side: Its side.
    """

    def __init__(self, side):
        self.side = side

    @property
    def volume(self) -> float:
        """Its volume."""
        return self.side ** 3


def outer() -> None:
    """Nothing to document."""
    def inner(y):
        return y
    inner(1)
'''
    assert problems(source) == []


def test_dataclass_fields_and_class_arguments_are_checked():
    source = '''"""A module."""
from dataclasses import dataclass


@dataclass
class Point:
    """A point.

    Attributes:
        x: Across.
    """

    x: float
    y: float


class Box:
    """A box.

    Args:
        side: Its side.
    """

    def __init__(self, side, colour):
        self.side = side
'''
    assert problems(source) == ["Point: Attributes lists ['x'], the fields are ['x', 'y']",
                                "Box.__init__: the class's Args list ['side'], the signature has ['side', 'colour']"]


def test_a_missing_docstring_is_found():
    assert problems("def f(x):\n    return x\n") == ["module: no docstring", "f: no docstring"]


# --- the package ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", MODULES, ids=[p.stem for p in MODULES])
def test_every_public_docstring_is_complete(path):
    assert problems(path.read_text(encoding="utf-8")) == []
