#!/usr/bin/env python3
"""Check optimized paths in src/cl/shufl.cl against their plain implementations.

For every parameter combination supported by the FFT shape space, evaluate the write and read indices for every lane and
element, checking that:

  1. All indices are within the LDS block the workgroup may use, whose pointer is where the allocation puts the block.
  2. No two writes target the same slot, and every read targets a written slot.
  3. Each lane ends with the same value in every component of u as in the plain implementation.
  4. A barrier separates every run of LDS writes from the reads around it.
  5. Every arm of one function leaves each lane the same value as the arm before it.
  6. When workgroups share LDS (LDSMUL), each path is one LDStx_start ... LDStx_end transaction with only LDSbar inside.

LDSMUL and the number of workgroups decide which arm runs and how large its block is, so every combination of them is
checked, and a path that addresses LDS in bytes from its start is replayed at each workgroup's block. The plain
implementations also undergo checks 1, 2, 4 and 6, including arms without optimized paths. Expressions, allocation
rules, type widths, and supported parameter combinations are read from the source; unreadable or unrecognized source is
an error, never a silent skip. No GPU is required.

The reference is the plain implementation in the same SHUFL_BYTES arm; incorrect permutations shared with that
implementation are not detected. An error common to every arm of a function is not detected.

Exit status: 0 if all checks pass, 1 if a check fails, 2 if the source cannot be read.
"""

from __future__ import annotations

import argparse
import ast
import functools
import itertools
import operator
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, NamedTuple

PROG: Final = "check-shufl-index"
REPO_ROOT: Final = Path(__file__).resolve().parent.parent

# The OpenCL scalar types base.cl builds its own out of, and the vector lengths it results in.
SCALAR_BYTES: Final = {"char": 1, "short": 2, "int": 4, "uint": 4, "float": 4, "long": 8, "ulong": 8, "double": 8}
VECTOR_LENGTHS: Final = (2, 4, 8, 16)

# The one concrete type shufl casts to by name rather than through an expand.cl macro.
PLAIN_ELEMENT: Final = "int"

IDENTIFIER: Final = r"[A-Za-z_][A-Za-z0-9_]*"

# The values SHUFL_BYTES takes.
SHUFL_BYTES_VALUES: Final = (4, 8, 16)

# The LDSMUL -use values checked, and the numWG a shufl sees: 1 or 2 from the tail kernels, and WMUL from carryFused.
# Both are taken as powers of two up to 4.
LDS_MUL_VALUES: Final = (1, 2, 4)
NUM_WG_VALUES: Final = (1, 2, 4)

# An arm is chosen by SBMUL(numWG) * SHUFL_BYTES, and SBMUL is at most LDSMUL; an arm guarded by ">= n" serves each
# value from n up.
ARM_BYTES_VALUES: Final = tuple(sorted({b * m for b in SHUFL_BYTES_VALUES for m in LDS_MUL_VALUES}))

# The shufl functions fftbase.cl may call.
FUNCTIONS: Final = ("shufl", "shufl_and_fft2")


class SourceError(Exception):
    """The source is not in a shape this checker knows how to read."""


#
# Source Text
#

_COMMENT: Final = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def strip_comments(text: str) -> str:
    """Blank out C and C++ comments, leaving every offset and line number unchanged."""
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def line_of(text: str, offset: int) -> int:
    """Return the 1-based line number corresponding to the given offset in the text."""
    return text.count("\n", 0, offset) + 1


_CLOSER: Final = {"(": ")", "[": "]", "{": "}"}


def skip_balanced(text: str, start: int) -> int:
    """Return the offset immediately after the balanced expression starting at `start`."""
    opener, closer = text[start], _CLOSER[text[start]]
    depth = 0

    for i in range(start, len(text)):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return i + 1

    msg = f"unbalanced {opener!r}/{closer!r} at line {line_of(text, start)}"
    raise SourceError(msg)


_SPACE: Final = re.compile(r"\s*")


def skip_space(text: str, start: int, end: int) -> int:
    """Return the offset of the first non-space character at or after `start`, or `end`."""
    match = _SPACE.match(text, start, end)

    if not match:
        msg = f"expected space at line {line_of(text, start)}"
        raise SourceError(msg)

    return match.end()


def element_widths(base: str, expand: str) -> dict[str, int | None]:
    """Return the byte width of every type shufl may cast its LDS block to.

    base.cl names the concrete types, and expand.cl maps each generic macro onto one of them once per arithmetic part.
    The LDS macros count bytes, so the element type decides how many slots an expression may index. A macro expand.cl
    instantiates at more than one width has no one slot size, and maps to None.
    """
    concrete = {
        f"{scalar}{length or ''}": width * (length or 1)
        for scalar, width in SCALAR_BYTES.items()
        for length in (0, *VECTOR_LENGTHS)
    }

    for m in re.finditer(rf"\btypedef\s+({IDENTIFIER})\s+({IDENTIFIER})\s*;", base):
        if m.group(1) in concrete:
            concrete[m.group(2)] = concrete[m.group(1)]

    seen: dict[str, set[int]] = {}

    for m in re.finditer(rf"^[ \t]*#[ \t]*define[ \t]+({IDENTIFIER})[ \t]+({IDENTIFIER})[ \t]*$", expand, re.MULTILINE):
        name, target = m.group(1), m.group(2)

        if name.startswith("as_"):
            continue

        if target not in concrete:
            msg = f"expand.cl: {name} expands to {target}, which base.cl does not typedef to a type of known width"
            raise SourceError(msg)

        seen.setdefault(name, set()).add(concrete[target])

    if not seen:
        msg = "expand.cl defines no generic type macros"
        raise SourceError(msg)

    widths: dict[str, int | None] = {
        name: concrete[name] for name in (PLAIN_ELEMENT, *(f"{PLAIN_ELEMENT}{n}" for n in VECTOR_LENGTHS))
    }

    for name, found in sorted(seen.items()):
        widths[name] = found.pop() if len(found) == 1 else None

    return widths


def slot_bytes(widths: Mapping[str, int | None], name: str, where: str) -> int:
    """Return the byte width of one slot of `name`, or refuse to guess at a type of no fixed width."""
    width = widths.get(name)

    if width is None:
        msg = f"{where}: {name} has no one width across expand.cl's instantiations, so it cannot size an LDS slot"
        raise SourceError(msg)

    return width


@dataclass(frozen=True)
class Sources:
    """The comment-stripped text of every file the checker reads."""

    shufl: str
    fftbase: str
    base: str
    expand: str
    fftconfig_cpp: str
    fftconfig_h: str

    @classmethod
    def read(cls, root: Path) -> Sources:
        def load(relative: str) -> str:
            try:
                return strip_comments((root / relative).read_text(encoding="utf-8"))
            except UnicodeDecodeError as e:
                msg = f"{relative} is not UTF-8: {e}"
                raise SourceError(msg) from None

        return cls(
            shufl=load("src/cl/shufl.cl"),
            fftbase=load("src/cl/fftbase.cl"),
            base=load("src/cl/base.cl"),
            expand=load("src/cl/expand.cl"),
            fftconfig_cpp=load("src/FFTConfig.cpp"),
            fftconfig_h=load("src/FFTConfig.h"),
        )


#
# C Integer Expressions
#

# Kernel expressions are translated token by token to Python, parsed with ast, and compiled into closures over a
# whitelist of node types. Python and C agree on the relative precedence of every operator allowed here except in three
# places, and each is rejected unless parentheses make the grouping explicit: a comparison with a bitwise operand, "!"
# before anything but an atom, and chained comparisons.

Env = Mapping[str, int]
_Compiled = Callable[[Env], int]


def _unsigned_div(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        msg = f"{a} / {b} is not an unsigned division"
        raise ArithmeticError(msg)

    return a // b


def _unsigned_mod(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        msg = f"{a} % {b} is not an unsigned modulo"
        raise ArithmeticError(msg)

    return a % b


def _shift(shift: Callable[[int, int], int]) -> Callable[[int, int], int]:
    def checked(a: int, b: int) -> int:
        if b < 0:
            msg = f"shift of {a} by {b}"
            raise ArithmeticError(msg)

        return shift(a, b)

    return checked


_BINARY_OPS: Final[dict[type[ast.operator], Callable[[int, int], int]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: _unsigned_div,
    ast.Mod: _unsigned_mod,
    ast.LShift: _shift(lambda a, b: a << b),
    ast.RShift: _shift(lambda a, b: a >> b),
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitXor: lambda a, b: a ^ b,
}

_UNARY_OPS: Final[dict[type[ast.unaryop], Callable[[int], int]]] = {
    ast.Invert: operator.invert,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: lambda a: int(not a),
}

_COMPARE_OPS: Final[dict[type[ast.cmpop], Callable[[int, int], bool]]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_BITWISE_OPS: Final = (ast.BitAnd, ast.BitOr, ast.BitXor)

_C_TOKEN: Final = re.compile(r"&&|\|\||!(?!=)|/|\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b")
_C_TO_PYTHON: Final = {"&&": " and ", "||": " or ", "!": " not ", "/": "//"}


def _python_token(m: re.Match[str]) -> str:
    return m.group(1) or _C_TO_PYTHON[m.group()]


def _parenthesized(py: str, node: ast.expr) -> bool:
    end = len(py) if node.end_col_offset is None else node.end_col_offset
    before, after = py[: node.col_offset].rstrip(), py[end:].lstrip()
    return before.endswith("(") and after.startswith(")")


def _compile(node: ast.expr, py: str) -> _Compiled:
    """Compile an AST expression node into a callable that evaluates it in a given environment."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = node.value
        return lambda _: value

    if isinstance(node, ast.Name):
        name = node.id
        return lambda env: env[name]

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        binary = _BINARY_OPS[type(node.op)]
        left, right = _compile(node.left, py), _compile(node.right, py)
        return lambda env: binary(left(env), right(env))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        atom = isinstance(node.operand, ast.Name | ast.Constant | ast.UnaryOp)

        if isinstance(node.op, ast.Not) and not atom and not _parenthesized(py, node.operand):
            msg = "'!' applies to more in C than it does here; add parentheses"
            raise SourceError(msg)

        unary, operand = _UNARY_OPS[type(node.op)], _compile(node.operand, py)
        return lambda env: unary(operand(env))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE_OPS:
            msg = "chained comparisons do not mean the same in C; add parentheses"
            raise SourceError(msg)

        for side in (node.left, node.comparators[0]):
            if isinstance(side, ast.BinOp) and isinstance(side.op, _BITWISE_OPS) and not _parenthesized(py, side):
                msg = "C binds '==' and '<' tighter than '&', '|' and '^'; add parentheses"
                raise SourceError(msg)

        compare = _COMPARE_OPS[type(node.ops[0])]
        left, right = _compile(node.left, py), _compile(node.comparators[0], py)
        return lambda env: int(compare(left(env), right(env)))

    if isinstance(node, ast.BoolOp):
        parts = [_compile(value, py) for value in node.values]
        if isinstance(node.op, ast.And):
            return lambda env: int(all(part(env) for part in parts))
        return lambda env: int(any(part(env) for part in parts))

    if isinstance(node, ast.IfExp):
        test, chosen, other = _compile(node.test, py), _compile(node.body, py), _compile(node.orelse, py)
        return lambda env: chosen(env) if test(env) else other(env)

    msg = f"unsupported construct {ast.unparse(node)!r}"
    raise SourceError(msg)


@dataclass(frozen=True, eq=False)
class CExpr:
    """A C integer expression from the source, checked and compiled for evaluation."""

    text: str
    names: frozenset[str]
    compiled: _Compiled

    def evaluate(self, env: Env) -> int:
        try:
            return self.compiled(env)
        except KeyError as e:
            msg = f"{self.text!r}: {e.args[0]} is not known here"
            raise SourceError(msg) from None
        except ArithmeticError as e:
            msg = f"{self.text!r} with {dict(env)}: {e}"
            raise SourceError(msg) from None


def _conditionals(c_text: str) -> str:
    """Rewrite each C "a ? b : c" as Python's "((b) if (a) else (c))", innermost parentheses first."""
    pieces: list[str] = []
    position = 0

    for m in re.finditer(r"[(\[]", c_text):
        if m.start() < position:
            continue
        close = skip_balanced(c_text, m.start())
        pieces += [c_text[position : m.end()], _conditionals(c_text[m.end() : close - 1]), c_text[close - 1]]
        position = close

    flat = "".join(pieces) + c_text[position:]

    if (question := flat.find("?")) < 0:
        return flat

    # Once the parentheses are rewritten, every "?" and ":" left is at this level; the ":" that closes the first "?" is
    # the first one not claimed by a "?" after it.
    depth = 0
    for colon in range(question + 1, len(flat)):
        depth += {"?": 1, ":": -1}.get(flat[colon], 0)
        if depth < 0:
            break
    else:
        msg = f"'?' without ':' in C expression {c_text!r}"
        raise SourceError(msg)

    condition, chosen, other = flat[:question], flat[question + 1 : colon], flat[colon + 1 :]
    return f"(({_conditionals(chosen)}) if ({condition}) else ({_conditionals(other)}))"


def _translate(c_text: str) -> tuple[str, ast.expr]:
    py = _C_TOKEN.sub(_python_token, _conditionals(c_text)).strip()

    try:
        return py, ast.parse(py, mode="eval").body
    except SyntaxError as e:
        msg = f"cannot read C expression {c_text!r}: {e.msg}"
        raise SourceError(msg) from None


def _make_expr(text: str, py: str, node: ast.expr) -> CExpr:
    try:
        compiled = _compile(node, py)
    except SourceError as e:
        msg = f"C expression {text!r}: {e}"
        raise SourceError(msg) from None

    names = frozenset(sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name))
    return CExpr(text, names, compiled)


@functools.cache
def parse_c(c_text: str) -> CExpr:
    """Parse one C integer expression."""
    text = " ".join(c_text.split())
    return _make_expr(text, *_translate(text))


@functools.cache
def c_conjuncts(c_text: str) -> tuple[CExpr, ...]:
    """Parse a C condition into its top-level &&-separated parts."""
    py, node = _translate(" ".join(c_text.split()))
    parts = node.values if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) else [node]
    return tuple(_make_expr(py[part.col_offset : part.end_col_offset], py, part) for part in parts)


class Condition(NamedTuple):
    """One requirement a preprocessor branch places on the code inside it.

    `text` is None for an #ifdef or #ifndef, which is never evaluated. `holds` is False for the condition of an earlier
    branch of the same chain, which must be false for this branch to be compiled.
    """

    text: str | None
    holds: bool = True


def _decide_condition(condition: str, known: Env) -> bool | None:
    """Whether a preprocessor condition is true, false, or undecided given the values of the symbols in `known`.

    Each &&-part that uses only known symbols is evaluated. A part that uses none of them is undecided. A part that
    mixes the two is an error.
    """
    result: bool | None = True

    for part in c_conjuncts(condition):
        unknown = sorted(name for name in part.names if name not in known)

        if not unknown:
            if not part.evaluate(known):
                return False
        elif len(unknown) < len(part.names):
            msg = (
                f"cannot decide {part.text!r} in '#if {condition}': it mixes "
                f"{sorted(part.names - set(unknown))} with {unknown}, which are not known here"
            )
            raise SourceError(msg)
        else:
            result = None

    return result


def decide(conditions: Sequence[Condition], known: Env) -> bool | None:
    """Determine whether code under all of `conditions` is compiled.

    Returns True, False, or None if it cannot be decided due to unknown symbols.
    """
    result: bool | None = True

    for condition in conditions:
        value = None if condition.text is None else _decide_condition(condition.text, known)

        if value is not None and not condition.holds:
            value = not value

        if value is False:
            return False

        if value is None:
            result = None

    return result


_DIRECTIVE: Final = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[ \t]*(.*?)[ \t]*$", re.MULTILINE)


class _Frame(NamedTuple):
    earlier: tuple[str | None, ...]
    current: str | None
    in_else: bool

    def conditions(self) -> tuple[Condition, ...]:
        negated = tuple(Condition(text, holds=False) for text in self.earlier)
        return negated if self.in_else else (*negated, Condition(self.current))


def preprocessor_frames(text: str, offset: int) -> list[tuple[Condition, ...]]:
    """Return the conditions of each #if chain enclosing `offset`, outermost first."""
    stack: list[_Frame] = []

    for m in _DIRECTIVE.finditer(text, 0, offset):
        kind, condition = m.groups()
        where = f"line {line_of(text, m.start())}"

        match kind:
            case "if":
                stack.append(_Frame((), condition, in_else=False))
            case "ifdef" | "ifndef":
                stack.append(_Frame((), None, in_else=False))
            case _ if not stack:
                msg = f"#{kind} without #if at {where}"
                raise SourceError(msg)
            case "elif" | "else" if stack[-1].in_else:
                msg = f"#{kind} after #else at {where}"
                raise SourceError(msg)
            case "elif":
                frame = stack[-1]
                stack[-1] = _Frame((*frame.earlier, frame.current), condition, in_else=False)
            case "else":
                frame = stack[-1]
                stack[-1] = _Frame((*frame.earlier, frame.current), None, in_else=True)
            case _:
                stack.pop()

    return [frame.conditions() for frame in stack]


def enclosing_conditions(text: str, offset: int) -> list[Condition]:
    """Return every condition the #if, #elif and #else branches enclosing `offset` place on it, outermost first."""
    return [condition for frame in preprocessor_frames(text, offset) for condition in frame]


#
# FFT Shape Space
#


class Shape(NamedTuple):
    radix: int
    wg: int

    def __str__(self) -> str:
        return f"RADIX={self.radix} WG={self.wg}"


def shape_space(fftconfig_cpp: str, fftconfig_h: str) -> list[Shape]:
    """Return every (RADIX, WG) an fft_WIDTH or fft_HEIGHT kernel is compiled for.

    A pass over N points is split into RADIX = nW() (or nH()) chunks by a workgroup of WG = N / RADIX threads, so the
    size lists in FFTConfig.cpp and those two functions decide every shape.
    """
    shapes: set[Shape] = set()

    for dimension, radix_fn in (("width", "nW"), ("height", "nH")):
        sizes = re.search(rf"for\s*\(\s*u32\s+const\s+{dimension}\s*:\s*\{{([^}}]*)\}}", fftconfig_cpp)

        if sizes is None:
            msg = f"cannot find the {dimension} list in FFTConfig.cpp"
            raise SourceError(msg)

        entries = [entry.strip() for entry in sizes.group(1).split(",")]
        if entries[-1] == "":
            entries.pop()

        if not entries or not all(entry.isdigit() for entry in entries):
            msg = f"cannot read the {dimension} list in FFTConfig.cpp as integer literals: {{{sizes.group(1).strip()}}}"
            raise SourceError(msg)

        rule = re.search(
            rf"\b{radix_fn}\(\)\s*const\s*\{{\s*return\s*\((.*?)\)\s*\?\s*(\d+)\s*:\s*(\d+)\s*;",
            fftconfig_h,
            re.DOTALL,
        )

        size_test = rf"\b{dimension}\s*==\s*(\d+)"
        if rule is None or re.sub(rf"{size_test}|\|\||\s", "", rule.group(1)):
            msg = f"cannot read {radix_fn}() in FFTConfig.h as '({dimension} == n || ...) ? a : b'"
            raise SourceError(msg)

        small_sizes = {int(size) for size in re.findall(size_test, rule.group(1))}
        small, large = int(rule.group(2)), int(rule.group(3))

        for size in map(int, entries):
            radix = small if size in small_sizes else large

            if radix < 2 or size < radix or size % radix:
                msg = f"{dimension} {size} with {radix_fn}() = {radix} is not a whole number of workgroups"
                raise SourceError(msg)

            shapes.add(Shape(radix, size // radix))

    return sorted(shapes)


class CallSite(NamedTuple):
    function: str
    f: int
    r: int


_CALL: Final = re.compile(r"\b(shufl\w*)\s*\(")
# The loop starts at s0 where a fused first radix step runs ahead of it; s0 is then 1 or RADIX, so the loop's f values
# are still powers of RADIX below WG.
_GENERIC_LOOP: Final = re.compile(
    r"\bfor\s*\(\s*u32\s+s\s*=\s*(?:1|s0)\s*;\s*s\s*<\s*WG\s*;\s*s\s*\*=\s*RADIX\s*\)\s*\{"
)


def _powers(shape: Shape) -> Iterator[int]:
    f = 1
    while f < shape.wg:
        yield f
        f *= shape.radix


def call_sites(fftbase: str, shapes: Sequence[Shape]) -> dict[Shape, set[CallSite]]:
    """Return every (function, f, r) that fft_WIDTH or fft_HEIGHT can call shufl with, by shape."""
    loops = [(m.start(), skip_balanced(fftbase, m.end() - 1)) for m in _GENERIC_LOOP.finditer(fftbase)]
    sites: dict[Shape, set[CallSite]] = {shape: set() for shape in shapes}
    found = False

    for m in _CALL.finditer(fftbase):
        found = True
        function, where = m.group(1), f"fftbase.cl:{line_of(fftbase, m.start())}"
        args_end = skip_balanced(fftbase, m.end() - 1) - 1
        args = [arg.strip() for arg in fftbase[m.end() : args_end].split(",")]

        if function not in FUNCTIONS:
            msg = f"{where}: call to {function}, which is not one of {list(FUNCTIONS)}"
            raise SourceError(msg)

        # shufl(lds, u, f, [r,] numWG, lowMe)
        f_arg = args[2] if len(args) > 2 else ""
        r_arg = args[3] if len(args) == 6 else None
        in_generic_loop = any(start < m.start() < end for start, end in loops)
        if len(args) not in (5, 6) or not (f_arg.isdigit() or (f_arg == "s" and in_generic_loop)):
            msg = f"{where}: cannot read the call {function}({', '.join(args)})"
            raise SourceError(msg)

        if r_arg is not None and not r_arg.isdigit():
            msg = f"{where}: r is {r_arg!r}, not a literal"
            raise SourceError(msg)

        # A call that may be compiled at a shape is a call site there; extra sites only add work.
        conditions = enclosing_conditions(fftbase, m.start())
        for shape in shapes:
            if decide(conditions, {"WG": shape.wg, "RADIX": shape.radix}) is not False:
                r = shape.radix if r_arg is None else int(r_arg)
                fs = [int(f_arg)] if f_arg.isdigit() else _powers(shape)
                sites[shape].update(CallSite(function, f, r) for f in fs)

    if not found:
        msg = "found no shufl calls in fftbase.cl"
        raise SourceError(msg)

    return sites


class Mode(NamedTuple):
    """The LDSPAD / LDSSWIZ setting a kernel is compiled with."""

    ldspad: int
    ldsswiz: int

    def __str__(self) -> str:
        return f"LDSPAD={self.ldspad} LDSSWIZ={self.ldsswiz}"


# LDSSWIZ is only tuned with LDSPAD off, so both on is not a configuration to check.
MODES: Final = (
    Mode(ldspad=1, ldsswiz=0),
    Mode(ldspad=0, ldsswiz=1),
    Mode(ldspad=0, ldsswiz=0),
)


class Layout(NamedTuple):
    """How many workgroups share one kernel's LDS, and whether they may share blocks of it."""

    lds_mul: int
    num_wg: int
    nvidia: int

    def __str__(self) -> str:
        return f"LDSMUL={self.lds_mul} numWG={self.num_wg} NVIDIAGPU={self.nvidia}"


# Sharing needs an nVidia GPU or a WG within the wavefront. Both lay LDS out alike, so WAVEFRONT is held at 0 and the
# two NVIDIAGPU values cover sharing and not sharing.
LAYOUTS: Final = tuple(itertools.starmap(Layout, itertools.product(LDS_MUL_VALUES, NUM_WG_VALUES, (0, 1))))


@dataclass(frozen=True)
class Allocation:
    """The LDS block each workgroup's shufl may use under one setting.

    `blocks[w]` is the (start, end) byte range of workgroup w: its own share of LDS, or the SBMUL shares its transaction
    locks when sharing. `arm_bytes` is SBMUL(numWG) * SHUFL_BYTES, which chooses the arm.
    """

    mode: Mode
    shufl_bytes: int
    layout: Layout
    sharing: bool
    arm_bytes: int
    blocks: tuple[tuple[int, int], ...]

    def __str__(self) -> str:
        return f"{self.mode} SHUFL_BYTES={self.shufl_bytes} {self.layout}"

    def slots(self, pointer_bytes: int, element_bytes: int) -> tuple[int, str | None]:
        """Return the slots every workgroup can index, and a problem if some workgroup's pointer misses its block.

        LDSptr and LDSsharing_ptr step the pointer they are given, so the offset is truncated to that pointer's type.
        """
        capacity = min(end - start for start, end in self.blocks) // element_bytes
        problem: str | None = None

        for w, (start, end) in enumerate(self.blocks):
            offset = start // pointer_bytes * pointer_bytes

            if problem is None and (offset != start or offset % element_bytes):
                problem = (
                    f"workgroup {w}'s LDS pointer lands on byte {offset}, not on its block at byte {start} in"
                    f" {element_bytes}-byte slots"
                )

            capacity = min(capacity, (end - offset) // element_bytes)

        return capacity, problem

    def regions(self, pointer_bytes: int) -> tuple[int, ...]:
        """Return the byte offsets from the start of LDS at which the workgroups' LDS pointers land."""
        return tuple(sorted({start // pointer_bytes * pointer_bytes for start, _ in self.blocks}))


_MACRO_DEFINE: Final = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(numWG\)[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_MACRO_CALL: Final = re.compile(r"\b(\w+)\s*\(\s*numWG\s*\)")

# The fftbase.cl macros that lay out LDS.
_LDS_MACROS: Final = ("SHARING_LDS", "SBMUL", "LDS_SHUFL_BYTES", "LDS_BYTES")


@dataclass(frozen=True)
class LdsMacros:
    """The LDS layout macros in fftbase.cl, expanded for each LDSMUL."""

    expressions: Mapping[tuple[str, int], CExpr]

    @classmethod
    def parse(cls, fftbase: str) -> LdsMacros:
        definitions: dict[str, list[tuple[list[Condition], str]]] = {}

        for m in _MACRO_DEFINE.finditer(fftbase):
            definitions.setdefault(m.group(1), []).append((enclosing_conditions(fftbase, m.start()), m.group(2)))

        if missing := [name for name in _LDS_MACROS if name not in definitions]:
            msg = f"cannot find '#define {missing[0]}(numWG) ...' in fftbase.cl"
            raise SourceError(msg)

        def expand(name: str, lds_mul: int, seen: tuple[str, ...]) -> str:
            if name not in definitions:
                msg = f"fftbase.cl: {name}(numWG) is used in {seen[-1]} but not defined as a macro of numWG"
                raise SourceError(msg)

            if name in seen:
                msg = f"fftbase.cl: {name}(numWG) is defined in terms of itself"
                raise SourceError(msg)

            known = {"LDSMUL": lds_mul}
            bodies = [body for conditions, body in definitions[name] if decide(conditions, known) is not False]

            if len(bodies) != 1 or not all(decide(c, known) is not None for c, _ in definitions[name]):
                msg = f"fftbase.cl: cannot pick one definition of {name}(numWG) at LDSMUL={lds_mul}"
                raise SourceError(msg)

            return _MACRO_CALL.sub(lambda m: f"({expand(m.group(1), lds_mul, (*seen, name))})", bodies[0])

        return cls({(name, m): parse_c(expand(name, m, ())) for name in _LDS_MACROS for m in LDS_MUL_VALUES})

    def allocation(self, mode: Mode, shufl_bytes: int, layout: Layout, shape: Shape) -> Allocation:
        env = {
            "LDSPAD": mode.ldspad,
            "LDSSWIZ": mode.ldsswiz,
            "SHUFL_BYTES": shufl_bytes,
            "RADIX": shape.radix,
            "WG": shape.wg,
            "LDSMUL": layout.lds_mul,
            "numWG": layout.num_wg,
            "NVIDIAGPU": layout.nvidia,
            "WAVEFRONT": 0,
        }

        def value(name: str) -> int:
            return self.expressions[name, layout.lds_mul].evaluate(env)

        sharing, sbmul, share = bool(value("SHARING_LDS")), value("SBMUL"), value("LDS_SHUFL_BYTES")
        span = sbmul if sharing else 1

        if sbmul < 1 or (not sharing and sbmul != 1):
            msg = f"fftbase.cl: SBMUL(numWG) is {sbmul} with SHARING_LDS(numWG) {int(sharing)} under {layout}"
            raise SourceError(msg)

        # The semaphores follow the shares, so no block may reach past them.
        end = min(layout.num_wg * share, value("LDS_BYTES"))
        blocks = tuple(
            (w // span * span * share, min((w // span + 1) * span * share, end)) for w in range(layout.num_wg)
        )

        return Allocation(mode, shufl_bytes, layout, sharing, sbmul * shufl_bytes, blocks)


_POINTER_FUNCTION: Final = re.compile(
    r"\blocal\s+(\w+)\s*\*\s*OVERLOAD\s+(LDSptr|LDSsharing_ptr)\s*\(\s*local\s+\1\s*\*\s*lds\s*,\s*const\s+u32\s+numWG\s*\)\s*\{",
)

# The bodies the Allocation model assumes, condensed, with the pointer type as {t}.
_POINTER_BODIES: Final = {
    "LDSptr": "return lds+((u32)get_local_id(0)/WG)*LDS_SHUFL_BYTES(numWG)/sizeof({t});",
    "LDSsharing_ptr": (
        "if(!SHARING_LDS(numWG))return LDSptr(lds,numWG);"
        "return lds+((u32)get_local_id(0)/WG/SBMUL(numWG))*SBMUL(numWG)*LDS_SHUFL_BYTES(numWG)/sizeof({t});"
    ),
}


def check_lds_pointers(shufl: str, widths: Mapping[str, int | None]) -> None:
    """Require shufl.cl's typed LDSptr and LDSsharing_ptr to compute the block starts Allocation models."""
    found: set[tuple[str, str]] = set()

    for m in _POINTER_FUNCTION.finditer(shufl):
        pointer_type, function = m.groups()
        end = skip_balanced(shufl, m.end() - 1) - 1

        if pointer_type not in widths or _condensed(shufl[m.end() : end]) != _POINTER_BODIES[function].format(
            t=pointer_type,
        ):
            where = f"shufl.cl:{line_of(shufl, m.start())}"
            msg = f"{where}: {function} for {pointer_type} is not the pointer this checker models"
            raise SourceError(msg)

        found.add((function, pointer_type))

    if len({function for function, _ in found}) != len(_POINTER_BODIES):
        msg = f"shufl.cl: expected typed definitions of {sorted(_POINTER_BODIES)}"
        raise SourceError(msg)


#
# Kernel
#

# The kernel is read with a small grammar rather than searched: every statement of a shufl function must be one this
# checker models, so a condition, a statement or an arm it does not understand stops it instead of being skipped.
#
#   function := preamble local* arm ("else" arm)*   preamble: "u32 mask = f - 1;" and assert(...)s
#   arm      := "if (SBMUL(numWG) * SHUFL_BYTES == n | >= n) {" base item* ["return;"] "}"
#   base     := "local T* lds = [(local T*)]LDSsharing_ptr([(local P *)]lds2, numWG);"
#   item     := #if/#elif/#else/#endif line | barrier | local | "local T *name;"
#               | "local char* root = (local char*)lds2;"
#               | "intN name[RADIX], ...;" | "if (SWIZ_RECOMPUTE) { OPAQUE(name); ... }" | special
#               | ["if (WG == n)" | "else if (WG == n)" | "else"] loop | statement+
#   local    := "u32 name = expr, ...;" | "u32 name[n];" | "name[k] = expr;" | "if (condition) name = expr, ...;"
#   barrier  := "LDStx_start(lds2, numWG);" | "LDSbar(numWG);" | "LDStx_end(lds2, numWG);" | "bar(WG);"
#               ("LDStx_start_with_fence(lds2, numWG);" is a start: the fence only strengthens the barrier.)
#   special  := "if (condition) {" item* "return; }"   (no directives or nested specials inside)
#   loop     := "for (u32 i = a; i < b; ++i | i += s) {" local* statement+ "}"
#
# A statement stores a component of u to LDS, loads one from LDS into u or into a gathered vector, butterflies two LDS
# values into u, or butterflies two gathered vectors into u; see the forms below. LDS is indexed through the arm's
# pointer (lds[k]), through a vector cast of it (((local int4*)lds)[k], whose stores take an int4_of(...)), or as a
# byte offset from the start of all LDS (*(local T*)(root + b)), where (u32)((local char*)lds - root) is this
# workgroup's region. A local is substituted into what follows it, so naming a subexpression reads the same as writing
# it out. Only special cases may be inside an #if within an arm. bar(WG) is read so that it can be reported: while a
# workgroup holds a shared block, the others wait on its semaphore rather than at a barrier.

LoopKind = Literal["write", "read", "compute"]
OpKind = Literal["store", "load", "butterfly", "combine"]

# The name the replay binds to the byte offset of this workgroup's LDS block from the start of all LDS.
REGION: Final = "region"


class WgGuard(NamedTuple):
    """One arm of an "if (WG == n) / else if (WG == m) / ... / else" chain in front of a loop."""

    wg: int | None  #: the workgroup size this arm tests, or None for the trailing "else"
    earlier: frozenset[int]  #: the sizes the arms ahead of it test

    def admits(self, wg: int) -> bool:
        return wg == self.wg if self.wg is not None else wg not in self.earlier

    def then(self, wg: int | None) -> WgGuard:
        """Return the guard of the arm that follows this one, testing `wg` (None for the trailing "else")."""
        if self.wg is None:
            msg = "'else' closes the chain it would continue"
            raise SourceError(msg)
        return WgGuard(wg, self.earlier | {self.wg})


@dataclass(frozen=True, eq=False)
class SlotRef:
    """An LDS location, in slots of the arm's element type counted from the arm's lds pointer.

    `expr` is the slot itself, or a vector index of `scale` slots plus `offset`, or for a byte address from the start
    of LDS (`byte_size` set) the byte offset, from which the workgroup's region is subtracted.
    """

    expr: CExpr
    scale: int = 1
    offset: int = 0
    byte_size: int | None = None

    def evaluate(self, env: Env) -> tuple[int, bool]:
        """Return the slot and whether the address lands between two slots."""
        value = self.expr.evaluate(env)

        if self.byte_size is None:
            return value * self.scale + self.offset, False

        relative = value - env[REGION]
        return relative // self.byte_size, relative % self.byte_size != 0


class Place(NamedTuple):
    """A component of an element of u, or of a gathered vector: `array`[`element`].`component`."""

    array: str
    element: CExpr
    component: str


class Op(NamedTuple):
    """One statement of a loop.

    A store writes `place` to its slot; a load reads its slot into `place`; a butterfly reads two slots and adds them in
    the lower half of the lanes and subtracts them in the upper half, into `place`; a combine does the same with the
    vectors gathered into the arrays `sources`, without touching LDS.
    """

    kind: OpKind
    slots: tuple[SlotRef, ...]
    place: Place
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Loop:
    """A loop over i, or a run of statements outside one, that runs the same statements at every i of its range.

    A write loop only stores and a read loop only loads (or butterflies what it loads); a compute loop does not touch
    LDS. `barred` says whether a barrier comes between this loop and the loop before it.
    """

    kind: LoopKind
    start: CExpr
    stop: CExpr
    step: CExpr
    ops: tuple[Op, ...]
    guard: WgGuard | None
    barred: bool
    line: int

    def indices(self, env: Env) -> range:
        step = self.step.evaluate(env)
        if step <= 0:
            msg = f"shufl.cl:{self.line}: the loop does not advance (step {step})"
            raise SourceError(msg)
        return range(self.start.evaluate(env), self.stop.evaluate(env), step)

    @property
    def regional(self) -> bool:
        return any(slot.byte_size is not None for op in self.ops for slot in op.slots)


@dataclass(frozen=True, eq=False)
class Phase:
    """A run of write loops and the read loops after them: one pass of values through LDS."""

    writes: tuple[Loop, ...]
    reads: tuple[Loop, ...]


@dataclass(frozen=True, eq=False)
class SpecialBlock:
    """An "if (... f == n ...) { ... return; }" special case ahead of an arm's plain implementation."""

    condition: CExpr
    guards: tuple[Condition, ...]
    loops: tuple[Loop, ...]
    line: int
    transaction: tuple[str, ...]

    def enabled(self, mode: Mode) -> bool | None:
        return decide(self.guards, {"LDSPAD": mode.ldspad, "LDSSWIZ": mode.ldsswiz})

    def selects(self, shape: Shape, call: CallSite) -> bool:
        return bool(self.condition.evaluate({"f": call.f, "r": call.r, "RADIX": shape.radix, "WG": shape.wg}))

    @property
    def regional(self) -> bool:
        return any(loop.regional for loop in self.loops)


class ArmGuard(NamedTuple):
    """The "SBMUL(numWG) * SHUFL_BYTES == n" or ">= n" test of an arm."""

    op: str
    bytes: int

    def admits(self, arm_bytes: int) -> bool:
        return arm_bytes == self.bytes if self.op == "==" else arm_bytes >= self.bytes


@dataclass(frozen=True, eq=False)
class Arm:
    """One "if (SBMUL(numWG) * SHUFL_BYTES ...)" branch of a shufl function.

    `guards` are the "SBMUL(numWG) * SHUFL_BYTES" tests of the arms before it and, last, its own. `pointer_type` is the
    type of the pointer passed to LDSsharing_ptr, whose steps the block offset is counted in. `transaction` lists what
    is wrong with the plain implementation as a shared-LDS transaction.
    """

    function: str
    value_type: str
    element_type: str
    pointer_type: str
    element_bytes: int
    pointer_bytes: int
    guards: tuple[ArmGuard, ...]
    line: int
    blocks: tuple[SpecialBlock, ...]
    plain: tuple[Loop, ...]
    transaction: tuple[str, ...]

    def serves(self, arm_bytes: int) -> bool:
        """Whether this arm runs when SBMUL(numWG) * SHUFL_BYTES is `arm_bytes`."""
        *earlier, own = self.guards
        return own.admits(arm_bytes) and not any(guard.admits(arm_bytes) for guard in earlier)

    @property
    def regional(self) -> bool:
        return any(loop.regional for loop in self.plain)

    def __str__(self) -> str:
        return f"{self.function} {self.element_type} arm (shufl.cl:{self.line})"


_OVERLOAD: Final = re.compile(r"\bvoid\s+OVERLOAD\s+(\w+)\s*\(")
_SIGNATURE: Final = re.compile(
    r"\s*local\s+(\w+)\s*\*\s*lds2\s*,\s*\1\s*\*\s*u\s*,\s*u32\s+f\s*,\s*(u32\s+r\s*,\s*)?u32\s+numWG\s*,\s*u32\s+lowMe\s*",
)
_SHORTCUT_BODY: Final = re.compile(
    r"\s*shufl\s*\(\s*lds2\s*,\s*u\s*,\s*f\s*,\s*RADIX\s*,\s*numWG\s*,\s*lowMe\s*\)\s*;\s*",
)
_MASK: Final = re.compile(r"\s*u32\s+mask\s*=\s*f\s*-\s*1\s*;")
_ASSERT: Final = re.compile(r"\s*assert\s*\([^;]*\)\s*;")
_ARM: Final = re.compile(r"if\s*\(\s*SBMUL\s*\(\s*numWG\s*\)\s*\*\s*SHUFL_BYTES\s*(==|>=)\s*(\d+)\s*\)\s*\{")
_ELSE_ARM: Final = re.compile(r"\s*else\b")
_LDS_BASE: Final = re.compile(
    r"local\s+(\w+)\s*\*\s*lds\s*=\s*(?:\(\s*local\s+\1\s*\*\s*\)\s*)?"
    r"LDSsharing_ptr\s*\(\s*(?:\(\s*local\s+(\w+)\s*\*\s*\)\s*)?lds2\s*,\s*numWG\s*\)\s*;",
)
_DIRECTIVE_LINE: Final = re.compile(r"#[ \t]*(\w*)[^\n]*")

# The barriers, by what they are called in the transaction model.
_BARRIERS: Final = {
    "start": re.compile(r"LDStx_start(?:_with_fence)?\s*\(\s*lds2\s*,\s*numWG\s*\)\s*;"),
    "bar": re.compile(r"LDSbar\s*\(\s*numWG\s*\)\s*;"),
    "end": re.compile(r"LDStx_end\s*\(\s*lds2\s*,\s*numWG\s*\)\s*;"),
    "bar(WG)": re.compile(r"bar\s*\(\s*WG\s*\)\s*;"),
}
_DECLARATION: Final = re.compile(r"local\s+\w+\s*\*\s*\w+\s*;")
_ROOT: Final = re.compile(r"local\s+char\s*\*\s*(\w+)\s*=\s*\(\s*local\s+char\s*\*\s*\)\s*lds2\s*;")
_GATHER_DECLARATION: Final = re.compile(
    rf"({IDENTIFIER})\s+(\w+\s*\[\s*RADIX\s*\](?:\s*,\s*\w+\s*\[\s*RADIX\s*\])*)\s*;",
)
_OPAQUE: Final = re.compile(r"if\s*\(\s*SWIZ_RECOMPUTE\s*\)\s*\{((?:\s*OPAQUE\s*\(\s*\w+\s*\)\s*;)+)\s*\}")
_U32_DECLARATION: Final = re.compile(r"u32\s+([^;]*);")
_U32_ARRAY: Final = re.compile(rf"\s*({IDENTIFIER})\s*\[\s*(\d+)\s*\]\s*")
_U32_INIT: Final = re.compile(rf"\s*({IDENTIFIER})\s*=\s*(.+?)\s*", re.DOTALL)
_ELEMENT_ASSIGNMENT: Final = re.compile(rf"({IDENTIFIER})\s*\[\s*(\d+)\s*\]\s*=\s*([^;]+);")
_CONDITIONAL_ASSIGNMENT: Final = re.compile(r"if\s*\(")
_IF: Final = re.compile(r"if\s*\(")
_ELSE_LOOP: Final = re.compile(r"else\s+(?=for\b)")
_ELSE_IF: Final = re.compile(r"else\s+(?=if\b)")
_WG_CONDITION: Final = re.compile(r"\s*WG\s*==\s*(\d+)\s*")
_RETURN_AT_END: Final = re.compile(r"\breturn\s*;\s*\}$")
_FOR: Final = re.compile(
    r"for\s*\(\s*u32\s+i\s*=\s*([^;]+);\s*i\s*<\s*([^;]+);\s*(?:\+\+\s*i|i\s*\+=\s*([^)]+))\s*\)\s*\{",
)
_REGION: Final = re.compile(r"\(\s*u32\s*\)\s*\(\s*\(\s*local\s+char\s*\*\s*\)\s*lds\s*-\s*(\w+)\s*\)")

# LDS references, in the source and before anything is condensed: the arm's pointer, a vector cast of it, and a byte
# address from the start of LDS.
_LDS_INDEX: Final = re.compile(r"\blds\s*\[")
_VECTOR_INDEX: Final = re.compile(r"\(\s*\(\s*local\s+(\w+)\s*\*\s*\)\s*lds\s*\)\s*\[")
_BYTE_ADDRESS: Final = re.compile(r"\*\s*\(\s*local\s+(\w+)\s*\*\s*\)\s*\(")

# Statements, with whitespace reduced to what separates two words and each LDS reference replaced by @n. An index of
# u is anything up to its closing bracket and is parsed as an expression of i.
_U: Final = r"u\[([^\]]+)\]"
_SOURCES: Final = (
    (re.compile(rf"as_int4\({_U}\)\.([xyzw])"), "int4."),
    (re.compile(rf"as_int\({_U}\.([xy])\)"), ""),
    (re.compile(rf"{_U}(?:\.([xyzw]))?"), ""),
)
_STORE: Final = re.compile(r"@(\d+)=([^;]+);")
_VECTOR_STORE: Final = re.compile(r"@(\d+)=(\w+)_of\(([^;]*)\);")
_LOAD: Final = re.compile(rf"{_U}(?:\.([xyzw]))?=@(\d+);")
_INT4_LOAD: Final = re.compile(rf"int4 tmp=as_int4\({_U}\);tmp\.([xyzw])=@(\d+);u\[\1\]=as_(\w+)\(tmp\);")
_GATHER: Final = re.compile(r"(\w+)\[([^\]]+)\]\.([xyzw])=@(\d+);")
_BUTTERFLY: Final = re.compile(
    rf"(\w+) val1=@(\d+);\1 val2=@(\d+);if\(lowMe<WG/2\){_U}(?:\.([xyzw]))?=addq\(val1,val2\);"
    r"else u\[\4\](?:\.\5)?=subq\(val1,val2\);",
)
_COMBINE: Final = re.compile(
    rf"(\w+) val1=as_\1\((\w+)\[([^\]]+)\]\);\1 val2=as_\1\((\w+)\[\3\]\);if\(lowMe<WG/2\){_U}=addq\(val1,val2\);"
    r"else u\[\5\]=subq\(val1,val2\);",
)
_LEADING_LOCAL: Final = re.compile(r"u32 (\w+)=([^;]+);")


def _condensed(text: str) -> str:
    return re.sub(r" (?=\W)|(?<=\W) ", "", " ".join(text.split()))


def _family(component: str) -> str:
    return "all of u[i]" if not component else "as_int4(u[i])" if component.startswith("int4.") else "u[i].x/.y"


def _split_top(text: str, separator: str = ",") -> list[str]:
    """Split `text` at each `separator` outside parentheses and brackets."""
    pieces: list[str] = []
    depth, start = 0, 0

    for position, char in enumerate(text):
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(char, 0)
        if char == separator and depth == 0:
            pieces.append(text[start:position])
            start = position + 1

    pieces.append(text[start:])
    return pieces


@dataclass(frozen=True)
class Locals:
    """The u32 locals in scope, each as the C text it stands for, with the locals before it already substituted."""

    values: Mapping[str, str] = field(default_factory=dict[str, str])
    arrays: frozenset[str] = frozenset()

    def substitute(self, text: str) -> str:
        """Return `text` with every local in scope written out."""
        for name in self.arrays:
            text = re.sub(rf"\b{name}\s*\[\s*(\d+)\s*\]", rf"{name}__\1", text)

        if not self.values:
            return text

        pattern = re.compile(
            r"(?<![.\w])(" + "|".join(map(re.escape, sorted(self.values, key=len, reverse=True))) + r")\b",
        )
        return pattern.sub(lambda m: f"({self.values[m.group(1)]})", text)

    def bind(self, name: str, value: str, where: str) -> Locals:
        """Return these locals with `name` standing for `value`, which may use the locals already in scope."""
        if not re.fullmatch(IDENTIFIER, name):
            msg = f"{where}: cannot read the local {name!r}"
            raise SourceError(msg)

        written = " ".join(self.substitute(_REGION.sub(REGION, value)).split())
        if written == REGION and name == REGION:
            return self

        parse_c(written)  # it must be an integer expression the replay can evaluate
        return Locals({**self.values, name: written}, self.arrays)

    def declare_array(self, name: str) -> Locals:
        return Locals(self.values, self.arrays | {name})


def _parse_function_locals(text: str, start: int, end: int, where: str) -> tuple[Locals, int]:
    """Read the u32 locals a function sets up before its first arm."""
    scope, position = Locals(), start

    while True:
        position = skip_space(text, position, end)

        if m := _U32_DECLARATION.match(text, position, end):
            scope, position = _declare(scope, m.group(1), f"shufl.cl:{line_of(text, position)}"), m.end()
            continue

        if (m := _CONDITIONAL_ASSIGNMENT.match(text, position, end)) and not _ARM.match(text, position, end):
            close = skip_balanced(text, m.end() - 1)
            statement_end = text.find(";", close)
            scope = _assign_if(scope, text[m.end() : close - 1], text[close:statement_end], where)
            position = statement_end + 1
            continue

        return scope, position


def _declare(scope: Locals, declarators: str, where: str) -> Locals:
    """Bind each "name = value" (or declare each "name[n]") of a "u32 ...;" declaration."""
    for declarator in _split_top(declarators):
        if array := _U32_ARRAY.fullmatch(declarator):
            scope = scope.declare_array(array.group(1))
        elif init := _U32_INIT.fullmatch(declarator):
            scope = scope.bind(init.group(1), init.group(2), where)
        else:
            msg = f"{where}: cannot read the declaration 'u32 {' '.join(declarators.split())};'"
            raise SourceError(msg)

    return scope


def _assign_if(scope: Locals, condition: str, assignments: str, where: str) -> Locals:
    """Bind "if (condition) a = x, b = y;" as each name becoming "condition ? x : what it was"."""
    updates: list[tuple[str, str]] = []

    for assignment in _split_top(assignments):
        if (init := _U32_INIT.fullmatch(assignment)) is None or init.group(1) not in scope.values:
            msg = f"{where}: cannot read 'if ({' '.join(condition.split())}) {' '.join(assignments.split())};'"
            raise SourceError(msg)
        updates.append((init.group(1), init.group(2)))

    # Every right-hand side reads the values from before the statement.
    condition, before = scope.substitute(condition), scope
    for name, value in updates:
        scope = scope.bind(name, f"({condition}) ? ({before.substitute(value)}) : ({before.values[name]})", where)

    return scope


def parse_shufl(text: str, widths: Mapping[str, int | None]) -> list[Arm]:
    """Parse every SHUFL_BYTES arm of every shufl and shufl_and_fft2 overload, in source order."""
    arms: list[Arm] = []
    defined: set[tuple[str, str]] = set()
    shortcuts: set[str] = set()

    for overload in _OVERLOAD.finditer(text):
        function = overload.group(1)
        if not function.startswith("shufl"):
            continue

        where = f"shufl.cl:{line_of(text, overload.start())}"
        params_end = skip_balanced(text, overload.end() - 1)
        signature = _SIGNATURE.fullmatch(text, overload.end(), params_end - 1)
        brace = skip_space(text, params_end, len(text))

        if function not in FUNCTIONS or signature is None or not text.startswith("{", brace):
            msg = f"{where}: cannot read the definition of {function}"
            raise SourceError(msg)

        body_end = skip_balanced(text, brace) - 1
        value_type, takes_r = signature.group(1), signature.group(2) is not None

        if function == "shufl" and not takes_r:
            if not _SHORTCUT_BODY.fullmatch(text, brace + 1, body_end):
                msg = f"{where}: expected the shortcut body 'shufl(lds2, u, f, RADIX, numWG, lowMe);'"
                raise SourceError(msg)

            shortcuts.add(value_type)
            continue

        if (function, value_type) in defined:
            msg = f"{where}: a second {function} for {value_type}"
            raise SourceError(msg)

        defined.add((function, value_type))
        arms.extend(_parse_function(text, function, value_type, brace + 1, body_end, widths))

    if not arms:
        msg = "found no SHUFL_BYTES arms in shufl.cl"
        raise SourceError(msg)

    if missing := sorted(t for t in shortcuts if ("shufl", t) not in defined):
        msg = f"shufl.cl: the shufl shortcut for {missing} has no shufl with r to forward to"
        raise SourceError(msg)

    return arms


def _parse_function(
    text: str,
    function: str,
    value_type: str,
    start: int,
    end: int,
    widths: Mapping[str, int | None],
) -> list[Arm]:
    arms: list[Arm] = []
    guards: list[ArmGuard] = []
    has_mask, position = False, start

    while m := _MASK.match(text, position, end) or _ASSERT.match(text, position, end):
        has_mask = has_mask or m.re is _MASK
        position = m.end()

    if not has_mask:
        msg = f"shufl.cl:{line_of(text, start)}: expected 'u32 mask = f - 1;' at the start of {function}"
        raise SourceError(msg)

    scope, position = _parse_function_locals(text, position, end, f"shufl.cl:{line_of(text, position)}")

    while True:
        position = skip_space(text, position, end)
        where = f"shufl.cl:{line_of(text, position)}"
        arm = _ARM.match(text, position, end)

        if arm is None:
            msg = f"{where}: expected 'if (SBMUL(numWG) * SHUFL_BYTES == n) {{' or '... >= n) {{' in {function}"
            raise SourceError(msg)

        guards.append(ArmGuard(arm.group(1), int(arm.group(2))))
        position = skip_balanced(text, arm.end() - 1)
        arms.append(_parse_arm(text, function, value_type, arm, position, tuple(guards), widths, scope))

        if not any(arms[-1].serves(value) for value in ARM_BYTES_VALUES):
            guard = f"SBMUL(numWG) * SHUFL_BYTES {guards[-1].op} {guards[-1].bytes}"
            msg = f"{where}: '{guard}' serves none of {ARM_BYTES_VALUES} not already served"
            raise SourceError(msg)

        if (else_arm := _ELSE_ARM.match(text, position, end)) is None:
            break

        position = else_arm.end()

    if text[position:end].strip():
        msg = f"shufl.cl:{line_of(text, skip_space(text, position, end))}: unexpected code after the last arm"
        raise SourceError(msg)

    if missing := [value for value in ARM_BYTES_VALUES if not any(arm.serves(value) for arm in arms)]:
        msg = (
            f"shufl.cl:{line_of(text, start)}: {function} for {value_type} has no arm for"
            f" SBMUL(numWG) * SHUFL_BYTES={missing}"
        )
        raise SourceError(msg)

    return arms


@dataclass(frozen=True)
class _ArmContext:
    text: str
    value_type: str
    element_type: str
    frames: tuple[tuple[Condition, ...], ...]
    widths: Mapping[str, int | None]
    element_bytes: int


@dataclass
class _Scope:
    """What a body has declared so far: its locals, the start of LDS if named, and the arrays it gathers into."""

    locals: Locals
    root: str | None = None
    gathers: dict[str, str] = field(default_factory=dict[str, str])

    def nested(self) -> _Scope:
        return _Scope(self.locals, self.root, dict(self.gathers))


def _parse_arm(
    text: str,
    function: str,
    value_type: str,
    arm: re.Match[str],
    end: int,
    guards: tuple[ArmGuard, ...],
    widths: Mapping[str, int | None],
    scope: Locals,
) -> Arm:
    line = line_of(text, arm.start())
    base = _LDS_BASE.match(text, skip_space(text, arm.end(), end))

    pointer_type = value_type if base is None or base.group(2) is None else base.group(2)

    if base is None or base.group(1) not in widths or pointer_type not in widths:
        msg = (
            f"shufl.cl:{line}: expected 'local T* lds = [(local T*)]LDSsharing_ptr([(local P *)]lds2, numWG);' with T"
            f" and P among {sorted(widths)}"
        )
        raise SourceError(msg)

    where = f"shufl.cl:{line}"
    element_bytes = slot_bytes(widths, base.group(1), where)
    pointer_bytes = slot_bytes(widths, pointer_type, where)

    context = _ArmContext(
        text,
        value_type,
        base.group(1),
        tuple(preprocessor_frames(text, arm.start())),
        widths,
        element_bytes,
    )
    body_end = end - 1

    if ending := _RETURN_AT_END.search(text, base.end(), end):
        body_end = ending.start()

    plain, blocks, events = _parse_items(context, _Scope(scope), base.end(), body_end, in_special=False)

    loops = (*plain, *(loop for block in blocks for loop in block.loops))
    # A combine rebuilds whole values and a gather fills its own array, so neither says how the arm splits a value.
    components = {
        op.place.component for loop in loops for op in loop.ops if op.kind != "combine" and op.place.array == "u"
    }

    if len(families := sorted({_family(component) for component in components})) > 1:
        msg = f"shufl.cl:{line}: the arm mixes {' and '.join(families)}"
        raise SourceError(msg)

    return Arm(
        function,
        value_type,
        base.group(1),
        pointer_type,
        element_bytes,
        pointer_bytes,
        guards,
        line,
        tuple(blocks),
        tuple(plain),
        transaction_problems(events),
    )


def _chain_from(previous: Loop | None, where: str) -> tuple[WgGuard, bool]:
    """Return the guard an "else" or "else if" continues and whether a barrier came before it.

    Raise a SourceError if there is no chain to continue.
    """
    if previous is None or previous.guard is None or previous.guard.wg is None:
        msg = f"{where}: 'else' does not follow an 'if (WG == n)' loop"
        raise SourceError(msg)
    return previous.guard, previous.barred


def _parse_items(
    context: _ArmContext,
    scope: _Scope,
    start: int,
    end: int,
    *,
    in_special: bool,
) -> tuple[list[Loop], list[SpecialBlock], list[Event]]:
    """Parse the items of an arm or special case body, from `start` to `end`.

    The events are this body's own barriers and the loops that touch LDS, in order, without those of the special
    cases inside it.
    """
    text = context.text
    loops: list[Loop] = []
    blocks: list[SpecialBlock] = []
    events: list[Event] = []
    position, depth, barred = start, 0, False
    previous: Loop | None = None

    def top_level(where: str) -> None:
        if depth:
            msg = f"{where}: only special cases may be inside an #if within an arm"
            raise SourceError(msg)

    def add(loop: Loop) -> None:
        nonlocal barred
        loops.append(loop)
        if loop.kind != "compute":
            events.append(Event("loop", loop.line))
            barred = False

    while (position := skip_space(text, position, end)) < end:
        where = f"shufl.cl:{line_of(text, position)}"
        follows_loop, previous = previous, None

        if directive := _DIRECTIVE_LINE.match(text, position, end):
            kind = directive.group(1)

            if in_special or kind not in ("if", "ifdef", "ifndef", "elif", "else", "endif"):
                msg = f"{where}: unexpected directive {directive.group().strip()!r}"
                raise SourceError(msg)

            if kind in ("elif", "else", "endif") and not depth:
                msg = f"{where}: #{kind} closes an #if opened outside the arm"
                raise SourceError(msg)

            depth += {"if": 1, "ifdef": 1, "ifndef": 1, "endif": -1}.get(kind, 0)
            position = directive.end()
            continue

        candidates = ((kind, pattern.match(text, position, end)) for kind, pattern in _BARRIERS.items())

        if barrier := next(((kind, m) for kind, m in candidates if m), None):
            top_level(where)
            events.append(Event(barrier[0], line_of(text, position)))
            barred, position = True, barrier[1].end()
            continue

        if m := _ROOT.match(text, position, end):
            top_level(where)
            scope.root, position = m.group(1), m.end()
            continue

        if m := _DECLARATION.match(text, position, end):
            top_level(where)
            position = m.end()
            continue

        if (m := _GATHER_DECLARATION.match(text, position, end)) and m.group(1) in context.widths:
            top_level(where)
            for name in _split_top(m.group(2)):
                scope.gathers[name.split("[")[0].strip()] = m.group(1)
            position = m.end()
            continue

        if m := _U32_DECLARATION.match(text, position, end):
            top_level(where)
            scope.locals, position = _declare(scope.locals, m.group(1), where), m.end()
            continue

        if (m := _ELEMENT_ASSIGNMENT.match(text, position, end)) and m.group(1) in scope.locals.arrays:
            top_level(where)
            scope.locals = scope.locals.bind(f"{m.group(1)}__{m.group(2)}", m.group(3), where)
            position = m.end()
            continue

        if m := _OPAQUE.match(text, position, end):
            top_level(where)
            for name in re.findall(r"OPAQUE\s*\(\s*(\w+)\s*\)", m.group(1)):
                if name not in scope.locals.values:
                    msg = f"{where}: OPAQUE({name}) names no local"
                    raise SourceError(msg)
            position = m.end()
            continue

        guard, chained = None, None

        # "else if" continues the chain the loop before it started, so its guard must also exclude every size that
        # chain has already tested; a bare "else" closes it and admits the rest.
        if m := _ELSE_IF.match(text, position, end):
            (chained, barred), position = _chain_from(follows_loop, where), m.end()

        if m := _IF.match(text, position, end):
            close = skip_balanced(text, m.end() - 1)
            condition = text[m.end() : close - 1]
            after = skip_space(text, close, end)

            if text.startswith("{", after) and not in_special:
                if chained is not None:
                    msg = f"{where}: 'else if' does not introduce a loop"
                    raise SourceError(msg)
                block, position = _parse_special(context, scope.nested(), condition, position, after)
                blocks.append(block)
                continue

            wg = _WG_CONDITION.fullmatch(condition)
            if wg is None or not _FOR.match(text, after, end):
                msg = f"{where}: unexpected 'if ({' '.join(condition.split())})'"
                raise SourceError(msg)

            size = int(wg.group(1))
            guard = chained.then(size) if chained is not None else WgGuard(size, frozenset())
            position = after
        elif chained is not None:
            msg = f"{where}: 'else if' is not followed by 'if (WG == n)'"
            raise SourceError(msg)
        elif m := _ELSE_LOOP.match(text, position, end):
            (chain, barred), position = _chain_from(follows_loop, where), m.end()
            guard = chain.then(None)

        if header := _FOR.match(text, position, end):
            top_level(where)
            position = skip_balanced(text, header.end() - 1)
            bounds = (header.group(1), header.group(2), header.group(3) or "1")
            previous = _parse_loop(
                context,
                scope,
                header.end(),
                position - 1,
                bounds,
                guard,
                barred=barred,
                where=where,
            )
            add(previous)
            continue

        if guard is not None:
            msg = f"{where}: 'if (WG == n)' and 'else' only guard loops"
            raise SourceError(msg)

        # A statement outside any loop runs once, as a loop of one index that it does not use.
        statement_end = text.find(";", position, end)
        if statement_end < 0 or not any(
            pattern.search(text, position, statement_end) for pattern in (_LDS_INDEX, _VECTOR_INDEX, _BYTE_ADDRESS)
        ):
            snippet = " ".join(text[position:end].split())[:60]
            loop = "for (u32 i = 0; i < RADIX; ++i) {"
            msg = f"{where}: expected a barrier, a special case or '{loop}', found {snippet!r}"
            raise SourceError(msg)

        top_level(where)
        statement = _parse_loop(
            context,
            scope,
            position,
            statement_end + 1,
            ("0", "1", "1"),
            None,
            barred=barred,
            where=where,
        )
        if any("i" in expr.names for op in statement.ops for expr in (op.place.element, *(s.expr for s in op.slots))):
            msg = f"{where}: a statement outside a loop indexes u by i"
            raise SourceError(msg)
        add(statement)
        position = statement_end + 1

    if depth:
        msg = f"shufl.cl:{line_of(text, end)}: #if without #endif"
        raise SourceError(msg)

    return loops, blocks, events


def _parse_special(
    context: _ArmContext,
    scope: _Scope,
    condition: str,
    start: int,
    brace: int,
) -> tuple[SpecialBlock, int]:
    text = context.text
    end = skip_balanced(text, brace)
    line = line_of(text, start)
    ending = _RETURN_AT_END.search(text, brace + 1, end)

    if ending is None:
        msg = f"shufl.cl:{line}: special case does not end in 'return;'"
        raise SourceError(msg)

    frames = preprocessor_frames(text, start)
    if tuple(frames[: len(context.frames)]) != context.frames:
        msg = f"shufl.cl:{line}: an #elif or #else inside the arm continues an #if from outside it"
        raise SourceError(msg)

    loops, _, events = _parse_items(context, scope, brace + 1, ending.start(), in_special=True)
    guards = tuple(condition for frame in frames[len(context.frames) :] for condition in frame)

    return SpecialBlock(parse_c(condition), guards, tuple(loops), line, transaction_problems(events)), end


class Event(NamedTuple):
    """A barrier ("start", "bar", "end" or "bar(WG)") or a "loop", at a line of shufl.cl."""

    kind: str
    line: int


def transaction_problems(events: Sequence[Event]) -> tuple[str, ...]:
    """Check that a path is one shared-LDS transaction: LDStx_start first, LDStx_end last, and only LDSbar between."""
    if not any(event.kind == "loop" for event in events):
        return ()

    problems: list[str] = []

    if events[0].kind != "start":
        problems.append(f"the path at shufl.cl:{events[0].line} does not begin with LDStx_start")

    if events[-1].kind != "end":
        problems.append(f"the path ending at shufl.cl:{events[-1].line} does not end with LDStx_end")

    for event in events[1:-1]:
        if event.kind in ("start", "end"):
            name = "LDStx_start" if event.kind == "start" else "LDStx_end"
            problems.append(f"{name} at shufl.cl:{event.line} is inside the transaction")

    problems.extend(
        f"bar(WG) at shufl.cl:{event.line} waits for workgroups that are waiting on the shared block; use LDSbar(numWG)"
        for event in events
        if event.kind == "bar(WG)"
    )

    return tuple(problems)


def _lds_references(context: _ArmContext, scope: _Scope, source: str, where: str) -> tuple[str, list[SlotRef]]:
    """Replace each LDS reference in `source` by @n, returning the text and the n-th reference's slot."""
    refs: list[SlotRef] = []
    pieces: list[str] = []
    position = 0
    patterns = (_VECTOR_INDEX, _BYTE_ADDRESS, _LDS_INDEX)

    while found := [m for pattern in patterns if (m := pattern.search(source, position))]:
        m = min(found, key=lambda each: each.start())
        close = skip_balanced(source, m.end() - 1)
        inner = source[m.end() : close - 1]

        if m.re is _LDS_INDEX:
            ref = SlotRef(parse_c(inner))
        elif m.re is _VECTOR_INDEX:
            width = context.widths.get(m.group(1))
            if width is None or width % context.element_bytes:
                msg = f"{where}: ((local {m.group(1)}*)lds) does not hold a whole number of {context.element_type}"
                raise SourceError(msg)
            ref = SlotRef(parse_c(inner), scale=width // context.element_bytes)
        else:
            if scope.root is None or not (base := re.match(rf"\s*{scope.root}\s*\+(.*)$", inner, re.DOTALL)):
                msg = (
                    f"{where}: a byte address that is not 'root + offset' from 'local char* root = (local char*)lds2;'"
                )
                raise SourceError(msg)
            if context.widths.get(m.group(1)) != context.element_bytes:
                msg = f"{where}: *(local {m.group(1)}*) is not a {context.element_type} slot of this arm"
                raise SourceError(msg)
            ref = SlotRef(parse_c(base.group(1)), byte_size=context.element_bytes)

        pieces += [source[position : m.start()], f" @{len(refs)} "]
        refs.append(ref)
        position = close

    pieces.append(source[position:])
    return "".join(pieces), refs


def _source_place(text: str, where: str) -> Place:
    """Return the component of u a store or vector element writes, e.g. "as_int4(u[i]).x" or "u[i + 4].y"."""
    for pattern, prefix in _SOURCES:
        if m := pattern.fullmatch(text):
            return Place("u", parse_c(m.group(1)), prefix + (m.group(2) or ""))

    msg = f"{where}: {text!r} is not a component of u this checker reads"
    raise SourceError(msg)


def _parse_loop(
    context: _ArmContext,
    scope: _Scope,
    start: int,
    end: int,
    bounds: tuple[str, str, str],
    guard: WgGuard | None,
    *,
    barred: bool,
    where: str,
) -> Loop:
    line = line_of(context.text, start)
    body = _condensed(context.text[start:end])

    # Locals declared at the top of the body, in terms of i and of the locals around it.
    local_scope = scope.locals
    while m := _LEADING_LOCAL.match(body):
        local_scope, body = local_scope.bind(m.group(1), m.group(2), where), body[m.end() :]

    body, refs = _lds_references(context, scope, local_scope.substitute(_REGION.sub(REGION, body)), where)
    body = _condensed(body)
    ops: list[Op] = []
    used: list[int] = []

    def slot(n: str) -> SlotRef:
        used.append(int(n))
        return refs[int(n)]

    def element(text: str) -> CExpr:
        return parse_c(text)

    while body:
        if m := _VECTOR_STORE.match(body):
            target = refs[int(m.group(1))]
            parts = _split_top(m.group(3))
            if len(parts) != target.scale:
                msg = f"{where}: {m.group(2)}_of() of {len(parts)} values stored to {target.scale} slots"
                raise SourceError(msg)
            used.append(int(m.group(1)))
            ops.extend(
                Op("store", (SlotRef(target.expr, target.scale, k, target.byte_size),), _source_place(part, where))
                for k, part in enumerate(parts)
            )
        elif m := _INT4_LOAD.match(body):
            if m.group(4) != context.value_type:
                msg = f"{where}: as_{m.group(4)}(tmp) is not the value type {context.value_type}"
                raise SourceError(msg)
            ops.append(Op("load", (slot(m.group(3)),), Place("u", element(m.group(1)), f"int4.{m.group(2)}")))
        elif m := _BUTTERFLY.match(body):
            if m.group(1) != context.element_type:
                msg = f"{where}: a butterfly of {m.group(1)} values in an arm of {context.element_type} slots"
                raise SourceError(msg)
            place = Place("u", element(m.group(4)), m.group(5) or "")
            ops.append(Op("butterfly", (slot(m.group(2)), slot(m.group(3))), place))
        elif m := _COMBINE.match(body):
            names = (m.group(2), m.group(4))
            if m.group(1) != context.value_type or m.group(3) != m.group(5):
                msg = f"{where}: a combine must butterfly {context.value_type} values of one index into u"
                raise SourceError(msg)
            for name in names:
                if scope.gathers.get(name) != "int4" or context.widths.get(context.value_type) != 16:
                    msg = f"{where}: {name} is not an int4[RADIX] holding whole {context.value_type} values"
                    raise SourceError(msg)
            if names[0] == names[1]:
                msg = f"{where}: both values of the butterfly come from {names[0]}"
                raise SourceError(msg)
            ops.append(Op("combine", (), Place("u", element(m.group(3)), ""), names))
        elif (m := _GATHER.match(body)) and m.group(1) in scope.gathers:
            ops.append(Op("load", (slot(m.group(4)),), Place(m.group(1), element(m.group(2)), m.group(3))))
        elif m := _LOAD.match(body):
            ops.append(Op("load", (slot(m.group(3)),), Place("u", element(m.group(1)), m.group(2) or "")))
        elif m := _STORE.match(body):
            ops.append(Op("store", (slot(m.group(1)),), _source_place(m.group(2), where)))
        else:
            msg = f"{where}: loop body is not a form this checker reads: {body!r}"
            raise SourceError(msg)

        body = body[m.end() :]

    if sorted(used) != list(range(len(refs))):
        msg = f"{where}: an LDS reference is not part of a statement this checker reads"
        raise SourceError(msg)

    kinds = {op.kind for op in ops}
    if not ops or ("store" in kinds and len(kinds) > 1) or ("combine" in kinds and len(kinds) > 1):
        msg = f"{where}: a loop must only store to LDS, only load from it, or only combine gathered values"
        raise SourceError(msg)

    kind: LoopKind = "write" if "store" in kinds else "compute" if "combine" in kinds else "read"
    start_expr, stop_expr, step_expr = (parse_c(local_scope.substitute(bound)) for bound in bounds)
    return Loop(kind, start_expr, stop_expr, step_expr, tuple(ops), guard, barred, line)


def group_phases(loops: Sequence[Loop]) -> tuple[Phase, ...]:
    """Group loops into passes: a run of write loops, then the read (and compute) loops that follow them."""
    phases: list[Phase] = []
    writes: list[Loop] = []
    reads: list[Loop] = []

    for loop in loops:
        if loop.kind == "write" and reads:
            phases.append(Phase(tuple(writes), tuple(reads)))
            writes, reads = [], []

        (writes if loop.kind == "write" else reads).append(loop)

    if writes or reads:
        phases.append(Phase(tuple(writes), tuple(reads)))

    return tuple(phases)


def for_wg(loops: Sequence[Loop], wg: int) -> list[Loop]:
    """Return the loops that run at this WG, once "if (WG == n) / else" guards are applied."""
    return [loop for loop in loops if loop.guard is None or loop.guard.admits(wg)]


#
# Replay
#


class ValueId(NamedTuple):
    """One component of u[i] of one lane: a place, or the value it held when shufl was called."""

    i: int
    lane: int
    component: str

    def __str__(self) -> str:
        if self.component.startswith("int4."):
            return f"as_int4(u[{self.i}]).{self.component[5:]} of lane {self.lane}"
        return f"u[{self.i}]{'.' if self.component else ''}{self.component} of lane {self.lane}"


class Butterfly(NamedTuple):
    operation: Literal["add", "sub"]
    left: LaneValue
    right: LaneValue

    def __str__(self) -> str:
        return f"{self.operation}q({_show(self.left)}, {_show(self.right)})"


class Join(NamedTuple):
    """One whole component rebuilt from the two halves that crossed LDS in separate passes."""

    low: LaneValue
    high: LaneValue

    def __str__(self) -> str:
        return f"join({_show(self.low)}, {_show(self.high)})"


class Whole(NamedTuple):
    """One value rebuilt from the four 32-bit pieces gathered into an int4, which are not one element's own pieces."""

    parts: tuple[LaneValue, ...]

    def __str__(self) -> str:
        return f"whole({', '.join(_show(part) for part in self.parts)})"


LaneValue = ValueId | Butterfly | Join | Whole | None

_PIECES: Final = ("int4.x", "int4.y", "int4.z", "int4.w")


def _whole(parts: Sequence[LaneValue]) -> LaneValue:
    """Return the value four gathered pieces make: an element itself when they are its own four pieces, in order."""
    first = parts[0]

    if isinstance(first, ValueId) and all(
        isinstance(part, ValueId) and (part.i, part.lane, part.component) == (first.i, first.lane, piece)
        for part, piece in zip(parts, _PIECES, strict=True)
    ):
        return ValueId(first.i, first.lane, "")

    return Whole(tuple(parts))


def _show(value: LaneValue) -> str:
    return "an unwritten slot" if value is None else str(value)


class Access(NamedTuple):
    slot: int
    phase: int
    i: int
    lane: int
    write: bool

    def __str__(self) -> str:
        action = "write" if self.write else "read"
        return f"pass {self.phase + 1} {action} of u[{self.i}] lane {self.lane} -> slot {self.slot}"


@dataclass(frozen=True)
class Trace:
    """Everything one replay of a path did."""

    values: Mapping[ValueId, LaneValue]
    slots: tuple[int, ...]
    lowest: Access | None
    highest: Access | None
    collisions: tuple[tuple[Access, LaneValue], ...]
    unwritten_reads: tuple[Access, ...]
    unread_phases: tuple[int, ...]
    misaligned: tuple[Access, ...] = ()
    stray: tuple[str, ...] = ()


def replay(phases: Sequence[Phase], shape: Shape, call: CallSite, region: int = 0) -> Trace:
    """Run the passes of a path over every lane, as the kernel would, and return what each place in u holds.

    `region` is the byte offset of the workgroup's LDS block from the start of all LDS, which byte addresses count from.
    """
    env = {
        "RADIX": shape.radix,
        "WG": shape.wg,
        "f": call.f,
        "r": call.r,
        "mask": call.f - 1,
        "i": 0,
        "lowMe": 0,
        REGION: region,
    }
    lanes, half = range(shape.wg), shape.wg // 2
    values: dict[ValueId, LaneValue] = {}
    gathered: dict[tuple[str, int, int, str], LaneValue] = {}
    slots: list[int] = []
    lowest: Access | None = None
    highest: Access | None = None
    collisions: list[tuple[Access, LaneValue]] = []
    unwritten_reads: list[Access] = []
    unread_phases: list[int] = []
    misaligned: list[Access] = []
    stray: list[str] = []

    def track(ref: SlotRef, phase: int, i: int, lane: int, *, write: bool) -> Access:
        nonlocal lowest, highest
        slot, between = ref.evaluate(env)
        access = Access(slot, phase, i, lane, write)
        slots.append(slot)

        if between:
            misaligned.append(access)

        if lowest is None or slot < lowest.slot:
            lowest = access

        if highest is None or slot > highest.slot:
            highest = access

        return access

    def element(place: Place, loop: Loop) -> int:
        index = place.element.evaluate(env)
        if not 0 <= index < shape.radix and len(stray) < 1:
            stray.append(f"the loop at shufl.cl:{loop.line} uses {place.array}[{index}], outside [0, {shape.radix})")
        return index

    for p, phase in enumerate(phases):
        memory: dict[int, LaneValue] = {}

        for loop in phase.writes:
            for i in loop.indices(env):
                env["i"] = i

                for lane in lanes:
                    env["lowMe"] = lane

                    for op in loop.ops:
                        place = ValueId(element(op.place, loop), lane, op.place.component)
                        value = values.get(place, place)
                        access = track(op.slots[0], p, place.i, lane, write=True)
                        owner = memory.setdefault(access.slot, value)
                        if owner != value:
                            collisions.append((access, owner))

        if not any(loop.kind == "read" for loop in phase.reads):
            unread_phases.append(p)

        for loop in phase.reads:
            for i in loop.indices(env):
                env["i"] = i

                for lane in lanes:
                    env["lowMe"] = lane
                    operation: Literal["add", "sub"] = "add" if lane < half else "sub"

                    for op in loop.ops:
                        index = element(op.place, loop)
                        loaded: list[LaneValue] = []

                        for ref in op.slots:
                            access = track(ref, p, index, lane, write=False)

                            if access.slot not in memory:
                                unwritten_reads.append(access)

                            loaded.append(memory.get(access.slot))

                        place = ValueId(index, lane, op.place.component)

                        if op.kind == "combine":
                            left, right = (
                                _whole([gathered.get((name, index, lane, piece[5:])) for piece in _PIECES])
                                for name in op.sources
                            )
                            values[place] = Butterfly(operation, left, right)
                        elif op.kind == "butterfly":
                            values[place] = Butterfly(operation, loaded[0], loaded[1])
                        elif op.place.array != "u":
                            gathered[op.place.array, index, lane, op.place.component] = loaded[0]
                        else:
                            values[place] = loaded[0]

    return Trace(
        values,
        tuple(slots),
        lowest,
        highest,
        tuple(collisions),
        tuple(unwritten_reads),
        tuple(unread_phases),
        tuple(misaligned),
        tuple(stray),
    )


def integrity_problems(trace: Trace) -> list[str]:
    """Check the integrity of a trace and return a list of problems."""
    problems: list[str] = [*trace.stray]

    if trace.misaligned:
        problems.append(
            f"{len(trace.misaligned)} byte address(es) fall between two slots, e.g. {trace.misaligned[0]}",
        )

    if trace.collisions:
        access, owner = trace.collisions[0]
        problems.append(
            f"{len(trace.collisions)} write(s) land on a slot another lane"
            f" already holds, e.g. {access} holding {_show(owner)}",
        )

    if trace.unwritten_reads:
        problems.append(
            f"{len(trace.unwritten_reads)} read(s) of a slot nothing wrote, e.g. {trace.unwritten_reads[0]}",
        )

    problems.extend(f"pass {p + 1} writes values that are never read" for p in trace.unread_phases)

    return problems


def barrier_problems(loops: Sequence[Loop]) -> list[str]:
    """Check that a barrier separates each change between writing LDS and reading it."""
    touching = [loop for loop in loops if loop.kind != "compute"]
    return [
        f"the {'write' if loop.kind == 'write' else 'read'} at shufl.cl:{loop.line} is not separated by a barrier"
        f" from the {'write' if previous.kind == 'write' else 'read'} before it"
        for previous, loop in itertools.pairwise(touching)
        if (previous.kind == "write") != (loop.kind == "write") and not loop.barred
    ]


def bounds_problem(trace: Trace, capacity: int) -> str | None:
    """Check whether a path indexes outside an LDS block of `capacity` slots."""
    if trace.lowest is None or trace.highest is None:
        return None

    if trace.lowest.slot >= 0 and trace.highest.slot < capacity:
        return None

    outside = sum(1 for slot in trace.slots if not 0 <= slot < capacity)
    worst = trace.highest if trace.highest.slot >= capacity else trace.lowest

    return f"{outside} index(es) outside the {capacity}-slot LDS block, e.g. {worst}"


def value_problem(plain: Trace, special: Trace) -> str | None:
    """Check whether a special case leaves any place in u with a different value than the plain case."""
    places = plain.values.keys() | special.values.keys()
    wrong = sorted(place for place in places if plain.values.get(place, place) != special.values.get(place, place))

    if not wrong:
        return None

    place = wrong[0]

    return (
        f"{len(wrong)} of {len(places)} lane values differ from the plain implementation, e.g. "
        f"{place} gets {_show(special.values.get(place, place))}, should get {_show(plain.values.get(place, place))}"
    )


#
# Cross-Arm Equivalence
#

# The as_int4 components the two halves of each whole component are written as.
_HALVES: Final = {"x": ("int4.x", "int4.y"), "y": ("int4.z", "int4.w")}


def _canonical(value: LaneValue, component: str) -> LaneValue:
    """Rewrite a lane value as an expression over whole components of the values shufl was called with."""
    if value is None:
        return None

    if isinstance(value, ValueId):
        return ValueId(value.i, value.lane, component) if not value.component else value

    if isinstance(value, Butterfly):
        return Butterfly(value.operation, _canonical(value.left, component), _canonical(value.right, component))

    if isinstance(value, Whole):
        low, high = (value.parts[_PIECES.index(half)] for half in _HALVES[component])
        return _joined(low, high, component)

    return _joined(value.low, value.high, component)


def _joined(low: LaneValue, high: LaneValue, component: str) -> LaneValue:
    if (
        isinstance(low, ValueId)
        and isinstance(high, ValueId)
        and (low.i, low.lane) == (high.i, high.lane)
        and (low.component, high.component) == _HALVES[component]
    ):
        return ValueId(low.i, low.lane, component)

    return Join(_canonical(low, component), _canonical(high, component))


def canonical_values(trace: Trace, shape: Shape) -> dict[ValueId, LaneValue]:
    """Compute the canonical lane values for a given trace and shape."""
    canonical: dict[ValueId, LaneValue] = {}

    for i, lane, component in itertools.product(range(shape.radix), range(shape.wg), ("x", "y")):
        place = ValueId(i, lane, component)
        whole = trace.values.get(ValueId(i, lane, ""))
        direct = trace.values.get(place)
        low, high = (trace.values.get(ValueId(i, lane, half)) for half in _HALVES[component])

        if whole is not None:
            canonical[place] = _canonical(whole, component)
        elif direct is not None:
            canonical[place] = _canonical(direct, component)
        elif low is not None or high is not None:
            canonical[place] = _joined(low, high, component)
        else:
            canonical[place] = place

    return canonical


def cross_arm_problem(reference: Mapping[ValueId, LaneValue], arm: Mapping[ValueId, LaneValue]) -> str | None:
    """Check whether two arms of one function leave any lane a different value."""
    wrong = sorted(place for place in reference if reference[place] != arm[place])

    if not wrong:
        return None

    place = wrong[0]

    return (
        f"{len(wrong)} of {len(reference)} lane values differ from the arm above it, e.g. "
        f"{place} gets {_show(arm[place])}, should get {_show(reference[place])}"
    )


def check_cross_arms(arms: Sequence[Arm], sites: Mapping[Shape, set[CallSite]], tally: _Tally) -> None:
    """Check every arm of each shufl function against the arm above it, at every shape and call."""
    by_function: dict[tuple[str, str], list[Arm]] = {}

    for arm in arms:
        by_function.setdefault((arm.function, arm.value_type), []).append(arm)

    for group in by_function.values():
        for reference, arm in itertools.pairwise(group):
            for shape, calls in sorted(sites.items()):
                for call in sorted(call for call in calls if call.function == arm.function):
                    tally.cross_checked += 1
                    values = [
                        canonical_values(replay(group_phases(for_wg(each.plain, shape.wg)), shape, call), shape)
                        for each in (reference, arm)
                    ]

                    if problem := cross_arm_problem(values[0], values[1]):
                        where = f"{arm} | {shape} f={call.f} r={call.r} | against the {reference}"
                        tally.failures.append(Failure(where, (problem,)))


@dataclass(frozen=True)
class Failure:
    where: str
    problems: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    shapes: tuple[Shape, ...]
    special_checked: int
    plain_checked: int
    cross_checked: int
    failures: tuple[Failure, ...]
    unreachable: tuple[tuple[Arm, SpecialBlock], ...]

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass
class _Tally:
    """What run_checks has found so far."""

    failures: list[Failure] = field(default_factory=list[Failure])
    special_checked: int = 0
    plain_checked: int = 0
    cross_checked: int = 0
    reached: set[SpecialBlock] = field(default_factory=set[SpecialBlock])


def run_checks(sources: Sources) -> Report:
    """Run all checks on the given sources and return a report."""
    shapes = shape_space(sources.fftconfig_cpp, sources.fftconfig_h)
    sites = call_sites(sources.fftbase, shapes)
    macros = LdsMacros.parse(sources.fftbase)
    widths = element_widths(sources.base, sources.expand)
    check_lds_pointers(sources.shufl, widths)
    arms = parse_shufl(sources.shufl, widths)

    called = {site.function for calls in sites.values() for site in calls}
    defined = {(arm.function, arm.value_type) for arm in arms}
    if missing := sorted({(f, t) for f in called for _, t in defined} - defined):
        msg = f"fftbase.cl calls shufl functions that shufl.cl does not define: {missing}"
        raise SourceError(msg)

    allocations = {
        shape: [
            macros.allocation(mode, shufl_bytes, layout, shape)
            for mode, shufl_bytes, layout in itertools.product(MODES, SHUFL_BYTES_VALUES, LAYOUTS)
        ]
        for shape in shapes
    }

    tally = _Tally()
    chosen = {allocation.arm_bytes for each in allocations.values() for allocation in each}

    for (function, value_type), served in itertools.groupby(arms, lambda arm: (arm.function, arm.value_type)):
        group = list(served)
        for value in sorted(value for value in chosen if not any(arm.serves(value) for arm in group)):
            tally.failures.append(
                Failure(f"{function} for {value_type}", (f"no arm serves SBMUL(numWG) * SHUFL_BYTES = {value}",)),
            )

    for arm in arms:
        for shape in shapes:
            plain_loops = for_wg(arm.plain, shape.wg)
            calls = sorted(site for site in sites[shape] if site.function == arm.function)

            for call in calls:
                _check_call(arm, shape, call, plain_loops=plain_loops, allocations=allocations[shape], tally=tally)

    check_cross_arms(arms, sites, tally)
    unreachable = tuple((arm, block) for arm in arms for block in arm.blocks if block not in tally.reached)

    return Report(
        tuple(shapes),
        tally.special_checked,
        tally.plain_checked,
        tally.cross_checked,
        tuple(tally.failures),
        unreachable,
    )


def _check_call(
    arm: Arm,
    shape: Shape,
    call: CallSite,
    *,
    plain_loops: Sequence[Loop],
    allocations: Sequence[Allocation],
    tally: _Tally,
) -> None:
    """Check one call of one arm at one shape, under every allocation that chooses the arm."""
    context = f"{shape} f={call.f} r={call.r}"
    plain_where = f"{arm} plain implementation | {context}"
    phases = group_phases(plain_loops)
    plains: dict[int, Trace] = {}

    def plain_at(region: int) -> Trace:
        if region not in plains:
            plains[region] = replay(phases, shape, call, region)
        return plains[region]

    plain = plain_at(0)
    plain_broken = integrity_problems(plain) + barrier_problems(plain_loops)

    if plain_broken:
        tally.failures.append(Failure(plain_where, tuple(plain_broken)))

    # Each special case's trace and size-independent problems, shared across settings, by the region it was replayed at.
    specials: dict[tuple[SpecialBlock, int], tuple[Trace, list[str]]] = {}
    # Allocations that differ only in ways this arm cannot see are checked once.
    checked: set[tuple[Mode, bool, int, str | None, tuple[int, ...]]] = set()
    regional = arm.regional or any(block.regional for block in arm.blocks)

    for allocation in allocations:
        if not arm.serves(allocation.arm_bytes):
            continue

        capacity, misplaced = allocation.slots(arm.pointer_bytes, arm.element_bytes)
        regions = allocation.regions(arm.pointer_bytes) if regional else (0,)
        key = (allocation.mode, allocation.sharing, capacity, misplaced, regions)

        if key in checked:
            continue

        checked.add(key)
        setting = str(allocation)

        if misplaced:
            tally.failures.append(Failure(f"{arm} LDS pointer | {context} {setting}", (misplaced,)))

        # The first selecting special case that is compiled runs. One whose #if cannot be decided may or may not be, so
        # it is checked and so is whatever would run without it.
        for block in arm.blocks:
            enabled = block.enabled(allocation.mode) if block.selects(shape, call) else False

            if enabled is False:
                continue

            # A byte address counts from the start of LDS, so the path is replayed where each workgroup's block is.
            for region in regions if block.regional else (0,):
                _check_special(
                    arm,
                    block,
                    shape,
                    call,
                    capacity,
                    region,
                    sharing=allocation.sharing,
                    plain=plain,
                    plain_broken=bool(plain_broken),
                    specials=specials,
                    where=f"{context} {setting}" + (f" region={region}" if block.regional else ""),
                    tally=tally,
                )

            if enabled:
                break
        else:
            tally.plain_checked += 1
            problems = [*arm.transaction] if allocation.sharing else []

            for region in regions if arm.regional else (0,):
                problems.extend(integrity_problems(plain_at(region)) if region else [])
                if problem := bounds_problem(plain_at(region), capacity):
                    problems.append(problem)

            if problems:
                tally.failures.append(Failure(f"{plain_where} {setting}", tuple(problems)))


def _check_special(
    arm: Arm,
    block: SpecialBlock,
    shape: Shape,
    call: CallSite,
    capacity: int,
    region: int,
    *,
    sharing: bool,
    plain: Trace,
    plain_broken: bool,
    specials: dict[tuple[SpecialBlock, int], tuple[Trace, list[str]]],
    where: str,
    tally: _Tally,
) -> None:
    tally.special_checked += 1
    tally.reached.add(block)

    if (block, region) not in specials:
        loops = for_wg(block.loops, shape.wg)
        trace = replay(group_phases(loops), shape, call, region)
        problems = integrity_problems(trace) + barrier_problems(loops)
        if not plain_broken and (problem := value_problem(plain, trace)):
            problems.append(problem)
        specials[block, region] = (trace, problems)

    trace, problems = specials[block, region]

    if sharing:
        problems = [*problems, *block.transaction]

    if problem := bounds_problem(trace, capacity):
        problems = [*problems, problem]

    if problems:
        location = f"{arm} special case shufl.cl:{block.line} '{block.condition.text}' | {where}"
        tally.failures.append(Failure(location, tuple(problems)))


def print_report(report: Report) -> None:
    """Print a report of the shufl index checks."""
    radixes = sorted({shape.radix for shape in report.shapes})
    shapes = ", ".join(
        f"RADIX={radix}: WG " + "/".join(str(shape.wg) for shape in report.shapes if shape.radix == radix)
        for radix in radixes
    )
    print(f"{PROG}: shape space from FFTConfig -- {shapes}")

    for arm, block in report.unreachable:
        print(
            f"{PROG}: unreachable (no shape selects it) -- {arm.function} {arm.element_type}"
            f" shufl.cl:{block.line} '{block.condition.text}'",
        )

    if report.ok:
        print(
            f"{PROG}: OK -- {report.special_checked} (path, shape) combinations replayed, all match the plain shufl and"
            f" stay inside LDS; {report.plain_checked} more run the plain shufl and stay inside LDS;"
            f" {report.cross_checked} arm-against-arm comparisons agree",
        )
        return

    print(f"\n{PROG}: these shufl paths are wrong.")

    for failure in report.failures:
        print(f"  {failure.where}")

        for problem in failure.problems:
            print(f"    {problem}")

    print(f"\n{len(report.failures)} broken combination(s).")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the the shufl index checker."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository to check (default: %(default)s)")

    args = parser.parse_args(argv)

    try:
        report = run_checks(Sources.read(args.root))
    except (SourceError, OSError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 2

    print_report(report)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
