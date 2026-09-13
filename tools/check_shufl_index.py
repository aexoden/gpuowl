#!/usr/bin/env python3
"""Check optimized paths in src/cl/shufl.cl against their plain implementations.

For every parameter combination supported by the FFT shape space, evaluate the write and read indices for every lane and
element, checking that:

  1. All indices are within the LDS block allocated by LDS_BYTES.
  2. No two writes target the same slot, and every read targets a written slot.
  3. Each lane ends with the same value in every component of u as in the plain implementation.
  4. A bar(WG) separates every run of LDS writes from the reads around it.

The plain implementations also undergo checks 1, 2 and 4, including arms without optimized paths. Expressions,
allocation rules, and supported parameter combinations are read from the source; unreadable or unrecognized source is an
error, never a silent skip. No GPU is required.

The reference is the plain implementation in the same SHUFL_BYTES arm; incorrect permutations shared with that
implementation are not detected. Cross-arm equivalence is not checked.

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

# Bytes per element of each type shufl casts its LDS block to. These are the expand.cl macro names, and both
# instantiations of each have the same width. LDS_BYTES is a byte count, so the element type decides how many slots the
# expressions may index.
ELEMENT_BYTES: Final = {"T2_GF61": 16, "T_Z61": 8, "F2_GF31": 8, "F_Z31": 4, "int": 4}

# The values SHUFL_BYTES takes; an arm guarded by "SHUFL_BYTES >= n" serves each one from n up.
SHUFL_BYTES_VALUES: Final = (4, 8, 16)

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


@dataclass(frozen=True)
class Sources:
    """The comment-stripped text of every file the checker reads."""

    shufl: str
    fftbase: str
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


def _translate(c_text: str) -> tuple[str, ast.expr]:
    py = _C_TOKEN.sub(_python_token, c_text).strip()

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
_GENERIC_LOOP: Final = re.compile(r"\bfor\s*\(\s*u32\s+s\s*=\s*1\s*;\s*s\s*<\s*WG\s*;\s*s\s*\*=\s*RADIX\s*\)\s*\{")


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

_LDS_ARM: Final = re.compile(
    r"^[ \t]*#[ \t]*(?:el)?if[ \t]+(.+?)[ \t]*\n\s*#[ \t]*define[ \t]+LDS_BYTES[ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)

_LDS_ELSE: Final = re.compile(
    r"^[ \t]*#[ \t]*else[ \t]*\n\s*#[ \t]*define[ \t]+LDS_BYTES[ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)

_LDS_DEFINE: Final = re.compile(r"#[ \t]*define[ \t]+LDS_BYTES\b")


@dataclass(frozen=True)
class LdsBytes:
    """The LDS_BYTES #if / #elif / #else chain in fftbase.cl."""

    arms: tuple[tuple[CExpr, CExpr], ...]
    fallback: CExpr

    @classmethod
    def parse(cls, fftbase: str) -> LdsBytes:
        arms = _LDS_ARM.findall(fftbase)
        fallback = _LDS_ELSE.findall(fftbase)

        if len(fallback) != 1 or len(arms) + 1 != len(_LDS_DEFINE.findall(fftbase)):
            msg = "cannot read the LDS_BYTES chain in fftbase.cl as '#if c / #define LDS_BYTES e' lines"
            raise SourceError(msg)

        return cls(tuple((parse_c(cond), parse_c(size)) for cond, size in arms), parse_c(fallback[0]))

    def slots(self, mode: Mode, shufl_bytes: int, shape: Shape, element_bytes: int) -> int:
        env = {
            "LDSPAD": mode.ldspad,
            "LDSSWIZ": mode.ldsswiz,
            "SHUFL_BYTES": shufl_bytes,
            "RADIX": shape.radix,
            "WG": shape.wg,
        }

        size = next((size for condition, size in self.arms if condition.evaluate(env)), self.fallback)

        return size.evaluate(env) // element_bytes


#
# Kernel
#

# The kernel is read with a small grammar rather than searched: every statement of a shufl function must be one this
# checker models, so a condition, a statement or an arm it does not understand stops it instead of being skipped.
#
#   function := preamble arm ("else" arm)*          preamble: "u32 mask = f - 1;" and assert(...)s
#   arm      := "if (SHUFL_BYTES == n | >= n) {" base item* "}"
#   base     := "local T* lds = lds2;" "if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T);"
#   item     := #if/#elif/#else/#endif line | "bar(WG);" | "local T *name;" | special | ["if (WG == n)" | "else"] loop
#   special  := "if (condition) {" item* "return; }"   (no directives or nested specials inside)
#   loop     := "for (u32 i = 0; i < RADIX; ++i) {" one of the loop bodies below "}"
#
# Only special cases may be inside an #if within an arm.

LoopKind = Literal["write", "read", "butterfly"]


class WgGuard(NamedTuple):
    """The "if (WG == n)" (equal) or its "else" (not equal) in front of a loop."""

    wg: int
    equal: bool

    def admits(self, wg: int) -> bool:
        return (wg == self.wg) == self.equal


@dataclass(frozen=True, eq=False)
class Loop:
    """A "for (u32 i = 0; i < RADIX; ++i)" loop that touches LDS.

    A write loop stores one component of u[i] at its slot. A read loop loads one component of u[i] from its slot. A
    butterfly loop loads two slots and adds them in the lower half of the lanes and subtracts them in the upper half, as
    shufl_and_fft2 does. The component is "" for all of u[i], "x" for u[i].x, and "int4.x" for as_int4(u[i]).x. `barred`
    says whether a bar(WG) comes between this loop and the loop before it.
    """

    kind: LoopKind
    slots: tuple[CExpr, ...]
    component: str
    guard: WgGuard | None
    barred: bool
    line: int


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

    def enabled(self, mode: Mode) -> bool | None:
        return decide(self.guards, {"LDSPAD": mode.ldspad, "LDSSWIZ": mode.ldsswiz})

    def selects(self, shape: Shape, call: CallSite) -> bool:
        return bool(self.condition.evaluate({"f": call.f, "r": call.r, "RADIX": shape.radix, "WG": shape.wg}))


@dataclass(frozen=True, eq=False)
class Arm:
    """One "if (SHUFL_BYTES ...)" branch of a shufl function."""

    function: str
    value_type: str
    element_type: str
    shufl_bytes: tuple[int, ...]
    line: int
    blocks: tuple[SpecialBlock, ...]
    plain: tuple[Loop, ...]

    @property
    def element_bytes(self) -> int:
        return ELEMENT_BYTES[self.element_type]

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
_ARM: Final = re.compile(r"if\s*\(\s*SHUFL_BYTES\s*(==|>=)\s*(\d+)\s*\)\s*\{")
_ELSE_ARM: Final = re.compile(r"\s*else\b")
_LDS_BASE: Final = re.compile(
    r"local\s+(\w+)\s*\*\s*lds\s*=\s*(?:lds2|\(\s*\(\s*local\s+\1\s*\*\s*\)\s*lds2\s*\)|\(\s*local\s+\1\s*\*\s*\)\s*lds2)\s*;"
    r"\s*if\s*\(\s*numWG\s*>\s*1\s*\)\s*lds\s*\+=\s*\(\s*\(\s*u32\s*\)\s*get_local_id\s*\(\s*0\s*\)\s*/\s*WG\s*\)"
    r"\s*\*\s*LDS_BYTES\s*/\s*sizeof\s*\(\s*\1\s*\)\s*;",
)
_DIRECTIVE_LINE: Final = re.compile(r"#[ \t]*(\w*)[^\n]*")
_BAR: Final = re.compile(r"bar\s*\(\s*WG\s*\)\s*;")
_DECLARATION: Final = re.compile(r"local\s+\w+\s*\*\s*\w+\s*;")
_IF: Final = re.compile(r"if\s*\(")
_ELSE_LOOP: Final = re.compile(r"else\s+(?=for\b)")
_WG_CONDITION: Final = re.compile(r"\s*WG\s*==\s*(\d+)\s*")
_RETURN_AT_END: Final = re.compile(r"\breturn\s*;\s*\}$")
_I_LOOP: Final = re.compile(r"for\s*\(\s*u32\s+i\s*=\s*0\s*;\s*i\s*<\s*RADIX\s*;\s*\+\+i\s*\)\s*\{")
_LDS_INDEX: Final = re.compile(r"\blds\s*\[")

# Loop bodies, with whitespace reduced to what separates two words and each LDS index replaced by @n.
_WRITE_BODY: Final = re.compile(r"@0=(?:u\[i\](?:\.([xyzw]))?|as_int4\(u\[i\]\)\.([xyzw]));")
_READ_BODY: Final = re.compile(r"u\[i\](?:\.([xyzw]))?=@0;")
_INT4_READ_BODY: Final = re.compile(r"int4 tmp=as_int4\(u\[i\]\);tmp\.([xyzw])=@0;u\[i\]=as_(\w+)\(tmp\);")
_BUTTERFLY_BODY: Final = re.compile(
    r"(\w+) val1=@0;\1 val2=@1;if\(lowMe<WG/2\)u\[i\](?:\.([xyzw]))?=addq\(val1,val2\);"
    r"else u\[i\](?:\.\2)?=subq\(val1,val2\);",
)


def _condensed(text: str) -> str:
    return re.sub(r" (?=\W)|(?<=\W) ", "", " ".join(text.split()))


def _family(component: str) -> str:
    return "all of u[i]" if not component else "as_int4(u[i])" if component.startswith("int4.") else "u[i].x/.y"


def parse_shufl(text: str) -> list[Arm]:
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
        arms.extend(_parse_function(text, function, value_type, brace + 1, body_end))

    if not arms:
        msg = "found no SHUFL_BYTES arms in shufl.cl"
        raise SourceError(msg)

    if missing := sorted(t for t in shortcuts if ("shufl", t) not in defined):
        msg = f"shufl.cl: the shufl shortcut for {missing} has no shufl with r to forward to"
        raise SourceError(msg)

    return arms


def _parse_function(text: str, function: str, value_type: str, start: int, end: int) -> list[Arm]:
    arms: list[Arm] = []
    claimed: list[int] = []
    has_mask, position = False, start

    while m := _MASK.match(text, position, end) or _ASSERT.match(text, position, end):
        has_mask = has_mask or m.re is _MASK
        position = m.end()

    if not has_mask:
        msg = f"shufl.cl:{line_of(text, start)}: expected 'u32 mask = f - 1;' at the start of {function}"
        raise SourceError(msg)

    while True:
        position = skip_space(text, position, end)
        where = f"shufl.cl:{line_of(text, position)}"
        arm = _ARM.match(text, position, end)

        if arm is None:
            msg = f"{where}: expected 'if (SHUFL_BYTES == n) {{' or 'if (SHUFL_BYTES >= n) {{' in {function}"
            raise SourceError(msg)

        op, value = arm.group(1), int(arm.group(2))
        serves = tuple(v for v in SHUFL_BYTES_VALUES if (v == value if op == "==" else v >= value) and v not in claimed)

        if not serves:
            msg = f"{where}: 'SHUFL_BYTES {op} {value}' serves none of {SHUFL_BYTES_VALUES} not already served"
            raise SourceError(msg)

        claimed.extend(serves)
        position = skip_balanced(text, arm.end() - 1)
        arms.append(_parse_arm(text, function, value_type, arm, position, serves))

        if (else_arm := _ELSE_ARM.match(text, position, end)) is None:
            break

        position = else_arm.end()

    if text[position:end].strip():
        msg = f"shufl.cl:{line_of(text, skip_space(text, position, end))}: unexpected code after the last arm"
        raise SourceError(msg)

    if missing := sorted(set(SHUFL_BYTES_VALUES) - set(claimed)):
        msg = f"shufl.cl:{line_of(text, start)}: {function} for {value_type} has no arm for SHUFL_BYTES={missing}"
        raise SourceError(msg)

    return arms


@dataclass(frozen=True)
class _ArmContext:
    text: str
    value_type: str
    element_type: str
    frames: tuple[tuple[Condition, ...], ...]


def _parse_arm(text: str, function: str, value_type: str, arm: re.Match[str], end: int, serves: tuple[int, ...]) -> Arm:
    line = line_of(text, arm.start())
    base = _LDS_BASE.match(text, skip_space(text, arm.end(), end))

    if base is None or base.group(1) not in ELEMENT_BYTES:
        msg = (
            f"shufl.cl:{line}: expected 'local T* lds = lds2; if (numWG > 1) lds += ((u32) get_local_id(0) / WG) *"
            f" LDS_BYTES / sizeof(T);' with T one of {sorted(ELEMENT_BYTES)}"
        )
        raise SourceError(msg)

    context = _ArmContext(text, value_type, base.group(1), tuple(preprocessor_frames(text, arm.start())))
    plain, blocks = _parse_items(context, base.end(), end - 1, in_special=False)

    components = {loop.component for loop in (*plain, *(loop for block in blocks for loop in block.loops))}
    if len(families := sorted({_family(component) for component in components})) > 1:
        msg = f"shufl.cl:{line}: the arm mixes {' and '.join(families)}"
        raise SourceError(msg)

    return Arm(function, value_type, base.group(1), serves, line, tuple(blocks), tuple(plain))


def _parse_items(
    context: _ArmContext,
    start: int,
    end: int,
    *,
    in_special: bool,
) -> tuple[list[Loop], list[SpecialBlock]]:
    """Parse the items of an arm or special case body, from `start` to `end`."""
    text = context.text
    loops: list[Loop] = []
    blocks: list[SpecialBlock] = []
    position, depth, barred = start, 0, False
    previous: Loop | None = None

    def top_level(where: str) -> None:
        if depth:
            msg = f"{where}: only special cases may be inside an #if within an arm"
            raise SourceError(msg)

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

        if m := _BAR.match(text, position, end):
            top_level(where)
            barred, position = True, m.end()
            continue

        if m := _DECLARATION.match(text, position, end):
            top_level(where)
            position = m.end()
            continue

        guard = None

        if m := _IF.match(text, position, end):
            close = skip_balanced(text, m.end() - 1)
            condition = text[m.end() : close - 1]
            after = skip_space(text, close, end)

            if text.startswith("{", after) and not in_special:
                block, position = _parse_special(context, condition, position, after)
                blocks.append(block)
                continue

            wg = _WG_CONDITION.fullmatch(condition)
            if wg is None or not _I_LOOP.match(text, after, end):
                msg = f"{where}: unexpected 'if ({' '.join(condition.split())})'"
                raise SourceError(msg)

            guard, position = WgGuard(int(wg.group(1)), equal=True), after
        elif m := _ELSE_LOOP.match(text, position, end):
            if follows_loop is None or follows_loop.guard is None or not follows_loop.guard.equal:
                msg = f"{where}: 'else' does not follow an 'if (WG == n)' loop"
                raise SourceError(msg)

            guard, barred, position = follows_loop.guard._replace(equal=False), follows_loop.barred, m.end()

        header = _I_LOOP.match(text, position, end)
        if header is None:
            snippet = " ".join(text[position:end].split())[:60]
            msg = (
                f"{where}: expected bar(WG), a special case or 'for (u32 i = 0; i < RADIX; ++i) {{', found {snippet!r}"
            )
            raise SourceError(msg)

        top_level(where)
        position = skip_balanced(text, header.end() - 1)
        previous = _parse_loop(context, header.end(), position - 1, guard, barred=barred, where=where)
        loops.append(previous)
        barred = False

    if depth:
        msg = f"shufl.cl:{line_of(text, end)}: #if without #endif"
        raise SourceError(msg)

    return loops, blocks


def _parse_special(context: _ArmContext, condition: str, start: int, brace: int) -> tuple[SpecialBlock, int]:
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

    loops, _ = _parse_items(context, brace + 1, ending.start(), in_special=True)
    guards = tuple(condition for frame in frames[len(context.frames) :] for condition in frame)

    return SpecialBlock(parse_c(condition), guards, tuple(loops), line), end


def _parse_loop(context: _ArmContext, start: int, end: int, guard: WgGuard | None, *, barred: bool, where: str) -> Loop:
    text = context.text
    slots: list[CExpr] = []
    pieces: list[str] = []
    position = start

    for m in _LDS_INDEX.finditer(text, start, end):
        close = skip_balanced(text, m.end() - 1)
        pieces += [text[position : m.start()], f" @{len(slots)} "]
        slots.append(parse_c(text[m.end() : close - 1]))
        position = close

    pieces.append(text[position:end])
    body = _condensed("".join(pieces))
    line = line_of(text, start)

    def loop(kind: LoopKind, component: str) -> Loop:
        return Loop(kind, tuple(slots), component, guard, barred, line)

    if m := _WRITE_BODY.fullmatch(body):
        return loop("write", f"int4.{m.group(2)}" if m.group(2) else m.group(1) or "")

    if m := _READ_BODY.fullmatch(body):
        return loop("read", m.group(1) or "")

    if (m := _INT4_READ_BODY.fullmatch(body)) and m.group(2) == context.value_type:
        return loop("read", f"int4.{m.group(1)}")

    if (m := _BUTTERFLY_BODY.fullmatch(body)) and m.group(1) == context.element_type:
        return loop("butterfly", m.group(2) or "")

    msg = f"{where}: loop body is not a form this checker reads: {body!r}"
    raise SourceError(msg)


def group_phases(loops: Sequence[Loop]) -> tuple[Phase, ...]:
    """Group loops into passes: a run of write loops, then the read loops that follow them."""
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


LaneValue = ValueId | Butterfly | None


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


def replay(phases: Sequence[Phase], shape: Shape, call: CallSite) -> Trace:
    """Run the passes of a path over every lane, as the kernel would, and return what each place in u holds."""
    env = {
        "RADIX": shape.radix,
        "WG": shape.wg,
        "f": call.f,
        "r": call.r,
        "mask": call.f - 1,
        "i": 0,
        "lowMe": 0,
    }
    indices, lanes, half = range(shape.radix), range(shape.wg), shape.wg // 2
    values: dict[ValueId, LaneValue] = {}
    slots: list[int] = []
    lowest: Access | None = None
    highest: Access | None = None
    collisions: list[tuple[Access, LaneValue]] = []
    unwritten_reads: list[Access] = []
    unread_phases: list[int] = []

    def track(slot: int, phase: int, i: int, lane: int, *, write: bool) -> Access:
        nonlocal lowest, highest
        access = Access(slot, phase, i, lane, write)
        slots.append(slot)

        if lowest is None or slot < lowest.slot:
            lowest = access

        if highest is None or slot > highest.slot:
            highest = access

        return access

    for p, phase in enumerate(phases):
        memory: dict[int, LaneValue] = {}

        for loop in phase.writes:
            for expr in loop.slots:
                for i in indices:
                    env["i"] = i

                    for lane in lanes:
                        env["lowMe"] = lane
                        place = ValueId(i, lane, loop.component)
                        value = values.get(place, place)
                        access = track(expr.evaluate(env), p, i, lane, write=True)
                        owner = memory.setdefault(access.slot, value)
                        if owner != value:
                            collisions.append((access, owner))

        if not phase.reads:
            unread_phases.append(p)

        for loop in phase.reads:
            for i in indices:
                env["i"] = i

                for lane in lanes:
                    env["lowMe"] = lane
                    loaded: list[LaneValue] = []

                    for expr in loop.slots:
                        access = track(expr.evaluate(env), p, i, lane, write=False)

                        if access.slot not in memory:
                            unwritten_reads.append(access)

                        loaded.append(memory.get(access.slot))

                    place = ValueId(i, lane, loop.component)
                    if loop.kind == "butterfly":
                        values[place] = Butterfly("add" if lane < half else "sub", loaded[0], loaded[1])
                    else:
                        values[place] = loaded[0]

    return Trace(values, tuple(slots), lowest, highest, tuple(collisions), tuple(unwritten_reads), tuple(unread_phases))


def integrity_problems(trace: Trace) -> list[str]:
    """Check the integrity of a trace and return a list of problems."""
    problems: list[str] = []

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
    """Check that a bar(WG) separates each change between writing LDS and reading it."""
    return [
        f"the {'write' if loop.kind == 'write' else 'read'} at shufl.cl:{loop.line} is not separated by bar(WG)"
        f" from the {'write' if previous.kind == 'write' else 'read'} before it"
        for previous, loop in itertools.pairwise(loops)
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


@dataclass(frozen=True)
class Failure:
    where: str
    problems: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    shapes: tuple[Shape, ...]
    special_checked: int
    plain_checked: int
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
    reached: set[SpecialBlock] = field(default_factory=set[SpecialBlock])


def run_checks(sources: Sources) -> Report:
    """Run all checks on the given sources and return a report."""
    shapes = shape_space(sources.fftconfig_cpp, sources.fftconfig_h)
    sites = call_sites(sources.fftbase, shapes)
    lds_bytes = LdsBytes.parse(sources.fftbase)
    arms = parse_shufl(sources.shufl)

    called = {site.function for calls in sites.values() for site in calls}
    defined = {(arm.function, arm.value_type) for arm in arms}
    if missing := sorted({(f, t) for f in called for _, t in defined} - defined):
        msg = f"fftbase.cl calls shufl functions that shufl.cl does not define: {missing}"
        raise SourceError(msg)

    tally = _Tally()

    for arm in arms:
        for shape in shapes:
            plain_loops = for_wg(arm.plain, shape.wg)
            calls = sorted(site for site in sites[shape] if site.function == arm.function)

            for call in calls:
                _check_call(arm, shape, call, plain_loops=plain_loops, lds_bytes=lds_bytes, tally=tally)

    unreachable = tuple((arm, block) for arm in arms for block in arm.blocks if block not in tally.reached)

    return Report(tuple(shapes), tally.special_checked, tally.plain_checked, tuple(tally.failures), unreachable)


def _check_call(
    arm: Arm,
    shape: Shape,
    call: CallSite,
    *,
    plain_loops: Sequence[Loop],
    lds_bytes: LdsBytes,
    tally: _Tally,
) -> None:
    """Check one call of one arm at one shape, under every SHUFL_BYTES and LDSPAD/LDSSWIZ."""
    context = f"{shape} f={call.f} r={call.r}"
    plain = replay(group_phases(plain_loops), shape, call)
    plain_where = f"{arm} plain implementation | {context}"
    plain_broken = integrity_problems(plain) + barrier_problems(plain_loops)

    if plain_broken:
        tally.failures.append(Failure(plain_where, tuple(plain_broken)))

    # Each special case's trace and size-independent problems, shared across settings.
    specials: dict[SpecialBlock, tuple[Trace, list[str]]] = {}

    for shufl_bytes, mode in itertools.product(arm.shufl_bytes, MODES):
        setting = f"{mode} SHUFL_BYTES={shufl_bytes}"
        capacity = lds_bytes.slots(mode, shufl_bytes, shape, arm.element_bytes)

        # The first selecting special case that is compiled runs. One whose #if cannot be decided may or may not be, so
        # it is checked and so is whatever would run without it.
        for block in arm.blocks:
            enabled = block.enabled(mode) if block.selects(shape, call) else False

            if enabled is False:
                continue

            _check_special(
                arm,
                block,
                shape,
                call,
                capacity,
                plain=plain,
                plain_broken=bool(plain_broken),
                specials=specials,
                where=f"{context} {setting}",
                tally=tally,
            )

            if enabled:
                break
        else:
            tally.plain_checked += 1
            if problem := bounds_problem(plain, capacity):
                tally.failures.append(Failure(f"{plain_where} {setting}", (problem,)))


def _check_special(
    arm: Arm,
    block: SpecialBlock,
    shape: Shape,
    call: CallSite,
    capacity: int,
    *,
    plain: Trace,
    plain_broken: bool,
    specials: dict[SpecialBlock, tuple[Trace, list[str]]],
    where: str,
    tally: _Tally,
) -> None:
    tally.special_checked += 1
    tally.reached.add(block)

    if block not in specials:
        loops = for_wg(block.loops, shape.wg)
        trace = replay(group_phases(loops), shape, call)
        problems = integrity_problems(trace) + barrier_problems(loops)
        if not plain_broken and (problem := value_problem(plain, trace)):
            problems.append(problem)
        specials[block] = (trace, problems)

    trace, problems = specials[block]

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
            f" stay inside LDS; {report.plain_checked} more run the plain shufl and stay inside LDS",
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
