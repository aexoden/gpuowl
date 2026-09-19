#!/usr/bin/env python3
"""Check that every -use option the code understands is declared in the option table.

Three kinds of name are collected from the source:

  1. A kernel knob: a name the kernels give a default to, guarded either way round -- "#ifndef X", "#if !defined(X)"
     or "#if !X" holding the "#define X", or "#ifdef X" or "#if defined(X)" holding it in their "#else".
  2. An undefaulted kernel knob: a name a kernel conditional tests that neither the kernels nor the host ever define,
     so it reads as 0 unless the user sets it. A condition is scanned whole -- across a compound operator, across a
     backslash continuation, and through parentheses that only wrap a name -- so "#if !defined(X) && Y" says something
     about Y as well as about X.
  3. A host read: useValue(config, "X", ...), args.value("X", ...), config["X"], k == "X", use_override = "X", or
     emplace_back("X", ...).

The kernels are what genbundle.sh bundles: src/cl, plus the CUDA prelude in src/cuda that is prepended to every kernel
compiled through NVRTC.

Each must appear in src/OptionSpace.cpp's table or in IGNORE below. A table key the scan does not see is reported but is
not an error: several keys are reached only indirectly, through a value list or a kernel macro the scan cannot follow.

Exit status: 0 if every name is declared, 1 if one is not, 2 if the source cannot be read.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

PROG: Final = "check-option-inventory"
REPO_ROOT: Final = Path(__file__).resolve().parent.parent

# Names that look like options to the scan but are not -use keys.
IGNORE: Final = {
    "AMDGPU": "the vendor of the device, decided by the host",
    "NVIDIAGPU": "the vendor of the device, decided by the host",
    "__has_builtin": "a compiler feature test",
    "LL": "which carry a kernel is compiled with, set per kernel by Gpu::kernelDefines()",
    "HAS_PTX": "derived in base.cl from the compute capability the host reports",
    "M_PI": "a math constant",
    "M_SQRT1_2": "a math constant",
}

# Generated files to ignore.
GENERATED: Final = ("bundle.cpp",)

IDENT: Final = r"[A-Za-z_][A-Za-z0-9_]*"
KEY: Final = r"[A-Z][A-Z0-9_]{2,}"

# A name in an "#if" expression that could be an option.
_WORD: Final = re.compile(rf"\b{KEY}\b")

_COMMENT: Final = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_CONTINUATION: Final = re.compile(r"\\\n")

_HOST_READ: Final = re.compile(
    r"(?:useValue\s*\([^,()]+,\s*"
    r"|args(?:->|\.)value\s*\(\s*"
    r"|config\[\s*"
    r"|\bk\s*==\s*"
    r"|use_override\s*=\s*"
    r"|emplace_back\s*\(\s*"
    rf')"({IDENT})"',
)


class SourceError(Exception):
    """The source is not in a form this checker knows how to read."""


def strip_comments(text: str) -> str:
    """Remove C/C++ comments, leaving every line number unchanged."""
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def read(path: Path) -> str:
    """Return the comment-free text of one source file, its continued lines joined."""
    try:
        return _CONTINUATION.sub(" ", strip_comments(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError) as e:
        msg = f"cannot read {path}: {e}"
        raise SourceError(msg) from e


def body_of(text: str, signature: str, where: Path) -> str:
    """Return the braced body of the function whose definition starts with `signature`."""
    start = text.find(signature)
    if start < 0:
        msg = f"{where.name}: no '{signature}'"
        raise SourceError(msg)

    opening = text.find("{", start)
    if opening < 0:
        msg = f"{where.name}: '{signature}' has no body"
        raise SourceError(msg)

    depth = 0
    for i in range(opening, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : i]

    msg = f"{where.name}: unbalanced braces in '{signature}'"
    raise SourceError(msg)


#
# The table
#


def declared_keys(path: Path) -> set[str]:
    """Return the keys src/OptionSpace.cpp's table declares."""
    text = read(path)
    table = body_of(text, "vector<Option> buildTable()", path)

    written = set(re.findall(rf'\.key = "({KEY})"', table))

    for form in (rhs.strip() for rhs in re.findall(r"\.key = ([^,}]+)", table)):
        if not re.fullmatch(rf'"{KEY}"', form) and form not in ("key", "r.key"):
            msg = f"{path.name}: buildTable() sets .key from '{form}', which this checker cannot read"
            raise SourceError(msg)

    fixed = re.search(r"for \(const char\* key : \{(.*?)\}\)", table, re.DOTALL)
    if not fixed:
        msg = f"{path.name}: buildTable() has no 'for (const char* key : {{...}})' block"
        raise SourceError(msg)

    helpers = set(re.findall(r"\b(\w+)\s*\(\s*t\b", table))
    if helpers != {"addRegOptions"}:
        msg = f"{path.name}: buildTable() hands the table to {sorted(helpers)}, expected only addRegOptions"
        raise SourceError(msg)

    ladder = re.search(
        r"static const RegKey keys\[\] = \{(.*?)\n  \};",
        body_of(text, "void addRegOptions(", path),
        re.DOTALL,
    )
    if not ladder:
        msg = f"{path.name}: addRegOptions() has no 'static const RegKey keys[]' table"
        raise SourceError(msg)

    groups = {
        "written-out": written,
        "never-searched": set(re.findall(rf'"({KEY})"', fixed.group(1))),
        "register-count": set(re.findall(rf'\{{"({KEY})",', ladder.group(1))),
    }

    for name, keys in groups.items():
        if not keys:
            msg = f"{path.name}: the {name} keys of the table read as empty"
            raise SourceError(msg)

    return written | groups["never-searched"] | groups["register-count"]


#
# The kernels
#


def cl_defined(sources: dict[Path, str]) -> set[str]:
    """Return every name the kernels #define, conditionally or not."""
    pattern = re.compile(rf"^\s*#\s*define\s+({IDENT})", re.MULTILINE)
    return {m.group(1) for text in sources.values() for m in pattern.finditer(text)}


def unwrap(condition: str) -> str:
    """Drop parentheses that only wrap a name or a defined() test, so "!(X)" reads as the "!X" it means."""
    redundant = re.compile(rf"(?<!defined)\(({IDENT}|defined\({IDENT}\))\)")

    while (shorter := redundant.sub(r"\1", condition)) != condition:
        condition = shorter

    return condition


@dataclass
class Conditional:
    """One open preprocessor conditional."""

    condition: str
    name_only: bool
    undefined_first: bool
    in_else: bool = False
    exhausted: bool = False


def guards(frame: Conditional, name: str) -> bool:
    """Whether the branch the walk is in is the one that runs when `name` is unset."""
    if frame.exhausted:
        return False

    if frame.name_only:
        return frame.condition == name and frame.in_else != frame.undefined_first

    if frame.in_else:
        return frame.condition == name or bool(re.search(rf"(?<!!)defined\({name}\)", frame.condition))

    return bool(re.search(rf"!defined\({name}\)|!{name}(?![A-Za-z0-9_])", frame.condition))


def cl_defaults(sources: dict[Path, str]) -> dict[str, Path]:
    """Return each name given a default under its own guard."""
    guarded: dict[str, Path] = {}
    unguarded: set[str] = set()

    for path, text in sources.items():
        stack: list[Conditional] = []

        for line in text.splitlines():
            if directive := re.match(r"\s*#\s*(ifndef|ifdef|if|elif|else|endif)\b(.*)", line):
                which, rest = directive.group(1), re.sub(r"\s+", "", directive.group(2))

                if which in ("if", "ifdef", "ifndef"):
                    stack.append(
                        Conditional(
                            condition=unwrap(rest),
                            name_only=which != "if",
                            undefined_first=which == "ifndef",
                        ),
                    )
                elif not stack:
                    pass
                elif which == "else":
                    stack[-1].in_else = True
                elif which == "elif":
                    stack[-1].exhausted = True
                else:
                    stack.pop()

            elif m := re.match(rf"\s*#\s*define\s+({IDENT})", line):
                name = m.group(1)

                if any(guards(frame, name) for frame in stack):
                    guarded.setdefault(name, path)
                else:
                    unguarded.add(name)

    return {name: path for name, path in guarded.items() if name not in unguarded}


def call_arguments(text: str, function: str) -> Iterator[str]:
    """Yield the argument text of each call to `function`, parentheses balanced."""
    for call in re.finditer(rf"\b{re.escape(function)}\s*\(", text):
        depth = 0

        for i in range(call.end() - 1, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    yield text[call.end() : i]
                    break


def host_emitted(sources: dict[Path, str]) -> set[str]:
    """Return the names the host defines for the kernels."""
    names: set[str] = set()

    for text in sources.values():
        names |= set(re.findall(rf'toDefine\("({IDENT})"', text))
        names |= set(re.findall(rf"-D({IDENT})", text))

        # The width, height and middle of the shape reach the kernels through one initializer list. Brace-initialized
        # string keys are only read inside a toDefine() call: taken from anywhere in the host sources, an unrelated
        # table of strings would silence a kernel knob that happened to share a name with one of its entries.
        for arguments in call_arguments(text, "toDefine"):
            names |= set(re.findall(rf'\{{\s*"({KEY})"\s*,', arguments))

    return names


#
# The scan
#


@dataclass
class Report:
    """Results of one scan."""

    found: dict[str, str] = field(default_factory=dict[str, str])
    declared: set[str] = field(default_factory=set[str])
    undeclared: dict[str, str] = field(default_factory=dict[str, str])
    stale_ignore: list[str] = field(default_factory=list[str])
    unseen: list[str] = field(default_factory=list[str])

    @property
    def ok(self) -> bool:
        """Whether every name found in the source is declared."""
        return not self.undeclared


def where(root: Path, path: Path) -> str:
    """Return a source path as the report would display it."""
    relative = path.relative_to(root / "src")
    return str(relative) if relative.parent.name else path.name


def read_kernels(root: Path) -> dict[Path, str]:
    """Return the text of every kernel source."""
    sources = {
        path: read(path)
        for path in sorted((root / "src" / "cuda").glob("*.cuh")) + sorted((root / "src" / "cl").glob("*.cl"))
    }

    if not sources:
        msg = f"no kernel sources under {root / 'src' / 'cl'}"
        raise SourceError(msg)

    return sources


def read_host(root: Path) -> dict[Path, str]:
    """Return the text of every host source."""
    sources = {
        path: read(path)
        for directory in (root / "src", root / "src" / "cuda")
        for path in sorted(directory.glob("*.cpp")) + sorted(directory.glob("*.h"))
        if path.name not in GENERATED
    }

    if not sources:
        msg = f"no host sources under {root / 'src'}"
        raise SourceError(msg)

    return sources


def scan(root: Path) -> Report:
    """Collect every option-looking name the kernels or the host understand, and compare against the table."""
    kernels, host = read_kernels(root), read_host(root)
    defined, emitted = cl_defined(kernels), host_emitted(host)

    found: dict[str, str] = {}

    for path, text in kernels.items():
        for line in text.splitlines():
            if m := re.match(rf"\s*#\s*if\s+!\s*defined\s*\(\s*({IDENT})\s*\)", line) or re.match(
                rf"\s*#\s*ifndef\s+({IDENT})",
                line,
            ):
                found.setdefault(m.group(1), f"kernel knob with a default in {where(root, path)}")

            if re.match(r"\s*#\s*(if|ifdef|ifndef|elif)\b", line):
                for name in _WORD.findall(line):
                    if name not in defined and name not in emitted:
                        found.setdefault(name, f"undefaulted kernel knob in {where(root, path)}")

    for name, path in cl_defaults(kernels).items():
        found.setdefault(name, f"kernel knob with a default in {where(root, path)}")

    for path, text in host.items():
        for name in _HOST_READ.findall(text):
            found.setdefault(name, f"read by the host in {where(root, path)}")

    declared = declared_keys(root / "src" / "OptionSpace.cpp")

    return Report(
        found=found,
        declared=declared,
        undeclared={k: w for k, w in sorted(found.items()) if k not in declared and k not in IGNORE},
        stale_ignore=sorted(k for k in IGNORE if k not in found),
        unseen=sorted(k for k in declared if k not in found),
    )


def print_report(report: Report, *, verbose: bool = False) -> None:
    """Describe on stdout what the scan found."""
    if verbose:
        print(f"{PROG}: declared: {' '.join(sorted(report.declared))}")
        print(f"{PROG}: found: {' '.join(sorted(report.found))}")

    if report.stale_ignore:
        print(f"{PROG}: warning: IGNORE names the scan no longer sees: {' '.join(report.stale_ignore)}")

    if report.unseen:
        print(
            f"{PROG}: note: table keys the scan does not see, each reached through a value list or a kernel macro "
            f"instead: {' '.join(report.unseen)}",
        )

    if report.ok:
        print(
            f"{PROG}: OK -- {len(report.declared)} keys in the option table, {len(report.found)} option names found "
            f"in the source, none undeclared",
        )
        return

    print(f"\n{PROG}: these names are not in the option table.")

    width = max(len(name) for name in report.undeclared)
    for name, where in report.undeclared.items():
        print(f"  {name:<{width}}  {where}")

    print(f"\n{len(report.undeclared)} undeclared name(s).")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the option inventory checker."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository to check (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="also list every name on both sides")

    args = parser.parse_args(argv)

    try:
        report = scan(args.root)
    except (SourceError, OSError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 2

    print_report(report, verbose=args.verbose)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
