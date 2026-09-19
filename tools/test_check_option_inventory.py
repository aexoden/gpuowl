#!/usr/bin/env python3
"""Tests for check_option_inventory.py.

Run with: python3 -m unittest discover -s tools -p 'test_*.py'

Most tests build a small synthetic tree, so that a planted error is visible in the test itself. One test runs the
checker over this repository, and is therefore not independent of the sources it reads.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Final

import check_option_inventory as coi

if TYPE_CHECKING:
    from collections.abc import Generator

# A table with one key of each of the three forms the checker recognizes.
TABLE: Final = """\
namespace {

void addRegOptions(vector<Option>& t) {
  struct RegKey {
    const char* key;
    u32 touches;
  };

  static const RegKey keys[] = {
    {"REGCF64", KG_CARRY},
  };

  for (const RegKey& r : keys) { t.push_back({.key = r.key, .values = REG_LADDER}); }
}

vector<Option> buildTable() {
  vector<Option> t;

  t.push_back({.key = "PAD", .values = {0, 64}});
  t.push_back({.key = "INPLACE", .values = {0, 1}});
  addRegOptions(t);

  for (const char* key : {
         "STATS",
       }) {
    t.push_back({.key = key, .kind = Kind::Fixed});
  }

  return t;
}

}  // namespace
"""

KERNEL: Final = """\
#if !defined(PAD)
#define PAD 0
#endif

#if PAD
// ...
#endif
"""

HOST: Final = """\
int pad = args.value("PAD", 0);
int stats = useValue(config, "STATS", 0);
"""


@contextmanager
def tree(*, table: str = TABLE, kernel: str = KERNEL, host: str = HOST) -> Generator[Path]:
    """Yield the root of a synthetic source tree holding one kernel, one host source, and one option table."""
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "src" / "cl").mkdir(parents=True)

        (root / "src" / "cl" / "base.cl").write_text(kernel)
        (root / "src" / "Gpu.cpp").write_text(host)
        (root / "src" / "OptionSpace.cpp").write_text(table)

        yield root


class RepositoryTest(unittest.TestCase):
    def test_this_repository_passes(self) -> None:
        report = coi.scan(coi.REPO_ROOT)
        self.assertEqual(report.undeclared, {})
        self.assertEqual(report.stale_ignore, [])

    def test_every_ignored_name_is_still_found(self) -> None:
        # An IGNORE entry the scan no longer sees is a rule kept alive past the code that needed it.
        self.assertTrue(set(coi.IGNORE) <= set(coi.scan(coi.REPO_ROOT).found))


class SyntheticTreeTest(unittest.TestCase):
    def test_a_complete_tree_passes(self) -> None:
        with tree() as root:
            report = coi.scan(root)
            self.assertTrue(report.ok)
            self.assertEqual(report.declared, {"PAD", "INPLACE", "STATS", "REGCF64"})
            self.assertEqual(set(report.found), {"PAD", "STATS"})
            self.assertEqual(report.unseen, ["INPLACE", "REGCF64"])

    def test_a_kernel_knob_missing_from_the_table_fails(self) -> None:
        with tree(kernel=KERNEL + "\n#ifndef NEW_KNOB\n#define NEW_KNOB 1\n#endif\n") as root:
            report = coi.scan(root)
            self.assertFalse(report.ok)
            self.assertIn("NEW_KNOB", report.undeclared)
            self.assertIn("base.cl", report.undeclared["NEW_KNOB"])

    def test_an_undefaulted_kernel_knob_fails(self) -> None:
        with tree(kernel=KERNEL + "\n#if UNDEFAULTED\n#endif\n") as root:
            self.assertIn("UNDEFAULTED", coi.scan(root).undeclared)

    def test_a_knob_defaulted_without_defined_is_a_knob(self) -> None:
        with tree(kernel="#if !IN_WG\n#define IN_WG 128\n#endif\n") as root:
            self.assertIn("IN_WG", coi.scan(root).undeclared)

    def test_a_knob_defaulted_in_an_else_is_a_knob(self) -> None:
        for guard in ("#if defined(KNOB)", "#ifdef KNOB", "#if KNOB"):
            with self.subTest(guard=guard), tree(kernel=f"{guard}\n#else\n#define KNOB 0\n#endif\n") as root:
                self.assertIn("KNOB", coi.scan(root).undeclared)

    def test_a_define_in_the_else_of_its_own_negative_guard_is_not_a_default(self) -> None:
        # "#if !KNOB / #else / #define KNOB" defines the name only where it is already set, so it is not a default.
        with tree(kernel="#if !KNOB\n#else\n#define KNOB 1\n#endif\n") as root:
            self.assertTrue(coi.scan(root).ok)

    def test_the_rest_of_a_compound_guard_is_scanned(self) -> None:
        with tree(kernel="#if !defined(PAD) && OTHER_KNOB\n#define PAD 0\n#endif\n") as root:
            self.assertEqual(set(coi.scan(root).undeclared), {"OTHER_KNOB"})

    def test_a_positive_ifdef_is_scanned(self) -> None:
        # "#ifdef X" is "#if defined(X)", and says as much about X as the spelled-out form does.
        for guard in ("#ifdef NEW_KNOB", "#if defined(NEW_KNOB)"):
            with self.subTest(guard=guard), tree(kernel=f"{guard}\n#endif\n") as root:
                self.assertIn("NEW_KNOB", coi.scan(root).undeclared)

    def test_a_continued_condition_is_read_whole(self) -> None:
        with tree(kernel="#if PAD && \\\n    NEW_KNOB\n#endif\n") as root:
            self.assertIn("NEW_KNOB", coi.scan(root).undeclared)

    def test_a_parenthesized_guard_is_a_default(self) -> None:
        for guard in ("#if !(NEW_KNOB)", "#if !( NEW_KNOB )", "#if !(defined(NEW_KNOB))"):
            with self.subTest(guard=guard), tree(kernel=f"{guard}\n#define NEW_KNOB 1\n#endif\n") as root:
                report = coi.scan(root)
                self.assertIn("NEW_KNOB", report.undeclared)
                self.assertIn("default", report.undeclared["NEW_KNOB"])

    def test_a_define_after_an_elif_is_not_read_as_a_default(self) -> None:
        with tree(kernel="#if !KNOB\n#elif OTHER\n#define KNOB 1\n#endif\n") as root:
            self.assertNotIn("KNOB", coi.scan(root).undeclared)

    def test_the_cuda_prelude_is_a_kernel_source(self) -> None:
        with tree() as root:
            (root / "src" / "cuda").mkdir()
            (root / "src" / "cuda" / "prelude.cuh").write_text("#ifndef PRELUDE_KNOB\n#define PRELUDE_KNOB 0\n#endif\n")
            report = coi.scan(root)
            self.assertIn("PRELUDE_KNOB", report.undeclared)
            self.assertIn("cuda/prelude.cuh", report.undeclared["PRELUDE_KNOB"])

    def test_an_unconditional_define_is_not_a_knob(self) -> None:
        with tree(kernel="#define DERIVED 4\n#if DERIVED\n#endif\n") as root:
            self.assertTrue(coi.scan(root).ok)

    def test_a_host_emitted_name_is_not_a_knob(self) -> None:
        for host in ('toDefine("WIDTH", width)', 'toDefine({{"WIDTH", w}, {"PAD", p}})', 'string d = "-DWIDTH=4";'):
            with self.subTest(host=host), tree(kernel="#if WIDTH\n#endif\n", host=host) as root:
                self.assertTrue(coi.scan(root).ok)

    def test_a_string_table_outside_toDefine_does_not_hide_a_knob(self) -> None:
        # A brace-initialized string anywhere in the host sources used to count as a name the host -Ds.
        with tree(kernel="#if WIDTH\n#endif\n", host='static const char* junk[] = { "WIDTH", "x" };') as root:
            self.assertIn("WIDTH", coi.scan(root).undeclared)

    def test_a_generated_source_is_not_read(self) -> None:
        with tree() as root:
            (root / "src" / "bundle.cpp").write_text('args.value("BUNDLED_KEY", 0)')
            self.assertTrue(coi.scan(root).ok)

    def test_each_host_read_form_is_seen(self) -> None:
        forms = {
            "A_KEY": 'useValue(config, "A_KEY", 0)',
            "B_KEY": 'args.value("B_KEY", 0)',
            "C_KEY": 'args->value("C_KEY", 0)',
            "D_KEY": 'config["D_KEY"] = "1"',
            "E_KEY": 'if (k == "E_KEY") {}',
            "F_KEY": 'use_override = "F_KEY"',
            "G_KEY": 'v.emplace_back("G_KEY", 1)',
        }

        with tree(host="\n".join(forms.values())) as root:
            self.assertEqual(set(coi.scan(root).undeclared), set(forms))

    def test_a_host_read_in_a_comment_is_not_seen(self) -> None:
        with tree(host='// args.value("COMMENTED", 0)\n') as root:
            self.assertTrue(coi.scan(root).ok)

    def test_every_host_source_is_read(self) -> None:
        with tree() as root:
            (root / "src" / "cuda" / "clwrap_cuda.cpp").parent.mkdir()
            (root / "src" / "cuda" / "clwrap_cuda.cpp").write_text('args.value("BACKEND_KEY", 0)')
            self.assertIn("BACKEND_KEY", coi.scan(root).undeclared)

    def test_a_key_dropped_from_the_table_fails(self) -> None:
        with tree(table=TABLE.replace('t.push_back({.key = "PAD", .values = {0, 64}});', "")) as root:
            self.assertIn("PAD", coi.scan(root).undeclared)

    def test_a_never_searched_key_dropped_from_the_table_fails(self) -> None:
        with tree(table=TABLE.replace('"STATS",', '"OTHER",')) as root:
            self.assertIn("STATS", coi.scan(root).undeclared)

    def test_an_ignored_name_the_scan_no_longer_sees_is_reported(self) -> None:
        with tree() as root:
            self.assertEqual(coi.scan(root).stale_ignore, sorted(coi.IGNORE))


class UnreadableSourceTest(unittest.TestCase):
    def assert_source_error(self, **kwargs: str) -> None:
        with tree(**kwargs) as root, self.assertRaises(coi.SourceError):
            coi.scan(root)

    def test_a_table_without_buildTable_is_an_error(self) -> None:
        self.assert_source_error(table="int main() { return 0; }\n")

    def test_a_key_set_from_something_unreadable_is_an_error(self) -> None:
        self.assert_source_error(table=TABLE.replace('.key = "PAD"', ".key = someName"))

    def test_an_unknown_table_helper_is_an_error(self) -> None:
        self.assert_source_error(table=TABLE.replace("addRegOptions(t);", "addRegOptions(t);\n  addMoreOptions(t);"))

    def test_a_missing_never_searched_block_is_an_error(self) -> None:
        self.assert_source_error(table=TABLE.replace("for (const char* key : {", "for (const string& key : {"))

    def test_an_empty_never_searched_block_is_an_error(self) -> None:
        self.assert_source_error(table=TABLE.replace('"STATS",', ""))

    def test_a_missing_register_ladder_is_an_error(self) -> None:
        self.assert_source_error(table=TABLE.replace("static const RegKey keys[]", "static const RegKey ladder[]"))

    def test_unbalanced_braces_are_an_error(self) -> None:
        self.assert_source_error(table=TABLE.replace("  return t;\n}\n\n}  // namespace", "  return t;"))

    def test_a_tree_without_kernels_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "src" / "cl").mkdir(parents=True)

            with self.assertRaises(coi.SourceError):
                coi.scan(root)

    def test_undecodable_source_is_an_error(self) -> None:
        with tree() as root:
            (root / "src" / "cl" / "base.cl").write_bytes(b"\xff\n")

            with self.assertRaises(coi.SourceError):
                coi.scan(root)


class StripCommentsTest(unittest.TestCase):
    def test_slash_star_inside_a_line_comment_opens_nothing(self) -> None:
        text = "a\n//*****\nb\n/* c\n */ d\n"
        stripped = coi.strip_comments(text)
        self.assertEqual(len(stripped), len(text))
        self.assertEqual([line.strip() for line in stripped.splitlines()], ["a", "", "b", "", "d"])


class MainTest(unittest.TestCase):
    def run_main(self, *argv: str) -> int:
        """Run the checker's entry point, keeping the report it prints out of the test output."""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return coi.main(argv)

    def test_a_clean_tree_exits_zero(self) -> None:
        with tree() as root:
            self.assertEqual(self.run_main("--root", str(root), "--verbose"), 0)

    def test_an_undeclared_name_exits_one(self) -> None:
        with tree(kernel=KERNEL + "\n#ifndef NEW_KNOB\n#define NEW_KNOB 1\n#endif\n") as root:
            self.assertEqual(self.run_main("--root", str(root)), 1)

    def test_unreadable_source_exits_two(self) -> None:
        with tree(table="") as root:
            self.assertEqual(self.run_main("--root", str(root)), 2)


if __name__ == "__main__":
    unittest.main()
