#!/usr/bin/env python3
"""Tests for check_shufl_index.py.

Run with: python3 -m unittest discover -s tools -p 'test_*.py'

This test is not currently independent of the source files it relies on. Changes to those files may affect the outcome
of these tests.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Final, NamedTuple

import check_shufl_index as csi

TOOLS: Final = Path(__file__).resolve().parent


class StripCommentsTest(unittest.TestCase):
    def test_slash_star_inside_a_line_comment_opens_nothing(self) -> None:
        text = "a\n//*****\nb\n/* c\n */ d\n"
        stripped = csi.strip_comments(text)
        self.assertEqual(len(stripped), len(text))
        self.assertEqual([line.strip() for line in stripped.splitlines()], ["a", "", "b", "", "d"])


class SourcesTest(unittest.TestCase):
    def test_undecodable_source_is_a_source_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for relative in ("src/cl/shufl.cl", "src/cl/fftbase.cl", "src/FFTConfig.cpp", "src/FFTConfig.h"):
                path = Path(root, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\xff\n")

            with self.assertRaises(csi.SourceError):
                csi.Sources.read(Path(root))


class CExprTest(unittest.TestCase):
    def value(self, text: str, **env: int) -> int:
        return csi.parse_c(text).evaluate(env)

    def test_c_arithmetic(self) -> None:
        self.assertEqual(self.value("lowMe / 8 * 64 + (lowMe & 7)", lowMe=13), 69)
        self.assertEqual(self.value("(lowMe & ~mask) * r", lowMe=13, mask=3, r=8), 96)
        self.assertEqual(self.value("16u * 2"), 32)

    def test_c_logic(self) -> None:
        self.assertEqual(self.value("!(WG == 64) && RADIX == 8 || 0", WG=128, RADIX=8), 1)
        self.assertEqual(self.value("!LDSPAD", LDSPAD=1), 0)
        self.assertEqual(self.value("(a & b) == c", a=6, b=3, c=2), 1)

    def test_rejects_what_python_would_group_differently(self) -> None:
        for text in ("a & b == c", "a == b | c", "!a == b", "!a + b", "a < b < c"):
            with self.subTest(text=text), self.assertRaises(csi.SourceError):
                csi.parse_c(text)

    def test_rejects_unsupported_constructs(self) -> None:
        for text in ("a ? b", "f(x)", "u[i]", "a.x"):
            with self.subTest(text=text), self.assertRaises(csi.SourceError):
                csi.parse_c(text)

    def test_c_conditional(self) -> None:
        pad = "!LDSPAD ? 0 : RADIX == 4 ? 12 : SHUFL_BYTES >= 16 ? 7 : 56"
        self.assertEqual(self.value(pad, LDSPAD=0, RADIX=8, SHUFL_BYTES=8), 0)
        self.assertEqual(self.value(pad, LDSPAD=1, RADIX=4, SHUFL_BYTES=8), 12)
        self.assertEqual(self.value(pad, LDSPAD=1, RADIX=8, SHUFL_BYTES=16), 7)
        self.assertEqual(self.value(pad, LDSPAD=1, RADIX=8, SHUFL_BYTES=8), 56)
        self.assertEqual(self.value("a ? b ? 1 : 2 : 3", a=1, b=0), 2)
        self.assertEqual(self.value("n * 4 + (s > 1 ? 16 : 0)", n=2, s=2), 24)

    def test_unknown_symbol_is_an_error(self) -> None:
        with self.assertRaises(csi.SourceError):
            self.value("WG + 1")

    def test_negative_shift_is_an_error(self) -> None:
        with self.assertRaises(csi.SourceError):
            self.value("1 << -1")


class PreprocessorTest(unittest.TestCase):
    TEXT: Final = "#if WG == 64\n#if VARIANT == 0\n#endif\nX\n#elif WG == 128\nY\n#else\nZ\n#endif\n"

    def test_enclosing_conditions_follow_nesting_and_exclude_earlier_branches(self) -> None:
        def at(marker: str) -> list[csi.Condition]:
            return csi.enclosing_conditions(self.TEXT, self.TEXT.index(marker))

        condition = csi.Condition
        self.assertEqual(at("X"), [condition("WG == 64")])
        self.assertEqual(at("Y"), [condition("WG == 64", holds=False), condition("WG == 128")])
        self.assertEqual(at("Z"), [condition("WG == 64", holds=False), condition("WG == 128", holds=False)])

    def test_decide(self) -> None:
        condition = csi.Condition
        mixed = "WG == 64 && VARIANT == 2"
        self.assertIsNone(csi.decide([condition(mixed)], {"WG": 64, "RADIX": 8}))
        self.assertFalse(csi.decide([condition(mixed)], {"WG": 128, "RADIX": 8}))
        self.assertTrue(csi.decide([condition(mixed, holds=False)], {"WG": 128}))
        self.assertIsNone(csi.decide([condition(mixed, holds=False)], {"WG": 64}))
        self.assertIsNone(csi.decide([condition("!AMDGPU")], {"WG": 64}))
        self.assertIsNone(csi.decide([condition(None)], {}))

    def test_dead_else_branch_is_not_compiled(self) -> None:
        text = "#if 1\n#else\nZ\n#endif\n"
        conditions = csi.enclosing_conditions(text, text.index("Z"))
        self.assertFalse(csi.decide(conditions, {}))

    def test_mixing_known_and_unknown_symbols_is_an_error(self) -> None:
        with self.assertRaises(csi.SourceError):
            csi.decide([csi.Condition("WG == 64 || VARIANT == 2")], {"WG": 64, "RADIX": 8})

    def test_call_sites(self) -> None:
        fftbase = """
          for (u32 s = 1; s < WG; s *= RADIX) { shufl(lds, u, s, numWG, lowMe); }
          #if WG == 64 && RADIX == 8 && VARIANT == 2
            shufl(lds, u, 2, 4, numWG, lowMe);
          #elif WG == 128 && RADIX == 8
            shufl_and_fft2(lds, u, 8, numWG, lowMe);
          #endif
        """
        small, large = csi.Shape(8, 64), csi.Shape(8, 128)
        site = csi.CallSite
        self.assertEqual(
            csi.call_sites(fftbase, [small, large]),
            {
                small: {site("shufl", 1, 8), site("shufl", 8, 8), site("shufl", 2, 4)},
                large: {site("shufl", 1, 8), site("shufl", 8, 8), site("shufl", 64, 8), site("shufl_and_fft2", 8, 8)},
            },
        )

    def test_call_to_an_unknown_shufl_is_an_error(self) -> None:
        with self.assertRaises(csi.SourceError):
            csi.call_sites("shufl3(lds, u, 1, numWG, lowMe);", [csi.Shape(8, 64)])


class ShapeSpaceTest(unittest.TestCase):
    H: Final = "u32 nW() const { return (width == 256) ? 4 : 8; }\nu32 nH() const { return (height == 256) ? 4 : 8; }\n"

    def shapes(self, widths: str, h: str = H) -> list[csi.Shape]:
        cpp = f"for (u32 const width : {{{widths}}}) {{ for (u32 const height : {{256, 512}}) {{"
        return csi.shape_space(cpp, h)

    def test_reads_the_size_lists(self) -> None:
        self.assertEqual(self.shapes("256, 1024,"), [csi.Shape(4, 64), csi.Shape(8, 64), csi.Shape(8, 128)])

    def test_rejects_entries_that_are_not_literals(self) -> None:
        with self.assertRaises(csi.SourceError):
            self.shapes("256, 512 * 2")

    def test_rejects_sizes_that_are_not_whole_workgroups(self) -> None:
        for widths, h in (("256, 1001", self.H), ("256", self.H.replace("? 4", "? 1"))):
            with self.subTest(widths=widths), self.assertRaises(csi.SourceError):
                self.shapes(widths, h)


class Mutation(NamedTuple):
    """A change planted in shufl.cl: occurrence `occurrence` of `anchor`, found `copies` times, becomes `change`."""

    anchor: str
    change: str
    copies: int = 1
    occurrence: int = 0


class KernelTest(unittest.TestCase):
    """The real kernel passes, and each kind of error planted in it is caught."""

    sources: csi.Sources

    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = csi.Sources.read(TOOLS.parent)

    def mutate(self, text: str, mutation: Mutation) -> str:
        """Return `text` with `mutation` applied."""
        anchor = mutation.anchor
        self.assertEqual(text.count(anchor), mutation.copies, f"shufl.cl changed; update the anchor {anchor!r}")
        position = -1

        for _ in range(mutation.occurrence + 1):
            position = text.index(anchor, position + 1)

        return text[:position] + mutation.change + text[position + len(anchor) :]

    def planted(self, *mutations: Mutation, source: str = "shufl") -> csi.Report:
        """Check the kernel with each mutation applied in turn to `source`, shufl.cl or fftbase.cl."""
        text = getattr(self.sources, source)

        for mutation in mutations:
            text = self.mutate(text, mutation)

        return csi.run_checks(replace(self.sources, **{source: text}))

    def assertCaught(self, report: csi.Report, fragment: str) -> None:
        problems = [problem for failure in report.failures for problem in failure.problems]
        self.assertTrue(
            any(fragment in problem for problem in problems),
            f"no problem mentions {fragment!r}: {problems}",
        )

    def assertUnreadable(self, *mutations: Mutation) -> None:
        with self.assertRaises(csi.SourceError):
            self.planted(*mutations)

    def test_real_kernel_passes(self) -> None:
        report = csi.run_checks(self.sources)
        self.assertEqual(report.failures, ())
        self.assertEqual((report.special_checked, report.plain_checked, report.cross_checked), (196, 308, 36))

    def test_wrong_read_in_the_x_pass(self) -> None:
        # T_Z61 LDSPAD f == 8 && r == 8, WG != 64 read of .x; the .y pass is untouched.
        anchor = "u[i].x = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]"
        report = self.planted(Mutation(anchor, anchor.replace("(lowMe & 7)]", "((lowMe + 1) & 7)]"), copies=2))
        self.assertCaught(report, "lane values differ")

    def test_wrong_read_in_the_y_pass(self) -> None:
        anchor = "u[i].y = lds[i * (WG / 64) * 8 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 8) + (lowMe & 7)]"
        report = self.planted(Mutation(anchor, anchor.replace("(lowMe & 7)]", "((lowMe + 1) & 7)]"), copies=2))
        self.assertCaught(report, "lane values differ")

    def test_read_into_the_wrong_component(self) -> None:
        # T_Z61 LDSPAD f == 1 && r == 8: the .y pass stores its values into .x.
        anchor = "u[i].y = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 2)]"
        report = self.planted(Mutation(anchor, anchor.replace("u[i].y", "u[i].x")))
        self.assertCaught(report, "lane values differ")

    def test_write_of_another_element_is_an_error(self) -> None:
        # T2_GF61 LDSPAD f == 1 && r == 8 write.
        anchor = "lds[i * (WG + 1) + lowMe] = u[i];"
        self.assertUnreadable(Mutation(anchor, anchor.replace("u[i]", "u[0]")))

    def test_condition_inside_a_loop_is_an_error(self) -> None:
        anchor = "{ lds[i * (WG + 1) + lowMe] = u[i]; }"
        self.assertUnreadable(Mutation(anchor, "{ if (lowMe != 0) lds[i * (WG + 1) + lowMe] = u[i]; }"))

    def test_condition_around_a_loop_is_an_error(self) -> None:
        anchor = "for (u32 i = 0; i < RADIX; ++i) { lds[i * (WG + 1) + lowMe] = u[i]; }"
        self.assertUnreadable(Mutation(anchor, f"if (lowMe != 0) {{ {anchor} }}"))

    def test_unread_arm_header_is_an_error(self) -> None:
        # Equivalent for the values SBMUL(numWG) * SHUFL_BYTES takes, but not a form the checker reads.
        anchor = "if (SBMUL(numWG) * SHUFL_BYTES >= 16) {"
        self.assertUnreadable(Mutation(anchor, anchor.replace(">= 16", "> 8"), copies=2))

    def test_arm_chosen_by_shufl_bytes_alone_is_an_error(self) -> None:
        anchor = "if (SBMUL(numWG) * SHUFL_BYTES >= 16) {"
        self.assertUnreadable(Mutation(anchor, "if (SHUFL_BYTES >= 16) {", copies=2))

    def test_arm_chain_must_cover_every_shufl_bytes(self) -> None:
        anchor = "else if (SBMUL(numWG) * SHUFL_BYTES == 4) {"
        self.assertUnreadable(Mutation(anchor, anchor.replace("== 4", "== 2"), copies=4))

    def test_changed_lds_base_pointer_is_an_error(self) -> None:
        anchor = "local T2_GF61* lds = LDSsharing_ptr(lds2, numWG);"
        self.assertUnreadable(Mutation(anchor, anchor.replace("LDSsharing_ptr", "LDSptr"), copies=2))

    def test_changed_lds_pointer_function_is_an_error(self) -> None:
        anchor = "/ WG / SBMUL(numWG)) * SBMUL(numWG) * LDS_SHUFL_BYTES(numWG)"
        self.assertUnreadable(Mutation(anchor, "/ WG / SBMUL(numWG)) * LDS_SHUFL_BYTES(numWG)", copies=2))

    def test_missing_barrier(self) -> None:
        # T2_GF61 LDSPAD f == 1 && r == 8 reads without waiting for the other lanes' writes.
        anchor = "= u[i]; }\n      LDSbar(numWG);\n      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 8)"
        report = self.planted(Mutation(anchor, anchor.replace("LDSbar(numWG);", "")))
        self.assertCaught(report, "not separated by a barrier")

    def test_unclosed_transaction_fails_only_when_sharing(self) -> None:
        # The same T2_GF61 LDSPAD f == 1 && r == 8 case, which returns still holding the shared block.
        anchor = "(lowMe & 7) * (WG + 1)]; }\n      LDStx_end(lds2, numWG);"
        report = self.planted(Mutation(anchor, anchor.replace("LDStx_end(lds2, numWG);", "")))
        self.assertCaught(report, "does not end with LDStx_end")

        for failure in report.failures:
            self.assertIn("NVIDIAGPU=1", failure.where)
            self.assertNotIn("LDSMUL=1", failure.where)
            self.assertNotIn("numWG=1", failure.where)

    def test_bar_wg_in_a_shared_transaction(self) -> None:
        anchor = "= u[i]; }\n      LDSbar(numWG);\n      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG / 8)"
        report = self.planted(Mutation(anchor, anchor.replace("LDSbar(numWG);", "bar(WG);")))
        self.assertCaught(report, "use LDSbar(numWG)")

    def test_shared_block_without_room_for_its_padding(self) -> None:
        # Sharing gives an 8-byte arm the padding of a 16-byte one, which the 8-byte LDSPAD cases outgrow.
        anchor = "SBMUL(numWG) * SHUFL_BYTES >= 16 ? 7 : 56"
        report = self.planted(Mutation(anchor, anchor.replace(">= 16", ">= 8")), source="fftbase")
        self.assertCaught(report, "outside the")

    def test_share_not_a_whole_number_of_pointer_steps(self) -> None:
        anchor = "#define LDS_SHUFL_BYTES(numWG)    ((WG * RADIX + LDSPAD_COUNT(numWG)) * SHUFL_BYTES)"
        report = self.planted(
            Mutation(anchor, anchor.replace("* SHUFL_BYTES)", "* SHUFL_BYTES + 4)"), copies=2, occurrence=1),
            source="fftbase",
        )
        self.assertCaught(report, "LDS pointer lands on byte")

    def test_dead_else_branch_does_not_hide_a_special_case(self) -> None:
        # The T2_GF61 16-byte LDSPAD cases move into a dead #else, so they no longer shadow the LDSSWIZ case that has a
        # bad write planted in it.  The anchor carries its newline so that it names the six section guards and not the
        # "#if LDSPAD && RADIX == 4 && ..." build guard at the top of the file.
        swiz = "lds[(lowMe * 8 + i) ^ (lowMe & 7)] = u[i];"
        report = self.planted(
            Mutation("#if LDSPAD\n", "#if 1\n#else\n", copies=6),
            Mutation(swiz, "lds[((lowMe * 8 + i) ^ (lowMe & 7)) + WG * RADIX] = u[i];"),
        )
        self.assertCaught(report, "outside the")

        # Only the planted case fails: the dead cases are neither checked nor chosen in its place.
        for failure in report.failures:
            self.assertIn("'f == 1 && r == 8 && RADIX == 8' |", failure.where)
            self.assertIn("LDSSWIZ=1", failure.where)

    def test_undecidable_guard_checks_the_path_without_it(self) -> None:
        # The T2_GF61 16-byte LDSSWIZ section is under a condition that cannot be decided, and the plain shufl it may
        # shadow indexes outside LDS, only where the f == 1 && r == 8 special case would otherwise have run.
        plain = (
            "lds[i / (RADIX / r) * f + i % (RADIX / r) * WG * r + (lowMe & ~mask) * r + (lowMe & mask)] = u[i]; }\n"
            "    LDSbar(numWG);\n"
            "    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }"
        )
        shift = "(f == 1 && r == 8) * WG * RADIX"
        shifted = plain.replace("(lowMe & mask)]", f"(lowMe & mask) + {shift}]").replace(
            "lds[i * WG + lowMe]",
            f"lds[i * WG + lowMe + {shift}]",
        )
        report = self.planted(
            Mutation("#if LDSPAD\n", "#if 0\n", copies=6),
            Mutation("#if LDSSWIZ\n", "#if VARIANT == 2\n", copies=3),
            Mutation(plain, shifted, copies=2),
        )
        self.assertCaught(report, "outside the")

    def test_wrong_butterfly_pair(self) -> None:
        anchor = "F2_GF31 val2 = lds[(i / 2 + 4)"
        self.assertCaught(self.planted(Mutation(anchor, anchor.replace("+ 4", "+ 5"))), "lane values differ")

    def test_missing_read_pass(self) -> None:
        # T_Z61 LDSPAD f == 1 && r == 8 loses its .y read.
        anchor = "for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG / 8 + (lowMe / 8) + (lowMe & 7) * (WG + 2)]; }"
        self.assertCaught(self.planted(Mutation(anchor, "")), "never read")

    def test_two_lanes_on_one_slot(self) -> None:
        # T2_GF61 LDSSWIZ f == 1 && r == 8 write, with lanes 0 and 1 (2 and 3, ...) sharing a block.
        anchor = "lds[(lowMe * 8 + i) ^ (lowMe & 7)] = u[i];"
        report = self.planted(Mutation(anchor, anchor.replace("lowMe * 8", "lowMe / 2 * 8")))
        self.assertCaught(report, "another lane already holds")

    def test_write_outside_the_block(self) -> None:
        # T2_GF61 LDSPAD f == 4 && r == 4 write (the F2_GF31 arm has the same line).
        anchor = "lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i];"
        report = self.planted(Mutation(anchor, anchor.replace("(WG + 4)", "(WG + 5)"), copies=2))
        self.assertCaught(report, "outside the")

    def test_plain_only_arm_outside_the_block(self) -> None:
        # T2_GF61 SHUFL_BYTES == 4 shufl, which has no special cases.
        anchor = "tmp.x = lds[i * WG + lowMe]"
        report = self.planted(Mutation(anchor, "tmp.x = lds[i * WG + lowMe + 1]"))
        self.assertCaught(report, "outside the")

    def test_arm_that_butterflies_the_wrong_pair(self) -> None:
        # The T2_GF61 shufl_and_fft2 4-byte arm has no special case to check it, so only the other arms disagree.
        anchor = "lo2[i], lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]"
        report = self.planted(Mutation(anchor, anchor.replace("4 * WG", "2 * WG"), copies=2))
        self.assertCaught(report, "differ from the arm above it")

    def test_arm_that_loses_its_butterfly(self) -> None:
        # The bug this arm carried before: the shufl happens, the fft2 its callers expect does not. The two stash
        # loops are one text, so the .x/.y pass claims the first of them before the .z/.w pass takes what is left.
        stash = (
            "for (u32 i = 0; i < RADIX; ++i) { lo1[i] = lds[i * (WG / 2) + lowMe % (WG / 2)];"
            " lo2[i] = lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]; }"
        )

        def join(component: str) -> str:
            return (
                "for (u32 i = 0; i < RADIX; ++i) {\n"
                "      T_Z61 val1 = as_T_Z61((int2) (lo1[i], lds[         i * (WG / 2) + lowMe % (WG / 2)]));\n"
                "      T_Z61 val2 = as_T_Z61((int2) (lo2[i], lds[4 * WG + i * (WG / 2) + lowMe % (WG / 2)]));\n"
                f"      if (lowMe < WG / 2) u[i].{component} = addq(val1, val2);\n"
                f"      else u[i].{component} = subq(val1, val2);\n"
                "    }"
            )

        def plain(component: str) -> str:
            return (
                f"for (u32 i = 0; i < RADIX; ++i) {{ int4 tmp = as_int4(u[i]); tmp.{component} = lds[i * WG + lowMe];"
                " u[i] = as_T2_GF61(tmp); }"
            )

        report = self.planted(
            Mutation("int lo1[RADIX], lo2[RADIX];", ""),
            Mutation(stash, plain("x"), copies=2),
            Mutation(join("x"), plain("y")),
            Mutation(stash, plain("z")),
            Mutation(join("y"), plain("w")),
        )
        self.assertCaught(report, "differ from the arm above it")

    def test_join_from_an_undeclared_stash_is_an_error(self) -> None:
        self.assertUnreadable(Mutation("int lo1[RADIX], lo2[RADIX];", "int lo1[RADIX], lo3[RADIX];"))

    def test_join_of_one_stash_with_itself_is_an_error(self) -> None:
        anchor = "T_Z61 val2 = as_T_Z61((int2) (lo2[i]"
        self.assertUnreadable(Mutation(anchor, anchor.replace("lo2[i]", "lo1[i]"), copies=2))

    def test_unreadable_loop_is_an_error(self) -> None:
        anchor = "for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 7)] = u[i]; }"
        self.assertUnreadable(Mutation(anchor, anchor.replace("i < RADIX", "i < 8")))


if __name__ == "__main__":
    unittest.main()
