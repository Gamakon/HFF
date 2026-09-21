"""The race log is ONE table whose columns never move — whatever a fit returns."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))
import _rust_race as R

PREFIX = len(R.TABLE_HEADER) - len("  model")
HOSTILE = [
    ("feynman_I_14_3", True, True, 1.0, 1, 0.9, "early_stop", ["5.834e-15", "6.317e-15", "0", "-22.42"], "x_0*x_1*x_2", ""),
    ("strogatz_barmag1", False, False, -3.247739129437077e57, 250, 6.1, "time", ["6.986e-2", "2.973e-2", "2", "-2.81"], "-0.0433*exp(x_1**3/4)", ""),
    ("feynman_test_1", False, False, float("nan"), 135, 6.3, "time", ["4.953e-1", "6.157e-1", "1", "-0.91"], "15.8*sqrt(exp(-64*(0.25*x_2**3)))" * 9, ""),
    ("feynman_a_very_long_dataset_name_that_never_ends", False, False, 1e300, 123456789, 99999.123, "ENGINE FAIL", ["-", "-", "-", "-"], "", "ENGINE FAILED: a long\nmulti-line\tmessage " * 5),
    ("feynman_II_6_15b", False, True, -0.5, 0, "6.2", "n_gen", ["1e-300", "inf", "12", "-inf"], "x_0 +\n x_1", "REPORT FAULT: chromosome 0.9 string 0.4"),
    ("strogatz_lv1", True, False, float("inf"), 7, 0.0, "early_stop", ["nan", "nan", "0", "nan"], "0", ""),
]


def test_every_row_lines_up_with_the_header():
    for row in HOSTILE:
        line = R.format_row(*row)
        assert "\n" not in line and "\t" not in line, line
        assert line[PREFIX:PREFIX + 2] == "  ", f"the model column moved:\n{R.TABLE_HEADER}\n{line}"
        assert len(line) <= PREFIX + 2 + R.MODEL_WIDTH, f"too long ({len(line)}):\n{line}"
        # every column's cell is exactly its width, and ends where the header's does
        at = 0
        for name, width, align in R.TABLE_COLUMNS:
            cell = line[at:at + width]
            assert len(cell) == width
            if align == ">":
                assert cell[0] == " ", f"column {name!r} touches its neighbour: {cell!r} in\n{line}"
            at += width


def test_the_values_a_reader_needs_survive():
    line = R.format_row(*HOSTILE[0])
    for wanted in ("f_I_14_3", "Y", "1.0000", "early_stop", "5.834e-15", "-22.42", "x_0*x_1*x_2"):
        assert wanted in line, (wanted, line)
    assert "-3.2e+57" in R.format_row(*HOSTILE[1])
    assert " nan" in R.format_row(*HOSTILE[2]) and R.format_row(*HOSTILE[2]).endswith("...")
