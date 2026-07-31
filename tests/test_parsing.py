from flcva.parsing import answers_equal, extract_boxed, parse_gsm8k, parse_math


def test_extracts_nested_box() -> None:
    assert extract_boxed(r"work \boxed{\frac{1}{2}} tail") == r"\frac{1}{2}"


def test_gsm8k_numeric_normalization() -> None:
    assert parse_gsm8k("#### 1,200", gold=True).canonical == "1200"
    assert parse_gsm8k(r"\boxed{0.5}").canonical == "0.5"
    assert answers_equal("gsm8k", r"\boxed{1/2}", "#### 0.5")[0]


def test_math_string_and_symbolic_equivalence() -> None:
    assert answers_equal("math500", r"\boxed{\frac{1}{2}}", r"\frac{1}{2}")[0]
    parsed = parse_math(r"\boxed{p-q}")
    assert parsed.canonical is not None


def test_math_tuple_falls_back_after_latex_error() -> None:
    equal, prediction, gold = answers_equal(
        "math500",
        r"\boxed{\left( 3, \frac{\pi}{2} \right)}",
        r"\left( 3, \frac{\pi}{2} \right)",
    )
    assert equal
    assert prediction.canonical == "(3, pi/2)"
    assert gold.canonical == "(3, pi/2)"


def test_math_list_answer_is_simplified_elementwise() -> None:
    equal, prediction, gold = answers_equal(
        "math500",
        r"\boxed{[1, \frac{2}{4}]}",
        r"[1, \frac{1}{2}]",
    )
    assert equal
    assert prediction.canonical == "[1, 1/2]"


def test_math_nested_list_answer_is_simplified_recursively() -> None:
    equal, prediction, gold = answers_equal(
        "math500",
        r"\boxed{[[1, \frac{2}{4}], [3]]}",
        r"[[1, \frac{1}{2}], [3]]",
    )
    assert equal
    assert prediction.canonical == "[[1, 1/2], [3]]"


def test_malformed_math_answer_uses_deterministic_string_fallback() -> None:
    parsed = parse_math(r"\boxed{(1, 2]}")
    assert parsed.status == "ok_string"
    assert parsed.canonical == "(1,2]"


def test_parse_failure_is_explicit() -> None:
    parsed = parse_gsm8k("no numeric answer")
    assert parsed.status == "parse_failure"
    assert parsed.canonical is None
