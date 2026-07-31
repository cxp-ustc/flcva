from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache

import sympy


@dataclass(frozen=True)
class ParsedAnswer:
    raw: str
    extracted: str | None
    canonical: str | None
    status: str


_BOX_START = re.compile(r"\\boxed\s*\{")
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/[+-]?\d+)?")


def extract_boxed(text: str) -> str | None:
    starts = list(_BOX_START.finditer(text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return text[start:].strip() or None


def _canonical_number(value: str) -> str | None:
    normalized = value.replace(",", "").strip()
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            return str(
                Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
            )
        decimal = Decimal(normalized)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if decimal == decimal.to_integral_value():
        return str(decimal.to_integral_value())
    return format(decimal.normalize(), "f")


def parse_gsm8k(text: str, *, gold: bool = False) -> ParsedAnswer:
    candidate: str | None = None
    if gold and "####" in text:
        candidate = text.rsplit("####", 1)[-1].strip()
    if candidate is None:
        candidate = extract_boxed(text)
    if candidate is None:
        matches = _NUMBER.findall(text)
        candidate = matches[-1] if matches else None
    if candidate is None:
        return ParsedAnswer(text, None, None, "parse_failure")
    number_match = _NUMBER.search(candidate)
    canonical = _canonical_number(number_match.group(0)) if number_match else None
    status = "ok" if canonical is not None else "parse_failure"
    return ParsedAnswer(text, candidate, canonical, status)


def _normalized_math_string(value: str) -> str:
    value = value.strip().replace("$", "")
    value = re.sub(r"\\left|\\right", "", value)
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = re.sub(r"\s+", "", value)
    return value


def _latex_fraction_to_sympy(value: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = pattern.sub(r"(\1)/(\2)", value)
    return value.replace("^", "**").replace("\\cdot", "*")


def _simplified_string(expression: object) -> str:
    """Recursively simplify container answers without handing them to SymPy."""
    if isinstance(expression, list):
        return "[" + ", ".join(_simplified_string(item) for item in expression) + "]"
    if isinstance(expression, tuple):
        body = ", ".join(_simplified_string(item) for item in expression)
        if len(expression) == 1:
            body += ","
        return f"({body})"
    if isinstance(expression, set):
        return "{" + ", ".join(
            sorted(_simplified_string(item) for item in expression)
        ) + "}"
    if isinstance(expression, dict):
        items = sorted(
            (
                _simplified_string(key),
                _simplified_string(value),
            )
            for key, value in expression.items()
        )
        return "{" + ", ".join(f"{key}: {value}" for key, value in items) + "}"
    return str(sympy.simplify(expression))


def parse_math(text: str, *, gold: bool = False) -> ParsedAnswer:
    candidate = extract_boxed(text)
    if candidate is None and gold:
        candidate = text.strip()
    if candidate is None:
        return ParsedAnswer(text, None, None, "parse_failure")
    normalized = _normalized_math_string(candidate)
    try:
        import antlr4.error.ErrorListener  # noqa: F401
        from sympy.parsing.latex import parse_latex

        expression = parse_latex(candidate)
        canonical = _simplified_string(expression)
        return ParsedAnswer(text, candidate, canonical, "ok_symbolic")
    except Exception:
        # Model answers are untrusted strings. Any parser failure deterministically
        # falls through to the conservative normalised-string representation.
        try:
            expression = sympy.sympify(
                _latex_fraction_to_sympy(normalized)
                .replace("\\pi", "pi")
                .replace("\\sqrt", "sqrt"),
                evaluate=True,
            )
            canonical = _simplified_string(expression)
            return ParsedAnswer(text, candidate, canonical, "ok_symbolic")
        except Exception:
            pass
    if normalized:
        return ParsedAnswer(text, candidate, normalized, "ok_string")
    return ParsedAnswer(text, candidate, None, "parse_failure")


@lru_cache(maxsize=65_536)
def parse_answer(dataset_id: str, text: str, *, gold: bool = False) -> ParsedAnswer:
    if dataset_id in {"gsm8k", "smoke"}:
        return parse_gsm8k(text, gold=gold)
    if dataset_id in {"math500", "math-500"}:
        return parse_math(text, gold=gold)
    raise ValueError(f"unsupported dataset parser: {dataset_id}")


def answers_equal(dataset_id: str, prediction: str, gold: str) -> tuple[bool, ParsedAnswer, ParsedAnswer]:
    parsed_prediction = parse_answer(dataset_id, prediction)
    parsed_gold = parse_answer(dataset_id, gold, gold=True)
    equal = (
        parsed_prediction.canonical is not None
        and parsed_gold.canonical is not None
        and parsed_prediction.canonical == parsed_gold.canonical
    )
    if (
        not equal
        and dataset_id in {"gsm8k", "smoke"}
        and parsed_prediction.canonical is not None
        and parsed_gold.canonical is not None
    ):
        try:
            equal = Fraction(parsed_prediction.canonical) == Fraction(
                parsed_gold.canonical
            )
        except (ValueError, ZeroDivisionError):
            pass
    return equal, parsed_prediction, parsed_gold
