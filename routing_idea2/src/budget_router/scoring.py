"""Deterministic, fail-closed answer scoring; never executes generated text.

The v1 extraction policy is an explicit engineering choice, not the original
benchmark's entire evaluator. Pin this policy before comparing strategies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
import unicodedata
from collections.abc import Mapping, Sequence

SCORER_VERSION = "safe_exact_v1_1"
CHOICE_TASKS = frozenset({
    "date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton",
    "logical_deduction_three_objects", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "movie_recommendation", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
    "snarks", "temporal_sequences", "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
})
EXACT_VOCABULARIES = {
    "boolean_expressions": frozenset({"true", "false"}),
    "causal_judgement": frozenset({"yes", "no"}),
    "formal_fallacies": frozenset({"valid", "invalid"}),
    "navigate": frozenset({"yes", "no"}),
    "sports_understanding": frozenset({"yes", "no"}),
    "web_of_lies": frozenset({"yes", "no"}),
}
NUMERIC_TASKS = frozenset({"multistep_arithmetic_two", "object_counting"})
BBH_TASKS = CHOICE_TASKS | EXACT_VOCABULARIES.keys() | NUMERIC_TASKS | {"dyck_languages", "word_sorting"}
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FINAL_MARKER = re.compile(
    r"(?:\bfinal\s+answer\s*(?:is\b|[:=])|\bthe\s+answer\s+is\b|\banswer\s*[:=])\s*",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class ScoreResult:
    score: float
    prediction: str | None
    reference: str
    parsed: bool
    reason: str
    scorer_version: str = SCORER_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Answers must be text; convert cache schema explicitly.")
    if len(value) > 200_000:
        raise ValueError("Answer exceeds the pinned scoring input limit.")
    value = unicodedata.normalize("NFKC", value).replace("−", "-").strip()
    for marker in ("<|eot_id|>", "<|endoftext|>", "</s>"):
        value = value.replace(marker, "")
    return value.strip()


def _boxed(value: str) -> list[str]:
    values = []
    for hit in re.finditer(r"\\boxed\s*\{", value):
        start = hit.end()
        depth = 1
        for pos in range(start, len(value)):
            if value[pos] == "{": depth += 1
            elif value[pos] == "}": depth -= 1
            if depth == 0:
                values.append(value[start:pos].strip())
                break
    return values


def _candidate(value: str) -> tuple[str, bool]:
    value = _clean(value)
    if "####" in value:
        tail = value.rsplit("####", 1)[1].strip()
        return (tail.splitlines()[0].strip() if tail else ""), True
    xml = re.findall(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.I | re.S)
    if xml:
        return xml[-1].strip(), True
    hits = list(_FINAL_MARKER.finditer(value))
    if hits:
        tail = value[hits[-1].end():].strip()
        boxes = _boxed(tail)
        return (boxes[-1] if boxes else tail.splitlines()[0].strip() if tail else ""), True
    boxes = _boxed(value)
    if boxes:
        return boxes[-1], True
    return value, False


def _trim(value: str) -> str:
    value = value.strip().strip("`\"'")
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value


def _numeric(value: str) -> str | None:
    value = _trim(value).replace("\\,", "").replace("\\!", "")
    # Safe literal fractions only: expressions such as 2+2 are never evaluated.
    frac = re.fullmatch(r"\\(?:d?frac)\{(" + _NUMBER + r")\}\{(" + _NUMBER + r")\}", value)
    if frac:
        value = frac[1] + "/" + frac[2]
    value = re.sub(r"^[$£€]\s*", "", value).removesuffix("%").strip()
    value = value.removesuffix(".").strip()
    if len(value) > 256:
        return None
    literal = re.fullmatch(r"(" + _NUMBER + r")(?:\s*/\s*(" + _NUMBER + r"))?", value)
    if not literal:
        return None
    try:
        numerator = Decimal(literal[1].replace(",", ""))
        denominator = Decimal(literal[2].replace(",", "")) if literal[2] else Decimal(1)
        if not numerator.is_finite() or not denominator.is_finite() or denominator == 0:
            return None
        # Bound exponents before allocating arbitrarily large Fraction integers.
        if abs(numerator.adjusted()) > 1000 or abs(denominator.adjusted()) > 1000:
            return None
        number = Fraction(numerator) / Fraction(denominator)
        return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None


def _normal_text(value: str) -> str:
    return " ".join(_trim(value).removesuffix(".").split()).casefold()


def _choice(value: str, choices: Mapping[str, str] | Sequence[str] | None) -> str | None:
    value = _trim(value).strip()
    allowed = {chr(65 + i) for i in range(26)} if choices is None else (
        {str(k).strip("()").upper() for k in choices} if isinstance(choices, Mapping)
        else {chr(65+i) for i in range(len(choices))}
    )
    # Entire literal label, or a label followed by a punctuated explanation.
    match = re.fullmatch(r"(?:option\s+|choice\s+)?\(?([A-Za-z])\)?(?:[.):]\s*.*|\s*)", value, re.S | re.I)
    if match and match[1].upper() in allowed:
        return match[1].upper()
    if isinstance(choices, Mapping):
        matches = [str(k).strip("()").upper() for k, text in choices.items() if _normal_text(value) == _normal_text(text)]
        if len(matches) == 1:
            return matches[0]
    return None


def _canonical(task: str, value: str, choices=None) -> str | None:
    if task == "gsm8k" or task in NUMERIC_TASKS:
        return _numeric(value)
    if task in CHOICE_TASKS:
        return _choice(value, choices)
    if task in EXACT_VOCABULARIES:
        value = _normal_text(value)
        return value if value in EXACT_VOCABULARIES[task] else None
    if task == "dyck_languages":
        value = re.sub(r"\s+", "", _trim(value))
        return value if value and re.fullmatch(r"[()\[\]{}<>]+", value) else None
    if task == "word_sorting":
        value = _normal_text(value)
        # Preserve legacy tokens; additionally accept ASCII letter compounds
        # such as the official word-sorting vocabulary token form A&B.
        token = r"(?:[\w'\-]+|[A-Za-z]+(?:&[A-Za-z]+)+)"
        return value if value and re.fullmatch(token + r"(?:\s+" + token + r")*", value, re.UNICODE) else None
    raise ValueError(f"Unsupported task/subtask: {task!r}; no guessed evaluator.")


def score_answer(task: str, prediction: str, reference: str, *, subtask: str | None = None,
                 choices: Mapping[str, str] | Sequence[str] | None = None) -> ScoreResult:
    """Score a frozen output, with an explicit BBH subtask and safe exact rules.

    Free-form CoT is not searched for its last number or a stray answer letter.
    Require an explicit final marker/boxed answer or a literal final answer.
    Invalid predictions score zero; invalid references stop the run.
    """
    task = task.casefold()
    if task == "bbh":
        if subtask is None:
            raise ValueError("BBH requires a pinned subtask evaluator.")
        evaluator = subtask.casefold()
        if evaluator not in BBH_TASKS:
            raise ValueError(f"Unknown BBH subtask: {subtask!r}")
    elif task == "gsm8k":
        evaluator = task
    else:
        raise ValueError(f"Task {task!r} is outside the v1 locked task family.")
    pred_value, explicit = _candidate(prediction)
    ref_value, _ = _candidate(reference)
    gold = _canonical(evaluator, ref_value, choices)
    if gold is None:
        raise ValueError("Invalid reference answer; audit cache/target mapping before scoring.")
    parsed = _canonical(evaluator, pred_value, choices)
    if parsed is None:
        return ScoreResult(0.0, None, gold, False, "invalid_or_no_explicit_final_answer")
    return ScoreResult(float(parsed == gold), parsed, gold, True,
                       "explicit_final" if explicit else "literal_answer")
