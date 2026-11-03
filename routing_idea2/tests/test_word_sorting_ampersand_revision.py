"""Software regressions for the pre-E1 word-sorting vocabulary repair."""
import pytest
from budget_router.scoring import SCORER_VERSION, score_answer

@pytest.mark.parametrize("word", ["Acme&Co", "A&B&C", "R&D"])
def test_letter_compound_with_internal_ampersand_is_a_literal_token(word):
    answer = word + " cedar zulu"
    result = score_answer("bbh", "<answer>" + answer + "</answer>", answer,
                          subtask="word_sorting")
    assert result.parsed and result.score == 1
    assert result.prediction == answer.casefold()
    assert result.scorer_version == SCORER_VERSION == "safe_exact_v1_1"

@pytest.mark.parametrize("prediction", [
    "&", "&abc", "abc&", "a&&b", "é&b", "a&é", "a-&b", "a&-b",
    "a & b", "a&b; cedar", "a&b + cedar", "__import__('os').system('noop')",
])
def test_ampersand_does_not_enable_operators_or_nonletter_neighbors(prediction):
    result = score_answer("bbh", prediction, "alpha", subtask="word_sorting")
    assert result.score == 0 and not result.parsed

@pytest.mark.parametrize("task,subtask,prediction,reference", [
    ("gsm8k", None, "Final answer: 0.5", "Reasoning\n#### 1/2"),
    ("bbh", "date_understanding", "(A)", "(A)"),
    ("bbh", "boolean_expressions", "True", "True"),
    ("bbh", "causal_judgement", "yes", "yes"),
    ("bbh", "formal_fallacies", "invalid", "invalid"),
    ("bbh", "multistep_arithmetic_two", "-2", "-2"),
    ("bbh", "object_counting", "3", "3"),
    ("bbh", "dyck_languages", "] } )", "] } )"),
    ("bbh", "word_sorting", "can't co-op zulu", "can't co-op zulu"),
    ("bbh", "word_sorting", "café naïve zulu", "café naïve zulu"),
    ("bbh", "word_sorting", "_var co_op zulu", "_var co_op zulu"),
])
def test_existing_supported_syntax_keeps_its_result(task, subtask, prediction, reference):
    result = score_answer(task, prediction, reference, subtask=subtask)
    assert result.parsed and result.score == 1


def test_explicit_final_marker_still_controls_extraction():
    result = score_answer("bbh", "Analysis: not the answer. Final answer: Acme&Co cedar", 
                          "Acme&Co cedar", subtask="word_sorting")
    assert result.parsed and result.score == 1


def test_bad_word_sorting_reference_still_stops():
    with pytest.raises(ValueError, match="Invalid reference"):
        score_answer("bbh", "alpha", "a & b", subtask="word_sorting")


def test_ampersand_repair_is_not_applied_to_numeric_tasks():
    with pytest.raises(ValueError, match="Invalid reference"):
        score_answer("bbh", "A&B", "A&B", subtask="object_counting")
