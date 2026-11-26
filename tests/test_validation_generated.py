import pytest

from production.validation_generated import code_completion, numeric_answer, qa_scores, score_rows


def test_numeric_final_answer_and_multilingual_qa():
    assert numeric_answer("There were 40. Final answer: \\boxed{1,200}") == 1200
    assert numeric_answer("first 200; #### 3/2") == numeric_answer("1.5")
    assert numeric_answer("#### -.5") == numeric_answer("-0.5")
    assert numeric_answer("#### 1.2e3") == 1200
    assert numeric_answer("no answer") is None
    assert numeric_answer("#### 1/0") is None
    assert qa_scores("नई दिल्ली।", ["नई दिल्ली", "दिल्ली"])["exact_match"] == 1
    assert qa_scores("wrong", ["नई दिल्ली"])["token_f1"] == 0
    assert code_completion("    return x\n") == "    return x\n"
    assert code_completion("```python\n    return x\n```\ncomment") == "    return x\n"


def test_coverage_cannot_hide_missing_tasks():
    with pytest.raises(ValueError, match="no items"):
        score_rows([], [])
    items = [{"id": "math-1", "kind": "math", "answers": ["42"], "task": "math",
              "language": "en", "role": "development"}]
    result = score_rows(items, [{"id": "math-1", "completion": "#### 42"}])
    assert result["summary"]["math/en"]["exact_match"] == 1
    with pytest.raises(ValueError, match="coverage"):
        score_rows(items, [])
    with pytest.raises(ValueError, match="duplicate"):
        score_rows(items, [{"id": "math-1", "completion": "42"}] * 2)
    code = [{"id": "code-1", "kind": "code", "task": "code", "language": "en",
             "role": "development", "input": "def f():\n", "test": "assert f() == 1"}]
    with pytest.raises(ValueError, match="explicit execution"):
        score_rows(code, [{"id": "code-1", "completion": "    return 1"}])
