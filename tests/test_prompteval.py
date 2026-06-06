import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.task import CASES, LABELS, VARIANTS
from prompteval import scorers
from prompteval.llm import MockLLM
from prompteval.optimizer import compare, optimize
from prompteval.runner import evaluate
from prompteval.template import PromptTemplate


def test_template_render_and_vars():
    t = PromptTemplate("x", "Hello {name}, you are {role}")
    assert t.render(name="Al", role="dev") == "Hello Al, you are dev"
    assert t.variables() == {"name", "role"}


def test_scorers():
    assert scorers.exact_match("Billing", "billing") == 1.0
    assert scorers.contains("the answer is technical", "technical") == 1.0
    lm = scorers.label_match(LABELS)
    assert lm("billing", "billing") == 1.0
    # first label wins: output lists technical first -> not billing
    assert lm("technical, maybe billing", "billing") == 0.0
    assert lm("technical, maybe billing", "technical") == 1.0


# Simulates how prompt quality affects an LLM: when the prompt explicitly lists
# the labels, the model classifies correctly from keywords; a vague prompt rambles.
def _responder(messages):
    p = messages[-1]["content"].lower()
    import re as _re
    explicit = "billing, technical, account" in p
    # Look only at the actual ticket text (not label names or few-shot examples).
    if "ticket:" in p:
        ticket = _re.split(r"label:", p[p.rfind("ticket:") + 7:], flags=_re.I)[0]
    elif "classify:" in p:
        ticket = p[p.rfind("classify:") + 9:]
    else:
        ticket = p
    if not explicit:
        return "this could be billing, technical, or account"
    if any(w in ticket for w in ["charge", "invoice", "refund", "overcharged"]):
        return "billing"
    if any(w in ticket for w in ["crash", "error", "bug", "export"]):
        return "technical"
    if any(w in ticket for w in ["log in", "login", "password", "reset", "email on my"]):
        return "account"
    return "unknown"


def test_runner_aggregates_score():
    out = evaluate(VARIANTS[1], CASES, MockLLM(_responder), scorers.label_match(LABELS))
    assert out["n"] == 6 and out["score"] == 1.0          # explicit prompt -> all correct


def test_compare_ranks_better_prompt_first():
    ranking = compare(VARIANTS, CASES, MockLLM(_responder), scorers.label_match(LABELS))
    assert ranking[0]["version"] in ("v2-explicit", "v3-fewshot")
    assert ranking[0]["score"] == 1.0
    assert ranking[-1]["version"] == "v1-vague"
    assert ranking[-1]["score"] < 0.5                     # vague prompt is measurably worse


def test_optimize_returns_best_and_ranking():
    res = optimize(VARIANTS[1], CASES, MockLLM(_responder), scorers.label_match(LABELS))
    assert res["best"]["score"] == 1.0
    assert len(res["ranking"]) == 3                       # base + 2 generated variants
