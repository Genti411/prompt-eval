"""Compare prompt variants and search for a better one.

`compare` ranks candidate prompts by eval score (A/B/n testing). `generate_variants`
applies common prompt-engineering transformations to a base prompt so the
optimizer can search a small space automatically.
"""
from __future__ import annotations

from .runner import EvalCase, evaluate
from .template import PromptTemplate


def compare(templates: list[PromptTemplate], cases: list[EvalCase], llm, scorer) -> list[dict]:
    evals = [evaluate(t, cases, llm, scorer) for t in templates]
    return sorted(evals, key=lambda e: e["score"], reverse=True)


def best(templates: list[PromptTemplate], cases: list[EvalCase], llm, scorer) -> dict:
    return compare(templates, cases, llm, scorer)[0]


def generate_variants(base: PromptTemplate) -> list[PromptTemplate]:
    """A few standard prompt-engineering tweaks to try automatically."""
    return [
        base,
        PromptTemplate(base.id, base.template + "\nAnswer with only the single best label.",
                       base.version + "+format"),
        PromptTemplate(base.id, base.template + "\nThink step by step, then give the label.",
                       base.version + "+cot"),
    ]


def optimize(base: PromptTemplate, cases: list[EvalCase], llm, scorer) -> dict:
    """Generate variants of `base`, evaluate all, and return the best + the ranking."""
    ranking = compare(generate_variants(base), cases, llm, scorer)
    return {"best": ranking[0], "ranking": ranking}
