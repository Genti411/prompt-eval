"""Evaluate a prompt template over a dataset and aggregate the score."""
from __future__ import annotations

from dataclasses import dataclass

from .template import PromptTemplate


@dataclass
class EvalCase:
    inputs: dict
    expected: str


def evaluate(template: PromptTemplate, cases: list[EvalCase], llm, scorer) -> dict:
    results = []
    for case in cases:
        prompt = template.render(**case.inputs)
        output = llm.complete([{"role": "user", "content": prompt}])
        results.append({
            "inputs": case.inputs,
            "output": output,
            "expected": case.expected,
            "score": scorer(output, case.expected),
        })
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "template_id": template.id,
        "version": template.version,
        "score": round(avg, 3),
        "n": len(results),
        "results": results,
    }
