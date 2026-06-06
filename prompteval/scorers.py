"""Scorers grade a model output against the expected answer (0.0–1.0).

Choosing the right scorer is half of prompt evaluation: exact match is strict,
`contains` is lenient, and `label_match` is designed for classification (the
expected label must be the one the model actually picked).
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return s.strip().lower()


def exact_match(output: str, expected: str) -> float:
    return 1.0 if _norm(output) == _norm(expected) else 0.0


def contains(output: str, expected: str) -> float:
    return 1.0 if _norm(expected) in _norm(output) else 0.0


def regex_match(output: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, output, re.I) else 0.0


def label_match(labels: list[str]):
    """Build a classification scorer over a fixed label set: correct iff the
    expected label is the FIRST label that appears in the output (so a model that
    lists every label doesn't get credit)."""
    low = [l.lower() for l in labels]

    def score(output: str, expected: str) -> float:
        o = _norm(output)
        positions = [(o.find(l), l) for l in low if l in o]
        if not positions:
            return 0.0
        first = min(positions)[1]
        return 1.0 if first == _norm(expected) else 0.0

    return score
