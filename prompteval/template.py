"""Versioned prompt templates.

A prompt is treated as a versioned artifact with named variables, so you can
diff, test, and compare prompts like code instead of editing strings by hand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    id: str
    template: str
    version: str = "v1"

    def render(self, **variables) -> str:
        out = self.template
        for key, value in variables.items():
            out = out.replace("{" + key + "}", str(value))
        return out

    def variables(self) -> set[str]:
        return set(re.findall(r"\{(\w+)\}", self.template))
