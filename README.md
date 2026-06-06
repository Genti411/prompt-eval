# Prompt Engineering & Evaluation Framework

Treat prompts as **versioned, measured artifacts** instead of strings you tweak by
feel. This framework defines prompt templates, runs them over a labeled eval set,
**scores** the outputs, **A/B-compares** variants, and **optimizes** a base prompt
by searching standard transformations. The core is provider-agnostic and
unit-tested with a mock LLM, so it's verifiable with no API cost.

| Area | What's shown |
|------|--------------|
| **Prompt engineering** | versioned templates, variant comparison, automatic optimization |
| **LLM evaluation** | labeled dataset, scorers (exact / contains / regex / classification), aggregate scores |
| **Eval-driven workflow** | measure a prompt change instead of guessing whether it helped |
| **LLMOps** | pluggable backends (OpenAI / Ollama / Anthropic); reproducible runs |

## The idea

```
prompt variants ─┐
                 ├─▶ run over eval dataset ─▶ score each output ─▶ ranked results
eval dataset ────┘                                                 ▶ pick / optimize best
```

The bundled task is support-ticket classification with three prompt variants
(vague → explicit → few-shot); the framework shows the explicit/few-shot prompts
score measurably higher than the vague one — the whole point of evaluating
prompts rather than eyeballing them.

## Run against a real model

```bash
docker build -t prompt-eval .
docker run --rm --network host \
  -e LLM=ollama -e OPENAI_BASE_URL=http://localhost:11434/v1 -e OPENAI_MODEL=llama3.1 \
  prompt-eval                 # prints each prompt version and its score, best first
```

Or `-e LLM=anthropic -e ANTHROPIC_API_KEY=...` / `-e LLM=openai -e OPENAI_API_KEY=...`.

## Use it on your own prompts

```python
from prompteval.template import PromptTemplate
from prompteval.runner import EvalCase
from prompteval.optimizer import compare, optimize
from prompteval.scorers import label_match

variants = [PromptTemplate("t", "...{x}...", "v1"), PromptTemplate("t", "...{x}...", "v2")]
cases = [EvalCase({"x": "..."}, expected="...")]
ranking = compare(variants, cases, llm, label_match(["a", "b"]))   # A/B
best = optimize(variants[0], cases, llm, label_match(["a", "b"]))   # auto-search
```

## Tests (no API key)

```bash
pip install -r requirements.txt pytest && python -m pytest
```

A mock LLM simulates how prompt quality changes outputs, so the tests prove the
framework actually *ranks better prompts higher* and the optimizer/scoring work.

## Layout

```
prompteval/template.py   versioned prompt templates
prompteval/scorers.py    exact / contains / regex / classification scorers
prompteval/runner.py     evaluate a prompt over a dataset
prompteval/optimizer.py  compare variants + auto-generate/search
prompteval/llm.py        Mock / OpenAI-compatible / Anthropic backends
evals/task.py            sample task + prompt variants
run_eval.py              rank the variants against a real model
tests/                   framework tests with a mock LLM
```
