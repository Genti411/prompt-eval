"""Evaluate the prompt variants against a real LLM and print the ranking.

  LLM=ollama OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_MODEL=llama3.1 python run_eval.py
"""
import sys

from evals.task import CASES, LABELS, VARIANTS
from prompteval.llm import from_env
from prompteval.optimizer import compare
from prompteval.scorers import label_match


def main() -> int:
    llm = from_env()
    scorer = label_match(LABELS)
    ranking = compare(VARIANTS, CASES, llm, scorer)
    print(f"{'version':16} {'score':>6}")
    for e in ranking:
        print(f"{e['version']:16} {e['score']:>6.3f}")
    print(f"\nBest prompt: {ranking[0]['version']} (score {ranking[0]['score']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
