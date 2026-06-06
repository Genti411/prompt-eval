"""A concrete task to evaluate prompts on: support-ticket classification.

Three prompt variants of increasing quality (vague → explicit → few-shot) so the
framework can show measurable differences between them.
"""
from prompteval.runner import EvalCase
from prompteval.template import PromptTemplate

LABELS = ["billing", "technical", "account"]

CASES = [
    EvalCase({"text": "I was charged twice on my invoice this month"}, "billing"),
    EvalCase({"text": "The app crashes with an error when I upload a file"}, "technical"),
    EvalCase({"text": "I can't log in and my password reset isn't working"}, "account"),
    EvalCase({"text": "Please refund the duplicate billing charge"}, "billing"),
    EvalCase({"text": "There is a bug in the export feature"}, "technical"),
    EvalCase({"text": "How do I update the email on my account?"}, "account"),
]

VARIANTS = [
    PromptTemplate("ticket", "Classify: {text}", "v1-vague"),
    PromptTemplate(
        "ticket",
        "Classify the support ticket into exactly one of: billing, technical, account.\n"
        "Ticket: {text}\nLabel:",
        "v2-explicit"),
    PromptTemplate(
        "ticket",
        "Classify the support ticket into one of: billing, technical, account.\n"
        "Examples:\n  'I was overcharged' -> billing\n  'the page errors out' -> technical\n"
        "  'reset my login' -> account\nTicket: {text}\nLabel:",
        "v3-fewshot"),
]
