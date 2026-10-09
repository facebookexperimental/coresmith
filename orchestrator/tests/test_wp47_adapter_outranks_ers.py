"""WP-47: the task adapter outranks internal ERS/validation requirements."""
from orchestrator.langgraph import pipeline_graph as pg


def test_budget_guidance_names_the_ers():
    src = open(pg.__file__, encoding="utf-8").read()
    assert "THAT " in src and "requirement is wrong: revise it" in src
