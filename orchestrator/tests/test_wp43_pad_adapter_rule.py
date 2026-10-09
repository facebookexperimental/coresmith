"""WP-43: the prompts agree that the engine assembles the Caravel chip top; the
user_project_wrapper block is a pad adapter and is never asked to instantiate siblings."""
from pathlib import Path

from orchestrator.langgraph import pipeline_graph as pg

P = Path(pg.__file__).resolve().parent.parent / "langchain" / "prompts"


def test_integration_review_excludes_wrapper_instantiation_from_issues():
    s = (P / "integration_review.md").read_text()
    assert "Pad-adapter scope" in s and "are therefore NOT issues" in s


def test_uarch_spec_prompt_specifies_a_pad_adapter():
    s = (P / "uarch_spec_generator.md").read_text()
    assert "Pad-adapter block (locked chip boundary)" in s and "instantiates no other block" in s
