from orchestrator.langgraph.architecture_graph import (
    _feedback_revision_target,
    route_after_final_review,
)


def test_final_review_feedback_about_prd_routes_to_gather_requirements():
    assert _feedback_revision_target("PRD says 30 fps but manual answer was 1 fps") == (
        "Gather Requirements"
    )
    assert route_after_final_review({
        "human_response": {
            "action": "feedback",
            "feedback": "PRD has not answered sizing questions",
        }
    }) == "Gather Requirements"


def test_final_review_feedback_about_frd_routes_to_functional_requirements():
    assert _feedback_revision_target("FRD PSNR KPI is missing") == "Functional Requirements"


def test_final_review_feedback_defaults_to_block_diagram():
    assert _feedback_revision_target("Split the predictor into two blocks") == "Block Diagram"
