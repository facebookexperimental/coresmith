from orchestrator.architecture.specialists.prd_spec import _build_answers_context


def test_prd_answer_context_preserves_unmatched_answers():
    context = _build_answers_context(
        {
            "frame_rate_target": "1 fps sustained",
            "target_frame_rate": "Legacy answer: 1 fps, not 30 fps",
            "architect_final_review_feedback": "Fix PRD contradictions before OK2DEV.",
        },
        previous_questions=[
            {
                "id": "frame_rate_target",
                "question": "What sustained frame rate is required?",
            }
        ],
    )

    assert "frame_rate_target" in context
    assert "1 fps sustained" in context
    assert "ADDITIONAL ARCHITECT ANSWERS" in context
    assert "target_frame_rate: Legacy answer: 1 fps, not 30 fps" in context
    assert "architect_final_review_feedback: Fix PRD contradictions before OK2DEV." in context


def test_prd_answer_context_marks_missing_asked_answers():
    context = _build_answers_context(
        {"target_pdk": "sky130"},
        previous_questions=[
            {"id": "frame_rate_target", "question": "What frame rate?"},
        ],
    )

    assert "frame_rate_target: What frame rate?" in context
    assert "Answer: (not answered)" in context
    assert "target_pdk: sky130" in context
