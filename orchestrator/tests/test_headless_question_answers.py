import importlib.util
from pathlib import Path


def _load_runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_top_headless.py"
    spec = importlib.util.spec_from_file_location("run_top_headless_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_question_answer_ids_include_remaining_choice_questions():
    runner = _load_runner_module()
    payload = {
        "ask_question": {
            "questions": [{"id": "target_pdk_selection"}],
            "remaining_choice_questions": [{"id": "frame_rate_target"}],
            "auto_answerable": [{"id": "sky130_library_flavor"}],
        }
    }

    assert runner._question_answer_ids_from_escalation(payload) == [
        "target_pdk_selection",
        "frame_rate_target",
        "sky130_library_flavor",
    ]


def test_question_answer_validation_rejects_partial_answer_files():
    runner = _load_runner_module()
    payload = {
        "ask_question": {
            "questions": [{"id": "target_pdk_selection"}],
            "remaining_choice_questions": [{"id": "frame_rate_target"}],
        }
    }

    missing, extras = runner._validate_question_answers(
        {"target_pdk": "legacy sky130 answer"},
        payload,
    )

    assert missing == ["target_pdk_selection", "frame_rate_target"]
    assert extras == ["target_pdk"]


def test_normalize_answer_payload_accepts_wrapped_answers():
    runner = _load_runner_module()

    assert runner._normalize_answer_payload({"answers": {"q1": "a1"}}) == {"q1": "a1"}
