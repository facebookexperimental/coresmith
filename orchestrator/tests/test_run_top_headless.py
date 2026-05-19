from scripts.run_top_headless import _answer_prd_questions


def test_headless_prd_answers_do_not_invent_area_or_power_budgets():
    state = {
        "ask_question": {
            "questions": [
                {
                    "id": "area_gate_budget",
                    "category": "area",
                    "question": "What is the maximum gate count?",
                    "options": ["Under 25k gates", "Under 50k gates", "No explicit gate budget"],
                },
                {
                    "id": "power_total_budget",
                    "category": "power",
                    "question": "What is the active power budget?",
                    "options": ["Under 1 mW", "Under 5 mW", "No explicit active power constraint"],
                },
                {
                    "id": "cycles_per_macroblock_budget",
                    "category": "speed_and_feeds",
                    "question": "Maximum cycles per 8x8 macroblock?",
                    "options": [
                        "One macroblock accepted every cycle after pipeline fill",
                        "Derived from frame-rate target",
                    ],
                },
                {
                    "id": "frame_rate",
                    "category": "speed_and_feeds",
                    "question": "What frame rate is required?",
                    "options": ["1 fps", "10 fps", "No real-time requirement"],
                },
            ]
        }
    }

    answers = _answer_prd_questions(state, "H.264 codec with Mort GIF PSNR and bpp KPIs")

    assert answers["area_gate_budget"] == "No explicit gate budget"
    assert answers["power_total_budget"] == "No explicit active power constraint"
    assert answers["cycles_per_macroblock_budget"] == "Derived from frame-rate target"
    assert answers["frame_rate"] == "No real-time requirement"


def test_headless_prd_answers_ignore_empty_suggestions():
    state = {
        "ask_question": {
            "auto_answerable": [
                {"id": "area_gate_budget", "suggested_answer": ""},
                {"id": "power_total_budget", "suggested_answer": "No explicit active power constraint"},
            ]
        }
    }

    answers = _answer_prd_questions(state, "H.264 codec with Mort GIF PSNR and bpp KPIs")

    assert "area_gate_budget" not in answers
    assert answers["power_total_budget"] == "No explicit active power constraint"
