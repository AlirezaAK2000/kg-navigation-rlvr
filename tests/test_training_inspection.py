"""Renderer unit fixtures; these are not model-generated episodes."""

import copy

import pytest

from kg_rlvr.training.inspection import render_episode_html, render_traces_html


def trace(task_id="unit_1", reward=0):
    return {
        "task_id": task_id,
        "question": "Which city?",
        "hop_count": 2,
        "start_entity": "r",
        "submitted_answer": "c",
        "reward": reward,
        "tool_calls": 2,
        "termination_reason": "submitted",
        "prompt": [
            {"role": "system", "content": "unit instructions"},
            {"role": "user", "content": "unit question"},
        ],
        "turns": [
            {
                "raw_assistant": '<tool_call>{"name":"get_neighbors"}</tool_call>',
                "model_tokens": 12,
                "tool_call": {"name": "get_neighbors", "arguments": {"entity_id": "r"}},
                "observation": {"ok": True, "entities": [{"id": "c"}]},
                "observation_in_context": True,
            },
            {
                "raw_assistant": '<tool_call>{"name":"submit_answer"}</tool_call>',
                "model_tokens": 9,
                "tool_call": {"name": "submit_answer", "arguments": {"entity_id": "c"}},
                "observation": {"ok": True, "submitted": "c"},
                "observation_in_context": False,
            },
        ],
    }


def test_inspection_preserves_turn_order_and_escapes_all_untrusted_text():
    episode = trace()
    injection = '<script>alert("unit")</script>'
    episode["question"] = injection
    episode["prompt"][1]["content"] = injection
    episode["turns"][0]["observation"]["entities"][0]["id"] = injection
    original = copy.deepcopy(episode)
    result = render_episode_html(episode, label=injection, rendered_prompt=injection)
    assert "<script>" not in result and "&lt;script&gt;" in result
    assert "&lt;tool_call&gt;" in result  # visible raw output, not interpreted HTML
    assert result.index("Input prompt") < result.index("Turn 1") < result.index("Turn 2")
    assert result.index("Parsed tool call") < result.index("Tool response")
    assert "Appended to model context" in result and "Not appended to model context" in result
    assert "including tool schemas and chat tokens" in result
    assert episode == original


def test_inspection_uses_stored_slice_and_handles_empty_or_incomplete_episodes():
    episodes = [trace("first_failure", 0), trace("second_success", 1), trace("third_failure", 0)]
    result = render_traces_html(episodes, count=1, offset=0)
    assert "first_failure" in result and "second_success" not in result
    result = render_traces_html(episodes, count=1, offset=1)
    assert "second_success" in result and "third_failure" not in result
    assert "No recorded model trajectories" in render_traces_html([], tokenizer=object())
    malformed = trace()
    malformed["turns"] = [{"raw_assistant": "unparsed model output", "model_tokens": 3}]
    assert "No tool response was recorded" in render_episode_html(malformed)
    with pytest.raises(ValueError):
        render_traces_html(episodes, count=0)


def test_tool_error_response_is_visibly_marked():
    episode = trace()
    episode["turns"][0]["observation"] = {"ok": False, "error": "Unknown relation"}
    result = render_episode_html(episode)
    assert 'class="kg-block kg-tool kg-error"' in result
    assert "Unknown relation" in result
