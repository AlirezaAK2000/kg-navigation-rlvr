"""Escaped, dependency-light HTML for inspecting recorded model/tool episodes.

Rendering never samples a model, executes tools, or changes a recorded trace.
"""

import html
import json


STYLE = """<style>
.kg-inspection{font:14px/1.55 system-ui,sans-serif;color:#172b3a;max-width:1100px}
.kg-inspection *{box-sizing:border-box}
.kg-inspection .kg-episode{background:#fff;border:1px solid #ccd6df;border-radius:12px;margin:18px 0;overflow:hidden}
.kg-inspection .kg-header{background:#eef3f7;padding:18px 22px;border-bottom:1px solid #ccd6df}
.kg-inspection h3,.kg-inspection h4{color:#172b3a;margin:0 0 9px}
.kg-inspection .kg-meta{font-size:12px;color:#4a5f70;margin:8px 0}
.kg-inspection .kg-body{padding:18px 22px}
.kg-inspection .kg-block{border-left:4px solid #367db4;background:#f0f7fd;border-radius:5px;padding:12px 16px;margin:10px 0}
.kg-inspection .kg-assistant{border-color:#8962ba;background:#f7f2fc}
.kg-inspection .kg-call{border-color:#cd911d;background:#fff9eb}
.kg-inspection .kg-tool{border-color:#2a8b69;background:#eef9f4}
.kg-inspection .kg-error{border-color:#c24d50;background:#fff1f1}
.kg-inspection .kg-turn{border-top:1px solid #dbe3e9;padding-top:15px;margin-top:20px}
.kg-inspection pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto;font:12px/1.55 ui-monospace,monospace;color:#172b3a;background:transparent;border:0;margin:8px 0 0;padding:0}
.kg-inspection details{margin:10px 0}.kg-inspection summary{cursor:pointer;font-weight:600}
.kg-inspection .kg-badge{display:inline-block;padding:3px 9px;margin:3px 6px 3px 0;background:#e0e8ee;border-radius:20px;font-size:12px}
</style>"""


def _text(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return _text(json.dumps(value, ensure_ascii=False, indent=2))


def render_episode_html(trace, *, label="Model", rendered_prompt=None):
    """Preserve chronological raw assistant text and actual tool observations."""
    blocks = [f'<article class="kg-episode"><header class="kg-header"><h3>{_text(label)} · '
              f'{_text(trace.get("task_id", "Episode"))}</h3><div>{_text(trace.get("question", ""))}</div>']
    metadata = {"Hops": trace.get("hop_count"), "Start": trace.get("start_entity"),
                "Submitted": trace.get("submitted_answer"), "Reward": trace.get("reward"),
                "Answer correct": trace.get("answer_correct"),
                "Gold retrieved": trace.get("gold_entity_retrieved"),
                "Retrieval bonus": trace.get("retrieval_reward"), "Training reward": trace.get("training_reward"),
                "Tool calls": trace.get("tool_calls"), "End": trace.get("termination_reason")}
    blocks.append(''.join(f'<span class="kg-badge">{_text(key)}: {_text(value)}</span>'
                          for key, value in metadata.items() if value is not None))
    blocks.append('</header><div class="kg-body"><h4>Input prompt</h4>')
    for message in trace.get("prompt", []):
        blocks.append(f'<div class="kg-block"><strong>{_text(message["role"]).upper()}</strong>'
                      f'<pre>{_text(message["content"])}</pre></div>')
    if rendered_prompt is not None:
        blocks.append('<details><summary>Full initial Qwen input, including tool schemas and chat tokens</summary>'
                      f'<pre>{_text(rendered_prompt)}</pre></details>')
    for number, turn in enumerate(trace.get("turns", []), 1):
        blocks.append(f'<section class="kg-turn"><h4>Turn {number}</h4>'
                      '<div class="kg-block kg-assistant"><strong>Assistant · raw model output</strong>'
                      f'<div class="kg-meta">Generated tokens: {_text(turn.get("model_tokens", "unavailable"))}</div>'
                      f'<pre>{_text(turn.get("raw_assistant", ""))}</pre></div>')
        if "tool_call" in turn:
            blocks.append('<div class="kg-block kg-call"><strong>Parsed tool call</strong>'
                          f'<pre>{_json(turn["tool_call"])}</pre></div>')
        if "observation" in turn:
            observation = turn["observation"]
            failed = isinstance(observation, dict) and observation.get("ok") is False
            visibility = "Appended to model context" if turn.get("observation_in_context") else "Not appended to model context"
            blocks.append(f'<div class="kg-block kg-tool{" kg-error" if failed else ""}">'
                          f'<strong>Tool response</strong><div class="kg-meta">{visibility}</div>'
                          f'<pre>{_json(observation)}</pre></div>')
        else:
            blocks.append('<div class="kg-meta">No tool response was recorded for this turn.</div>')
        blocks.append('</section>')
    blocks.append('</div></article>')
    return '<div class="kg-inspection">' + ''.join(blocks) + '</div>'


def render_traces_html(traces, *, label="Model", count=3, offset=0, tokenizer=None):
    """Show an unranked slice; optional tokenizer renders the full initial input."""
    if count < 1 or offset < 0:
        raise ValueError("Inspection count must be positive and offset nonnegative")
    selected = traces[offset:offset + count]
    if not selected:
        return STYLE + '<div class="kg-inspection"><p>No recorded model trajectories in this selection. No inference was run by this inspection cell.</p></div>'
    codec = None
    if tokenizer is not None:
        from .rollouts import QwenChatCodec
        codec = QwenChatCodec(tokenizer)
    cards = []
    for trace in selected:
        rendered = codec.decode(codec.prompt_ids(trace["prompt"])) if codec is not None else None
        cards.append(render_episode_html(trace, label=label, rendered_prompt=rendered))
    caption = (f'<div class="kg-inspection"><p>{_text(label)}: recorded episodes '
               f'{offset + 1}–{offset + len(selected)} of {len(traces)}. '
               'Shown in stored order; no success-based selection or extra inference.</p></div>')
    return STYLE + caption + ''.join(cards)
