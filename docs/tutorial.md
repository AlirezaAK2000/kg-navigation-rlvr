# Code map for the tutorial

The training notebook compares vanilla, SFT, SFT + GRPO and GRPO only. All four
use the same saved graph, prompt, tool limits and fixed test questions.

| File | Responsibility |
| --- | --- |
| `rdf.py`, `graph_index.py` | Read the saved ontology and navigate asserted RDF facts |
| `data.py` | Load questions, select single-answer instances and build public prompts |
| `training/environment.py` | Maintain discovery, execute tools and end episodes |
| `training/rewards.py` | Score executed submissions and successful retrievals |
| `training/sampling.py` | Reserve separate SFT/GRPO instances and schedule hop exposure |
| `training/sft.py` | Execute reference demonstrations and mask non-assistant labels |
| `training/rollouts.py` | Generate tool calls, append observations and preserve token masks |
| `training/grpo.py` | Load QLoRA and delegate optimization to TRL |
| `training/experiments.py` | Run a training phase and persist diagnostics |
| `training/artifacts.py` | Save per-variant results, trajectories and ZIP exports |
| `training/presentation.py` | Load saved records for the notebook's existing slide plots |
| `eda.py` | Create dataset and ontology presentation figures |

The model receives question text and a starting entity, never the reference
program or gold answer. One assistant turn emits one tool call. Runtime checks for
IDs, arguments, relation types and budgets define the task; dataset-generation
validators and optimizer-verification gates have been removed.

The reward is correct submission plus a once-per-episode retrieval bonus of 0.2.
Standard TRL reward callables use private dataset columns and trusted rollout
metadata. Observation tokens are masked out of the policy loss. FP32 adapter
restoration after the TRL constructor preserves the T4 warm start.

Run Notebook 1 for read-only statistics and slide exports. Run Notebook 2 on Kaggle
for the experiment. Recorded figures in Notebook 2 remain the original outputs.
