# Tutorial refactor verification — 2026-10-27

The tutorial retains SFT, GRPO only and SFT + GRPO. This refactor removes generation
utilities and runtime audit gates while keeping tool contracts, stage separation,
reward semantics and observation-token masking.

| Check | Result |
| --- | --- |
| Focused tests | 22 passed in the training API environment; the additional SFT optimizer test also passed |
| Real TRL integration | Mixed rewards produce gradients and adapter changes; homogeneous rewards complete without a gate |
| SFT integration | A tiny CPU adapter takes an optimizer step with assistant-only labels |
| Native Qwen tokenizer | Executed graph traversal retains sampled action tokens and masks tool observations |
| Notebook 1 | 8 code cells, no errors, 4 inline presentation figures |
| Notebook 2 | 26 code cells execute in CPU mode with no errors; model execution skipped |
| Original Notebook 2 plots | All 7 embedded PNGs and their outputs preserved exactly; plotting code unchanged |
| Experiment fidelity | Selected SFT/GRPO/test IDs, all selected prompts, 100 reference demonstrations and the 20-group schedule match the pre-refactor implementation exactly |
| Saved data | `graph.ttl` and original train/test JSONL match the committed bytes |
| Figure inspection | Ontology routing, labels, counts and statistical plots visually inspected |

The API tests used Python 3.11, TRL 1.8.0, Transformers 5.3.0, PEFT 0.18.1,
Accelerate 1.10.1 and Torch 2.14.0 on CPU. They use a tiny random architecture and
the pinned Qwen tokenizer, not pretrained GPU training. The ordinary environment
passed 19 tests with 3 optional API tests skipped before the additional SFT test.

Notebook 2 preserves the user's completed Kaggle experiment (Torch 2.6.0/cu124,
T4, 100 SFT examples and 20 steps per RL variant). Those recorded results remain
historical outputs, not a fresh execution of the refactored GPU path. No new
pretrained CUDA/NF4 experiment is claimed. Earlier reports describe earlier code.
