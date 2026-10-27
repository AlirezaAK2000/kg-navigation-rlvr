"""Run independently initialized RL variants and save phase diagnostics."""

from pathlib import Path

from .artifacts import save_json, persist_rl_diagnostics
from .grpo import build_trainer, memory_snapshot


def release_trainer(trainer):
    """Release this phase's optimizer and gradients while retaining its policy."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.accelerator.free_memory()
    model.zero_grad(set_to_none=True)
    return model


def run_rl_phase(
    model,
    tokenizer,
    store,
    task_ids,
    limits,
    *,
    variant_dir,
    checkpoint_root,
    phase,
    steps,
    manifest,
    generations,
    accumulation,
    learning_rate,
    save_steps,
    seed,
    test_ids,
    eval_every,
    show_progress,
    hop_weights,
    excluded_sft_task_ids,
    resume_from_checkpoint=None,
):
    """Persist training evidence in finally, before returning or raising.

    Both RL policies call this same function with the same schedule and budgets.
    Their model initializations, manifests and output directories are separate.
    """
    variant_dir, checkpoint_root = Path(variant_dir), Path(checkpoint_root)
    directory = checkpoint_root / phase
    diagnostics_dir = variant_dir / "training" / phase
    resume = str(resume_from_checkpoint) if resume_from_checkpoint else None
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Training output already exists; use a fresh RUN_NAME or compatible resume checkpoint"
        )
    save_json(directory / "run_manifest.json", manifest)
    save_json(diagnostics_dir / "run_manifest.json", manifest)
    trainer = None
    status = {
        "variant": variant_dir.name,
        "phase": phase,
        "status": "training",
        "evaluation_complete": False,
    }
    save_json(variant_dir / "status.json", status)
    primary_error = None
    try:
        trainer, _ = build_trainer(
            model,
            tokenizer,
            store,
            task_ids,
            limits,
            directory,
            phase,
            steps,
            generations,
            accumulation,
            learning_rate,
            save_steps,
            seed,
            eval_task_ids=test_ids,
            eval_every=eval_every,
            show_progress=show_progress,
            hop_weights=hop_weights,
            excluded_sft_task_ids=excluded_sft_task_ids,
            diagnostics_dir=diagnostics_dir,
        )
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=resume)
        measured = {
            "global_step": trainer.state.global_step,
            "train_loss": float(result.training_loss),
            "verified_update": None,
            "memory": memory_snapshot(variant_dir.name + "_" + phase),
        }
        save_json(diagnostics_dir / "training_result.json", measured)
        save_json(
            variant_dir / "status.json",
            {**status, "status": phase + "_trained", "training_result": measured},
        )
        return trainer, measured
    except Exception as error:
        primary_error = error
        save_json(
            diagnostics_dir / "failure.json",
            {
                "error": repr(error),
                "configuration": manifest,
                "memory": memory_snapshot(variant_dir.name + "_failure"),
            },
        )
        save_json(
            variant_dir / "status.json",
            {**status, "status": "training_failed", "error": repr(error)},
        )
        raise
    finally:
        # Raw logs are already streamed to persistent phase directories. Copy
        # and render them even if verification or a later optimizer step fails.
        try:
            persist_rl_diagnostics(
                variant_dir, [checkpoint_root / p for p in ("smoke", "full")], tokenizer
            )
        except Exception as artifact_error:
            save_json(variant_dir / "artifact_failure.json", {"error": repr(artifact_error)})
            if primary_error is None:
                raise
            print("Partial logs remain on disk; diagnostic rendering failed:", repr(artifact_error))
