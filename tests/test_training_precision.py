"""CPU parameter/AMP fixtures; these do not establish CUDA GRPO success."""

import pytest

torch = pytest.importorskip("torch")

from kg_rlvr.training.grpo import assert_lora_fp32, restore_lora_fp32


def test_precision_repair_leaves_frozen_storage_and_allows_scaled_optimizer_step():
    model = torch.nn.Module()
    model.register_parameter("frozen_base", torch.nn.Parameter(torch.arange(4, dtype=torch.uint8), requires_grad=False))
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16)))
    frozen = model.frozen_base.detach().clone()
    identity = id(model.lora_A)
    with pytest.raises(RuntimeError, match="AFTER constructing GRPOTrainer"):
        assert_lora_fp32(model)
    report = restore_lora_fp32(model)
    assert report["restored_tensors"] == 1
    assert model.lora_A.dtype == torch.float32 and id(model.lora_A) == identity
    assert model.frozen_base.dtype == torch.uint8 and torch.equal(model.frozen_base, frozen)
    # Actual CPU loss scaling/unscale/clip/AdamW. No language-model/RL update.
    optimizer = torch.optim.AdamW([model.lora_A], lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    before = model.lora_A.detach().clone()
    scaler.scale(model.lora_A.square().sum()).backward()
    assert model.lora_A.grad.dtype == torch.float32
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([model.lora_A], 1.0)
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(norm) and not torch.equal(before, model.lora_A)
    assert torch.equal(frozen, model.frozen_base)


def test_precision_repair_refuses_existing_gradients():
    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)))
    model.lora_A.grad = torch.ones_like(model.lora_A)
    with pytest.raises(RuntimeError, match="Existing gradients"):
        restore_lora_fp32(model)
    assert model.lora_A.dtype == torch.bfloat16


def test_precision_repair_refuses_trainable_base_weights():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="non-LoRA"):
        restore_lora_fp32(model)
