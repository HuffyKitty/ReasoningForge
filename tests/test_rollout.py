"""完全离线的小模型验证：检查概率与 mask，不下载任何预训练权重。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from reasoning_forge.models import ModelBundle, compute_token_stats, load_bundle
from reasoning_forge.rollout import PytorchRolloutEngine, build_completion_mask


def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(13)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=16,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        attn_implementation="eager",
    ))


def test_completion_mask_keeps_first_eos_and_handles_multiple_eos() -> None:
    ids = torch.tensor([[5, 2, 2, 2], [7, 3, 2, 2], [5, 6, 7, 8]])
    mask, has_eos = build_completion_mask(ids, [2, 3])
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    assert has_eos.tolist() == [True, True, False]


def test_logps_match_next_token_targets_and_backpropagate() -> None:
    model = tiny_model().eval()
    ids = torch.tensor([[2, 2, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    actual, entropy = compute_token_stats(model, ids, mask, 4, return_entropy=True)
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    expected_logits = model(ids, attention_mask=mask, position_ids=positions).logits[:, 3:-1]
    expected = expected_logits.log_softmax(-1).gather(-1, ids[:, 4:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2)
    assert entropy is not None and not entropy.requires_grad
    (-actual.mean()).backward()
    assert model.transformer.h[0].attn.c_attn.weight.grad is not None
    # 左 padding 不应改变同一真实上下文的概率。
    short_logps, _ = compute_token_stats(model, ids[:1, 2:], mask[:1, 2:], 2)
    torch.testing.assert_close(actual[:1], short_logps, atol=1e-6, rtol=1e-6)


class FakeTokenizer:
    pad_token_id = 2
    eos_token_id = 2

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        rows = [[4, 5] if text == "short" else [6, 7, 8] for text in prompts]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[2] * (width - len(row)) + row for row in rows]),
            "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        }

    def batch_decode(self, rows: list[list[int]], **kwargs: object) -> list[str]:
        return [" ".join(str(token) for token in row if token not in {2, 3}) for row in rows]


def test_rollout_micro_batches_group_order_mask_and_first_ratio() -> None:
    model = tiny_model().train()
    model.generation_config.eos_token_id = [2, 3]
    bundle = ModelBundle(model, FakeTokenizer(), None, torch.device("cpu"))
    config = SimpleNamespace(
        num_generations=2, max_prompt_length=8, max_new_tokens=3,
        temperature=1.0, top_p=1.0, top_k=0, rollout_micro_batch_size=2,
    )
    outputs = [torch.tensor([[9, 2, 2], [10, 11, 12]]), torch.tensor([[8, 3], [7, 2]])]
    call_index = 0

    def generate(**kwargs: object) -> torch.Tensor:
        nonlocal call_index
        assert kwargs["use_model_defaults"] is False
        prompt_ids = kwargs["input_ids"]
        result = torch.cat([prompt_ids, outputs[call_index]], dim=-1)
        call_index += 1
        return result

    with patch.object(model, "generate", side_effect=generate):
        rollout = PytorchRolloutEngine(bundle, config).rollout(["short", "long"])
    assert model.training  # rollout 应恢复训练模式
    assert call_index == 2
    assert rollout.prompt_width == 3
    assert rollout.input_ids.shape == (4, 6)
    assert rollout.completion_mask.tolist() == [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0]]
    assert rollout.attention_mask[:, :3].tolist() == [[0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert rollout.truncated.tolist() == [False, True, False, False]
    assert rollout.completions == ["9", "10 11 12", "8", "7"]
    current_logps, _ = compute_token_stats(model, rollout.input_ids, rollout.attention_mask, rollout.prompt_width)
    valid = rollout.completion_mask.bool()
    ratios = torch.exp(current_logps[valid] - rollout.old_logps[valid])
    torch.testing.assert_close(ratios, torch.ones_like(ratios), atol=1e-6, rtol=1e-6)


def test_real_offline_generation_ignores_model_sampling_defaults() -> None:
    model = tiny_model().train()
    model.generation_config.top_k = 5
    model.generation_config.repetition_penalty = 1.8
    config = SimpleNamespace(
        num_generations=2, max_prompt_length=8, max_new_tokens=3,
        temperature=1.0, top_p=1.0, top_k=0, rollout_micro_batch_size=1,
    )
    engine = PytorchRolloutEngine(ModelBundle(model, FakeTokenizer(), None, torch.device("cpu")), config)
    generation = engine._generation_config(3)
    prepared, _ = model._prepare_generation_config(generation, use_model_defaults=False)
    assert prepared.top_k == 0 and prepared.repetition_penalty == 1.0
    result = engine.rollout(["short", "long"])
    assert result.input_ids.shape[0] == 4
    assert result.old_logps.shape == result.completion_mask.shape
    assert model.training


def test_load_bundle_offline_disables_dropout_and_freezes_reference(tmp_path: Path) -> None:
    model = tiny_model()
    model.config.resid_pdrop = model.config.embd_pdrop = model.config.attn_pdrop = 0.4
    model.save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(tmp_path)
    config = SimpleNamespace(
        model_name_or_path=str(tmp_path), revision=None, trust_remote_code=False,
        local_files_only=True, device="cpu", dtype="auto", attn_implementation="eager",
        lora=False, gradient_checkpointing=False, beta=0.04,
    )
    bundle = load_bundle(config)
    assert bundle.tokenizer.padding_side == "left"
    assert bundle.tokenizer.pad_token_id == bundle.tokenizer.eos_token_id == 2
    assert bundle.actor.training
    assert bundle.reference is not None and not bundle.reference.training
    assert all(not parameter.requires_grad for parameter in bundle.reference.parameters())
    assert all(module.p == 0.0 for module in bundle.actor.modules() if isinstance(module, torch.nn.Dropout))
    ids = torch.tensor([[3, 3, 3]])
    before, _ = compute_token_stats(bundle.actor.eval(), ids, torch.ones_like(ids), 2)
    after, _ = compute_token_stats(bundle.actor.train(), ids, torch.ones_like(ids), 2)
    torch.testing.assert_close(before, after)
    config.beta = 0.0
    assert load_bundle(config).reference is None


class RolloutTests(unittest.TestCase):
    """标准库 unittest 即可运行，无需额外安装 pytest。"""

    def test_eos_mask(self) -> None:
        test_completion_mask_keeps_first_eos_and_handles_multiple_eos()

    def test_logps_alignment_and_gradient(self) -> None:
        test_logps_match_next_token_targets_and_backpropagate()

    def test_micro_batches_and_ratio(self) -> None:
        test_rollout_micro_batches_group_order_mask_and_first_ratio()

    def test_offline_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test_load_bundle_offline_disables_dropout_and_freezes_reference(Path(directory))

    def test_real_generation(self) -> None:
        test_real_offline_generation_ignores_model_sampling_defaults()


if __name__ == "__main__":
    unittest.main()
