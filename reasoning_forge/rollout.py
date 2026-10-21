"""同步 PyTorch 采样器：先生成，再保存回答、mask 和旧策略 log-prob。

基于 Transformers 的模型生成接口，保持易读的数据流，
后续可以用具有相同返回值的高性能 rollout engine 替换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from transformers import GenerationConfig

from .models import ModelBundle, compute_token_stats


@dataclass
class RolloutBatch:
    """N=B×G；P 为左填充后的 prompt 宽度；R 为当前 batch 的回答宽度。"""

    input_ids: torch.Tensor          # [N, P+R]：prompt 与回答的完整序列
    attention_mask: torch.Tensor     # [N, P+R]：可参加 Attention 的真实 token
    completion_ids: torch.Tensor     # [N, R]：只包含回答
    completion_mask: torch.Tensor    # [N, R]：回答有效位置，包含第一次 EOS
    old_logps: torch.Tensor          # [N, R]：采样时旧 Actor 的原始分布 log-prob
    completions: list[str]           # 长度 N，供答案验证器计算 reward
    prompt_width: int
    truncated: torch.Tensor          # [N]：用完预算且没有 EOS 的回答


def _eos_ids(bundle: ModelBundle) -> list[int]:
    # 某些 Qwen 模型同时把多个 token 作为停止符；generation_config 优先。
    generation_config = getattr(bundle.actor, "generation_config", None)
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        eos = bundle.tokenizer.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(token_id) for token_id in eos]


def build_completion_mask(
    completion_ids: torch.Tensor, eos_token_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """保留第一次 EOS 本身，屏蔽它之后的补齐格子。

    不能简单写 ``ids != pad_id``：pad_id 经常等于 eos_id，这会错误地
    把真实结束动作也屏蔽。例如 [7, EOS, EOS] 的 mask 应为 [1,1,0]。
    第二个返回值表示该回答是否至少出现一次 EOS。
    """
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for token_id in eos_token_ids:
        is_eos |= completion_ids.eq(token_id)
    # 当前 token 前尚未遇到 EOS 就有效；因而第一个 EOS 也有效。
    previous_eos_count = is_eos.long().cumsum(-1) - is_eos.long()
    return previous_eos_count.eq(0).long(), is_eos.any(-1)


class PytorchRolloutEngine:
    def __init__(self, bundle: ModelBundle, config: Any) -> None:
        self.bundle = bundle
        self.config = config

    def _generation_config(self, budget: int) -> GenerationConfig:
        cfg = self.config
        if cfg.temperature <= 0:
            raise ValueError("GRPO 需要随机采样，temperature 必须大于 0。")
        # 从干净默认值开始，不继承模型仓库里未知的 repetition、stop 等限制。
        generation = GenerationConfig()
        generation.max_new_tokens = budget
        generation.do_sample = True
        generation.temperature = cfg.temperature
        generation.top_p = cfg.top_p
        generation.top_k = cfg.top_k
        generation.num_beams = 1
        generation.num_beam_groups = 1
        generation.num_return_sequences = 1
        generation.pad_token_id = self.bundle.tokenizer.pad_token_id
        eos = _eos_ids(self.bundle)
        generation.eos_token_id = eos or None
        generation.use_cache = True
        generation.return_dict_in_generate = False
        generation.output_scores = False
        # 清除模型目录中隐含的解码限制，使实验由本项目配置控制。
        for name, value in {
            "repetition_penalty": 1.0,
            "encoder_repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "force_words_ids": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "sequence_bias": None,
            "min_length": 0,
            "min_new_tokens": 0,
            "typical_p": 1.0,
            "min_p": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "penalty_alpha": None,
            "renormalize_logits": False,
            "remove_invalid_values": False,
            "stop_strings": None,
            "token_healing": False,
            "watermarking_config": None,
        }.items():
            setattr(generation, name, value)
        return generation

    def rollout(
        self,
        prompts: list[str],
        num_generations: int | None = None,
        max_new_tokens: int | None = None,
    ) -> RolloutBatch:
        """prompts 已由 data.render_prompt 排版；这里不重复套 Chat Template。

        排列固定为 [问题0回答0, 问题0回答1, ..., 问题1回答0, ...]，
        这样 rewards.view(B, G) 才能计算同一个问题的组内相对优势。
        """
        if not prompts:
            raise ValueError("rollout 至少需要一个 prompt。")
        group_size = self.config.num_generations if num_generations is None else num_generations
        budget = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        if group_size <= 0 or budget <= 0:
            raise ValueError("num_generations 与 max_new_tokens 必须大于 0。")
        tokenizer = self.bundle.tokenizer
        expanded_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        encoded = tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_ids = encoded["input_ids"].to(self.bundle.device)
        prompt_mask = encoded["attention_mask"].to(self.bundle.device)
        if prompt_ids.shape[1] == 0 or (prompt_mask.sum(-1) == 0).any():
            raise ValueError("prompt tokenize 后不能为空；plain base 模型也需要至少一个输入 token。")
        prompt_width = prompt_ids.shape[1]
        actor = self.bundle.actor
        was_training = actor.training
        micro_batch_size = getattr(self.config, "rollout_micro_batch_size", 2)
        if micro_batch_size <= 0:
            raise ValueError("rollout_micro_batch_size 必须大于 0。")
        sequence_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        logp_chunks: list[torch.Tensor] = []
        eos_chunks: list[torch.Tensor] = []
        actor.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(expanded_prompts), micro_batch_size):
                    end = start + micro_batch_size
                    generated_chunk = actor.generate(
                        input_ids=prompt_ids[start:end],
                        attention_mask=prompt_mask[start:end],
                        generation_config=self._generation_config(budget),
                        # Transformers>=4.50 否则可能又从仓库恢复 top-k 等默认值。
                        use_model_defaults=False,
                    )
                    completion_chunk = generated_chunk[:, prompt_width:]
                    mask_chunk, eos_chunk = build_completion_mask(completion_chunk, _eos_ids(self.bundle))
                    attention_chunk = torch.cat([prompt_mask[start:end], mask_chunk], dim=1)
                    # 不读取经过 top-p/温度等变换的 generation scores。旧/新策略都
                    # 使用原始模型分布计算新旧策略概率；首次更新 ratio≈1。
                    # 若 temperature!=1 或 top-p/top-k 截断，采样行为分布与这个
                    # 原始分布并不完全相同；严格 on-policy 基线请选择 1/1/0。
                    chunk_logps, _ = compute_token_stats(
                        actor, generated_chunk, attention_chunk, prompt_width, temperature=1.0
                    )
                    sequence_chunks.append(generated_chunk)
                    mask_chunks.append(mask_chunk)
                    logp_chunks.append(chunk_logps)
                    eos_chunks.append(eos_chunk)
        finally:
            actor.train(was_training)

        # 各 micro-batch 可在不同位置提前结束。只在回答右侧补齐，保持
        # prompt 宽度 P 与 next-token 对齐位置不变，同时限制峰值 KV/logits 显存。
        response_width = max(chunk.shape[1] for chunk in mask_chunks)
        generated = torch.cat([
            F.pad(chunk, (0, prompt_width + response_width - chunk.shape[1]), value=tokenizer.pad_token_id)
            for chunk in sequence_chunks
        ])
        completion_ids = generated[:, prompt_width:]
        completion_mask = torch.cat([
            F.pad(chunk, (0, response_width - chunk.shape[1]), value=0) for chunk in mask_chunks
        ])
        old_logps = torch.cat([
            F.pad(chunk, (0, response_width - chunk.shape[1]), value=0.0) for chunk in logp_chunks
        ])
        has_eos = torch.cat(eos_chunks)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # decode 使用有效前缀，防止 pad token 与普通文本混到答案验证中。
        text_ids = [row[mask.bool()].tolist() for row, mask in zip(completion_ids, completion_mask)]
        completions = tokenizer.batch_decode(text_ids, skip_special_tokens=True)
        truncated = ~has_eos & (completion_mask.sum(-1) >= budget)
        return RolloutBatch(
            input_ids=generated,
            attention_mask=attention_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_logps=old_logps.detach(),
            completions=completions,
            prompt_width=prompt_width,
            truncated=truncated,
        )
