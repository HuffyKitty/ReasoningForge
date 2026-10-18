"""Hugging Face 模型适配：把 GRPO 算法与具体模型结构分开。

训练循环只使用 ``ModelBundle`` 和 ``compute_token_stats``，因此可以替换
Qwen、Llama、GPT-2 等 decoder-only Dense 模型，而不需要改奖励与损失公式。
这里不实现 MoE 的辅助损失，也不支持 encoder-decoder 模型。
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    """Actor 可训练；Reference 冻结；两者必须使用相同的词表和初始模型。"""

    actor: nn.Module
    tokenizer: Any
    reference: nn.Module | None
    device: torch.device


def _resolve_device_and_dtype(config: Any) -> tuple[torch.device, torch.dtype]:
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("当前教学实现支持 CPU 或 CUDA 设备。")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    if config.dtype not in {"auto", "float32", "bfloat16"}:
        raise ValueError("dtype 请选择 auto、float32 或 bfloat16；本实现不使用 float16。")
    # CPU 使用 fp32，便于初次验证；CUDA 仅在硬件支持时使用 bf16。
    if device.type == "cpu":
        return device, torch.float32
    bf16_supported = torch.cuda.is_bf16_supported()
    if config.dtype == "bfloat16" and not bf16_supported:
        raise ValueError("当前 CUDA 设备不支持 bfloat16，请使用 float32。")
    dtype = torch.bfloat16 if config.dtype != "float32" and bf16_supported else torch.float32
    return device, dtype


def _check_dense_causal_model(model_config: Any) -> None:
    if getattr(model_config, "is_encoder_decoder", False):
        raise ValueError("当前 GRPO 适配层仅支持 decoder-only 因果语言模型。")
    for name in ("num_local_experts", "num_experts", "n_routed_experts"):
        if (getattr(model_config, name, None) or 0) > 1:
            raise ValueError("当前实现支持 Dense 模型；MoE 需要单独处理路由辅助损失。")


def _disable_dropout(model: nn.Module) -> None:
    """概率比应比较参数的变化，不能混入每次随机 Dropout 的变化。

    同时处理 nn.Dropout 与依赖 config 的函数式 Dropout。模型仍可处于
    train 模式，梯度检查点和 autograd 不受影响。
    """
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.p = 0.0
        # Qwen/Llama 的 Attention 可能把概率保存为普通浮点属性。
        for key in ("attention_dropout", "dropout", "activation_dropout"):
            if isinstance(getattr(module, key, None), (int, float)):
                setattr(module, key, 0.0)
    model_config = getattr(model, "config", None)
    if model_config is not None:
        _zero_config_dropout(model_config)


def _zero_config_dropout(model_config: Any) -> None:
    for key, value in vars(model_config).items():
        if ("dropout" in key or key.endswith("_pdrop")) and isinstance(value, (int, float)):
            setattr(model_config, key, 0.0)


def load_bundle(config: Any, resume_model_path: Path | None = None) -> ModelBundle:
    """加载通用 HF Actor，按需创建 LoRA 和冻结的 Reference。

    full checkpoint 直接作为 Actor 的权重目录；LoRA checkpoint 只包含
    adapter，因此仍先加载初始 base，再恢复 adapter。Reference 总是初始
    base，不会随着 Actor 恢复或更新。当 beta=0 时省去 Reference 的显存。
    """
    device, dtype = _resolve_device_and_dtype(config)
    common = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    base_config = AutoConfig.from_pretrained(config.model_name_or_path, **common)
    _check_dense_causal_model(base_config)
    # 初始化前清零，覆盖模块构造时复制 config 概率的架构。
    _zero_config_dropout(base_config)

    # 恢复时优先使用 checkpoint 中保存的 tokenizer，防止模板发生漂移。
    tokenizer_path = config.model_name_or_path
    if resume_model_path is not None and (resume_model_path / "tokenizer_config.json").exists():
        tokenizer_path = str(resume_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **common)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer 同时缺少 pad_token 和 eos_token，无法安全构造生成 batch。")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        **common,
        "dtype": dtype,
        "attn_implementation": config.attn_implementation,
    }
    actor_path = config.model_name_or_path
    if resume_model_path is not None and not config.lora:
        actor_path = str(resume_model_path)
    actor = AutoModelForCausalLM.from_pretrained(actor_path, config=copy.deepcopy(base_config), **model_kwargs)
    _disable_dropout(actor)

    if config.lora:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as error:
            raise ImportError("启用 LoRA 需要安装 peft，请使用本项目的 requirements.txt。") from error
        if config.lora_dropout != 0:
            raise ValueError("GRPO 使用确定的策略概率；请将 lora_dropout 设为 0。")
        if resume_model_path is None:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            actor = get_peft_model(actor, lora_config)
        else:
            actor = PeftModel.from_pretrained(actor, str(resume_model_path), is_trainable=True)
        _disable_dropout(actor)
    actor.to(device)
    actor.config.use_cache = False
    if config.gradient_checkpointing:
        # 非重入方式适用于冻结 embedding 的 LoRA，不依赖输入本身有梯度。
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    actor.train()

    reference = None
    if config.beta > 0:
        reference = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, config=copy.deepcopy(base_config), **model_kwargs
        )
        _disable_dropout(reference)
        reference.to(device)
        reference.requires_grad_(False)
        reference.eval()
    return ModelBundle(actor=actor, tokenizer=tokenizer, reference=reference, device=device)


def _supports_position_ids(model: nn.Module) -> bool:
    """PEFT 外层可能只有 **kwargs，因此继续检查它包住的基础模型。"""
    candidates = [model]
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())
    for candidate in candidates:
        try:
            if "position_ids" in inspect.signature(candidate.forward).parameters:
                return True
        except (TypeError, ValueError):
            continue
    return False


def compute_token_stats(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    *,
    temperature: float = 1.0,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """计算回答 token 的条件 log-prob，返回 [N, R]。

    若整句是 [prompt | a | b | c]，a 的概率来自 prompt 最后一个位置的
    logits。因此 logits 从 P-1 开始，标签从 P 开始：这就是 next-token
    prediction 的错位对齐。padding/EOS 后的无效格子由调用方的 mask 排除。
    """
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids 与 attention_mask 必须是相同形状的二维矩阵。")
    if not 1 <= prompt_width <= input_ids.shape[1]:
        raise ValueError("prompt_width 必须包含至少一个 prompt token，且不超过总长度。")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    if input_ids.shape[1] == prompt_width:
        empty = torch.empty((input_ids.shape[0], 0), device=input_ids.device, dtype=torch.float32)
        return empty, empty.clone() if return_entropy else None

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    if _supports_position_ids(model):
        # 左侧 padding 不应占用真实的位置编号：[0,0,1,1] -> [0,0,0,1]。
        positions = attention_mask.long().cumsum(-1) - 1
        kwargs["position_ids"] = positions.masked_fill(attention_mask == 0, 0)
    outputs = model(**kwargs)
    logits = outputs.logits[:, prompt_width - 1 : -1, :]
    target_ids = input_ids[:, prompt_width:]

    logp_rows: list[torch.Tensor] = []
    entropy_rows: list[torch.Tensor] = []
    for row_logits, row_targets in zip(logits, target_ids):
        logp_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        # Qwen 的词表很大。分行、分块做 fp32 log_softmax，避免一次创建
        # [整个 batch, 整个序列, 词表] 的额外 fp32 概率矩阵。
        for chunk_logits, chunk_targets in zip(row_logits.split(128), row_targets.split(128)):
            log_probs = torch.log_softmax(chunk_logits.float() / temperature, dim=-1)
            logp_chunks.append(log_probs.gather(-1, chunk_targets[:, None]).squeeze(-1))
            if return_entropy:
                # 熵只是监控指标；detach 避免为它额外保留反向计算图。
                with torch.no_grad():
                    detached = log_probs.detach()
                    entropy_chunks.append(-(detached.exp() * detached).sum(-1))
        logp_rows.append(torch.cat(logp_chunks))
        if return_entropy:
            entropy_rows.append(torch.cat(entropy_chunks))
    return torch.stack(logp_rows), torch.stack(entropy_rows) if return_entropy else None
