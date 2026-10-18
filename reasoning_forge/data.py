"""统一数据格式与 Prompt 渲染；这里不依赖 MiniMind 的特殊 token。

数据流：本地 JSONL -> PromptSample -> 模型自己的 chat template -> rollout。
训练文件和评测文件由调用方分别加载，本模块不会自动混合数据集或划分。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class PromptSample:
    """一道题与标准答案；frozen 防止采样时意外改写标准答案。"""

    prompt: str
    answer: str
    id: str
    dataset: str = "custom"


def load_jsonl(path: Union[str, Path]) -> list[PromptSample]:
    """读取统一格式；兼容 question，但不猜测 conversations 等复杂格式。

    一行示例：{"id":"train-1","prompt":"1+1=?","answer":"2","dataset":"gsm8k"}
    answer 应当已经是最终答案，原始 GSM8K/MATH 请先运行 prepare_data.py。
    """

    samples: list[PromptSample] = []
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{source}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: 无效 JSON：{exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{location}: 每一行必须是 JSON 对象。")

            # prompt 是正式字段；question 仅是便于迁移本地题库的别名。
            prompt = record.get("prompt", record.get("question"))
            answer = record.get("answer")
            for name, value in (("prompt/question", prompt), ("answer", answer)):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{location}: {name} 必须是非空字符串。")
            sample_id = record.get("id", str(line_number))
            if not isinstance(sample_id, (str, int)) or isinstance(sample_id, bool):
                raise ValueError(f"{location}: id 必须是字符串或整数。")
            if not str(sample_id).strip():
                raise ValueError(f"{location}: id 不能为空。")
            dataset = record.get("dataset", "custom")
            if not isinstance(dataset, str) or not dataset.strip():
                raise ValueError(f"{location}: dataset 必须是非空字符串。")
            samples.append(PromptSample(prompt, answer.strip(), str(sample_id), dataset))

    if not samples:
        raise ValueError(f"{source}: 数据集为空。")
    return samples


def _config_value(config: Any, key: str, default: Any) -> Any:
    """同时支持配置 dataclass / argparse Namespace 和普通字典。"""

    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def render_prompt(sample_or_text: Union[PromptSample, str], tokenizer: Any, config: Any) -> str:
    """将一道题转成实际送给模型的字符串，标准答案绝不放进 Prompt。

    chat：沿用 HF tokenizer 自带模板，适合 Instruct 模型。
    plain：只拼普通文本，适合没有聊天模板的 Base 模型，不伪造特殊 token。
    """

    text = sample_or_text.prompt if isinstance(sample_or_text, PromptSample) else sample_or_text
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt 必须是非空字符串。")
    system = _config_value(config, "system_prompt", "")
    if not isinstance(system, str):
        raise ValueError("system_prompt 必须是字符串。")
    prompt_format = _config_value(config, "prompt_format", "chat")
    if prompt_format == "plain":
        return f"{system}\n\n{text}" if system else text
    if prompt_format != "chat":
        raise ValueError("prompt_format 只能是 'chat' 或 'plain'。")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "该 tokenizer 没有 chat_template。请使用匹配的 Instruct tokenizer，"
            "或将 prompt_format 设为 'plain' 后为 Base 模型提供普通文本提示。"
        )
    kwargs = _config_value(config, "chat_template_kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ValueError("chat_template_kwargs 必须是字典。")
    if {"tokenize", "add_generation_prompt", "conversation"}.intersection(kwargs):
        raise ValueError("chat_template_kwargs 不能覆盖 tokenize/add_generation_prompt/conversation。")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    # 如 enable_thinking=False，应通过配置传递，而不是把某模型专属字段写死。
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **dict(kwargs)
    )
