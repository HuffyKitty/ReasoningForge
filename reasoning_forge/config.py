"""配置边界：模型、采样、奖励和优化各自负责什么，在这里集中说明。"""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    # HF 仓库名或本地 save_pretrained 目录；通过 Transformers 加载模型。
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "sdpa"

    # 仅 Actor 的 LoRA 参数更新；Reference 保持初始模型不变。
    lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    gradient_checkpointing: bool = True

    data_path: str = "data/sample_math.jsonl"
    # 固定开发集只用于观测，不参与梯度更新；eval_steps=0 时不做周期评测。
    dev_data_path: str | None = None
    output_dir: str = "runs/qwen05b_grpo"
    prompt_format: str = "chat"  # 没有 chat_template 的原始 Base 模型可显式选择 plain。
    system_prompt: str = "Solve the problem carefully. End with the final answer in \\boxed{...}."
    chat_template_kwargs: dict = field(default_factory=dict)
    reward_function: str = "reasoning_forge.rewards:math_reward"
    answer_format: str = "boxed"  # boxed: GSM8K；answer_line: DAPO-Math 原始提示。

    # B 个题目各生成 G 个回答。micro batch 只控制显存，不改变每题的 G。
    prompt_batch_size: int = 1
    num_generations: int = 4
    max_prompt_length: int = 256
    max_new_tokens: int = 128
    rollout_micro_batch_size: int = 2
    train_micro_batch_size: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    epochs: int = 1
    max_steps: int = 20  # 优化器更新次数；0 表示只由 epochs 决定。
    gradient_accumulation_steps: int = 1  # 每次更新积累多少个 prompt batch。
    updates_per_rollout: int = 1  # 同一批 old_logps 固定的回答复用几次；>1 时 clipping 才可能生效。
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.02
    clip_low: float = 0.2
    clip_high: float = 0.2  # 与 clip_low 分开，方便后续 Clip-Higher 消融。
    loss_reduction: str = "sequence"  # sequence: 每条回答平均后再平均；token: 全局有效 token 平均。
    dynamic_sampling: bool = False
    # 最多生成目标有效题目数的多少倍候选；耗尽时失败，不偷偷缩小训练 batch。
    dynamic_sampling_max_candidates_factor: int = 16
    seed: int = 42
    save_steps: int = 10
    log_steps: int = 1
    eval_steps: int = 0
    eval_num_samples: int = 8
    eval_limit: int = 32
    probe_prompts: int = 8
    probe_response_prefix: str = "Let's solve this step by step."

    def validate(self) -> None:
        positive_ints = (
            "prompt_batch_size", "num_generations", "max_prompt_length", "max_new_tokens",
            "rollout_micro_batch_size", "train_micro_batch_size", "epochs",
            "gradient_accumulation_steps", "updates_per_rollout", "save_steps", "log_steps", "lora_rank", "lora_alpha",
            "eval_num_samples", "eval_limit", "probe_prompts",
            "dynamic_sampling_max_candidates_factor",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.num_generations < 2:
            raise ValueError("GRPO 至少需要每题两个回答，num_generations >= 2")
        for name in ("max_steps", "top_k", "seed", "eval_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        for name in ("learning_rate", "temperature", "beta", "clip_low", "clip_high", "max_grad_norm", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数")
        if self.learning_rate == 0 or self.temperature == 0:
            raise ValueError("learning_rate 和 temperature 必须大于 0")
        if not 0 <= self.clip_low < 1 or self.clip_high < 0:
            raise ValueError("clip_low 要在 [0,1)，clip_high >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 要在 (0,1]")
        if self.dtype not in {"auto", "float32", "bfloat16"}:
            raise ValueError("当前训练支持 auto/float32/bfloat16，未实现 FP16 GradScaler")
        if self.attn_implementation not in {"sdpa", "eager"}:
            raise ValueError("attn_implementation 请使用 sdpa 或 eager")
        if self.prompt_format not in {"chat", "plain"}:
            raise ValueError("prompt_format 只能是 chat 或 plain")
        if self.loss_reduction not in {"sequence", "token"}:
            raise ValueError("loss_reduction 只能是 sequence 或 token")
        if self.answer_format not in {"boxed", "answer_line"}:
            raise ValueError("answer_format 只能是 boxed 或 answer_line")
        if not isinstance(self.dynamic_sampling, bool):
            raise ValueError("dynamic_sampling 必须是布尔值")
        if self.lora_dropout != 0:
            raise ValueError("为使采样和训练 log-prob 可比较，当前基线要求 lora_dropout=0")
        if self.lora and not self.lora_target_modules:
            raise ValueError("启用 LoRA 时必须提供 lora_target_modules")
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs 必须是 JSON 对象")
        if not self.model_name_or_path or not self.reward_function:
            raise ValueError("模型路径和奖励函数不能为空")
        if self.dev_data_path is not None and (not isinstance(self.dev_data_path, str) or not self.dev_data_path.strip()):
            raise ValueError("dev_data_path 必须是非空路径，或设为 null")
        if self.eval_steps and self.dev_data_path is None:
            raise ValueError("eval_steps > 0 时必须指定 dev_data_path")
        if not isinstance(self.probe_response_prefix, str) or not self.probe_response_prefix.strip():
            raise ValueError("probe_response_prefix 必须是非空字符串")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    """JSON 内的数据、输出目录相对项目根目录，不受 PyCharm 工作目录影响。"""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    unknown = set(raw) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"未知配置项：{sorted(unknown)}")
    config = TrainConfig(**raw)
    for name in ("data_path", "dev_data_path", "output_dir"):
        if getattr(config, name) is None:
            continue
        value = Path(getattr(config, name)).expanduser()
        setattr(config, name, str(value if value.is_absolute() else (PROJECT_ROOT / value).resolve()))
    # HF 仓库 ID 里也有 '/'，只有确实存在的项目内目录才视为本地模型路径。
    candidate = PROJECT_ROOT / config.model_name_or_path
    if candidate.is_dir():
        config.model_name_or_path = str(candidate.resolve())
    config.validate()
    return config
