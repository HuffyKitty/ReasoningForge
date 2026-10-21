"""离线端到端自检：随机小 Qwen2 -> GRPO 一步 -> checkpoint -> 恢复一步。

运行：python scripts/smoke_test.py
不访问网络、不下载权重，不修改现有环境；临时模型与训练产物会自动清理。
再与连续训练两步比较最终权重，检查优化器、数据游标和 RNG 是否正确恢复。
这验证工程通路和真实参数更新，不证明随机模型具有数学能力。
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_tiny_model(path: Path) -> dict[str, torch.Tensor]:
    """使用 HF 官方模型类构造随机权重，验证标准 CausalLM 接口的训练通路。"""
    torch.manual_seed(17)
    # 所有 token 都是本地定义的；保留独立 PAD/EOS，便于检查序列 mask。
    words = ["<pad>", "<eos>", "<unk>", "system", "user", "assistant", "Solve", "1", "2", "+", "=", "?"]
    words += [f"word_{index}" for index in range(64 - len(words))]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
        model_max_length=128,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, eos_token_id=1, bos_token_id=None, tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def offline_environment(temporary_root: Path) -> dict[str, str]:
    """训练和评测共用离线环境，并保留调用者临时提供的 PEFT 等依赖路径。"""
    environment = os.environ.copy()
    # 即使运行者机器可以联网，也强制这个测试仅使用本地文件。
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(temporary_root), str(PROJECT_ROOT), environment.get("PYTHONPATH", "")]
    )
    return environment


def run_training(config_path: Path, temporary_root: Path, resume: Path | None = None) -> None:
    """走真实 CLI，而非跳过模型加载/优化器/存盘的模拟训练。"""
    command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(config_path)]
    if resume is not None:
        command += ["--resume", str(resume)]
    subprocess.run(command, cwd=PROJECT_ROOT, env=offline_environment(temporary_root), check=True, timeout=180)


def run_evaluation(root: Path, config: dict, model_path: Path, name: str) -> None:
    """检验 full/adapter 导出物都能通过独立评测 CLI 重新加载和生成。

    评测换回无状态的正确性 verifier；随机小模型答错完全正常，不能把工程
    通路检查误写成数学能力断言。这里只核对样本、指标以及禁用 Reference。
    """
    evaluation_config = dict(config, reward_function="reasoning_forge.rewards:math_reward")
    config_path = root / f"{name}-evaluation-config.json"
    output_path = root / f"{name}-evaluation.json"
    config_path.write_text(json.dumps(evaluation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        sys.executable, str(PROJECT_ROOT / "evaluate.py"), "--config", str(config_path),
        "--model", str(model_path), "--data", str(config["data_path"]),
        "--num-samples", "2", "--k", "1", "2", "--output", str(output_path),
    ]
    print(f"[smoke] {name}: 从 final 加载并运行独立评测", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=offline_environment(root), check=True, timeout=180)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result["config"]["beta"] != 0.0 or result["config"]["gradient_checkpointing"]:
        raise AssertionError("评测应禁用 Reference KL 和训练用梯度检查点。")
    if len(result["samples"]) != 2 or any(len(sample["completions"]) != 2 for sample in result["samples"]):
        raise AssertionError("评测应对两道题分别生成两个回答。")
    overall = result["metrics"]["overall"]
    if overall["prompts"] != 2 or overall["responses"] != 4:
        raise AssertionError("评测聚合中的题目数或回答数错误。")
    for key in ("pass@1", "pass@2"):
        if not 0.0 <= overall[key] <= 1.0:
            raise AssertionError(f"评测指标 {key} 不在概率区间内。")


def assert_artifacts(output_dir: Path, expected_step: int, *, lora: bool) -> Path:
    """存出的模型必须真的能恢复，日志也必须记录每次优化器更新。"""
    checkpoint = output_dir / f"checkpoint-{expected_step:06d}"
    expected = [
        checkpoint / "model" / "config.json" if not lora else checkpoint / "model" / "adapter_config.json",
        checkpoint / "trainer_state.pt", checkpoint / "config.json", checkpoint / "COMPLETE",
        output_dir / "final" / "tokenizer_config.json", output_dir / "metrics.jsonl",
    ]
    for artifact in expected:
        if not artifact.is_file():
            raise AssertionError(f"缺少预期产物：{artifact}")
    model_weight = "adapter_model.safetensors" if lora else "model.safetensors"
    if not (output_dir / "final" / model_weight).is_file():
        raise AssertionError(f"final 未导出 {model_weight}")
    rows = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < expected_step:
        raise AssertionError("恢复训练后 metrics.jsonl 应追加记录，而不是覆盖旧日志。")
    for key in ("mixed_group_fraction", "answer_extraction_rate", "response_length_p90",
                "prompt_truncation_fraction", "ratio_p99", "positive_clip_fraction"):
        if key not in rows[-1]:
            raise AssertionError(f"训练日志缺少诊断指标：{key}")
    dev_rows = [json.loads(line) for line in (output_dir / "dev_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(dev_rows) != expected_step or "probe_entropy_p10" not in dev_rows[-1]["probe"]:
        raise AssertionError("固定开发集指标或熵探针缺失")
    sample_rows = [json.loads(line) for line in (output_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(sample_rows) != expected_step * 2 or any("response_tokens" not in row for row in sample_rows):
        raise AssertionError("应保存每个新 rollout 的全部回答及长度")
    # 这里只读取本测试生成的可信临时 checkpoint，里面包含优化器和 RNG 状态。
    state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise AssertionError("trainer_state.pt 应包含可恢复的训练状态字典。")
    if state.get("global_step") != expected_step:
        raise AssertionError("checkpoint 目录编号与实际优化器更新次数不一致。")
    return checkpoint


def exercise_mode(root: Path, base_config: dict, initial_weights: dict[str, torch.Tensor], *, lora: bool) -> None:
    name = "lora" if lora else "full"
    config = dict(base_config, output_dir=str(root / name), lora=lora, max_steps=1)
    if lora:
        config.update(
            lora_rank=4, lora_alpha=8, lora_target_modules=["q_proj", "v_proj"],
            # 特意覆盖冻结 embedding 的场景，验证非重入 gradient checkpoint 正常反传。
            gradient_checkpointing=True,
        )
    config_path = root / f"{name}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] {name}: 训练第 1 步", flush=True)
    run_training(config_path, root)
    output = Path(config["output_dir"])
    checkpoint = assert_artifacts(output, 1, lora=lora)
    if not lora:
        trained = AutoModelForCausalLM.from_pretrained(output / "final", local_files_only=True)
        changed = any(not torch.equal(value.cpu(), initial_weights[key]) for key, value in trained.state_dict().items())
        if not changed:
            raise AssertionError("训练完成但没有参数发生变化；应检查 advantage 和计算图。")
        del trained
    else:
        # 不把“写出 adapter 文件”当作训练成功：LoRA 的 B 矩阵必须离开零初始化。
        from safetensors.torch import load_file
        adapter = load_file(str(output / "final" / "adapter_model.safetensors"))
        if not any("lora_B" in key and torch.count_nonzero(value).item() for key, value in adapter.items()):
            raise AssertionError("LoRA adapter 没有出现实际参数更新。")
    config["max_steps"] = 2
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] {name}: 从 checkpoint 恢复到第 2 步", flush=True)
    run_training(config_path, root, resume=checkpoint)
    assert_artifacts(output, 2, lora=lora)
    # CPU、单线程、固定种子和相同训练配置：恢复结果应与未中断的训练完全相同。
    # 仅“恢复后没有报错”不足以检查 RNG/优化器/数据游标是否真的正确。
    continuous_output = root / f"{name}-continuous"
    config["output_dir"] = str(continuous_output)
    continuous_config_path = root / f"{name}-continuous.json"
    continuous_config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] {name}: 与连续训练 2 步比较最终权重", flush=True)
    run_training(continuous_config_path, root)
    assert_artifacts(continuous_output, 2, lora=lora)
    from safetensors.torch import load_file
    filename = "adapter_model.safetensors" if lora else "model.safetensors"
    resumed_weights = load_file(str(output / "final" / filename))
    continuous_weights = load_file(str(continuous_output / "final" / filename))
    if resumed_weights.keys() != continuous_weights.keys():
        raise AssertionError("恢复训练与连续训练导出的参数集合不一致。")
    different = [key for key in resumed_weights if not torch.equal(resumed_weights[key], continuous_weights[key])]
    if different:
        raise AssertionError(f"恢复训练与连续训练的最终权重不一致：{different[:3]}")
    print(f"[smoke] {name}: 恢复训练与连续训练的最终权重逐位一致", flush=True)
    run_evaluation(root, config, output / "final", name)


def main() -> None:
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="reasoning-forge-smoke-") as directory:
        root = Path(directory)
        model_path = root / "tiny-qwen2"
        initial_weights = create_tiny_model(model_path)
        data_path = root / "tiny.jsonl"
        examples = [
            {"id": "tiny-0", "prompt": "1 + 1 = ?", "answer": "2", "dataset": "smoke"},
            {"id": "tiny-1", "prompt": "2 + 1 = ?", "answer": "3", "dataset": "smoke"},
        ]
        data_path.write_text("\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8")
        dev_path = root / "tiny-dev.jsonl"
        dev_path.write_text(json.dumps({
            "id": "tiny-dev-0", "prompt": "4 + 1 = ?", "answer": "5", "dataset": "smoke",
        }) + "\n", encoding="utf-8")
        # 纯测试 fixture：用交替奖励保证组内优势非零，不假装随机模型会答对数学题。
        # 正式实验必须使用真实 verifier；这个函数只存在于自动清理的临时目录。
        (root / "smoke_reward_fixture.py").write_text(
            "counter = 0\n"
            "def score(completion, answer):\n"
            "    global counter\n"
            "    counter += 1\n"
            "    return float(counter % 2)\n",
            encoding="utf-8",
        )
        config = {
            "model_name_or_path": str(model_path), "data_path": str(data_path),
            "dev_data_path": str(dev_path), "eval_steps": 1, "eval_num_samples": 2,
            "eval_limit": 1, "probe_prompts": 1,
            "device": "cpu", "dtype": "float32", "local_files_only": True,
            "beta": 0.02, "epochs": 2, "prompt_batch_size": 1, "num_generations": 2,
            "max_prompt_length": 32, "max_new_tokens": 4,
            "rollout_micro_batch_size": 2, "train_micro_batch_size": 1,
            "gradient_accumulation_steps": 1, "learning_rate": 1e-3,
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "save_steps": 1, "log_steps": 1, "seed": 23,
            "gradient_checkpointing": False, "prompt_format": "chat", "system_prompt": "Solve",
            "reward_function": "smoke_reward_fixture:score",
        }
        exercise_mode(root, config, initial_weights, lora=False)
        if importlib.util.find_spec("peft") is not None:
            exercise_mode(root, config, initial_weights, lora=True)
        else:
            print("[smoke] SKIP LoRA：当前解释器未安装 peft，全参数路径已实际验证。", flush=True)
    print("[smoke] PASS：离线训练、恢复一致性、full/adapter 导出与独立评测。", flush=True)


if __name__ == "__main__":
    main()
