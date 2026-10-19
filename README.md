# ReasoningForge · 推理工坊

把读懂的 MiniMind GRPO 变成可以替换基座、检查训练行为、逐步做消融的独立实验项目。

保留 MiniMind 的主线：**多回答采样 → 奖励 → 组内相对优势 → policy ratio/clipping → KL → mask → 反向传播**。模型改用 Hugging Face `AutoModelForCausalLM` / `AutoTokenizer`；可以加载 Qwen 等 Dense decoder-only 模型的官方权重，也可以训练完整参数或 PEFT LoRA。代码中的中文注释解释整体关系、张量维度和关键设计。

当前提供标准 GRPO、非对称 clipping、sequence/token 两种损失归一化、规则奖励、同步 PyTorch rollout、训练恢复、固定开发集监控和 pass@k 评估。**Dynamic Sampling 与 Overlong Reward Shaping 尚未实现；这是一份复现实验基础，并非完整 DAPO 或图片中成绩的复现结果。**

## 从这里开始

在 PowerShell 中切到本项目目录。建议建立独立虚拟环境，PyCharm 解释器选择 `ReasoningForge/.venv/Scripts/python.exe`。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 与本机已有 MiniMind 验证环境一致的 CUDA 12.6 构建；CPU 可换成 /whl/cpu。
.\.venv\Scripts\python.exe -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts/check_env.py
.\.venv\Scripts\python.exe train.py --config configs/qwen05b_grpo.json --check-config
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

`smoke_test.py` 完全离线，临时创建随机小 Qwen2，实际训练、检查参数变化、保存、恢复，并验证恢复训练与连续训练结果一致。它验证工程通路，不能证明数学能力提升。测试产物会自动清理；没有 PEFT 时明确跳过 LoRA 测试。

本项目未自动下载 Qwen 或运行正式训练。首次正式运行若配置保留 HF 仓库名，`from_pretrained` 会访问 Hugging Face。也可预先下载并使用本地模型目录：

```powershell
.\.venv\Scripts\hf.exe download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

之后把 JSON 中 `model_name_or_path` 改为 `models/Qwen2.5-0.5B-Instruct`；需要完全离线时再设 `local_files_only: true`。

## 运行第一轮训练

```powershell
.\.venv\Scripts\python.exe train.py --config configs/qwen05b_grpo.json
```

默认是 0.5B + LoRA、每题 4 个回答、每批回答复用更新 2 次、最多 20 次优化器更新。示例数据是 16 道自行编写的小题，**只用于检查通路**，不能用来报告 GSM8K/MATH 成绩。`max_steps` 与 `epochs` 任一预算耗尽就停止。

PyCharm Run Configuration：Script 选 `train.py`，Parameters 填 `--config configs/qwen05b_grpo.json`，Working directory 选项目目录。数据、输出和已有本地模型目录在 JSON 中相对项目根目录解析，不依赖当前终端目录。

想使用其他模型，主要改 `model_name_or_path` 与对应 LoRA 层名；Instruct 模型使用自身 `chat_template`，没有模板的 Base 模型设置 `prompt_format: "plain"` 并自行提供合适的普通文本提示。当前明确支持 Dense、decoder-only、CPU/单 CUDA 设备；未实现 MoE 辅助损失、多模态、DDP、vLLM/SGLang 和量化训练。**接口兼容不代表所有模型都已经实测。**

训练显存取决于模型、prompt/回答长度、微批大小和 Reference。先用 0.5B 配置做真实采样；`rollout_micro_batch_size` / `train_micro_batch_size` 控制显存，`num_generations` 控制组大小。若 OOM，先减长度和微批；把 `beta` 改为 0 可不加载 Reference，但这会改变实验目标，必须作为新实验记录。1.5B 配置仅提供迁移起点，不承诺在当前笔记本显存中可运行。

## 数据与评估

训练统一使用 JSONL，每行至少包含：

```json
{"id":"gsm8k-train-0","dataset":"gsm8k","prompt":"A question...","answer":"42"}
```

`answer` 只给 verifier，不拼进 prompt。默认奖励只认显式最终答案 `\boxed{42}` 或单独的 `#### 42` 行，正确 1、错误 0；支持简单数值等价（例如 `1/2` 与 `0.5`），**不是完整的 MATH 符号验证器**。复杂 MATH 正式实验前应替换 `reward_function` 并检查等价表达式、单位、超时和错误提取。

准备本地 GSM8K/MATH 原始 JSONL 后，转换为统一格式：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --format gsm8k --input data/raw/gsm8k_train.jsonl --output data/gsm8k_train.jsonl --split train
.\.venv\Scripts\python.exe scripts/prepare_data.py --format gsm8k --input data/raw/gsm8k_test.jsonl --output data/gsm8k_test.jsonl --split test
.\.venv\Scripts\python.exe scripts/split_data.py --input data/gsm8k_train.jsonl --train-output data/gsm8k_train_split.jsonl --dev-output data/gsm8k_dev.jsonl --dev-count 500 --seed 42
.\.venv\Scripts\python.exe scripts/audit_data.py --train data/gsm8k_train_split.jsonl --dev data/gsm8k_dev.jsonl --test data/gsm8k_test.jsonl
.\.venv\Scripts\python.exe scripts/check_prompt_lengths.py --config configs/qwen15b_gsm8k_pilot.json --test data/gsm8k_test.jsonl
```

MATH 用 `--format math`，输入字段为 `problem` / `solution`。脚本不下载数据，不自动混合 train/test，也不覆盖已有文件。修改训练配置的 `data_path` 后再开始新实验。

准备好数据后可用 [1.5B GSM8K 40 步试跑配置](configs/qwen15b_gsm8k_pilot.json)在单张大显存 GPU 上验证整条流程。它配置独立开发集，每 20 步评估固定 8 道题、每题采样 8 次，并测固定回答前缀熵；40 步与 8 道开发题均不足以证明算法复现效果。实际租用服务器的命令见 [AutoDL 迁移准备与运行](docs/AutoDL_迁移准备与运行.md)。

固定独立测试集、prompt 和采样设置，先测原基座，再测导出的模型：

```powershell
# 原基座（不传 --model）
.\.venv\Scripts\python.exe evaluate.py --config configs/qwen05b_grpo.json --data data/gsm8k_test.jsonl --num-samples 8 --k 1 8 --output runs/base_eval.json

# LoRA adapter：评估时仍需要原基座
.\.venv\Scripts\python.exe evaluate.py --config configs/qwen05b_grpo.json --model runs/qwen05b_grpo/final --data data/gsm8k_test.jsonl --num-samples 8 --k 1 8 --output runs/grpo_eval.json
```

在**同一开发集**先保存基座评估 JSON 后，训练后评估可加 `--baseline-results runs/base_dev_eval.json`，按基座答题情况把题目分为 `unsolved/mixed/all_correct` 三档。程序会检查数据 SHA256 与逐题题面是否一致。

这里的 pass@1 是随机采样的成功概率估计；pass@8 是一次采 8 个回答至少答对一个的概率估计，均不等同于 greedy accuracy。输出保存逐题回答、奖励、截断和按 `dataset` 分层统计。评估要求二元正确性奖励，不接受任意 shaped reward 作为成功标签。

## 产物与恢复

```text
runs/qwen05b_grpo/
  config.json             # 展开默认值的实际配置
  environment.json        # Python/依赖/CUDA 信息
  data_manifest.json      # 数据 SHA256，阻止换数据后按旧位置续训
  dev_manifest.json       # 固定开发集 SHA256；配置开发集时生成
  model_manifest.json     # 模型类型与可取得的实际 HF revision
  metrics.jsonl           # 每次更新的 reward/优势/格式/长度/ratio/KL/熵等指标
  samples.jsonl           # 每个新 rollout 的全部 G 个回答、奖励、长度和截断
  dev_metrics.jsonl       # 定期固定开发集 pass@k、格式率、固定探针熵等
  dev_samples.jsonl       # 开发集逐题回答；方便人工检查
  events.jsonl            # 数值异常或 OOM 时记录阶段；无异常则不创建
  checkpoint-000010/
    model/                # 完整 HF 模型或 LoRA adapter，加 tokenizer
    config.json
    trainer_state.pt      # 优化器、数据游标、随机数状态
    COMPLETE              # 检查点完整标记
  final/                  # 推理用 HF 模型 / LoRA adapter
```

把配置中的 `max_steps` / `epochs` 增大后，恢复：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/qwen05b_grpo.json --resume runs/qwen05b_grpo/checkpoint-000010
```

检查点在整批 rollout 的复用更新结束后保存；若 `save_steps` 落在中间，会推迟到窗口结束。`max_steps` 在复用中间耗尽时，剩余复用次数丢弃，恢复后从下一批题开始。因此比较“连续训练与断点恢复”时，应在完整复用窗口边界停止。LoRA 的 `final/` 只含 adapter，不是能脱离基座使用的完整模型。

## 阅读路线

1. `config.py`：B/G、两种 micro batch、累积与复用分别控制什么。
2. `data.py` / `rewards.py`：题目如何进入模型，答案如何被判定。
3. `models.py`：Actor/Reference/LoRA，以及 `[N,T,V] → [N,R]` 的 next-token 对齐。
4. `rollout.py`：样本排列、EOS/padding、固定旧策略概率。
5. `objectives.py`：逐行读优势、ratio、裁剪、KL、两种平均。
6. `trainer.py`：把以上模块连起来，理解为何固定 old、为何重新 forward。
7. `checkpoint.py` / `evaluate.py`：复现实验与独立评测。

架构、公式和具体消融步骤见 [架构与实验指南](docs/architecture_and_experiments.md)。测试记录见 [验证记录](docs/validation.md)。
指标的分母、空值和排查方式见 [GRPO 训练监控指标字典](docs/GRPO_训练监控指标字典.md)。

## 实现依据

算法主线参考本工作区 MiniMind `trainer/train_grpo.py` / `trainer/rollout_engine.py`，通用适配和工程边界在本目录实现；没有修改原 MiniMind 文件。保留 Apache-2.0 许可证，来源记录见 [NOTICE](NOTICE)。

- [Hugging Face Auto Classes](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/auto)
- [Hugging Face Generation API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)
- [PEFT checkpoint 格式](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/checkpoint)
- [Qwen2.5-0.5B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PyTorch 历史版本安装命令](https://pytorch.org/get-started/previous-versions/)
