# ReasoningForge：GSM8K GRPO 训练与评测方案

> 方案日期：2026-10-19。本文是实验设计；正式数据、官方 Qwen 权重和训练结果尚未准备或生成。

## 1. 这轮实验要回答什么

先复现一条可检查的数学 RLVR 主线：**Qwen2.5-0.5B-Instruct → 每题采样 4 个回答 → 用最终数值答案打 0/1 分 → 组内相对优势 → clipped GRPO 更新 LoRA → 在独立题目上评估**。首轮目标是确认真实数据上的训练信号与评估流程，然后再比较 Clip-Higher、Dynamic Sampling 等改动。截图中的具体分数和 DAPO 全套效果不属于首轮可直接宣称的复现结果。

## 2. Benchmark 与数据边界

| 用途 | 数据 | 规模与处理 | 本轮是否使用 |
| --- | --- | --- | --- |
| 训练 | [GSM8K `main/train`](https://huggingface.co/datasets/openai/gsm8k) | 官方 7,473 题；固定划出 500 题做开发集，剩余 **6,973 题**训练 | 是 |
| 开发验证 | GSM8K `main/train` 中保留的 500 题 | 只用于检查提示词、截断、选训练检查点；永不参与优化 | 是 |
| 最终主评估 | [GSM8K `main/test`](https://github.com/openai/grade-school-math) | 官方 1,319 题；训练与选检查点期间不看其结果 | 是 |
| 鲁棒性评估 | [Apple GSM-Symbolic](https://github.com/apple-aiml-research/ml-gsm-symbolic) | 由 GSM8K *test* 题改写；只做训练外测试，按原题模板聚合 | 第二阶段 |
| 跨难度评估 | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 只有 500 道测试题，含符号表达式 | 待符号验证器完成后 |

GSM8K 原始答案是解题文字，最后一行用 `#### 数字` 表示最终答案。训练文件仅保存题目和提取出的最终答案，**不把原始解题过程拼进模型输入**；这轮做的是 GRPO，不是对标准解答做 SFT。只取 `main`，不混合 `socratic`。GSM-Symbolic 与 MATH-500 均不进入这轮训练。GSM-Symbolic 里同一模板有多次实例化，统计时不能把它们当成互相独立的新题型。

数据准备采用[原始 GSM8K 仓库](https://github.com/openai/grade-school-math)中的 `grade_school_math/data/train.jsonl`、`test.jsonl`。下载时固定仓库 commit，记录原始文件及转换后文件的 SHA256。先用 `scripts/prepare_data.py` 分别转换官方 train/test；再运行 `scripts/split_data.py --input data/gsm8k_train.jsonl --train-output data/gsm8k_train_split.jsonl --dev-output data/gsm8k_dev.jsonl --dev-count 500 --seed 42`，生成稳定 SHA256 排名的划分及题目 ID 清单。运行 `scripts/audit_data.py --train data/gsm8k_train_split.jsonl --dev data/gsm8k_dev.jsonl --test data/gsm8k_test.jsonl` 校验数值答案、重复题和跨集交集，并人工抽查 prompt 不含答案。不要使用 Python 内置 `hash()` 划分，因为它可能随进程变化。

## 3. 本机第一轮配置

本机是 6 GiB 的 RTX 3060 Laptop GPU。优先跑 **Qwen2.5-0.5B-Instruct + BF16 + LoRA**；1.5B 加独立 Reference 模型需要更多显存，本轮不把它作为本地前置条件。项目当前只有 16 道演示题和默认 20 步配置，必须新建实验配置与输出目录。

| 参数 | 起点 | 原因 |
| --- | --- | --- |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`，固定 revision | 项目已适配的 Dense Instruct 模型，模型版本可追溯 |
| LoRA | `q_proj`、`v_proj`，rank 8、alpha 16、dropout 0 | 沿用现有可验证实现；减少可训练参数和显存 |
| 每题回答数 `G` | 4 | 能形成组内好/坏对照，采样成本仍可控 |
| prompt batch / 累积窗口 / 两种 micro batch | 1 / 1 / 1 | 先确认 6 GiB 显存可运行；之后才尝试扩大吞吐 |
| 回答复用 | 2 次 | 第二次更新起 ratio 可偏离 1，能观察 clipping |
| 最长 prompt / 回答 | 先用 256 / 256 tokens | 先统计题目 token 长度，再在试跑中看回答截断率；正式比较前锁定 |
| 采样 | 温度 1、`top_p=1`、`top_k=0` | 与项目按原模型概率计算 old/current log-prob 的基线一致 |
| 优化 | LR `1e-5`、clip `0.2/0.2`、sequence 平均 | 沿用现有 vanilla GRPO 配置 |
| Reference KL | `beta=0.02` 起步 | 保持偏离初始模型的约束；若 OOM 才做独立的 `beta=0` 无 Reference 实验 |

`max_prompt_length` 会截断过长题目，所以在正式训练前统计渲染后 prompt 的 token 长度。如果有题目被截断，应先调整上限并重新做基线；不要让模型读到残缺题目。`max_new_tokens=256` 只是试跑起点。如果大量回答因长度上限无法输出 `\boxed{...}`，需在显存允许范围内提高上限，随后将新上限用于所有对照组。格式指令固定为项目现有的“认真解题，最后以 `\boxed{...}` 输出答案”；若调整提示词，基线也要重测。

## 4. 实施顺序与停止检查点

1. **准备环境和素材**：在 ReasoningForge 独立 `.venv` 中安装项目依赖；固定并下载 Qwen 官方权重；取原始 GSM8K JSONL，转换并划分为 `train/dev/test` 三份。先跑环境、配置和数据校验。
2. **训练前评估**：先在开发集 100 题、每题 4 个回答上观察基座的奖励、显式最终答案格式率、回答截断率和“组内同时有对有错”的比例。近乎所有组都全错/全对时，GRPO 的答案奖励几乎没有相对学习信号；先修正格式或长度问题，再评估是否需要更强基座、更多回答或训练集内难度采样。
3. **短试跑**：`configs/qwen15b_gsm8k_pilot.json` 设定 40 次优化器更新、`G=4`、复用 2 次，即约 20 个题目组和 80 个新回答，并在第 20/40 步各对固定 8 道开发题采样 8 次。检查梯度/损失是否有限、奖励抽取、KL、熵、混合组、ratio 尾部和答案截断；不把这段成绩当 benchmark。
4. **首轮 GRPO 曲线**：在冻结的数据划分与配置上跑到 400 次优化器更新，约 200 个题目组、800 个新回答；按 100 步保存检查点。若试跑吞吐和学习信号允许，再延长到 1,600 次更新，约 800 个题目组、3,200 个新回答。这里的题目数按 `prompt_batch_size=1`、`updates_per_rollout=2` 计算；记录真实 `generated_tokens` 和运行时间。即使到 1,600 步，也只使用了训练集的一部分，不能声称完成了一个 epoch。
5. **选模型并最终测试**：用完整 500 题开发集比较预先指定的检查点（例如 400 与 1,600 步），固定验证设置选一个。随后对初始基座与所选 LoRA 用同一套 prompt、答案验证器、随机采样参数分别评估官方 1,319 道 test 题；各题采样 8 次，报告 `pass@1`、`pass@8` 及按题平均的结果。此处 `pass@1` 是随机采样正确概率的估计，**不是 greedy accuracy**。

若目标是让 6,973 道训练题都经历至少一次采样，在当前 `prompt_batch_size=1`、`G=4`、复用 2 次下，需要约 **27,892 个新回答、13,946 次优化器更新**。这是比 1,600 步更大的完整训练实验，应根据短试跑测得的真实吞吐决定是否继续在本机跑，或迁到更大显存设备；不能将首轮短预算结果写成完整数据集训练的成绩。

短试跑与正式训练使用不同输出目录；`train.py --check-config` 通过后再启动。可以从完整窗口边界的 checkpoint 恢复并延长 `max_steps`，每次记录实际配置与模型/data revision。开发评估与最终测试也各用独立输出文件，避免覆盖结果。

## 5. 结果表必须能解释什么

主要结果列为：模型/检查点、开发 `pass@1`、官方 test `pass@1` 与 `pass@8`、题数、每题样本数、prompt/回答 token 上限和采样设置。诊断列为：全错/全对/混合组、答案格式率、截断且无答案比例、长度分位数、reward、KL、训练与固定探针熵、ratio 分位数、正负优势裁剪率以及实际新生成 token 数。本轮按要求不加入显存峰值、累计墙钟时间和 GPU 小时统计。

验收标准不是“达到某个截图分数”，而是能给出 **同一模型、同一测试题、同一生成与判题规则**下，训练前后的可核对结果。开发集用于调整配置；官方 test 只做最终报告。若第一轮无提升，也要用回答格式、全零组、截断、KL/熵和样本量区分原因，不能单凭 loss 的下降声称推理能力提升。

当前 `samples.jsonl` 已保存每组全部回答的奖励、长度、截断和提取出的答案，便于核对零方差组；`dev_metrics.jsonl` 和 `dev_samples.jsonl` 记录固定开发集结果。MATH-500 暂不使用现有数值验证器评估：先接入如 [Math-Verify](https://github.com/huggingface/Math-Verify) 的符号/LaTeX 等价判题，并手工审查一批正确与错误样例，再另开数学跨难度实验。

### 坍缩与异常的观测口径

现有 `metrics.jsonl` 每次参数更新记录一次平均 token 熵、奖励均值/标准差、组内零方差比例、KL、ratio、裁剪比例、回答平均长度、截断率、梯度范数、生成 token 数与耗时。这里的熵是**当前 Actor 在当批已生成回答的有效 token 位置上，对完整词表的预测分布求熵，再平均**；它不参与损失，也不是回答文本的词汇多样性。不同批次的题目与回答会改变熵的统计对象，因而一条下降曲线不能单独证明策略坍缩。

代码已补齐三类诊断：① 在固定开发题及固定前缀上定期测预测熵、保存这些题的生成回答和开发 `pass@1/pass@8`；② 将零方差组拆成全错/全对/混合组，再记录格式率、答案多样性与全文重复率；③ 每组保存全部 4 个回答的奖励、长度、截断状态，以便回查具体失败案例。看到熵明显下降时，联合检查开发准确率、全错组、回答多样性、KL、裁剪比例和截断率，再判断是正常变得确定、模式重复、答案格式失败还是长度预算问题。固定题评测不使用官方 test 来选训练设置。

逐项定义、当前实现状态及异常判读见[GRPO 训练监控指标字典](GRPO_训练监控指标字典.md)。

## 6. 下一轮消融

完成 vanilla GRPO 基线后，依次比较 `clip_high > clip_low`（现有代码可配置）与按题目组动态采样（尚未实现）。每组保持相同模型、训练/开发/test 划分、提示词、回答长度、奖励和选检查点规则。动态采样会额外生成并丢弃探测回答，所以公平比较要按**所有实际生成 token 和总耗时**计费，不能只按优化器步数比较。初期曲线仅作探索；若要宣称稳定改进，补足多个随机种子及按题目的置信区间。

## 官方依据

- [OpenAI GSM8K 原始仓库与数据格式](https://github.com/openai/grade-school-math)
- [OpenAI GSM8K 数据集卡与划分](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5-0.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Apple GSM-Symbolic 生成规则与评估设置](https://github.com/apple-aiml-research/ml-gsm-symbolic)
- [Hugging Face H4 MATH-500 数据集](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [Math-Verify 验证器](https://github.com/huggingface/Math-Verify)
