# DAPO 消融与 AutoDL 执行队列

## 原始数据与划分

- 原始仓库：`BytedTsinghua-SIA/DAPO-Math-17k`，固定 revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`。
- 原始 Parquet SHA-256：`534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`。
- 官方文件有 1,791,700 行，但仅 17,398 个不同题面。7 个题面有互相冲突的标准答案，全部排除；可用题目 17,391 道。
- 固定随机种子 42，训练 16,391、开发 500、测试 500。划分和哈希保存在 `data/dapo_math_17k_manifest.json`。
- 保留官方 `Answer: ...` 提示格式；DAPO 实验使用 `dapo_math_reward` 和 `answer_format=answer_line`。GSM8K 的 `\boxed{}` 奖励规则保持独立。
- 原始文件由 `scripts/download_dapo_math_17k.sh` 下载并核验；`scripts/prepare_dapo_math_17k.py` 流式去重与划分。

## 单因素实验

| 任务 | 基线与变体 | 改动 |
|---|---|---|
| GSM8K | GRPO 第 1200 步检查点 vs `qwen15b_gsm8k_dynamic_sampling_full.json` | 仅开启 Dynamic Sampling；目标有效题目数和优化步数相同 |
| DAPO-Math-17k | `qwen15b_dapo_math_17k_baseline_pilot.json` | 200 步无 DAPO 改动基准 |
| DAPO-Math-17k | `qwen15b_dapo_math_17k_clip_higher_pilot.json` | 仅 `clip_high: 0.2 → 0.28` |
| DAPO-Math-17k | `qwen15b_dapo_math_17k_token_loss_pilot.json` | 仅 `loss_reduction: sequence → token` |

DAPO-Math-17k 的基线、Clip-Higher、Token-level Loss 试跑和完整配置统一使用每题 4 个训练回答（`num_generations: 4`）；GSM8K Dynamic Sampling 仍使用每题 8 个回答。

后三个实验必须从同一初始 Qwen2.5-1.5B-Instruct 开始，使用相同训练/开发划分、种子、奖励规则、最大长度、优化预算与评估采样参数；不能从另一个实验的 checkpoint 接着训练。测试集不用于挑选配置。

另已生成对应的 `*_full.json`，用于将来条件允许时开展完整一轮对比；本次 DAPO-Math 试跑已取消，当前不会启动任何 DAPO-Math 训练。

Dynamic Sampling 按 `seed+cycle` 洗牌候选训练题目，整组采样 8 个回答，仅保留组内奖励有差异的题目。每次更新仍补满目标有效题目数；最多尝试目标数的 16 倍，超过则明确失败。`candidate_index` 随检查点保存以供恢复。指标另记 `sampled_prompts`、`rejected_prompts`、`generated_tokens`，对比效率时要看生成成本，不能只看优化步数。

## 远程部署与顺序

原 GRPO 已停止，选用 `/root/autodl-tmp/ReasoningForge/runs/qwen15b_gsm8k_full_4090_fast_v2/checkpoint-001200` 作为基线。新版本代码和实验输出在独立目录 `/root/autodl-tmp/ReasoningForge-next`，其 `.conda` 和 `data` 是指向原目录的符号链接；当前训练的源码没有被替换。

远程 `dapo_experiment_queue` 已完成 Dynamic Sampling 烟测和 1200 步训练。GSM8K 四次采样与八次采样对照均已完成，执行顺序为：

1. GRPO 第 1200 步与 Dynamic Sampling 第 1200 步模型在相同 500 题 GSM8K 开发集上各采样 4 次，比较 pass@1 和 pass@4。原 GRPO 第 1201–1208 步的日志不用于模型或训练成本对比。
2. 两个模型在相同 500 题上各采样 8 次，比较 pass@1、pass@4、pass@8。两组评测分别保存逐题 JSON 和对比 JSON。
3. 八次采样评测结束后，`gsm8k_finish_after_pass8` 生成对照 JSON 和 SHA-256 清单，关闭原队列，不启动 DAPO-Math。

DAPO-Math 试跑已按用户的费用要求取消；相应本地同步计划任务也已撤销。数据集文件仍保存在本地 `data/`。GSM8K 的 Dynamic Sampling 完整训练目录、原 GRPO 第 1200 步检查点和 GSM8K 数据已复制到本地，关键文件 SHA-256 与远程一致。

本地 `scripts/sync_gsm8k_final_to_local.ps1` 由 Windows 计划任务 `ReasoningForge-GSM8K-FinalSync` 每分钟执行一次。它在八次采样对照完成后下载六个结果 JSON 并核对 SHA-256，向远程写入同步确认。远程随后执行 `/usr/bin/shutdown` 关机；即使本地未能在五分钟内确认同步，远程也会关机以避免继续计费，结果文件仍保留在远程数据盘。

2026-10-21 21:33:55（北京时间），六份结果文件已在本地完成校验并向远程确认。上述记录描述本次实验的执行过程；模型、原始数据和逐题日志保存在本地，不随源码提交到 Git。

发布后的进程接管脚本要求显式设置当前 `TRAIN_PID` 或 `EVAL_PID`，以及 `PAUSED_QUEUE_PID`；启动时核对进程命令、父子关系及暂停状态，终止前核对启动时间。`AUTO_SHUTDOWN` 默认为 `0`，只有显式设置 `AUTO_SHUTDOWN=1` 才关机。两个训练队列的 `RUN_DAPO_MATH_PILOT` 默认为 `0`。这些脚本使用原 AutoDL 目录布局，复用前须按新部署核对路径及进程；不会自动复用本次实验的旧 PID。

## 本次 GSM8K 对照结果

两组均使用 Qwen2.5-1.5B-Instruct + LoRA，比较第 1200 步；评估数据是从 GSM8K 训练集留出的同一组 500 道开发题，不是官方测试集。每题八次采样、温度为 1，pass@1 为随机采样成功率估计，不等同于贪心准确率。

| 指标 | 标准 GRPO | Dynamic Sampling | 差值（百分点） |
|---|---:|---:|---:|
| pass@1 | 81.275% | 81.675% | +0.400 |
| pass@4 | 93.743% | 92.940% | -0.803 |
| pass@8 | 95.600% | 95.000% | -0.600 |

两组有效训练题次均为 4800；标准 GRPO 生成约 1146 万 token，Dynamic Sampling 生成约 2312 万 token（含丢弃候选）。本次单种子对照没有显示稳定的多样本覆盖收益，不能据此推断其他模型、数据集或训练预算下的结论。

Clip-Higher 与 Token-level Loss 的配置和代码已备好，未加入本次远程队列。若任一步失败，队列停止，不会继续消耗 GPU 启动后续训练。

在**单独的 SSH 终端**查看队列和实验日志：

```bash
tail -f /root/autodl-tmp/logs/gsm8k_finish_after_pass8.status.log
tail -f /root/autodl-tmp/logs/gsm8k_dynamic_dev500_8.log
```

不要进入训练所在的 tmux 窗口查看；退出上述 `tail` 命令时按 `Ctrl+C` 只会停止日志查看。
