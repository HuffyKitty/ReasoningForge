# 验证记录

日期：2026-10-18。验证均未下载官方 Qwen 权重，未修改原 MiniMind 源码或其已安装依赖。

## 环境

- Windows，Python 3.11.4。
- PyTorch 2.7.0+cu126，Transformers 4.57.6。
- Accelerate 1.14.0，Tokenizers 0.22.2，Safetensors 0.8.0。
- PEFT 0.17.1 仅安装至临时测试目录，通过子进程 `PYTHONPATH` 加载，原虚拟环境未安装 PEFT。
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB，BF16 可用。

测试使用工作区原有可用解释器；本项目没有预创建 `.venv`。实际使用时按 README 建立独立环境、安装 requirements。系统默认的 Conda Python 3.9 不适合直接运行本项目。

## 已实际通过

| 验证 | 结果 |
|---|---|
| `python -m unittest discover -s tests -v` | 29 项通过 |
| GRPO 优势/正负优势裁剪/梯度方向/KL | 通过 |
| 无效 mask 不影响损失和梯度 | 通过 |
| left padding、next-token 对齐、pad==EOS、多 EOS | 通过 |
| 真实本地随机 GPT-2 生成和 HF 模型加载 | 通过 |
| 规则奖励、显式答案、数值等价与奖励投机边界 | 通过 |
| 不等长/不等批窗口的 sequence/token 归一化 | loss 与全部参数梯度均与整窗计算一致 |
| pass@k 公式、按题等权和数据集分层 | 通过 |
| 4 个实验 JSON、16 条自写演示数据 | 配置与数据校验通过 |
| 本地 GSM8K/MATH 格式转换及回读 | 通过 |

## 离线端到端

`scripts/smoke_test.py` 在临时目录创建随机小 Qwen2（词表64、hidden32、一层），使用真实 `train.py` / `evaluate.py` 子进程。

- 全参数模式：13,472 个可训练参数，确有权重变化。
- LoRA 模式：448 个可训练参数，B 矩阵离开零初始化；启用非重入梯度检查点。
- 两模式均完成：训练1步 → HF保存 → 恢复到第2步 → 导出 final。
- 同一模式、同一CPU种子下，与连续训练2步比较，最终 safetensors 权重逐位一致。
- 两模式均从 final 重新加载评测，每题2回答，保存 pass@1/pass@2 与逐题记录。
- 测试中使用交替奖励 fixture 强制产生非零优势；它仅用于检查参数更新，不是数学奖励或项目成果。

额外小实验验证 `updates_per_rollout=2`、梯度累积2批、token reduction：第一遍 ratio≈1，第二遍 ratio 偏离1；人为设置较大学习率及较小 clip 范围时，裁剪实际生效。检查点仅在窗口完成后保存，步数上限在中间触发时不重放剩余复用。

另在本机 CUDA 上运行小型随机 Qwen2：BF16 + LoRA + gradient checkpointing，连续2次优化，梯度范数均为正，导出检查点成功。

## 验证范围

这些结果确认算法与工程接口可运行，不证明数学能力提升，也不保证官方0.5B/1.5B模型的显存占用、训练稳定性或论文成绩。正式实验尚需下载/指定基座、准备独立数据划分、核验答案验证器、固定版本与采样预算，再建立真实基线。

完整 DAPO 的 Dynamic Sampling、Overlong Reward Shaping 尚未实现。复杂 MATH 的完整符号等价验证不在默认规则奖励范围内。CPU 的逐位恢复一致性不能推导为所有 GPU、算子和环境下的位级确定性。

## 2026-10-19：训练监控与 AutoDL 准备

- 单元测试扩至 **45 项**，覆盖组内全错/全对/混合、格式率、答案/全文多样性、长度/截断、prompt 长度预检、跨 microbatch 的 ratio 分位数和正优势裁剪分母、固定开发评估随机流隔离、基座难度分层、GSM8K 稳定划分与跨集数据审计。
- 离线 `scripts/smoke_test.py` 再次通过全参数和 LoRA 两条端到端路径；训练时开发集 `pass@k` 与固定探针熵写入日志，恢复训练和连续训练的最终权重仍逐位一致；训练回答全部 `G` 条均保存。
- 数据划分/审计/长度检查脚本的离线测试与 CLI `--help` 通过。尚无官方 GSM8K 文件或 Qwen2.5-1.5B 权重；因此 `configs/qwen15b_gsm8k_pilot.json --check-config` 在本机因缺少 `data/gsm8k_train_split.jsonl` 而停止，这是预期的准备缺口。
- 尚未租用 AutoDL 实例，也未实测 1.5B 显存、吞吐或正式 benchmark 成绩。按 [AutoDL 迁移指南](AutoDL_迁移准备与运行.md)完成数据、模型和环境准备后才能验证这些部分。
