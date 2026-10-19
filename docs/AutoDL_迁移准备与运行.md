# ReasoningForge 迁移到 AutoDL：准备与运行

本文对应当前 **单 GPU、Qwen2.5-1.5B-Instruct + LoRA + GSM8K + GRPO** 实现。云端租机尚未执行，以下命令是部署清单，实际显存与吞吐仍需在租到的机器上短跑验证。实验设计见 [GSM8K 方案](GRPO_GSM8K_训练与评测方案.md)，指标口径见[监控字典](GRPO_训练监控指标字典.md)。

## 迁移前，先确认手里有这些东西

| 物品 | 当前项目状态 | 云端要做什么 |
| --- | --- | --- |
| 代码 | GitHub 项目 `HuffyKitty/ReasoningForge` | 将**包含本次监控改动的提交**推到远端，再在实例上克隆；记录 `git rev-parse HEAD`。 |
| 基座 | 配置指向 `Qwen/Qwen2.5-1.5B-Instruct`，不会随 Git 克隆下载权重 | 在云端下载官方模型，固定 commit/revision；模型和 tokenizer 必须来自同一 revision。 |
| 数据 | 仓库仅有 16 道演示题 `data/sample_math.jsonl` | 准备互不重叠的 `gsm8k_train_split.jsonl`、`gsm8k_dev.jsonl`、`gsm8k_test.jsonl`；训练不能继续用演示题。 |
| 环境 | `requirements.txt` 已固定 Transformers/PEFT 版本 | 建 Python 3.11 独立环境，按实例驱动选 CUDA PyTorch wheel，运行自检和测试。 |
| 训练配置 | `configs/qwen15b_gsm8k_pilot.json` 是 40 步 GSM8K **通路试跑**配置 | 复制它并固定模型 revision；正式复现实验还需扩大训练与评估预算。 |
| 已有 checkpoint | `runs/` 被 Git 忽略，不会随代码上传 | 若要续训，单独传整个 `checkpoint-xxxxxx/` 和原数据；检查跨系统限制，见下文。 |

### 租什么实例

先租 **单卡 RTX 4090 24 GB** 或预算更紧时 **单卡 RTX 3090 24 GB**，选择支持 BF16、驱动和镜像版本匹配、数据盘空间足够的实例。当前实现只有 CPU/单 CUDA 设备，租两张卡不会自动把 Actor 和 Reference 分摊到两卡；增加卡数不能代替单卡显存。更长回答、更多并发或更大模型可考虑 40/48 GB 卡。24 GB 是这份小微批 LoRA 配置的试跑起点，**不是显存保证**；参考 [AutoDL GPU 型号表](https://www.autodl.com/docs/gpu/) 后仍须在实例上实测。

普通容器实例的 `/root/autodl-tmp` 是高速本地数据盘，系统盘通常 30 GB；先看租用页显示的数据盘容量，再在机器上执行 `df -h /root/autodl-tmp`。模型缓存、代码、数据和 `runs/` 均建议放数据盘，避免权重与 checkpoint 挤满系统盘。AutoDL 说明本地盘**无冗余**，普通实例连续关机 15 天会释放；停止使用前备份 checkpoint、指标和评估 JSON。[磁盘说明](https://www.autodl.com/docs/env) · [数据保留规则](https://www.autodl.com/docs/instance_data/)

## 1. 进机器、准备独立环境

在 AutoDL 控制台开机后，复制该实例自己的 SSH 地址与端口；也可以先用网页 JupyterLab 的 Terminal。以下是**实例上的 Bash**，不是本机 PowerShell：

```bash
nvidia-smi
df -h /root/autodl-tmp
cd /root/autodl-tmp
git clone https://github.com/HuffyKitty/ReasoningForge.git
cd ReasoningForge
git rev-parse HEAD

# AutoDL 官方镜像提供 Miniconda。把新环境放数据盘，避免系统盘被依赖占满。
conda create -y -p /root/autodl-tmp/ReasoningForge/.conda python=3.11
eval "$(conda shell.bash hook)"
conda activate /root/autodl-tmp/ReasoningForge/.conda
python -m pip install --upgrade pip
```

PyTorch 必须与实例的 NVIDIA 驱动兼容。**仅在所租机器支持 CUDA 12.6 构建时**，可沿用本项目已验证的版本：

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python scripts/check_env.py
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

若 CUDA 检查失败，先核对 `nvidia-smi` 所示驱动与镜像，再从 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)选择对应 wheel，例如 PyTorch 2.7.0 也提供 `cu118`；不要以“`pip install` 成功”代替 `torch.cuda.is_available()` 的验证。pilot 配置明确使用 `dtype=bfloat16`；若 `check_env.py` 显示设备不支持 BF16，此配置会报错，应换支持 BF16 的卡或明确另开 FP32 实验（显存需求更大）。[AutoDL 环境与依赖说明](https://www.autodl.com/docs/deps/)

`scripts/smoke_test.py` 使用临时小型随机模型，验证前向、反向、保存和恢复；它不验证真实 1.5B 模型的显存、速度或数学成绩。之后仍需一次真实基座的短试跑。

如果 GitHub/Hugging Face 访问慢，AutoDL 官方文档给出了可在当前终端执行的 `source /etc/network_turbo`；它是辅助下载办法，不保证永远可用。不要把镜像站地址或访问令牌硬编码进训练配置。[学术资源加速](https://www.autodl.com/docs/network_turbo/) · [Git 说明](https://www.autodl.com/docs/git/)

## 2. 把模型和数据放到数据盘

先将 HF 缓存指向数据盘；每次新开 shell 或 tmux 会话都要确保该环境变量仍在：

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
mkdir -p "$HF_HOME"
```

在能访问 Hugging Face 时，查询并固定官方模型 commit，再把该 revision 下载到缓存。以下命令不需要把公开模型复制进项目 `models/`：

```bash
export QWEN_REV="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen2.5-1.5B-Instruct").sha)')"
printf 'Qwen revision: %s\n' "$QWEN_REV"
hf download Qwen/Qwen2.5-1.5B-Instruct --revision "$QWEN_REV"
printf '%s\n' "$QWEN_REV" > data/qwen15b_revision.txt
```

配置中保持 `model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"`，并填同一个 `revision`；如已完整缓存且希望训练时不碰网络，再设 `local_files_only: true`。模型版本和本项目 commit 一同记录到实验记录里。Hugging Face 的 [`hf download`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 支持 `--revision`；AutoDL 建议设置 [`HF_HOME`](https://www.autodl.com/docs/huggingface/) 到数据盘。

数据文件需是每行一个 JSON 对象的 JSONL，例如 `{"id":"gsm8k-train-0","dataset":"gsm8k","split":"train","prompt":"...","answer":"42"}`。`answer` 只供验证器使用，不能拼进模型 prompt。官方 GSM8K 原始仓库是 [OpenAI grade-school-math](https://github.com/openai/grade-school-math)；`scripts/prepare_data.py` 能将其中 train/test 原始 JSONL 转成项目格式：

```bash
git clone https://github.com/openai/grade-school-math.git data/raw/grade-school-math
git -C data/raw/grade-school-math rev-parse HEAD
python scripts/prepare_data.py --format gsm8k --input data/raw/grade-school-math/grade_school_math/data/train.jsonl --output data/gsm8k_train.jsonl --split train
python scripts/prepare_data.py --format gsm8k --input data/raw/grade-school-math/grade_school_math/data/test.jsonl --output data/gsm8k_test.jsonl --split test
sha256sum data/gsm8k_train.jsonl data/gsm8k_test.jsonl
```

**转换不等于开发集划分**。使用项目脚本从官方 train 中按固定种子取 500 题做 dev、留下 6,973 题训练，然后审计三份数据：

```bash
python scripts/split_data.py --input data/gsm8k_train.jsonl --train-output data/gsm8k_train_split.jsonl --dev-output data/gsm8k_dev.jsonl --dev-count 500 --seed 42
python scripts/audit_data.py --train data/gsm8k_train_split.jsonl --dev data/gsm8k_dev.jsonl --test data/gsm8k_test.jsonl
sha256sum data/gsm8k_train_split.jsonl data/gsm8k_dev.jsonl data/gsm8k_test.jsonl
```

拆分脚本默认额外写出 `data/gsm8k_dev.split_manifest.json`，其中有拆分信息与题目 ID，应与三个数据文件一同备份。两个脚本默认拒绝覆盖已有输出；重复操作前先确认是否已生成，不能随手重划 dev。官方 test 不参与训练和选配置。不要把演示文件、完整 train 或官方 test 当作 dev。正式文件与 ID 清单应另行备份，不能依赖 Git 克隆恢复。

若三份文件已经在 Windows 本机准备好，可在**本机 PowerShell** 进入项目目录后用 SCP 上传。将主机和端口改成控制台显示的值；[AutoDL 官方 SCP 上传文档](https://www.autodl.com/docs/scp/)也有 FileZilla 等替代方式：

```powershell
Set-Location 'F:\学习笔记\AI大模型开发笔记\ReasoningForge'
$rfHost = 'region-1.autodl.com'  # 替换为当前实例 SSH 主机
$rfPort = 12345                  # 替换为当前实例 SSH 端口
scp -P $rfPort .\data\gsm8k_train_split.jsonl .\data\gsm8k_dev.jsonl .\data\gsm8k_test.jsonl .\data\gsm8k_dev.split_manifest.json "root@${rfHost}:/root/autodl-tmp/ReasoningForge/data/"
```

## 3. 新建短试跑配置，先做基线

项目已经提供 `configs/qwen15b_gsm8k_pilot.json`，建议在云端复制一份，加入固定模型 revision。它只用于**40 步通路试跑**；核心配置如下：

| 字段 | 短试跑起点 | 说明 |
| --- | --- | --- |
| `model_name_or_path`, `revision` | 官方 Qwen 仓库 ID 与上一步 `QWEN_REV` | 避免绝对本地模型路径影响跨机器续训。 |
| `data_path`, `dev_data_path` | `data/gsm8k_train_split.jsonl`, `data/gsm8k_dev.jsonl` | 开发题不能进入优化。 |
| `output_dir` | `runs/qwen15b_gsm8k_autodl_trial` | 训练拒绝覆盖非空目录；新实验用新路径。 |
| `eval_steps`, `eval_num_samples`, `eval_limit`, `probe_prompts` | `20`, `8`, `8`, `8` | 每 20 步在固定 8 道 dev 题上采 8 次，并测 8 道固定探针；只验证监控通路，正式实验需扩大 dev 覆盖。 |
| `max_steps`, `save_steps` | `40`, `20` | 仅用于观察真实显存、日志、奖励和截断，不能报告为基准成绩。 |
| `max_prompt_length`, `max_new_tokens` | `512`, `256` | 若 prompt 被截断或回答因预算无法交答案，重新定上限并重测基线。 |

`probe_response_prefix` 保持配置默认值；它是固定探针输入的一部分，实验中不要随意修改。可用下面的命令生成新配置，`x` 模式在文件已存在时拒绝覆盖：

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path('configs/qwen15b_gsm8k_pilot.json')
target = Path('configs/qwen15b_gsm8k_autodl_trial.json')
config = json.loads(source.read_text(encoding='utf-8'))
config.update({
    'revision': Path('data/qwen15b_revision.txt').read_text(encoding='utf-8').strip(),
    'local_files_only': True,
    'output_dir': 'runs/qwen15b_gsm8k_autodl_trial',
})
with target.open('x', encoding='utf-8') as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(target)
PY
```

之后执行：

```bash
python scripts/audit_data.py --train data/gsm8k_train_split.jsonl --dev data/gsm8k_dev.jsonl --test data/gsm8k_test.jsonl
python scripts/check_prompt_lengths.py --config configs/qwen15b_gsm8k_autodl_trial.json --test data/gsm8k_test.jsonl
python train.py --config configs/qwen15b_gsm8k_autodl_trial.json --check-config

# 训练前用开发集测原始基座；输出不要放在尚未开始的训练目录内。
mkdir -p runs/evals
python evaluate.py --config configs/qwen15b_gsm8k_autodl_trial.json --data data/gsm8k_dev.jsonl --num-samples 8 --k 1 8 --limit 100 --output runs/evals/qwen15b_base_dev100.json
```

先看基线 `pass@1`、格式可提取率、回答截断与无答案比例。这里 100 题只是快速基线；选检查点应在固定、完整的 500 道 dev 题上比较。`evaluate.py` 的采样 `pass@1` 不是 greedy accuracy。长度预检只加载 tokenizer 并扫描 train/dev/test 全部题目；任意 prompt 超过 `max_prompt_length` 时以退出码 2 停止，正式跑前先调上限并重测基线。

## 4. 训练、观察、恢复、最终评估

长训练建议在 tmux/screen 或 AutoDL JupyterLab 终端中启动，避免本机 PyCharm/SSH 断线结束进程。以下命令在实例 Bash 中执行；日志放在训练输出目录**外面**，因为训练一开始会检查输出目录必须为空：[AutoDL 守护进程说明](https://www.autodl.com/docs/daemon/)。

```bash
mkdir -p /root/autodl-tmp/logs
tmux new -s reasoningforge
# 进入 tmux 后：
cd /root/autodl-tmp/ReasoningForge
eval "$(conda shell.bash hook)"
conda activate /root/autodl-tmp/ReasoningForge/.conda
export HF_HOME=/root/autodl-tmp/hf-cache
set -o pipefail
python -u train.py --config configs/qwen15b_gsm8k_autodl_trial.json 2>&1 | tee /root/autodl-tmp/logs/qwen15b_trial.log
```

tmux 中按 `Ctrl+B`，再按 `D` 离开；之后用 `tmux attach -t reasoningforge` 回来。若镜像没有 tmux，可参考 AutoDL 文档安装或改用 screen。训练自动输出 `runs/qwen15b_gsm8k_autodl_trial/metrics.jsonl`、`samples.jsonl`、定期开发指标 `dev_metrics.jsonl`、按设置保存的开发采样 `dev_samples.jsonl`、模型检查点和 `final/`。开发集评估与固定探针会占用额外采样时间；它们不应使用官方 test 集。

先读短试跑日志：`format_rate`、混合组比例、截断且无答案比例、prompt 截断比例、ratio 尾部、正/负优势裁剪、KL、当前批熵、固定探针熵、dev `pass@1` 和 `grad_norm`。每题保存全部回答后，再抽查奖励是否把错误答案判成正确或将等价答案判错。遇到 CUDA OOM，优先调小 `rollout_micro_batch_size` / `train_micro_batch_size`，再调回答长度；改变 `beta`、裁剪或采样设置需要另开实验，不能当作同一基线的无痕续训。不要只凭 `loss` 下降判断数学能力进步。

在完整 checkpoint 边界需要延长训练时，只增加允许改变的预算（如 `max_steps`、`epochs`）并使用 `--resume`：

```bash
python train.py --config configs/qwen15b_gsm8k_autodl_trial.json --resume runs/qwen15b_gsm8k_autodl_trial/checkpoint-000040
```

这里的路径只是 40 步示例；使用实际存在、含 `COMPLETE` 的最新检查点。续训需要整个 `checkpoint-xxxxxx/`，包括 `model/`、`trainer_state.pt`、`config.json` 和 `COMPLETE`，且训练数据**字节级 SHA256 不变**。允许延长 `max_steps`/`epochs`、调整 `eval_steps` 和日志/保存频率；`eval_limit`、回答上限、奖励规则等改变后不能按原实验续训。若只是想看全部 500 道 dev 题，直接单独运行下方 `evaluate.py`，无需改训练配置。`final/` 只是部署/评估用导出物，不包含优化器和数据游标；LoRA 的 `final/` 只是 adapter，仍需原基座。续训与评估命令见 [README](../README.md)。

短试跑结束后，在**独立评估文件**中比较基座与 LoRA：

```bash
python evaluate.py --config configs/qwen15b_gsm8k_autodl_trial.json --model runs/qwen15b_gsm8k_autodl_trial/final --data data/gsm8k_dev.jsonl --num-samples 8 --k 1 8 --output runs/evals/qwen15b_trial_dev500.json
```

若先对完整 500 道 dev 题保存了基座评估，例如 `runs/evals/qwen15b_base_dev500.json`，可以在训练后评估命令加 `--baseline-results runs/evals/qwen15b_base_dev500.json`，得到基座全错/混合/全对题的分层 `pass@1`。基线只测前 100 题时，不能用于完整 500 题的分层。

选好 checkpoint、冻结所有配置后，才用官方 `data/gsm8k_test.jsonl` 对原始基座和被选中的模型分别做最终评估；结果分别写入不同 JSON，不能覆盖。短试跑若按 16/100 题开发集选过配置，要回到完整 500 题复核，然后再看 test。

## 5. 本机 PyCharm、跨系统 checkpoint 与备份

AutoDL [PyCharm 远程开发说明](https://www.autodl.com/docs/pycharm/)指出：**PyCharm Professional** 才有 On SSH 远程解释器；Community 可用 SSH 终端、JupyterLab 或 VS Code。专业版在 PyCharm 中填控制台给的 Host/Port/User，远程解释器设为 `/root/autodl-tmp/ReasoningForge/.conda/bin/python`，远程项目映射到 `/root/autodl-tmp/ReasoningForge`。调试可以从 PyCharm 启动，长期训练仍建议 tmux。不要把 SSH 密码、HF token 或 `.env` 提交到 Git；公开 Qwen 模型通常无需 HF token。

**从 Windows 本机 checkpoint 继续到 Linux 并非简单复制目录。** 本项目恢复时会比较训练语义字段，还会比较训练文件的 SHA256；如果本机 `model_name_or_path` 是 `F:/.../models/...` 等本地绝对路径，或 `dev_data_path` 在 Windows 被解析为绝对路径，Linux 路径变化会被拒绝。新实验推荐一直用稳定的 HF repo ID + 固定 `revision`，并在云端从头训练；如果确需迁移已有训练状态，先确认原配置字段一致、原数据字节一致，完整复制 checkpoint 和日志，再执行一次短续训核对。**不要直接编辑 `trainer_state.pt` 来绕过校验。**

普通实例的数据盘没有冗余；启用同地区的[文件存储 `/root/autodl-fs`](https://www.autodl.com/docs/fs/) 后，可将关键 checkpoint、manifest、`metrics.jsonl`、`dev_metrics.jsonl`、`samples.jsonl`、评估 JSON 和数据 ID 清单定期复制过去，或下载回本机。文件存储可跨实例保留，训练时仍从本地数据盘读取以取得较高 I/O 性能。AutoDL 也支持 [SCP 下载](https://www.autodl.com/docs/down/)和 FileZilla。若要在本机保存整个短试跑结果，可在**本机 PowerShell** 使用刚才设置的 `$rfHost`、`$rfPort`：

```powershell
New-Item -ItemType Directory -Force .\autodl_backup | Out-Null
scp -r -P $rfPort "root@${rfHost}:/root/autodl-tmp/ReasoningForge/runs/qwen15b_gsm8k_autodl_trial" .\autodl_backup\
```

这会包含 checkpoint，可能较大；实例停止前要检查备份的大小与文件完整性。GPU 运行状态开始计费，不使用时在控制台关机，但要留意实例释放期限。[快速开始](https://www.autodl.com/docs/quick_start/)

## 离开本机前的最终核对

1. 代码已推送且记录 Git commit；云端同一 commit 的测试、离线 smoke 和真实基座短试跑通过。
2. `python scripts/check_env.py` 显示 CUDA、GPU、BF16 正常；1.5B 短试跑无 OOM，`nvidia-smi` 与训练日志可交叉检查。
3. 官方 Qwen 模型 revision、原始 GSM8K 来源 commit、三份转换数据的 SHA256、固定 500 题 dev ID 清单均已记录；`audit_data.py` 核对通过。
4. 配置的 `data_path` 不是 `sample_math.jsonl`，`dev_data_path` 不是 train/test；prompt 截断和答案验证器已抽查。
5. 训练输出目录是新的空目录；日志在外部，checkpoint 和结果已放到数据盘并有独立备份。
