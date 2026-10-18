"""在 PyCharm 选中的解释器中检查依赖；不安装包，不加载/下载模型。"""

import argparse
import importlib.metadata
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="只检查全参数路径，不要求 PEFT")
    args = parser.parse_args()
    print(f"当前解释器：{sys.executable}")
    print(f"Python：{sys.version.split()[0]}")
    missing = []
    required = ["torch", "transformers", "accelerate", "safetensors"] + ([] if args.full else ["peft"])
    for name in required:
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            print(f"{name}: 未安装")
    if sys.version_info < (3, 10):
        raise SystemExit("本项目需要 Python >=3.10")
    if missing:
        raise SystemExit(f"缺少 {missing}，请在当前项目虚拟环境执行 python -m pip install -r requirements.txt")
    import torch
    import transformers
    if not args.full:
        import peft  # noqa: F401：实际导入，检测依赖间接口是否兼容。
    if tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2]) < (2, 6):
        raise SystemExit("需要 PyTorch 2.6 或更新的兼容版本")
    if transformers.__version__ != "4.57.6":
        print("提示：本项目测试版本为 transformers 4.57.6，请优先使用 requirements.txt。")
    print(f"CUDA：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        print(f"GPU：{torch.cuda.get_device_name(index)}")
        print(f"显存：{torch.cuda.get_device_properties(index).total_memory / 1024 ** 3:.1f} GiB")
        print(f"BF16：{torch.cuda.is_bf16_supported()}")
    print("环境导入检查通过；模型/数据是否适合训练需继续进行 smoke 和真实采样检查。")


if __name__ == "__main__":
    main()
