"""检查这台电脑能不能运行本地配音工作室。双击「检查环境.bat」运行。"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import paths
import trainer

paths.setup_env()
ok = True


def line(good, msg, fix=""):
    global ok
    ok = ok and good
    print(("  [OK]   " if good else "  [问题] ") + msg + (f"\n         → {fix}" if fix and not good else ""), flush=True)


print("=" * 60)
print("  本地配音工作室 · 环境检查")
print("=" * 60)
print(f"文件夹：{paths.ROOT}\n")

prob = paths.path_problem()
line(prob is None, "文件夹路径是纯英文、没有空格", prob or "")

free = shutil.disk_usage(paths.ROOT).free / 2**30
line(free > 10, f"所在磁盘剩余 {free:.0f} GB", "至少留 10 GB（训练、生成都要写文件）")

try:
    import torch
    cuda = torch.cuda.is_available()
    line(True, f"PyTorch {torch.__version__}（CUDA {torch.version.cuda}）")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 2**30
        line(True, f"显卡：{p.name}，显存 {vram:.1f} GB")
        line(vram >= 7.5, "显存 ≥ 8 GB", "8 GB 以下可能生成失败或训练爆显存")
        x = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")
        line(float(x[0, 0]) == 1024.0, "显卡能正常计算")
    else:
        line(False, "没有检测到可用的 NVIDIA 显卡（CUDA）",
             "需要 NVIDIA 显卡，并把显卡驱动更新到支持 CUDA 12.8 的版本（570 以上；RTX 50 系必须）")
except Exception as e:
    line(False, f"PyTorch 加载失败：{e}", "运行 setup.bat 安装 Python 依赖")

try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    if smi:
        major = int(smi.split(".")[0])
        line(major >= 570, f"显卡驱动 {smi}", "到 NVIDIA 官网更新驱动（需要 570 或更新）")
except Exception:
    pass

need = {
    "底模 config.yaml": paths.CKPT / "config.yaml",
    "底模 gpt.pth": paths.CKPT / "gpt.pth",
    "s2mel.pth": paths.CKPT / "s2mel.pth",
    "codec.pth": paths.CKPT / "codec.pth",
    "w2v-bert": paths.HF_CACHE / "w2v-bert-2.0" / "model.safetensors",
    "bigvgan": paths.HF_CACHE / "bigvgan" / "bigvgan_generator.pt",
    "CAMPPlus": paths.HF_CACHE / "campplus_cn_common.bin",
    "去伴奏模型 HTDemucs": paths.HF_CACHE / "models--adefossez--HTDemucs-ft",
    "中文识别 paraformer": paths.FUNASR["model"] / "model.pt",
    "语音活动检测 FSMN-VAD": paths.FUNASR["vad_model"] / "model.pt",
    "标点恢复 CT-Punc": paths.FUNASR["punc_model"] / "model.pt",
    "Whisper": paths.MODELS / "whisper" / "large-v3-turbo.pt",
}
for name, p in need.items():
    line(p.exists(), f"模型文件：{name}", f"缺少 {p}，运行 setup.bat 下载")

for k, v in trainer.load_speakers().items():
    lora = Path(v["lora"]) / "adapter_config.json"
    line(lora.exists(), f"说话人 {v.get('label', k)}（{k}）", f"缺少模型 {lora.parent}")

for mod in ("indextts", "gradio", "funasr", "whisper", "demucs", "librosa", "peft"):
    try:
        __import__(mod)
        line(True, f"组件 {mod}")
    except Exception as e:
        line(False, f"组件 {mod} 加载失败：{e}")

edge = Path(os.environ.get("ProgramFiles(x86)", r"C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
print("  [可选] Microsoft Edge " + ("已安装" if edge.exists() else "未找到，可从网页上传训练素材"))

print("\n" + ("✅ 全部通过，可以双击「启动本地配音工作室.bat」。" if ok else "❌ 有问题，按上面的提示处理后再运行本检查。"))
sys.exit(0 if ok else 1)
