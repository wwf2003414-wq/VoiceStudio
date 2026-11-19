"""工作室路径以 app 文件夹的上一级为根目录。

在 import 引擎/其他模块之前调用 setup_env()。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
IX = ROOT / "engine" / "index-tts"
CKPT = IX / "checkpoints"
HF_CACHE = CKPT / "hf_cache"          # w2v-bert、bigvgan、HTDemucs 都在这里
TOOLS = ROOT / "tools"
SPEAKERS_DIR = ROOT / "speakers"
SPEAKERS_JSON = APP / "speakers.json"
OUT = ROOT / "output"
TRAIN = ROOT / "train"
MODELS = ROOT / "models"              # whisper 等
PY = Path(sys.executable)

FUNASR = dict(model=CKPT / "paraformer-zh", vad_model=CKPT / "fsmn-vad", punc_model=CKPT / "ct-punc")


def ffmpeg_dir():
    import imageio_ffmpeg
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    plain = exe.with_name("ffmpeg.exe")
    if not plain.exists():
        import shutil
        shutil.copy(exe, plain)  # whisper、pydub 要找名为 ffmpeg 的程序
    return exe.parent


def env_vars():
    """子进程也要用的一组环境变量（全部指向本文件夹，且强制离线，模型缺了会直接报错而不是偷偷下载到别处）。"""
    no_proxy = os.environ.get("NO_PROXY", "")
    local_hosts = ",".join(dict.fromkeys(filter(None, (*no_proxy.split(","), "127.0.0.1", "localhost"))))
    return {
        "HF_HUB_CACHE": str(HF_CACHE),
        "HF_HOME": str(MODELS / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_CACHE": str(MODELS / "modelscope"),
        "XDG_CACHE_HOME": str(MODELS),
        "TORCH_HOME": str(MODELS / "torch"),
        "GRADIO_ANALYTICS_ENABLED": "False",
        "NO_PROXY": local_hosts,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
        "VOICE_STUDIO_ROOT": str(ROOT),
    }


def setup_env():
    os.environ.update(env_vars())
    try:
        os.environ["PATH"] = str(ffmpeg_dir()) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    for p in (str(IX), str(TOOLS), str(APP)):
        if p not in sys.path:
            sys.path.insert(0, p)


def child_env():
    env = dict(os.environ)
    env.update(env_vars())
    return env


def to_rel(p):
    """登记表里存相对路径，换电脑、换盘符都不受影响。"""
    if not p:
        return ""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def to_abs(p):
    if not p:
        return ""
    p = Path(p)
    return str(p if p.is_absolute() else (ROOT / p))


def path_problem():
    """返回路径问题说明；没问题返回 None。"""
    s = str(ROOT)
    if not s.isascii():
        return f"这个文件夹的路径里有中文或特殊字符：{s}\n请把整个文件夹放到纯英文路径下，例如 D:\\VoiceStudio，再重新启动。"
    if " " in s:
        return f"路径里有空格：{s}\n建议放到没有空格的路径，例如 D:\\VoiceStudio。"
    return None
