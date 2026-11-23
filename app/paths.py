"""工作室路径以 app 文件夹的上一级为根目录。

在 import 引擎/其他模块之前调用 setup_env()。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
DATA = Path(os.environ.get("FENGSTUDIO_DATA", ROOT)).expanduser().resolve()
IX = DATA / "engine" / "index-tts"
CKPT = IX / "checkpoints"
HF_CACHE = CKPT / "hf_cache"          # w2v-bert、bigvgan、HTDemucs 都在这里
TOOLS = ROOT / "tools"
SPEAKERS_DIR = DATA / "speakers"
SPEAKERS_JSON = DATA / "app" / "speakers.json"
OUT = DATA / "output"
TRAIN = DATA / "train"
MODELS = DATA / "models"              # whisper 等
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
        "VOICE_STUDIO_ROOT": str(DATA),
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
        return p.resolve().relative_to(DATA).as_posix()
    except ValueError:
        return str(p)


def to_abs(p):
    if not p:
        return ""
    p = Path(p)
    return str(p if p.is_absolute() else (DATA / p))


def path_problem():
    """返回路径问题说明；没问题返回 None。"""
    for folder in (ROOT, DATA):
        s = str(folder)
        if not s.isascii() or " " in s:
            return f"工作目录需要英文且无空格：{s}\n例如 D:\\FengStudio。"
    return None
