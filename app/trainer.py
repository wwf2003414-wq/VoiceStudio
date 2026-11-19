"""训练说话人：素材 → 去伴奏挑选 → 切片 → 转录 → 特征提取 → LoRA 训练 → 登记。每步都是子进程，日志写文件。"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import paths as P

ROOT = P.ROOT
APP = P.APP
IX = P.IX
PY_IX = sys.executable    # 便携版只有一个 Python 环境，所有步骤都用它
PY_VOICE = sys.executable
SKILL = P.TOOLS
SPEAKERS = P.SPEAKERS_JSON
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,30}$")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mov", ".mkv")


def paths(spk):
    base = P.TRAIN / spk
    return dict(base=base, raw=base / "1_原始素材", train=base / "train", refs=base / "refs",
                clips=base / "3_切片", lst=base / "4_转录.lst", feats=base / "5_特征",
                lora=P.SPEAKERS_DIR / spk / "model", spkdir=P.SPEAKERS_DIR / spk,
                log=base / "训练日志.txt", edge=base / "edge_profile")


def has_audio(d):
    return Path(d).is_dir() and any(f.suffix.lower() in AUDIO_EXTS for f in Path(d).iterdir())


def status(spk):
    p = paths(spk)
    reg = load_speakers().get(spk, {})
    trained = (Path(p["lora"]) / "adapter_config.json").is_file() or \
        (bool(reg.get("lora")) and (Path(reg["lora"]) / "adapter_config.json").is_file())
    return {
        "素材": has_audio(p["raw"]),
        "去伴奏挑选": has_audio(p["train"]),
        "切片": Path(p["clips"]).is_dir() and any(Path(p["clips"]).glob("*.wav")),
        "转录": Path(p["lst"]).is_file() and Path(p["lst"]).stat().st_size > 0,
        "特征提取": (Path(p["feats"]) / "train.json").is_file(),
        "LoRA训练": trained,
    }


def status_md(spk):
    if not spk or not NAME_RE.match(spk):
        return "先填一个说话人名字（英文字母开头，可含数字和下划线，例如 `xiaoli`）。"
    st = status(spk)
    rows = [f"**{spk}** 的进度："]
    for k, v in st.items():
        rows.append(f"{'✅' if v else '⬜'} {k}")
    n_raw = len([f for f in Path(paths(spk)["raw"]).glob("*") if f.suffix.lower() in AUDIO_EXTS]) if st["素材"] else 0
    if n_raw:
        rows.append(f"\n素材文件 {n_raw} 个，位置：`{paths(spk)['raw']}`")
    if st["LoRA训练"]:
        lora = load_speakers().get(spk, {}).get("lora") or paths(spk)["lora"]
        rows.append(f"\n🎉 已训练好，模型在 `{lora}`，到「文本配音」页刷新说话人就能选。")
    return "\n\n".join(rows)


def _env():
    env = P.child_env()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(cmd, log, label, cwd=None):
    """跑子进程，每 2 秒 yield 一次日志末尾。成功返回时最后一条以 ✅ 开头。"""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%H:%M:%S')} {label} =====\n$ {' '.join(map(str, cmd))}\n")
    t0 = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd or IX), stdout=f, stderr=subprocess.STDOUT,
                                env=_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while proc.poll() is None:
            time.sleep(2)
            yield f"⏳ {label} 进行中… 已用 {int(time.time() - t0)} 秒\n\n" + tail(log)
    ok = proc.returncode == 0
    yield (f"✅ {label} 完成（{int(time.time() - t0)} 秒）" if ok else f"❌ {label} 失败（退出码 {proc.returncode}）") + "\n\n" + tail(log)
    if not ok:
        raise RuntimeError(f"{label} 失败，详见日志 {log}")


def tail(log, n=18):
    try:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    lines = [l for l in lines if l.strip() and "it/s]" not in l and "s/it]" not in l]
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------- 各步骤 --
def save_uploads(spk, files):
    p = paths(spk)
    p["raw"].mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files or []:
        src = Path(f if isinstance(f, str) else getattr(f, "name", ""))
        if src.is_file() and src.suffix.lower() in AUDIO_EXTS:
            shutil.copy(src, p["raw"] / src.name)
            n += 1
    return n


def douyin_login(spk, url):
    p = paths(spk)
    subprocess.run([str(PY_VOICE), str(SKILL / "browser_collect.py"), "launch", "--profile-dir", str(p["edge"]),
                    "--urls", url], env=_env(), check=True)


def douyin_collect(spk, url):
    p = paths(spk)
    bc = SKILL / "browser_collect.py"
    yield from run([PY_VOICE, bc, "status"], p["log"], "检查登录")
    if '"douyin_logged_in": true' not in tail(p["log"], 3):
        raise RuntimeError("Edge 窗口里还没登录抖音，请先扫码登录")
    yield from run([PY_VOICE, bc, "collect", "--url", url, "--out", p["base"] / "list.json"], p["log"], "收集视频链接")
    yield from run([PY_VOICE, bc, "media", "--list", p["base"] / "list.json", "--out", p["base"] / "media.json",
                    "--profile-url", url], p["log"], "读取视频信息")
    yield from run([PY_VOICE, SKILL / "download.py", "--media", p["base"] / "media.json", "--out", p["raw"],
                    "--profile-url", url], p["log"], "下载视频音频")
    yield from run([PY_VOICE, bc, "close", "--profile-dir", p["edge"]], p["log"], "关闭登录窗口、删除登录数据")


def step_prepare(spk, minutes):
    p = paths(spk)
    yield from run([PY_VOICE, SKILL / "prepare_data.py", "--raw", p["raw"], "--work", p["base"],
                    "--minutes", str(minutes)], p["log"], "去伴奏 + 挑选训练集 + 截参考音频", cwd=p["base"])


def step_slice(spk):
    p = paths(spk)
    shutil.rmtree(p["clips"], ignore_errors=True)
    yield from run([PY_IX, "-u", "slice_audio.py", "--in_dir", p["train"], "--out_dir", p["clips"]],
                   p["log"], "切片")


def step_transcribe(spk):
    p = paths(spk)
    yield from run([PY_IX, "-u", "build_lst.py", "--audio_dir", p["clips"], "--out", p["lst"], "--engine", "funasr"],
                   p["log"], "转录")


def step_extract(spk, lang):
    p = paths(spk)
    yield from run([PY_IX, "-u", "extract_features.py", "--metadata", p["lst"], "--out", p["feats"],
                    "--lang", lang.lower()], p["log"], "特征提取")


def step_train(spk, epochs, batch):
    p = paths(spk)
    p["lora"].mkdir(parents=True, exist_ok=True)
    yield from run([PY_IX, "-u", "train_lora.py", "--feats", p["feats"], "--out", p["lora"],
                    "--epochs", str(int(epochs)), "--batch_size", str(int(batch))], p["log"], "LoRA 训练")


def register(spk, label=None):
    """训练完：参考音频拷到 speakers/<spk>/refs，写进 speakers.json（存相对路径）。"""
    p = paths(spk)
    refdir = p["spkdir"] / "refs"
    refdir.mkdir(parents=True, exist_ok=True)
    for f in Path(p["refs"]).glob("*.wav"):
        shutil.copy(f, refdir / f.name)
    calm = sorted(refdir.glob("平稳_1_*.wav")) or sorted(refdir.glob("*.wav"))
    reg = _load_raw()
    entry = reg.get(spk, {"label": label or spk, "presets": {}})
    entry["lora"] = P.to_rel(p["lora"])
    if calm and "默认" not in entry["presets"]:
        entry["presets"]["默认"] = {"ref": P.to_rel(calm[0]), "emo_method": "与音色参考音频相同", "emo_ref": "",
                                   "emo_weight": 0.6, "temperature": 0.75, "speed": 0.97, "pause_ms": 150}
    reg[spk] = entry
    save_speakers(reg)
    return entry


def _load_raw():
    try:
        d = json.loads(SPEAKERS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_speakers():
    """读登记表，把里面的相对路径换成本机的绝对路径。"""
    reg = _load_raw()
    for v in reg.values():
        v["lora"] = P.to_abs(v.get("lora"))
        for pr in v.get("presets", {}).values():
            for k in ("ref", "emo_ref"):
                pr[k] = P.to_abs(pr.get(k))
    return reg


def save_speakers(reg):
    d = {"_说明": "说话人登记表。路径相对于本工作室文件夹；新训练的说话人会自动加进来。", **reg}
    SPEAKERS.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
