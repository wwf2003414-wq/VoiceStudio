"""训练说话人：素材 → 去伴奏挑选 → 切片 → 转录 → 特征提取 → LoRA 训练 → 登记。每步都是子进程，日志写文件。"""
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from uuid import uuid4

import dubbing
import languages
import paths as P
import social_assets

ROOT = P.ROOT
APP = P.APP
IX = P.IX
PY_IX = sys.executable    # 便携版只有一个 Python 环境，所有步骤都用它
PY_VOICE = sys.executable
SKILL = P.TOOLS
SPEAKERS = P.SPEAKERS_JSON
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,30}")
RESERVED_NAMES = re.compile(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mov", ".mkv")
TRAIN_HELPERS = {dubbing.QWEN: "train_qwen.py", dubbing.OMNI: "train_omni.py"}
TRAIN_ENGINES = (dubbing.INDEX, *TRAIN_HELPERS)


def resolve_speaker(value):
    name = unicodedata.normalize("NFC", (value or "").strip())
    if not name:
        raise ValueError("请先填写上方的「说话人名称」，例如「小李」。")

    profiles = {key: profile for key, profile in _load_raw().items()
                if NAME_RE.fullmatch(key) and not RESERVED_NAMES.fullmatch(key)}
    normalized = name.casefold()
    for key, profile in profiles.items():
        if key.casefold() == normalized:
            return key, profile.get("label") or name

    matching = [(key, profile["label"]) for key, profile in profiles.items()
                if (profile.get("label") or "").strip().casefold() == normalized]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise ValueError("有多个同名说话人。请填写「说话人和历史」里对应的说话人标识。")

    if (len(name) > 40 or not any(char.isalnum() for char in name)
            or any(not char.isalnum() and char not in " _-" for char in name)):
        raise ValueError("「说话人名称」支持 1–40 个文字、数字、空格、下划线或连字符。")
    if NAME_RE.fullmatch(name) and not RESERVED_NAMES.fullmatch(name):
        if P.TRAIN.is_dir():
            for folder in P.TRAIN.iterdir():
                if folder.name.casefold() == normalized and folder.is_dir():
                    return folder.name, name
        return normalized, name

    key = "speaker_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return key, profiles.get(key, {}).get("label") or name


def paths(spk, revision=None):
    base = P.TRAIN / spk
    work = base / ".runs" / revision if revision else base
    return dict(base=base, work=work, raw=base / "1_原始素材", train=work / "train", refs=work / "refs",
                clips=work / "3_切片", lst=work / "4_转录.lst", feats=work / "5_特征",
                lora=P.SPEAKERS_DIR / spk / "model", spkdir=P.SPEAKERS_DIR / spk,
                log=base / "训练日志.txt", edge=base / "edge_profile", social_edge=base / "social_edge_profile")


def has_audio(d):
    return bool(media_files(d))


def media_files(directory):
    return sorted(f for f in Path(directory).glob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS)


def has_adapter(directory):
    if not directory:
        return False
    root = Path(directory)
    return (root / "adapter_config.json").is_file() and any(file.is_file() for file in root.glob("adapter_model.*"))


def workspace_revision(spk):
    try:
        revision = (paths(spk)["base"] / ".latest").read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    return revision if re.fullmatch(r"[a-f0-9]{20}", revision) else None


def input_revision(spk, minutes, lang):
    records = []
    for file in media_files(paths(spk)["raw"]):
        stat = file.stat()
        records.append((file.name, stat.st_size, stat.st_mtime_ns))
    key = json.dumps([records, int(minutes), lang], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:20]


def status(spk, engine=dubbing.INDEX):
    p = paths(spk, workspace_revision(spk))
    reg = load_speakers().get(spk, {})
    adapter = reg.get("adapters", {}).get(engine)
    trained = has_adapter(adapter) or (engine == dubbing.INDEX and has_adapter(p["lora"]))
    common = {
        "素材": has_audio(p["raw"]),
        "去伴奏挑选": has_audio(p["train"]),
        "切片": Path(p["clips"]).is_dir() and any(Path(p["clips"]).glob("*.wav")),
        "转录": Path(p["lst"]).is_file() and Path(p["lst"]).stat().st_size > 0,
    }
    if engine == dubbing.INDEX:
        common["特征提取"] = (p["feats"] / "train.json").is_file()
    return {**common, "LoRA训练": trained}


def status_md(spk, engine=dubbing.INDEX, name=None):
    profile = load_speakers().get(spk, {})
    st = status(spk, engine)
    rows = [f"**{profile.get('label') or name or spk} · {engine}**"]
    for k, v in st.items():
        rows.append(f"{'✅' if v else '⬜'} {k}")
    n_raw = len(media_files(paths(spk)["raw"]))
    if n_raw:
        rows.append(f"\n素材文件 {n_raw} 个，位置：`{paths(spk)['raw']}`")
    if st["LoRA训练"]:
        lora = profile.get("adapters", {}).get(engine) or paths(spk)["lora"]
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
        if not src.is_file() or src.suffix.lower() not in AUDIO_EXTS:
            continue
        dest = p["raw"] / src.name
        if dest.exists():
            with src.open("rb") as stream:
                src_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            with dest.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == src_hash:
                    continue
            dest = p["raw"] / f"{src.stem[:80]}_{src_hash[:16]}{src.suffix.lower()}"
            if dest.exists():
                continue
        pending = p["raw"] / f".{uuid4().hex}.tmp"
        shutil.copyfile(src, pending)
        pending.replace(dest)
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


def social_import(spk, posts):
    p = paths(spk)
    p["base"].mkdir(parents=True, exist_ok=True)
    yield f"正在导入 {len(posts)} 条链接…"
    with tempfile.TemporaryDirectory(prefix="social-links-", dir=p["base"]) as temporary:
        queue = Path(temporary) / "links.txt"
        queue.write_text("\n".join(posts.values()), encoding="utf-8")
        yield from run([PY_VOICE, SKILL / "social_assets.py", "--links", queue, "--raw", p["raw"]],
                       p["log"], "导入 Instagram / X 素材", cwd=SKILL)


def step_prepare(p, minutes):
    yield from run([PY_VOICE, SKILL / "prepare_data.py", "--raw", p["raw"], "--work", p["work"],
                    "--minutes", str(minutes)], p["log"], "转码 + 去伴奏 + 挑选训练集 + 截参考音频", cwd=p["work"])


def step_slice(p):
    shutil.rmtree(p["clips"], ignore_errors=True)
    yield from run([PY_IX, "-u", "slice_audio.py", "--in_dir", p["train"], "--out_dir", p["clips"]],
                   p["log"], "切片")


def step_transcribe(p, lang):
    if lang == "ZH":
        cmd = [PY_IX, "-u", "build_lst.py", "--audio_dir", p["clips"], "--out", p["lst"], "--engine", "funasr"]
    else:
        cmd = [PY_VOICE, APP / "helpers" / "score_batch.py", "--train", p["clips"], p["lst"], lang]
    yield from run(cmd, p["log"], "转录")


def step_extract(p, lang):
    yield from run([PY_IX, "-u", "extract_features.py", "--metadata", p["lst"], "--out", p["feats"],
                    "--lang", lang.lower()], p["log"], "特征提取")


def step_train(p, epochs, batch):
    p["lora"].mkdir(parents=True, exist_ok=True)
    yield from run([PY_IX, "-u", "train_lora.py", "--feats", p["feats"], "--out", p["lora"],
                    "--epochs", str(int(epochs)), "--batch_size", str(int(batch))], p["log"], "LoRA 训练")


def training_records(transcripts):
    records = []
    for line in transcripts.read_text(encoding="utf-8-sig").splitlines():
        audio, _, text = line.partition("\t")
        if not text.strip() or not Path(audio).is_file():
            continue
        records.append(dict(audio_path=str(Path(audio).resolve()), text=text.strip()))
    if not records:
        raise ValueError("没有可用于训练的音频与转写。")
    return records


def step_train_qwen(p, epochs, batch, python, config):
    records = training_records(p["lst"])
    references = sorted(p["refs"].glob("平稳_1_*.wav")) or sorted(p["refs"].glob("*.wav"))
    source = references[0] if references else records[0]["audio_path"]
    with tempfile.TemporaryDirectory(prefix="qwen-", dir=p["work"]) as temporary:
        work = Path(temporary)
        dataset = work / "clips.json"
        reference = work / "reference.wav"
        dubbing.write_json(dataset, records)
        dubbing.reference_audio(source, reference)
        yield from run([python, APP / "helpers" / TRAIN_HELPERS[dubbing.QWEN], "--device", config["device"],
                        "--model", config["model"], "--dataset", dataset, "--reference", reference,
                        "--output", p["lora"], "--epochs", int(epochs), "--batch", int(batch)],
                       p["log"], "Qwen3-TTS LoRA 训练", cwd=work)


def step_train_omni(p, lang, epochs, batch, python, config):
    model = Path(config["model"])
    device = config["device"]
    records = [dict(record, id=f"clip-{i:06d}", language_id=languages.OMNI_CODES[lang])
               for i, record in enumerate(training_records(p["lst"]))]
    with tempfile.TemporaryDirectory(prefix="omni-", dir=p["work"]) as temporary:
        work = Path(temporary)
        records_path = work / "clips.jsonl"
        records_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                                encoding="utf-8")
        yield from run(
            [python, "-m", "omnivoice.scripts.extract_audio_tokens", "--input_jsonl", records_path,
             "--tar_output_pattern", "tokens/audios/shard-%06d.tar",
             "--jsonl_output_pattern", "tokens/txts/shard-%06d.jsonl",
             "--tokenizer_path", model / "audio_tokenizer", "--nj_per_gpu", "1",
             "--loader_workers", "0", "--min_length", "1", "--max_length", "20"],
            p["log"], "OmniVoice 编码训练音频", cwd=work,
        )
        manifest = work / "tokens" / "data.lst"
        shards = [line.rsplit(maxsplit=3) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if not shards:
            raise ValueError("OmniVoice 未生成训练音频编码。请检查录音质量和日志。")
        sample_count = sum(int(count) for _, _, count, _ in shards)
        # WebDataset treats a Windows drive letter as a URL scheme.
        manifest.write_text("".join(
            f"{Path(audio).relative_to(work).as_posix()} {Path(labels).relative_to(work).as_posix()} {count} {seconds}\n"
            for audio, labels, count, seconds in shards
        ), encoding="utf-8")
        data = work / "data.json"
        dubbing.write_json(data, {"train": [{"manifest_path": [str(manifest)]}]})
        steps = max(2, math.ceil(sample_count / int(batch)) * int(epochs))
        settings = work / "train.json"
        dubbing.write_json(settings, {
            "init_from_checkpoint": str(model), "use_lora": True, "attn_implementation": "sdpa",
            "mixed_precision": "bf16", "num_workers": 1, "batch_tokens": 1600,
            "max_batch_size": int(batch), "min_sample_tokens": 24, "max_sample_tokens": 1600,
            "language_ratio": 1.0, "use_pinyin_ratio": 0.0, "instruct_ratio": 0.0,
            "only_instruct_ratio": 0.0, "steps": steps, "warmup_ratio": 0.01,
            "logging_steps": min(10, steps), "eval_steps": steps, "save_steps": steps,
            "keep_last_n_checkpoints": 1,
        })
        output = work / "checkpoints"
        yield from run([python, APP / "helpers" / TRAIN_HELPERS[dubbing.OMNI], "--device", device, "--config", settings,
                        "--data", data, "--output", output], p["log"], "OmniVoice LoRA 训练", cwd=work)
        checkpoint = output / f"checkpoint-{steps}"
        if not has_adapter(checkpoint):
            raise RuntimeError("OmniVoice 训练未生成可用的 LoRA。")
        p["lora"].mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(checkpoint / name, p["lora"] / name)


def register(spk, label=None, run_paths=None, engine=dubbing.INDEX):
    """训练完：参考音频拷到 speakers/<spk>/refs，写进 speakers.json（存相对路径）。"""
    p = run_paths or paths(spk)
    if not has_adapter(p["lora"]):
        raise RuntimeError("训练没有产出 LoRA 适配器配置。")
    refdir = p["spkdir"] / "refs" / p["lora"].name if run_paths else p["spkdir"] / "refs"
    refdir.mkdir(parents=True, exist_ok=True)
    for f in Path(p["refs"]).glob("*.wav"):
        shutil.copy(f, refdir / f.name)
    calm = sorted(refdir.glob("平稳_1_*.wav")) or sorted(refdir.glob("*.wav"))
    if not calm:
        clip = next(iter(sorted(Path(p["clips"]).glob("*.wav"))), None)
        if clip:
            calm = [Path(shutil.copy(clip, refdir / "默认参考.wav"))]
    reg = _load_raw()
    entry = reg.get(spk, {"label": label or spk, "presets": {}})
    if label:
        entry["label"] = label
    if engine == dubbing.INDEX:
        entry["lora"] = P.to_rel(p["lora"])
    else:
        entry.setdefault("adapters", {})[engine] = P.to_rel(p["lora"])
    references = {"默认": calm}
    for label, prefix in (("讲解", "平稳"), ("开场", "活泼"), ("闲聊", "放松")):
        references[label] = sorted(refdir.glob(f"{prefix}_*.wav"))
    presets = entry.setdefault("presets", {})
    for label, refs in references.items():
        if refs and label not in presets:
            presets[label] = {"ref": P.to_rel(refs[0]), "emo_method": "与音色参考音频相同", "emo_ref": "",
                              "emo_weight": 0.6, "temperature": 0.75, "speed": 0.97, "pause_ms": 150}
    reg[spk] = entry
    save_speakers(reg)
    return entry


def train_speaker(spk, label, minutes, lang, epochs, batch, engine=dubbing.INDEX):
    if engine not in TRAIN_ENGINES:
        raise ValueError("请选择支持 LoRA 的训练引擎。")
    if lang not in dubbing.supported_languages(engine):
        raise ValueError(f"{engine} 不支持该素材语言。")
    revision = input_revision(spk, minutes, lang)
    p = paths(spk, revision)
    if engine in TRAIN_HELPERS:
        python, config = dubbing.model_config(engine)
        yield from run([python, APP / "helpers" / TRAIN_HELPERS[engine], "--device", config["device"], "--check"],
                       p["log"], f"检查 {engine} 训练环境")
    p["work"].mkdir(parents=True, exist_ok=True)
    (p["base"] / ".latest").write_text(revision, encoding="ascii")
    stages = [
        ("prepare", step_prepare(p, minutes), lambda: has_audio(p["train"])),
        ("slice", step_slice(p), lambda: has_audio(p["clips"])),
        ("asr", step_transcribe(p, lang), lambda: p["lst"].is_file() and p["lst"].stat().st_size > 0),
    ]
    if engine == dubbing.INDEX:
        stages.append(("features", step_extract(p, lang), lambda: (p["feats"] / "train.json").is_file()))
    invalidate = False
    for name, task, ready in stages:
        marker = p["work"] / f".{name}.complete"
        if marker.is_file() and ready() and not invalidate:
            continue
        marker.unlink(missing_ok=True)
        if name != "features":
            (p["work"] / ".features.complete").unlink(missing_ok=True)
        invalidate = True
        yield from task
        if not ready():
            raise RuntimeError(f"训练步骤 {name} 未生成可用数据，请检查源素材和日志。")
        marker.touch()
    p["lora"] = p["spkdir"] / "models" / uuid4().hex[:12]
    if engine == dubbing.INDEX:
        yield from step_train(p, epochs, batch)
    elif engine == dubbing.QWEN:
        yield from step_train_qwen(p, epochs, batch, python, config)
    else:
        yield from step_train_omni(p, lang, epochs, batch, python, config)
    register(spk, label, p, engine)


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
        v["adapters"] = {engine: P.to_abs(directory) for engine, directory in v.get("adapters", {}).items()
                         if directory}
        if v["lora"]:
            v["adapters"][dubbing.INDEX] = v["lora"]
        for pr in v.get("presets", {}).values():
            for k in ("ref", "emo_ref"):
                pr[k] = P.to_abs(pr.get(k))
    return reg


def save_speakers(reg):
    d = {"_说明": "说话人登记表。路径相对于本工作室文件夹；新训练的说话人会自动加进来。", **reg}
    SPEAKERS.parent.mkdir(parents=True, exist_ok=True)
    pending = SPEAKERS.with_name(f".speakers-{uuid4().hex}.tmp")
    pending.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(SPEAKERS)
