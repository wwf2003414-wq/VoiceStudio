"""供工作室调用的配音引擎适配器。"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

import engine
import languages
import paths

INDEX = "IndexTTS 2.5"
QWEN = "Qwen3-TTS"
AUK = "AuK-Flash"
OMNI = "OmniVoice"
COMPARE_MODELS = (INDEX, QWEN, AUK)
MODELS = (*COMPARE_MODELS, OMNI)
MODEL_IDS = {INDEX: "index", QWEN: "qwen", AUK: "auk", OMNI: "omni"}
COMPARE = "三模型对比"
AUTO = "按语言自动选择"
MODEL_LANGUAGES = {
    INDEX: set(engine.LANGS.values()),
    QWEN: set(languages.QWEN_NAMES),
    AUK: {"ZH", "EN"},
    OMNI: set(languages.OMNI_CODES),
}
COMPARE_LANGUAGES = set.intersection(*(MODEL_LANGUAGES[model] for model in COMPARE_MODELS))
ALL_LANGUAGES = set.union(*MODEL_LANGUAGES.values())
RATE = 24000
MAX_CHUNK = 85


def preferred_model(lang):
    return next(model for model in MODELS if lang in MODEL_LANGUAGES[model])


def supported_languages(selection):
    if selection == AUTO:
        return ALL_LANGUAGES
    if selection == COMPARE:
        return COMPARE_LANGUAGES
    return MODEL_LANGUAGES.get(selection, ())


def write_json(file, data):
    temporary = file.with_name(f".{file.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(file)


def process(args):
    return subprocess.run(
        [str(arg) for arg in args], env=paths.child_env(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def ffmpeg(*args):
    result = process(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args])
    if result.returncode:
        raise ValueError("无法读取或处理音频。请检查文件格式及 FFmpeg 安装。")


def lines_for(text):
    lines = []
    for paragraph in (text or "").strip().splitlines():
        if not paragraph.strip():
            continue
        for sentence in re.split(r"(?<=[。！？!?؟।॥])\s*|(?<=\.)\s+", paragraph.strip()):
            while len(sentence) > MAX_CHUNK:
                breaks = list(re.finditer(r"[，、；：,;:\s]", sentence[:MAX_CHUNK]))
                end = next((m.end() for m in reversed(breaks) if m.end() >= MAX_CHUNK // 2), MAX_CHUNK)
                lines.append(sentence[:end].strip())
                sentence = sentence[end:].lstrip()
            if sentence:
                lines.append(sentence.strip())
    if not lines:
        raise ValueError("请输入要配音的文案。")
    return lines


def estimated_duration(text):
    cjk = re.findall(r"[\u3040-\u9fff\uac00-\ud7af]", text)
    rest = re.sub(r"[\u3040-\u9fff\uac00-\ud7af]", " ", text)
    units = len(cjk) / 3.7 + len(re.findall(r"[^\W_]+", rest)) / 2.45
    return max(1.1, min(28.0, units + 0.45))


def reference_audio(source, target):
    if not source or not Path(source).is_file():
        raise ValueError("请上传音色参考，或选择已有参考音频的说话人。")
    ffmpeg("-i", source, "-vn", "-ac", 1, "-ar", RATE, "-t", 30.1, target)
    y, sr = sf.read(target, dtype="float32")
    if not 1 <= len(y) / sr <= 30:
        raise ValueError("参考音频需为 1–30 秒，建议选 8–12 秒干净的单人录音。")
    if np.max(np.abs(y)) < 0.003:
        raise ValueError("参考音频接近静音，请换一段清楚的人声。")
    # 只裁首尾长静音，保留句内停顿和少量边界余量。
    active = np.flatnonzero(np.abs(y) >= max(0.001, float(np.max(np.abs(y)) * 0.02)))
    start = max(0, int(active[0]) - int(sr * 0.10))
    end = min(len(y), int(active[-1]) + int(sr * 0.20))
    if end - start < sr:
        raise ValueError("参考音频中的有效声音太短。")
    y = y[start:end]
    y *= min(3.0, 0.5 / float(np.max(np.abs(y))))
    sf.write(target, y, sr, subtype="PCM_16")


def model_config(model):
    if model == QWEN:
        prefix = "QWEN"
        model_dir = Path(os.environ.get("FENG_QWEN_MODEL", paths.MODELS / "Qwen3-TTS-12Hz-1.7B-Base"))
        required = [model_dir / name for name in ("config.json", "model.safetensors",
                                                 "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors")]
        default_device = "cuda:0"
    elif model == AUK:
        prefix = "AUK"
        model_dir = Path(os.environ.get("FENG_AUK_MODEL", paths.MODELS / "AuK-Flash"))
        encoder = Path(os.environ.get("FENG_AUK_ENCODER", paths.MODELS / "Qwen2.5-Omni-3B"))
        required = [model_dir / name for name in ("config.yaml", "auk_flash.safetensors", "vae.safetensors")]
        required.extend([encoder / "config.json", encoder / "model.safetensors.index.json"])
        default_device = "cpu"
    elif model == OMNI:
        prefix = "OMNI"
        model_dir = Path(os.environ.get("FENG_OMNI_MODEL", paths.MODELS / "OmniVoice"))
        required = [model_dir / name for name in ("config.json", "model.safetensors", "tokenizer.json",
                                                 "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors",
                                                 "audio_tokenizer/preprocessor_config.json")]
        default_device = "cuda:0"
    else:
        raise ValueError("找不到所选配音引擎。")
    expected_python = paths.ROOT / f".{prefix.lower()}-venv" / "Scripts" / "python.exe"
    python = Path(os.environ.get(f"FENG_{prefix}_PYTHON", expected_python))
    if model == AUK and required[-1].is_file():
        index = json.loads(required[-1].read_text(encoding="utf-8"))
        required.extend(encoder / shard for shard in set(index["weight_map"].values()))
    if not python.is_file() or not all(file.is_file() for file in required):
        raise ValueError(f"{model} 的运行环境或模型文件还未就绪。请检查 README 中的 FENG_{prefix}_* 配置。")
    result = {"model": str(model_dir.resolve()), "device": os.environ.get(f"FENG_{prefix}_DEVICE", default_device)}
    if model == AUK:
        result["encoder"] = str(encoder.resolve())
    return python, result


def run_model_worker(python, request_file, items, model, progress):
    log_file = request_file.with_suffix(".log")
    with log_file.open("w", encoding="utf-8") as log:
        for attempt in range(2):
            with subprocess.Popen(
                [str(python), "-X", "faulthandler", "-u", str(paths.APP / "helpers" / "clone_batch.py"),
                 str(request_file)],
                env=paths.child_env(), stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ) as child:
                while child.poll() is None:
                    done = sum(Path(item["path"]).exists() for item in items)
                    progress((done + 1) / (len(items) + 2), desc=f"{done}/{len(items)} 段")
                    time.sleep(1)
            code = child.returncode
            if not code:
                return
            print(f"Worker exit code: {code}", file=log, flush=True)
            if attempt or 0 < code < 0x80000000:
                break
            for item in items:
                Path(item["path"]).unlink(missing_ok=True)
            progress(0, desc="引擎进程意外退出，正在重试")
            time.sleep(1)
    raise ValueError(f"{model} 生成中断（退出码 {code}）。")


def generate_external(index_engine, model, text, lang, source, transcript, speed, pause_ms, log_dir, progress,
                      adapter=None):
    python, config = model_config(model)
    lines = lines_for(text)
    started = time.monotonic()
    index_engine.unload()
    with tempfile.TemporaryDirectory(prefix="feng-audio-") as temporary:
        folder = Path(temporary)
        reference = folder / "reference.wav"
        reference_audio(source, reference)
        transcript = (transcript or "").strip()
        if model == OMNI and not transcript:
            progress(0.05, desc="转写参考录音")
            result = process([sys.executable, paths.APP / "helpers" / "score_batch.py", "--ref", reference])
            lines_with_text = [line[len("REFJSON "):] for line in result.stdout.splitlines()
                               if line.startswith("REFJSON ")]
            if result.returncode or not lines_with_text or not (transcript := json.loads(lines_with_text[-1])):
                raise ValueError("无法自动转写参考录音。请手动填写参考录音逐字稿后重试。")
        items = [
            {"text": line, "lang": lang, "seconds": estimated_duration(line),
             "path": str(folder / f"{number:04d}.wav"), "seed": int.from_bytes(os.urandom(4), "big")}
            for number, line in enumerate(lines)
        ]
        request_file = folder / "batch.json"
        write_json(request_file, {
            "model_id": MODEL_IDS[model], **config, "reference": str(reference),
            "transcript": transcript, "items": items, "adapter": adapter,
        })
        log_file = request_file.with_suffix(".log")
        try:
            run_model_worker(python, request_file, items, model, progress)
        except ValueError as error:
            log_dir.mkdir(parents=True, exist_ok=True)
            saved_log = log_dir / f"{MODEL_IDS[model]}_{uuid4().hex[:12]}.log"
            shutil.copyfile(log_file, saved_log)
            raise ValueError(f"{error}日志：{saved_log}") from error

        gap = np.zeros(int(RATE * pause_ms / 1000), dtype=np.float32)
        tracks = []
        for number, item in enumerate(items):
            source_wav = Path(item["path"])
            if not source_wav.is_file() or not sf.info(source_wav).frames:
                raise ValueError(f"{model} 未生成有效音频。")
            adjusted = folder / "adjusted.wav"
            ffmpeg("-i", source_wav, "-af", f"atempo={1 / speed}", "-ac", 1, "-ar", RATE, adjusted)
            audio, _ = sf.read(adjusted, dtype="float32")
            if not audio.size:
                raise ValueError(f"{model} 返回了空音频。")
            if number:
                tracks.append(gap)
            tracks.append(audio)
        result = np.concatenate(tracks)
    return result, RATE, {"segments": len(items), "seconds": round(len(result) / RATE, 2),
                          "gen_seconds": round(time.monotonic() - started, 1)}
