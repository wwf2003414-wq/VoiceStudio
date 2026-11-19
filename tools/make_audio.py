"""一条命令出口播音频：文案 txt → IndexTTS 2.5 工作台（云端）生成 → 最终音频 + 参数记录 + 自动质检。

  python make_audio.py 文案.txt 输出.wav
  python make_audio.py 文案.txt 输出.wav --speaker xiaoge
  python make_audio.py 文案.txt 输出.wav --speaker cool --style 闲聊 --url https://9000-cpod-xxxx.pod.compshare.cn/

必须：云端工作台（优云智算 compshare 实例上的 IndexTTS 2.5 一体化工作台 WebUI）在运行，网址用 --url 或
环境变量 KOUBO_WORKBENCH_URL 传入；工作台里要已有对应说话人的 LoRA（runs/exp1_<speaker>/）。

输出：
  输出.wav                 工作台原样输出（22050 Hz、单声道、16 bit PCM；不做降噪/EQ/响度处理——这是已确认的交付做法）
  输出.params.json         本次用的全部参数
  输出.qc.json             质检：把音频转回文字和文案比对的结果（字错率、可疑片段）
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("KOUBO_ROOT", os.environ.get("VOICE_STUDIO_ROOT", ".")))
COOL = ROOT / "03_说话人模型" / "cool_交易员cool_配音工具包" / "2_参考音频"
XIAOGE = ROOT / "03_说话人模型" / "xiaoge"

# 每个说话人/风格的参数。出处见 README「C. 语音生成」。
PRESETS = {
    ("cool", "口播"): dict(  # 用户 2026-10-28 确认的 E2 设置；调速 2026-10-29 应用户要求从 0.9 改为 0.97
        ref=COOL / "cool_音色参考.wav", emo_method="使用情感参考音频", emo_ref=COOL / "cool_情感参考.wav",
        emo_weight=0.6, temperature=0.7, speed=0.97, pause_ms=180),
    ("cool", "闲聊"): dict(  # 闲聊 v2 样品里的设置（S02），用户没有明确选定，属于试验档
        ref=COOL / "cool_闲聊放松参考.wav", emo_method="与音色参考音频相同", emo_ref=None,
        emo_weight=0.6, temperature=0.8, speed=0.97, pause_ms=120),
    ("xiaoge", "口播"): dict(  # 用户 2026-10-30 确认的第 3 条
        ref=XIAOGE / "xiaoge_音色参考.wav", emo_method="与音色参考音频相同", emo_ref=None,
        emo_weight=0.6, temperature=0.75, speed=0.97, pause_ms=150),
}
COMMON = dict(fp16=True, top_p=0.8, top_k=20, num_beams=3, repetition_penalty=10.0, length_penalty=0.0,
              max_mel_tokens=1500, diffusion_steps=50, max_seg=120, fade_out_ms=30, do_sample=True)


def say(m):
    print(m, flush=True)


def detect_lang(text):
    cjk = len(re.findall(r"[一-鿿]", text))
    return "ZH" if cjk >= len(re.findall(r"[A-Za-z]", text)) / 4 else "EN"


def generate(url, speaker, p, text, lang, out_wav):
    from gradio_client import Client, handle_file
    c = Client(url, verbose=False)
    spk = [x[0] for x in c.predict("---选择说话人模型---", api_name="/refresh_speakers")["choices"]]
    if speaker not in spk:
        raise SystemExit(f"工作台里没有说话人 {speaker}（现有：{spk[1:]}）。需要先把 LoRA 放到工作台 runs/exp1_{speaker}/。")
    res = c.predict(
        speaker, COMMON["fp16"], p["emo_method"], handle_file(str(p["ref"])), text, lang,
        handle_file(str(p["emo_ref"])) if p["emo_ref"] else None, p["emo_weight"],
        0, 0, 0, 0, 0, 0, 0, 0, "",
        COMMON["max_seg"], p["speed"], p["pause_ms"], COMMON["fade_out_ms"], False,
        COMMON["do_sample"], COMMON["top_p"], COMMON["top_k"], p["temperature"], COMMON["length_penalty"],
        COMMON["num_beams"], COMMON["repetition_penalty"], COMMON["max_mel_tokens"], COMMON["diffusion_steps"],
        api_name="/do_generate",
    )
    flat = [r.get("value") if isinstance(r, dict) else r for r in res]
    flat = [r.get("path") if isinstance(r, dict) else r for r in flat]
    audio = next((r for r in flat if isinstance(r, str) and os.path.isfile(r)), None)
    msg = next((r for r in flat if isinstance(r, str) and not os.path.isfile(r)), "")
    if not audio:
        hint = "（CUDA error 时要重启整个云实例，只重启 WebUI 不够）" if "CUDA" in str(msg) else ""
        raise SystemExit(f"工作台没有返回音频：{str(msg)[:300]}{hint}")
    shutil.copy(audio, out_wav)
    return msg


CONTRACTIONS = {"n't": " not", "'re": " are", "'ll": " will", "'ve": " have", "'m": " am", "'d": " would",
                "let's": "let us", "it's": "it is", "here's": "here is", "that's": "that is", "there's": "there is",
                "what's": "what is", "he's": "he is", "she's": "she is", "who's": "who is"}
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                   "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                   "hundred thousand million billion point percent".split())


def en_words(s):
    """英文比对用：展开缩写，去掉数字和数字单词（ASR 常把 forty-three hundred 写成 4,300，不算读错）。"""
    s = s.lower().replace("’", "'")
    for k in sorted(CONTRACTIONS, key=len, reverse=True):
        s = s.replace(k, CONTRACTIONS[k])
    words = re.findall(r"[a-z0-9']+", s.replace("-", " "))
    return [w for w in words if not re.fullmatch(r"\d[\d,.]*", w) and w not in NUMBER_WORDS]


def norm_text(s):
    return re.sub(r"[^\w]", "", s.lower().replace("_", ""))


def edit_ops(ref, hyp):
    """字符级编辑距离，返回 (距离, 有差异的参考片段列表)。"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, bad = n, m, []
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
            continue
        bad.append(i - 1)
        if d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    bad += list(range(i))
    spans, cur = [], []
    for k in sorted(set(bad)):
        if cur and k - cur[-1] > 2:
            spans.append(cur)
            cur = []
        cur.append(k)
    if cur:
        spans.append(cur)
    sep = "" if isinstance(ref, list) and ref and len(ref[0]) == 1 else " "
    return d[n][m], [sep.join(ref[max(0, s[0] - 4): s[-1] + 5]) for s in spans]


def _local_funasr():
    """便携版：FunASR 模型放在 engine/index-tts/checkpoints 下，存在就直接用本地目录（不联网）。"""
    import os
    from pathlib import Path
    root = os.environ.get("VOICE_STUDIO_ROOT")
    if root:
        ck = Path(root) / "engine" / "index-tts" / "checkpoints"
        d = {k: ck / v for k, v in (("model", "paraformer-zh"), ("vad_model", "fsmn-vad"), ("punc_model", "ct-punc"))}
        if all(p.is_dir() for p in d.values()):
            return {k: str(v) for k, v in d.items()}
    return {"model": "paraformer-zh", "vad_model": "fsmn-vad", "punc_model": "ct-punc"}

def qc(wav, text, lang):
    """中文用 FunASR、英文用 Whisper 转回文字，和文案比对。"""
    import logging
    logging.disable(logging.INFO)
    import soundfile as sf
    import librosa
    if lang == "ZH":
        from funasr import AutoModel
        y, sr = sf.read(wav, dtype="float32")
        y = librosa.resample(y.mean(axis=1) if y.ndim > 1 else y, orig_sr=sr, target_sr=16000)
        m = AutoModel(**_local_funasr(), disable_update=True)
        hyp = m.generate(input=y, batch_size_s=300)[0]["text"]
        ref_n, hyp_n = list(norm_text(text)), list(norm_text(hyp))
    else:
        import imageio_ffmpeg
        import whisper
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not (exe.parent / "ffmpeg.exe").exists():
            shutil.copy(exe, exe.parent / "ffmpeg.exe")  # whisper 要求 PATH 里有名为 ffmpeg 的程序
        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
        hyp = whisper.load_model("turbo").transcribe(wav, language="en", condition_on_previous_text=False)["text"]
        ref_n, hyp_n = en_words(text), en_words(hyp)
    dist, spans = edit_ops(ref_n, hyp_n)
    rate = dist / max(1, len(ref_n))
    return {"asr_engine": "FunASR paraformer-zh" if lang == "ZH" else "Whisper turbo",
            "error_rate": round(rate, 4), "unit": "字" if lang == "ZH" else "词",
            "asr_text": hyp, "suspect_spans": spans[:20],
            "verdict": "通过" if rate <= 0.05 else ("需人工听" if rate <= 0.15 else "建议重做")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text_file")
    ap.add_argument("out_wav")
    ap.add_argument("--speaker", default="cool", choices=sorted({k[0] for k in PRESETS}))
    ap.add_argument("--style", default="口播", choices=["口播", "闲聊"])
    ap.add_argument("--url", default=os.environ.get("KOUBO_WORKBENCH_URL", ""))
    ap.add_argument("--speed", type=float, help="覆盖调速（数值越大越慢）")
    ap.add_argument("--no-qc", action="store_true")
    a = ap.parse_args()

    if not a.url:
        raise SystemExit("缺少工作台网址：加 --url，或设置环境变量 KOUBO_WORKBENCH_URL")
    key = (a.speaker, a.style)
    if key not in PRESETS:
        raise SystemExit(f"没有 {a.speaker} 的「{a.style}」设置，可用：{[k for k in PRESETS if k[0] == a.speaker]}")
    p = dict(PRESETS[key])
    if a.speed:
        p["speed"] = a.speed
    for k in ("ref", "emo_ref"):
        if p[k] and not Path(p[k]).is_file():
            raise SystemExit(f"找不到参考音频：{p[k]}")
    text = Path(a.text_file).read_text(encoding="utf-8-sig").strip()
    text = "".join(line.strip() for line in text.splitlines())  # 段落空行去掉，分句只看标点
    lang = detect_lang(text)
    out = Path(a.out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    say(f"说话人 {a.speaker}（{a.style}），语言 {lang}，{len(text)} 字符 → {out}")
    t0 = time.time()
    msg = generate(a.url.split("?")[0], a.speaker, p, text, lang, out)
    t_gen = time.time() - t0
    import soundfile as sf
    info = sf.info(str(out))
    say(f"生成完成：{info.duration:.1f} 秒音频，用时 {t_gen:.0f} 秒（{t_gen / max(info.duration, 1) * 60:.0f} 秒/每分钟音频）  {msg}")

    params = {"speaker": a.speaker, "style": a.style, "lang": lang, "workbench_url": a.url.split("?")[0],
              **{k: (str(v) if isinstance(v, Path) else v) for k, v in p.items()}, **COMMON,
              "text_file": str(Path(a.text_file).resolve()), "text_chars": len(text),
              "output": {"path": str(out.resolve()), "samplerate": info.samplerate, "channels": info.channels,
                         "subtype": info.subtype, "duration_s": round(info.duration, 2)},
              "generate_seconds": round(t_gen, 1), "workbench_message": msg,
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")

    if not a.no_qc:
        say("质检：转回文字和文案比对…")
        r = qc(str(out), text, lang)
        out.with_suffix(".qc.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        say(f"质检：{r['unit']}错误率 {r['error_rate'] * 100:.1f}% → {r['verdict']}")
        for s in r["suspect_spans"][:8]:
            say(f"  可疑：…{s}…")
    say(f"总用时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
