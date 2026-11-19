"""一键配音：开头保留真人原声，后面用工作台里训练好的说话人模型生成，输出纯人声。

两种模式：
  1 = 只给开头一小段真人音频（比如前 10 秒），后面念你自己写的文案
  2 = 给完整音频，保留前 N 秒，后面的内容自动转写后再用说话人模型重念

用法：
  双击桌面上的「<说话人>一键配音」（用 make_launcher.py 生成），选择音频/视频文件；或者把文件拖到图标上。
  音频和 .txt 文案一起拖上去，会直接按模式 1 用这份文案。
  命令行：python oneclick.py --speaker 名字 音频 [文案.txt] [--mode 1|2] [--cut 10] [--no-review]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "oneclick_config.json"
SPEAKER = "cool"  # 由 --speaker 覆盖
REPLACE_FILE = HERE / "替换词表.txt"  # 所有说话人共用；错字一行一个：错的=对的
FINAL_PUNCT = "。！？.!?"
WHISPER_TO_WB = {"zh": "ZH", "en": "EN", "ja": "JA", "es": "ES", "ar": "AR"}
LANG_NAME = {"ZH": "中文", "EN": "英文", "JA": "日语", "ES": "西班牙语", "AR": "阿拉伯语"}

# 已确认的生成设置（与工具包 3_预设 一致；参考音频改用本条音频开头她本人的声音）
GEN = dict(temperature=0.75, top_p=0.8, top_k=20, num_beams=3, repetition_penalty=10.0,
           max_mel_tokens=1500, diffusion_steps=50, pause_ms=150, max_seg=120)
# 工作台"调速"参数：数值越大越慢。三档：1 慢一点 / 2 标准 / 3 快一点
PACE = {
    "ZH": {"1": 1.03, "2": 0.97, "3": 0.9},
    "OTHER": {"1": 1.13, "2": 1.07, "3": 1.0},
}
PACE_NAME = {"1": "慢一点", "2": "标准", "3": "快一点"}


def speed_for(lang, pace):
    """pace 是档位 1/2/3，或者直接输入的调速数值。"""
    if pace in PACE_NAME:
        return PACE["ZH" if lang == "ZH" else "OTHER"][pace]
    return float(pace)


def ask_pace():
    say("\n语速选哪档？")
    say("  1 = 慢一点    2 = 标准（推荐）    3 = 快一点")
    say("  也可以直接输入调速数值，例如 1.0（数值越大越慢，0.85–1.15 之间比较自然）")
    s = input("直接回车 = 2：").strip()
    if not s:
        return "2"
    if s in PACE_NAME:
        return s
    try:
        v = float(s)
        if 0.5 <= v <= 2.0:
            return s
    except ValueError:
        pass
    say("没看懂，按标准语速处理。")
    return "2"
GAP_S = 0.15  # 真人段和 AI 段之间的停顿
# 中文转写提示：写上这个人常聊的领域，能明显减少专有名词识别错误
ASR_PROMPT = "以下是普通话的口播，内容可能涉及财经、黄金、美股、美联储、降息、美元指数、地缘局势。"


def say(msg=""):
    print(msg, flush=True)


def step(n, total, msg):
    say(f"\n[{n}/{total}] {msg}")


# ------------------------------------------------------------------ 准备 --
def ffmpeg_exe():
    import imageio_ffmpeg
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    plain = exe.with_name("ffmpeg.exe")
    if not plain.exists():
        shutil.copy(exe, plain)  # whisper 需要名为 ffmpeg 的程序
    os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(exe)


def pick_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    p = filedialog.askopenfilename(
        title="选择要处理的音频或视频",
        filetypes=[("音频/视频", "*.mp3 *.wav *.m4a *.aac *.flac *.mp4 *.mov *.mkv"), ("所有文件", "*.*")])
    root.destroy()
    return p


def url_ok(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/gradio_api/info", timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_url():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    url = cfg.get("url", "")
    if url and url_ok(url):
        say(f"工作台连接正常：{url}")
        return url
    if url:
        say(f"上次的工作台地址连不上：{url}")
        say("（实例关机或重启后地址会变，请到优云智算控制台确认实例在运行、WebUI 已启动）")
    while True:
        url = input("请粘贴工作台网址（例如 https://9000-cpod-xxxx.pod.compshare.cn/），然后按回车：").strip()
        if not url:
            continue
        if not url.startswith("http"):
            url = "https://" + url
        url = url.split("?")[0]
        if url_ok(url):
            CONFIG.write_text(json.dumps({"url": url}, ensure_ascii=False), encoding="utf-8")
            say("连接成功，已记住这个地址。")
            return url
        say("还是连不上。WebUI 刚启动的话等 1 分钟再试；也请检查网址是否复制完整。")


def ask_cut(default):
    s = input(f"前面保留多少秒真人声音？直接回车 = {default} 秒：").strip()
    try:
        return float(s) if s else float(default)
    except ValueError:
        say(f"没看懂，按 {default} 秒处理。")
        return float(default)


# ---------------------------------------------------------------- 音频处理 --
def to_wav(ff, src, dst):
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", dst],
                   check=True)


def separate(src_wav, sep_dir):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs_ft", "--two-stems=vocals", "-d", dev,
                    "-o", str(sep_dir), "--filename", "{stem}.{ext}", str(src_wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = sep_dir / "htdemucs_ft"
    return d / "vocals.wav", d / "no_vocals.wav"


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

def transcribe(vocals):
    """返回 (工作台语言代码, 断句 [(start, end, text)], 中文逐字 [(start, 字)] 或 None)

    中文：断句时间用 FunASR（句子边界准），文字用 Whisper（不容易出现"升升已已经"这种重复乱码），
    再用 FunASR 的标点模型补标点。其他语言全部用 Whisper。
    """
    import torch
    import whisper
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = whisper.load_model("turbo", device=dev)
    a = whisper.pad_or_trim(whisper.load_audio(str(vocals)))
    mel = whisper.log_mel_spectrogram(a, n_mels=m.dims.n_mels).to(m.device)
    _, probs = m.detect_language(mel)
    code = max(probs, key=probs.get)
    lang = WHISPER_TO_WB.get(code)
    if not lang:
        raise SystemExit(f"识别到的语言是 {code}，工作台不支持（只支持中/英/日/西/阿）。")
    say(f"识别到语言：{LANG_NAME[lang]}")
    if lang != "ZH":
        r = m.transcribe(str(vocals), language=code, condition_on_previous_text=False)
        return lang, [(s["start"], s["end"], s["text"].strip()) for s in r["segments"]], None
    r = m.transcribe(str(vocals), language="zh", word_timestamps=True, condition_on_previous_text=False,
                     initial_prompt=ASR_PROMPT)
    words = [(w["start"], w["word"].strip()) for s in r["segments"] for w in s.get("words", [])]
    del m
    torch.cuda.empty_cache()
    import librosa
    import soundfile as sf
    from funasr import AutoModel
    y, sr = sf.read(str(vocals), dtype="float32")
    y = librosa.resample(y.mean(axis=1) if y.ndim > 1 else y, orig_sr=sr, target_sr=16000)
    asr = AutoModel(**_local_funasr(), disable_update=True, device=dev)
    r = asr.generate(input=y, batch_size_s=300, sentence_timestamp=True)[0]
    return lang, [(s["start"] / 1000, s["end"] / 1000, s["text"]) for s in r.get("sentence_info", [])], words


def find_cut(vocals, sents, target):
    """在 target 附近找一句话真正说完的地方，再在那段停顿里找最安静的点。"""
    import numpy as np
    import soundfile as sf
    idx = next((i for i, s in enumerate(sents) if s[2].rstrip()[-1:] in FINAL_PUNCT and s[1] >= target - 0.5), None)
    if idx is None:
        idx = next((i for i, s in enumerate(sents) if s[1] >= target - 0.5), None)
    if idx is None or idx == len(sents) - 1:
        raise SystemExit("这条音频在你指定的秒数之后已经没有内容了，请把秒数改小一点。")
    end, nxt = sents[idx][1], sents[idx + 1][0]
    y, sr = sf.read(str(vocals), dtype="float32")
    mono = y.mean(axis=1) if y.ndim > 1 else y
    lo, hi = int(max(0, end - 0.4) * sr), int((max(nxt, end) + 0.2) * sr)
    win = int(0.01 * sr)
    seg = mono[lo:hi]
    db = 20 * np.log10(np.sqrt((seg[: len(seg) // win * win].reshape(-1, win) ** 2).mean(axis=1)) + 1e-9)
    run = np.convolve(db, np.ones(6) / 6, mode="valid")
    cut = (lo + (int(np.argmin(run)) + 3) * win) / sr
    return cut, idx


def load_replacements():
    if not REPLACE_FILE.exists():
        return []
    pairs = []
    for line in REPLACE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            a, b = line.split("=", 1)
            pairs.append((a.strip(), b.strip()))
    return pairs


def build_tail_text(sents, idx, lang, words=None, cut=None):
    if words is not None:
        # 中文：取切点之后 Whisper 的逐字结果，再补标点
        from funasr import AutoModel
        raw = "".join(w for s, w in words if s >= cut - 0.05)
        raw = re.sub(r"[，。？！、；：,.?!;:\s]+", "", raw)  # 去掉 Whisper 自带的零散标点，统一交给标点模型
        punc = AutoModel(model=_local_funasr()["punc_model"], disable_update=True)
        text = punc.generate(input=raw)[0]["text"]
        text = re.sub(r"(?<=[一-鿿\d]) +(?=[一-鿿\d])", "", text)
        text = text.replace(",", "，").replace("?", "？").replace("!", "！")
        text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)
        text = re.sub(r"([，。？！])[，。？！\s]+", r"\1", text)
    else:
        parts = [s[2] for s in sents[idx + 1:]]
        text = "".join(parts) if lang == "JA" else " ".join(parts)
    if lang == "ZH":
        text = text.replace("呃", "").replace("，，", "，")
    for a, b in load_replacements():
        text = text.replace(a, b)
    return text.strip()


def review(text_path):
    say(f"\n后段文案已打开（{text_path.name}）。")
    say("请在记事本里检查一遍：改掉识别错的字、数字尽量写成汉字（四千四百五十）、句子用逗号连起来。")
    say("改完按 Ctrl+S 保存，然后回到这个黑色窗口按回车继续。不需要改就直接按回车。")
    subprocess.Popen(["notepad.exe", str(text_path)])
    input()


def generate(url, ref_path, text, lang, out_wav, speed):
    from gradio_client import Client, handle_file
    c = Client(url, verbose=False)
    spk = [x[0] for x in c.predict("---选择说话人模型---", api_name="/refresh_speakers")["choices"]]
    # 这次 refresh 也会更新工作台下拉框的可选项，新训练的说话人才能被选中
    if SPEAKER not in spk:
        raise SystemExit(f"工作台里没有 {SPEAKER} 说话人模型（现有：{spk[1:]}）。新实例要先把它的两个 adapter 文件上传到工作台的 runs/exp1_{SPEAKER}/。")
    restart_hint = ("工作台在生成时出错了。通常是云端显卡卡住了：到优云智算控制台重启 IndexTTS 工作台（WebUI），"
                    "等它启动好后再双击运行一次。")
    try:
        res = c.predict(
            SPEAKER, True, "与音色参考音频相同", handle_file(str(ref_path)), text, lang,
            None, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, "",
            GEN["max_seg"], speed, GEN["pause_ms"], 30, False,
            True, GEN["top_p"], GEN["top_k"], GEN["temperature"], 0.0, GEN["num_beams"], GEN["repetition_penalty"],
            GEN["max_mel_tokens"], GEN["diffusion_steps"],
            api_name="/do_generate",
        )
    except Exception as e:
        raise SystemExit(f"{restart_hint}\n（详细信息：{e}）")
    flat = [r.get("value") if isinstance(r, dict) else r for r in res]
    flat = [r.get("path") if isinstance(r, dict) else r for r in flat]
    audio = next((r for r in flat if isinstance(r, str) and os.path.isfile(r)), None)
    if not audio:
        msg = next((r for r in flat if isinstance(r, str)), str(res))
        raise SystemExit(f"{restart_hint}\n（工作台提示：{msg[:200]}）" if "CUDA" in msg else f"工作台没有返回音频：{msg[:300]}")
    shutil.copy(audio, out_wav)


def active_rms(x, sr):
    import numpy as np
    w = int(0.02 * sr)
    f = np.sqrt((x[: len(x) // w * w].reshape(-1, w) ** 2).mean(axis=1))
    f = f[f > f.max() * 0.1]
    return float(np.sqrt((f ** 2).mean()))


def splice(vocals, cut, ai_wav, out_wav, head_out):
    import numpy as np
    import soundfile as sf
    import librosa
    y, sr = sf.read(str(vocals), dtype="float32")
    head = (y.mean(axis=1) if y.ndim > 1 else y)[: int(cut * sr)]
    sf.write(str(head_out), head, sr, subtype="PCM_16")
    ai, sr_a = sf.read(str(ai_wav), dtype="float32")
    ai = librosa.resample(ai.mean(axis=1) if ai.ndim > 1 else ai, orig_sr=sr_a, target_sr=sr)
    lead = librosa.effects.split(ai, top_db=40)[0][0]
    ai = ai[max(0, lead - int(0.03 * sr)):]
    ai *= active_rms(head, sr) / (active_rms(ai, sr) + 1e-9)
    out = np.concatenate([head, np.zeros(int(GAP_S * sr), dtype=np.float32), ai])
    peak = np.abs(out).max()
    if peak > 0.97:
        out *= 0.97 / peak
    sf.write(str(out_wav), out, sr, subtype="PCM_16")
    return len(head) / sr, len(ai) / sr


# -------------------------------------------------------------------- 主流程 --
TEXT_EXTS = (".txt",)


def detect_text_lang(text):
    cjk = len(re.findall(r"[一-鿿]", text))
    kana = len(re.findall(r"[぀-ヿ]", text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if kana > 5:
        return "JA"
    if cjk >= letters / 4:
        return "ZH"
    if re.search(r"[؀-ۿ]", text):
        return "AR"
    return "EN"


def ask_mode(dur):
    rec = "1" if dur <= 30 else "2"
    say(f"\n这条音频有 {dur:.0f} 秒。这次怎么做？")
    say("  1 = 这段音频全部保留当开头，后面念我自己写的文案")
    say(f"  2 = 这是完整音频，保留前几秒，后面的内容自动转写再用 AI 重念")
    s = input(f"输入 1 或 2，直接回车 = {rec}：").strip()
    return s if s in ("1", "2") else rec


def get_own_text(out, given_txt, no_review):
    text_path = out / "后段文案_AI念的内容.txt"
    if given_txt:
        text_path.write_text(Path(given_txt).read_text(encoding="utf-8-sig"), encoding="utf-8")
    elif not text_path.exists():
        text_path.write_text("", encoding="utf-8")
    while True:
        if not (given_txt and no_review):
            say(f"\n记事本已打开（{text_path.name}）。")
            say("把要让 AI 念的文案粘贴进去（中文或英文都行），按 Ctrl+S 保存，然后回到这个黑色窗口按回车。")
            subprocess.Popen(["notepad.exe", str(text_path)])
            input()
        text = text_path.read_text(encoding="utf-8-sig").strip()
        if text:
            return text_path, text
        say("文案还是空的，请粘贴后保存再按回车。")
        given_txt = None


def speech_end(vocals):
    """开头音频里最后一个字说完的位置（去掉结尾的静音，留 0.05 秒）。"""
    import librosa
    import soundfile as sf
    y, sr = sf.read(str(vocals), dtype="float32")
    mono = y.mean(axis=1) if y.ndim > 1 else y
    iv = librosa.effects.split(mono, top_db=40, frame_length=1024, hop_length=256)
    if len(iv) == 0:
        raise SystemExit("开头音频里没有检测到说话声音。")
    return min(len(mono), iv[-1][1] + int(0.05 * sr)) / sr


def main():
    import logging
    logging.disable(logging.INFO)  # 隐藏识别模型和网络请求的大量日志
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="cool", help="工作台里的说话人模型名")
    ap.add_argument("files", nargs="*", help="音频/视频文件，可以再加一个 .txt 文案")
    ap.add_argument("--cut", type=float)
    ap.add_argument("--mode", choices=["1", "2"])
    ap.add_argument("--pace", help="语速：1 慢一点 / 2 标准 / 3 快一点，或直接写调速数值")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    global SPEAKER
    SPEAKER = a.speaker

    say("=" * 56)
    say(f"  {SPEAKER} 一键配音：前段保留真人，后段用 {SPEAKER} 模型生成（纯人声）")
    say("=" * 56)
    txt = next((f for f in a.files if f.lower().endswith(TEXT_EXTS)), None)
    src = next((f for f in a.files if not f.lower().endswith(TEXT_EXTS)), None) or pick_file()
    if not src or not os.path.isfile(src):
        say("没有选择文件，已退出。")
        return
    src = Path(src)
    say(f"文件：{src.name}" + (f"   文案：{Path(txt).name}" if txt else ""))
    ff = ffmpeg_exe()

    out = src.parent / f"{src.stem}_{SPEAKER}成品"
    work = out / "_过程文件"
    stems = out / "分轨"
    for d in (work, stems):
        d.mkdir(parents=True, exist_ok=True)
    src_wav = work / "src.wav"
    to_wav(ff, str(src), str(src_wav))
    import soundfile as sf
    dur = sf.info(str(src_wav)).duration
    mode = a.mode or ("1" if txt else ask_mode(dur))
    cut_target = None
    if mode == "2":
        cut_target = a.cut if a.cut is not None else ask_cut(10)
    pace = a.pace or ask_pace()
    url = ensure_url()
    total = 5 if mode == "1" else 6
    t0 = time.time()

    step(1, total, "去掉背景音乐（约半分钟）…")
    vocals, bgm = separate(src_wav, work / "sep")

    if mode == "1":
        step(2, total, "准备文案…")
        text_path, text = get_own_text(out, txt, a.no_review)
        lang = detect_text_lang(text)
        say(f"文案语言：{LANG_NAME[lang]}，共 {len(text)} 个字符")
        cut = speech_end(vocals)
        say(f"开头真人部分：0–{cut:.2f} 秒（全部保留）")
        if dur > 30:
            say("（提示：开头音频比较长，会全部保留。只想保留前几秒的话，选模式 2）")
        gen_step = 3
    else:
        step(2, total, "识别语言并转写全文（约 1 分钟）…")
        lang, sents, words = transcribe(vocals)
        if not sents:
            raise SystemExit("没有识别出任何说话内容。")
        step(3, total, f"在第 {cut_target:g} 秒附近找断句点…")
        cut, idx = find_cut(vocals, sents, cut_target)
        head_text = ("" if lang in ("ZH", "JA") else " ").join(s[2] for s in sents[: idx + 1])
        say(f"切在 {cut:.2f} 秒。真人部分最后一句：…{head_text[-40:]}")
        if abs(cut - cut_target) > 2:
            say(f"（{cut_target:g} 秒时她还没说完这句话，所以往后顺延到 {cut:.2f} 秒，避免把一句话切成两半）")
        text_path = out / "后段文案_AI念的内容.txt"
        text_path.write_text(build_tail_text(sents, idx, lang, words, cut), encoding="utf-8")
        if not a.no_review:
            review(text_path)
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit("后段文案是空的，已退出。")
        gen_step = 4

    speed = speed_for(lang, pace)
    step(gen_step, total, f"用 {SPEAKER} 生成后段{LANG_NAME[lang]}，语速：{PACE_NAME.get(pace, '自定义')}（调速 {speed}）"
                          "，通常 2–4 分钟，期间不要关闭窗口…")
    import numpy as np
    y, sr = sf.read(str(vocals), dtype="float32")
    mono = y.mean(axis=1) if y.ndim > 1 else y
    ref = mono[int(max(0.05, cut - 12) * sr): int(cut * sr)]  # 离接缝最近的 12 秒，音色最容易接上
    ref_path = work / "参考音频_本条开头.wav"
    sf.write(str(ref_path), ref / (np.abs(ref).max() + 1e-9) * 0.89, sr, subtype="PCM_16")
    ai_wav = stems / "B_AI人声.wav"
    generate(url, ref_path, text, lang, ai_wav, speed)

    step(total, total, "拼接成品…")
    final = out / "成品_纯人声.wav"
    head_s, ai_s = splice(vocals, cut, ai_wav, final, stems / "A_真人人声.wav")
    shutil.copy(bgm, stems / "C_原背景音乐.wav")
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(final), "-c:a", "libmp3lame",
                    "-b:a", "192k", str(final.with_suffix(".mp3"))], check=True)

    say("\n" + "=" * 56)
    say(f"完成！用时 {(time.time() - t0) / 60:.1f} 分钟")
    say(f"  真人 0–{head_s:.2f} 秒 + AI {ai_s:.0f} 秒，成品共 {head_s + GAP_S + ai_s:.0f} 秒")
    say(f"  成品：{final.name} / {final.with_suffix('.mp3').name}")
    say(f"  文件夹：{out}")
    say("=" * 56)
    if not a.no_open:
        os.startfile(out)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            say(f"\n出错了：{e.code}")
    except Exception as e:
        say(f"\n出错了：{type(e).__name__}: {e}")
        say("可以把这个窗口截图发给 Claude 看。")
