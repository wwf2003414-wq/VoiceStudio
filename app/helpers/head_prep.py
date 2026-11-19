"""「真人开头 + AI」的音频处理部分（在配音工具环境 .venv 里运行，复用 oneclick.py 里实测过的函数）。

  python head_prep.py prep   --src 原音频 --work 工作目录 --mode auto|own --cut 秒数
  python head_prep.py splice --work 工作目录 --ai AI人声.wav --out 成品.wav

prep 输出 工作目录/prep.json：
  vocals   去伴奏后的人声
  bgm      伴奏
  cut      真人部分到第几秒（auto：在 --cut 附近找一句话说完的停顿；own：整段开头音频说完的位置）
  lang     语言（ZH/EN/JA/ES/AR）
  text     后段文案（auto：转写+纠错；own：空，由用户填写）
  ref      参考音频（切点前 12 秒本人声音）
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

paths.setup_env()
import oneclick as oc  # noqa: E402


def cmd_prep(a):
    import logging
    logging.disable(logging.INFO)
    import numpy as np
    import soundfile as sf
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ff = oc.ffmpeg_exe()
    src_wav = work / "src.wav"
    print("转换格式…", flush=True)
    oc.to_wav(ff, a.src, str(src_wav))
    print("去掉背景音乐（Demucs）…", flush=True)
    vocals, bgm = oc.separate(src_wav, work / "sep")
    text, lang = "", None
    if a.mode == "auto":
        print("识别语言并转写…", flush=True)
        lang, sents, words = oc.transcribe(vocals)
        if not sents:
            raise SystemExit("没有识别出任何说话内容。")
        cut, idx = oc.find_cut(vocals, sents, a.cut)
        text = oc.build_tail_text(sents, idx, lang, words, cut)
    else:
        cut = oc.speech_end(vocals)
    y, sr = sf.read(str(vocals), dtype="float32")
    mono = y.mean(axis=1) if y.ndim > 1 else y
    ref = mono[int(max(0.05, cut - 12) * sr): int(cut * sr)]
    ref_path = work / "参考音频_本条开头.wav"
    sf.write(str(ref_path), ref / (np.abs(ref).max() + 1e-9) * 0.89, sr, subtype="PCM_16")
    info = {"vocals": str(vocals), "bgm": str(bgm), "cut": round(float(cut), 3), "lang": lang,
            "text": text, "ref": str(ref_path), "duration": round(sf.info(str(src_wav)).duration, 2)}
    (work / "prep.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：真人部分 0–{cut:.2f} 秒", flush=True)


def cmd_splice(a):
    work = Path(a.work)
    info = json.loads((work / "prep.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    stems = out.parent / "分轨"
    stems.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.ai, stems / "B_AI人声.wav")
    head_s, ai_s = oc.splice(Path(info["vocals"]), info["cut"], Path(a.ai), out, stems / "A_真人人声.wav")
    shutil.copy(info["bgm"], stems / "C_原背景音乐.wav")
    ff = oc.ffmpeg_exe()
    import subprocess
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-c:a", "libmp3lame",
                    "-b:a", "192k", str(out.with_suffix(".mp3"))], check=True)
    print(json.dumps({"head_s": round(head_s, 2), "ai_s": round(ai_s, 2), "gap_s": oc.GAP_S}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--src", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--mode", choices=["auto", "own"], default="auto")
    p.add_argument("--cut", type=float, default=10)
    p = sub.add_parser("splice")
    p.add_argument("--work", required=True)
    p.add_argument("--ai", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": cmd_prep, "splice": cmd_splice}[a.cmd](a)


if __name__ == "__main__":
    main()
