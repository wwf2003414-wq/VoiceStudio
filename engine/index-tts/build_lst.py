#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_lst.py — 对一个文件夹里的短句 wav 做 ASR 转录，生成 IndexTTS 微调用的 .lst。

输出每行格式:  音频绝对路径 <Tab> 文本

用法（中文，推荐）:
    python build_lst.py --audio_dir clips --out finetune_data/my_voice.lst

用法（英文/其它语言，用 whisper）:
    python build_lst.py --audio_dir clips --out finetune_data/my_voice.lst --engine whisper

参数:
    --audio_dir  存放短句 wav 的文件夹
    --out        输出的 .lst 路径
    --engine     funasr(默认, 中文最佳) 或 whisper(多语言)
    --whisper_model  whisper 模型大小, 默认 medium (英文可用 small/medium)
"""
import argparse
import os
import re
import wave
import contextlib


def wav_duration(path):
    """返回 wav 时长(秒)，读不到就返回 None。"""
    try:
        with contextlib.closing(wave.open(path, "r")) as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
        return None


def clean_text(t):
    """轻量清洗: 去首尾空格, 合并中文字之间多余空格。"""
    t = t.strip()
    # FunParaformer 偶尔在汉字间插空格, 去掉汉字-汉字之间的空格
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t


def transcribe_funasr(wav_paths):
    from funasr import AutoModel
    print("[*] 加载 FunASR 模型(首次会自动下载, 稍等)...")
    # 本地版：云端工作台把模型放在 checkpoints/ 下；本机没有就用模型名，从 MODELSCOPE_CACHE 读或自动下载
    def pick(local, name):
        p = os.path.join("checkpoints", local)
        return p if os.path.isdir(p) else name
    model = AutoModel(
        model=pick("paraformer-zh", "paraformer-zh"),
        vad_model=pick("fsmn-vad", "fsmn-vad"),
        punc_model=pick("ct-punc", "ct-punc"),   # 自动加标点, 对学停顿很重要
        disable_update=True,
    )
    out = {}
    for i, p in enumerate(wav_paths, 1):
        try:
            res = model.generate(input=p)
            text = res[0]["text"] if res else ""
        except Exception as e:
            print(f"  [!] 转录失败 {os.path.basename(p)}: {e}")
            text = ""
        out[p] = clean_text(text)
        print(f"  ({i}/{len(wav_paths)}) {os.path.basename(p)} -> {out[p]}")
    return out


def transcribe_whisper(wav_paths, model_size):
    from faster_whisper import WhisperModel
    print(f"[*] 加载 faster-whisper [{model_size}] ...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    out = {}
    for i, p in enumerate(wav_paths, 1):
        try:
            segments, _ = model.transcribe(p, beam_size=5)
            text = "".join(s.text for s in segments)
        except Exception as e:
            print(f"  [!] 转录失败 {os.path.basename(p)}: {e}")
            text = ""
        out[p] = clean_text(text)
        print(f"  ({i}/{len(wav_paths)}) {os.path.basename(p)} -> {out[p]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["funasr", "whisper"], default="funasr")
    ap.add_argument("--whisper_model", default="medium")
    args = ap.parse_args()

    # 收集 wav, 用绝对路径
    wav_paths = sorted(
        os.path.abspath(os.path.join(args.audio_dir, f))
        for f in os.listdir(args.audio_dir)
        if f.lower().endswith(".wav")
    )
    if not wav_paths:
        print(f"[!] {args.audio_dir} 里没有 wav 文件")
        return
    print(f"[*] 找到 {len(wav_paths)} 个 wav")

    # 时长检查 (建议 3~10 秒)
    too_short, too_long = [], []
    total_dur = 0.0
    for p in wav_paths:
        d = wav_duration(p)
        if d is None:
            continue
        total_dur += d
        if d < 2.0:
            too_short.append((os.path.basename(p), round(d, 1)))
        elif d > 15.0:
            too_long.append((os.path.basename(p), round(d, 1)))

    # 转录
    if args.engine == "funasr":
        texts = transcribe_funasr(wav_paths)
    else:
        texts = transcribe_whisper(wav_paths, args.whisper_model)

    # 写 .lst (Tab 分隔!), 空文本的跳过并提示
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written, empty = 0, []
    with open(args.out, "w", encoding="utf-8") as f:
        for p in wav_paths:
            t = texts.get(p, "").strip()
            if not t:
                empty.append(os.path.basename(p))
                continue
            f.write(f"{p}\t{t}\n")
            written += 1

    # 汇总
    print("\n" + "=" * 50)
    print(f"[完成] 写入 {written} 条到: {args.out}")
    print(f"       总时长约 {total_dur/60:.1f} 分钟")
    if too_short:
        print(f"[提醒] {len(too_short)} 条偏短(<2s), 建议删掉或合并: {too_short[:5]}")
    if too_long:
        print(f"[提醒] {len(too_long)} 条偏长(>15s), 建议再切短: {too_long[:5]}")
    if empty:
        print(f"[提醒] {len(empty)} 条没转出文本(已跳过), 检查音频是否有效: {empty[:5]}")
    print("=" * 50)
    print("\n>> 下一步: 人工打开 .lst 快速校对错别字/标点, 然后跑 extract_codec.py")


if __name__ == "__main__":
    main()

