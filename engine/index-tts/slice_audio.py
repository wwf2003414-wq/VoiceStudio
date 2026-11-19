#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_audio.py — 把长录音按静音切成 3~10 秒的短句（改进版）。

相比原版的改动：
1) 超长段不再从正中间硬切，而是换更灵敏的参数在内部重新找停顿，
   实在切不开才退回到“音量最低点”切 —— 避免拦腰截断字词。
2) 每个成片再用反转音频精修一次末尾静音，比 keep_silence 更干净。
3) 收尾加一个短淡出，消除硬切的咔哒声和残留尾音。

用法:
    python slice_audio.py --in_dir raw_wav --out_dir clips

参数:
    --in_dir            放原始（长）wav 的文件夹
    --out_dir           切好的短片输出文件夹
    --min_silence_ms    判定为停顿的最短静音时长(毫秒)，默认 400
    --silence_thresh_db 静音音量阈值(dBFS)。注意方向：越接近 0（如 -30）
                        修剪越狠，越负（如 -50）保留越多。默认 -40。
                        如果还是有尾音，先试 -35 / -30。
    --min_keep_ms       片段最短保留时长，默认 2000 (2秒)
    --max_keep_ms       片段最长时长，超过会再尝试细切，默认 12000 (12秒)
    --keep_silence_ms   每段首尾保留的静音，默认 120（原来 200 容易带尾音）
    --fade_out_ms       收尾淡出时长，默认 40
"""
import argparse
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence


def quietest_split_point(seg):
    """在中间 1/3 区域找音量最低的位置作为切点，避免切在字词中间。"""
    step = 10          # 每 10ms 采样一次
    win = 50           # 用 50ms 窗口估计瞬时音量
    lo, hi = len(seg) // 3, len(seg) * 2 // 3
    best_i, best_db = len(seg) // 2, float("inf")
    for i in range(lo, hi, step):
        db = seg[i:i + win].dBFS  # 纯静音是 -inf，正好最优
        if db < best_db:
            best_db, best_i = db, i
    return best_i


def smart_split(seg, args):
    """把过长的段递归切开，优先在真正的停顿处切。"""
    if len(seg) <= args.max_keep_ms:
        return [seg]

    # 段太长：换更灵敏的参数，在内部再找停顿（而不是从中间硬切）
    subs = split_on_silence(
        seg,
        min_silence_len=max(150, args.min_silence_ms // 2),
        silence_thresh=args.silence_thresh_db,
        keep_silence=args.keep_silence_ms,
    )

    # 一口气说太长、内部确实没停顿，才退回到“最安静点”切
    if len(subs) <= 1:
        mid = quietest_split_point(seg)
        return smart_split(seg[:mid], args) + smart_split(seg[mid:], args)

    out = []
    for s in subs:
        out.extend(smart_split(s, args))  # 子段可能仍超长，继续递归
    return out


def clean_tail(seg, args):
    """精修末尾静音 + 淡出，消除尾音和硬切咔哒。"""
    # 用反转音频检测“开头静音”= 原音频的末尾静音，然后裁掉
    end_sil = detect_leading_silence(
        seg.reverse(),
        silence_threshold=args.silence_thresh_db,
        chunk_size=5,
    )
    if 0 < end_sil < len(seg):
        seg = seg[: len(seg) - end_sil]
    # 短淡出收尾
    if len(seg) > args.fade_out_ms:
        seg = seg.fade_out(args.fade_out_ms)
    return seg


def slice_one(path, out_dir, args, idx_start):
    audio = AudioSegment.from_file(path)
    # 统一为单声道 + 24kHz（IndexTTS 用 24k，提前转好更省事）
    audio = audio.set_channels(1).set_frame_rate(24000)

    chunks = split_on_silence(
        audio,
        min_silence_len=args.min_silence_ms,
        silence_thresh=args.silence_thresh_db,
        keep_silence=args.keep_silence_ms,
    )

    saved = 0
    idx = idx_start
    for ch in chunks:
        if len(ch) < args.min_keep_ms:      # 太短的丢掉（多半是噪声/气口）
            continue
        for p in smart_split(ch, args):     # 过长的智能细切
            if len(p) < args.min_keep_ms:
                continue
            p = clean_tail(p, args)          # 每个成片精修末尾
            if len(p) < args.min_keep_ms:    # 精修后可能又变短，再判一次
                continue
            out_path = os.path.join(out_dir, f"clip_{idx:04d}.wav")
            p.export(out_path, format="wav")
            idx += 1
            saved += 1
    return saved, idx


def main():
    print("正在切片...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_silence_ms", type=int, default=400)
    ap.add_argument("--silence_thresh_db", type=int, default=-40)
    ap.add_argument("--min_keep_ms", type=int, default=2000)
    ap.add_argument("--max_keep_ms", type=int, default=12000)
    ap.add_argument("--keep_silence_ms", type=int, default=120)
    ap.add_argument("--fade_out_ms", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wavs = sorted(
        f for f in os.listdir(args.in_dir)
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))
    )
    if not wavs:
        print(f"[!] {args.in_dir} 里没找到音频文件")
        return

    idx = 1
    total = 0
    for w in wavs:
        n, idx = slice_one(os.path.join(args.in_dir, w), args.out_dir, args, idx)
        total += n
        print(f"  {w} -> 切出 {n} 段")
    print(f"\n[完成] 共切出 {total} 段短句，输出在: {args.out_dir}")
    print("接下来对这个文件夹跑 build_lst.py 做转录。")


if __name__ == "__main__":
    main()

