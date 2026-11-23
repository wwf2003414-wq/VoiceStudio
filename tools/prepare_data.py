"""把下载好的原始音频变成训练集 + 参考音频。

  python prepare_data.py --raw raw_wav --work 工作目录 [--minutes 60] [--refs 已有参考1.wav;参考2.wav]

步骤：
  1. Demucs (htdemucs_ft) 去伴奏，只留人声（GPU 约每条 5 秒）
  2. 给每条打分：底噪越低越好 + 音色越接近"本人"越好
     - 给了 --refs：以这些本人片段为准
     - 没给：以所有视频的中位音色为准（多数视频是本人时成立）
  3. 去掉音色明显偏离的（嘉宾、换麦克风、重度处理），按分数挑到约 N 分钟 → train/
  4. 从训练集里截 8–12 秒参考片段，三种风格各几段 → refs/
       平稳（音色参考）  活泼（情感参考）  放松（闲聊风格）
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 24000


def separate(raw_dir, sep_dir):
    import torch
    todo = []
    for source in sorted(Path(raw_dir).glob("*.wav")):
        vocal = Path(sep_dir) / "htdemucs_ft" / f"{source.stem}_vocals.wav"
        try:
            complete = vocal.is_file() and sf.info(vocal).duration >= sf.info(source).duration - 0.1
        except RuntimeError:
            complete = False
        if not complete:
            todo.append(str(source))
    for offset in range(0, len(todo), 10):
        print(f"去伴奏 {offset + 1}–{min(len(todo), offset + 10)} / {len(todo)}…", flush=True)
        subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs_ft", "--two-stems=vocals",
                        "-d", "cuda" if torch.cuda.is_available() else "cpu", "-o", sep_dir,
                        "--filename", "{track}_{stem}.{ext}", *todo[offset:offset + 10]], check=True)
    for f in glob.glob(os.path.join(sep_dir, "htdemucs_ft", "*_no_vocals.wav")):
        os.remove(f)
    return sorted(glob.glob(os.path.join(sep_dir, "htdemucs_ft", "*_vocals.wav")))


def convert_media(raw, workspace):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from trainer import media_files

    folder = Path(workspace) / "converted"
    folder.mkdir(parents=True, exist_ok=True)
    for source in media_files(raw):
        identity = hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:12]
        target = folder / f"{source.stem[:64]}_{identity}.wav"
        if target.is_file():
            continue
        pending = target.with_suffix(".tmp.wav")
        result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                                 "-i", str(source), "-vn", "-ac", "1", "-ar", "44100", str(pending)],
                                capture_output=True, text=True, errors="replace")
        if result.returncode:
            pending.unlink(missing_ok=True)
            raise RuntimeError(f"无法读取训练素材 {source.name}：{result.stderr.strip()}")
        pending.replace(target)
    return str(folder)


def mfcc_mean(y):
    return librosa.feature.mfcc(y=y[: SR * 120], sr=SR, n_mfcc=20)[1:].mean(axis=1)


def noise_floor(y):
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=480)[0]
    return float(np.percentile(20 * np.log10(rms + 1e-9), 10))


def select(vocal_files, ref_files, out_dir, minutes):
    rows = []
    for f in vocal_files:
        y, _ = librosa.load(f, sr=SR, mono=True)
        if len(y) >= 2 * SR:
            rows.append({"f": f, "floor": noise_floor(y[:SR * 120]), "m": mfcc_mean(y)})
    if not rows:
        raise RuntimeError("没有解码出有效人声，请上传至少包含两秒人声的音频或视频。")
    if ref_files:
        target = np.mean([mfcc_mean(librosa.load(r, sr=SR, mono=True)[0]) for r in ref_files], axis=0)
    else:
        target = np.median([r["m"] for r in rows], axis=0)
    for r in rows:
        r["dist"] = float(np.linalg.norm(r["m"] - target))
    d = np.array([r["dist"] for r in rows])
    fl = np.array([r["floor"] for r in rows])
    cap = np.percentile(d, 75) + 1.0 * (np.percentile(d, 75) - np.percentile(d, 25))
    z = (fl - fl.mean()) / (fl.std() + 1e-9) + (d - d.mean()) / (d.std() + 1e-9)
    os.makedirs(out_dir, exist_ok=True)
    total = 0.0
    for i in np.argsort(z):
        r = rows[i]
        name = os.path.basename(r["f"]).replace("_vocals", "")
        keep = total < minutes * 60 and r["dist"] <= cap
        print(f"{'KEEP' if keep else 'skip'} {name:32s} 底噪 {r['floor']:6.1f} dB  音色距离 {r['dist']:5.1f}"
              f"{'  (偏离本人，排除)' if r['dist'] > cap else ''}")
        if keep:
            margin = 2 if sf.info(r["f"]).duration > 12 else 0
            remaining = max(0, minutes * 60 - total)
            y, _ = librosa.load(r["f"], sr=SR, mono=True, offset=margin,
                                duration=min(remaining, sf.info(r["f"]).duration - 2 * margin))
            if len(y) < 2 * SR:
                continue
            sf.write(os.path.join(out_dir, name), (y / (np.abs(y).max() + 1e-9) * 0.89).astype(np.float32), SR, subtype="PCM_16")
            total += len(y) / SR
    print(f"训练集 {total / 60:.1f} 分钟 -> {out_dir}（音色距离上限 {cap:.1f}）")
    if total < 30 * 60:
        print("⚠ 不足 30 分钟：效果会打折，建议补充素材。")
    return target


def cut_refs(train_dir, out_dir, per_style=3):
    """从训练集里截 8–12 秒、一口气说完、停顿短的片段，按三种风格各挑几段（每段来自不同视频）。"""
    cands = []
    for f in sorted(glob.glob(os.path.join(train_dir, "*.wav"))):
        y, _ = librosa.load(f, sr=SR, mono=True)
        iv = librosa.effects.split(y, top_db=38, frame_length=1024, hop_length=240)
        start = None
        for k in range(len(iv)):
            s, e = iv[k]
            start = s if start is None else start
            dur = (e - start) / SR
            nxt_gap = (iv[k + 1][0] - e) / SR if k + 1 < len(iv) else 9
            if 8 <= dur <= 12 and nxt_gap >= 0.35:
                seg = y[max(0, start - int(0.15 * SR)): e + int(0.2 * SR)]
                f0, vf, _ = librosa.pyin(seg, fmin=70, fmax=500, sr=SR, frame_length=1200)
                f0 = f0[vf & ~np.isnan(f0)]
                inner = [(iv[j + 1][0] - iv[j][1]) / SR for j in range(len(iv) - 1) if start <= iv[j][0] and iv[j + 1][1] <= e]
                if len(f0) >= 50:
                    st = 12 * np.log2(f0 / np.median(f0))
                    cands.append({"file": os.path.basename(f), "t": start / SR, "seg": seg, "f0": float(np.median(f0)),
                                  "range": float(np.percentile(st, 95) - np.percentile(st, 5)),
                                  "voiced": len(f0) / len(vf), "max_gap": max(inner) if inner else 0.0})
                start = None
            elif dur > 12:
                start = None
    if not cands:
        print("没有截到合适的参考片段")
        return
    f0_med = float(np.median([c["f0"] for c in cands]))
    styles = {
        # 平稳：本人常用音区、起伏中等 → 做"音色参考"
        "平稳": lambda c: abs(c["f0"] - f0_med) / 10 + abs(c["range"] - 11.5) / 2 - c["voiced"] * 3,
        # 活泼：起伏大、语速密 → 做"情感参考"
        "活泼": lambda c: abs(c["f0"] - f0_med * 1.03) / 10 + abs(c["range"] - 13.5) / 2 - c["voiced"] * 4 + max(0, c["max_gap"] - 0.35) * 6,
        # 放松：音区偏低（不提着）、起伏中等、停顿少 → 做"闲聊"参考，解决"吊嗓子/播音腔"
        "放松": lambda c: max(0.0, c["f0"] - f0_med * 0.94) / 6 + abs(c["range"] - 11.5) / 2 - c["voiced"] * 5 + max(0, c["max_gap"] - 0.3) * 6,
    }
    os.makedirs(out_dir, exist_ok=True)
    for style, score in styles.items():
        used, n = set(), 0
        for c in sorted(cands, key=score):
            if c["file"] in used:
                continue
            used.add(c["file"])
            n += 1
            name = f"{style}_{n}_{c['file'][:12]}_{c['t']:.0f}s.wav"
            sf.write(os.path.join(out_dir, name), c["seg"] / (np.abs(c["seg"]).max() + 1e-9) * 0.89, SR, subtype="PCM_16")
            print(f"{style}参考 {name:40s} 音高 {c['f0']:5.0f} Hz  起伏 {c['range']:4.1f} 半音")
            if n >= per_style:
                break
    print(f"本人中位音高约 {f0_med:.0f} Hz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="download.py 的输出目录")
    ap.add_argument("--work", required=True, help="工作目录（会建 sep/ train/ refs/）")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--refs", default="", help="可选：用分号分隔的本人干净片段，用来判断哪些是本人")
    a = ap.parse_args()
    vocals = separate(convert_media(a.raw, a.work), os.path.join(a.work, "sep"))
    select(vocals, [r for r in a.refs.split(";") if r], os.path.join(a.work, "train"), a.minutes)
    cut_refs(os.path.join(a.work, "train"), os.path.join(a.work, "refs"))


if __name__ == "__main__":
    main()
