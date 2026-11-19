"""Compare generated takes against the original recordings: pitch, pace, pauses, brightness, timbre distance."""
import sys
import glob
import os

import numpy as np
import librosa

SR = 22050


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def features(y, n_chars=None):
    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=500, sr=SR, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    semis = 12 * np.log2(f0 / np.median(f0))
    intervals = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    speech_sec = sum(e - s for s, e in intervals) / SR
    gaps = [(intervals[i + 1][0] - intervals[i][1]) / SR for i in range(len(intervals) - 1)]
    gaps = [g for g in gaps if g >= 0.08]
    spec = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    hi = spec[freqs > 7000].sum()
    mid = spec[(freqs > 500) & (freqs < 4000)].sum()
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    return {
        "f0_med": float(np.median(f0)),
        "range_st": float(np.percentile(semis, 95) - np.percentile(semis, 5)),
        "rate": (n_chars / speech_sec) if n_chars else None,
        "pause_med": float(np.median(gaps)) if gaps else 0.0,
        "hi_db": float(10 * np.log10(hi / mid)),
        "mfcc": mfcc[1:].mean(axis=1),
        "dur": len(y) / SR,
    }


def main():
    ref_files = sys.argv[1].split(";")
    gen_dir = sys.argv[2]
    n_chars = int(sys.argv[3])
    refs = [features(load(f)) for f in ref_files]
    ref_mfcc = np.mean([r["mfcc"] for r in refs], axis=0)
    print(f"{'file':42s} f0Hz  rangeSt  rate  pause  hi_dB  mfccDist  dur")
    for f, r in zip(ref_files, refs):
        print(f"{'[orig] ' + os.path.basename(f):42s} {r['f0_med']:5.0f} {r['range_st']:6.1f}   {'-':>4s}  {r['pause_med']:.2f}  {r['hi_db']:6.1f}  {np.linalg.norm(r['mfcc'] - ref_mfcc):6.1f}   {r['dur']:.1f}")
    for f in sorted(glob.glob(os.path.join(gen_dir, "*.wav"))):
        r = features(load(f), n_chars)
        d = np.linalg.norm(r["mfcc"] - ref_mfcc)
        print(f"{os.path.basename(f):42s} {r['f0_med']:5.0f} {r['range_st']:6.1f}  {r['rate']:4.1f}  {r['pause_med']:.2f}  {r['hi_db']:6.1f}  {d:6.1f}   {r['dur']:.1f}")


if __name__ == "__main__":
    main()
