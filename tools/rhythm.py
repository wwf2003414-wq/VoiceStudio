"""Measure how uneven the delivery is: local speech-rate variation, pause variety, and loudness movement."""
import glob
import os
import sys

import numpy as np
import librosa

SR = 16000
HOP = 160


def rhythm(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time")
    iv = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=HOP)
    # syllable-ish onset rate inside each speech run of >=1.5 s
    rates = []
    for s, e in iv:
        t0, t1 = s / SR, e / SR
        if t1 - t0 < 1.5:
            continue
        n = ((onsets >= t0) & (onsets < t1)).sum()
        rates.append(n / (t1 - t0))
    gaps = np.array([(iv[i + 1][0] - iv[i][1]) / SR for i in range(len(iv) - 1)])
    gaps = gaps[gaps >= 0.06]
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    db = 20 * np.log10(rms[rms > rms.max() * 0.05] + 1e-9)
    rates = np.array(rates)
    return {
        "rate_cv": float(rates.std() / rates.mean()) if len(rates) > 1 else 0.0,
        "pause_cv": float(gaps.std() / gaps.mean()) if len(gaps) > 1 else 0.0,
        "pause_med": float(np.median(gaps)) if len(gaps) else 0.0,
        "pause_iqr": float(np.percentile(gaps, 75) - np.percentile(gaps, 25)) if len(gaps) > 3 else 0.0,
        "loud_std": float(db.std()),
        "runs": len(rates),
    }


def main():
    print(f"{'file':40s} rateCV pauseCV pauseMed pauseIQR loudStd runs")
    for pattern in sys.argv[1:]:
        for f in sorted(glob.glob(pattern)):
            r = rhythm(f)
            print(f"{os.path.basename(f)[:40]:40s} {r['rate_cv']:6.2f} {r['pause_cv']:7.2f} {r['pause_med']:8.2f} {r['pause_iqr']:8.2f} {r['loud_std']:7.1f} {r['runs']:4d}")


if __name__ == "__main__":
    main()
