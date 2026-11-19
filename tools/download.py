"""下载 media.json 里本人的视频，转成单声道 44.1k wav。

  python download.py --media media.json --out raw_wav --profile-url 抖音主页 [--min-sec 15]
"""
import argparse
import json
import os
import re
import subprocess
import urllib.request

import imageio_ffmpeg

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36 Edg/156.0.0.0"


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://www.douyin.com/"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile-url", default="")
    ap.add_argument("--min-sec", type=float, default=15)
    a = ap.parse_args()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    owner = (re.search(r"/user/([^/?#]+)", a.profile_url) or [None, ""])[1]
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.media, encoding="utf-8"))
    n_ok = 0
    for i, it in enumerate(items, 1):
        if "is_owner" in it:  # media 步骤已经判断过谁是本人（支持 /user/self 链接）
            if not it["is_owner"]:
                continue
        elif owner and owner != "self" and it.get("author_sec_uid") and it["author_sec_uid"] != owner:
            continue
        if it.get("duration_ms", 0) < a.min_sec * 1000 or it.get("is_image"):
            continue
        wav = os.path.join(a.out, f"{i:03d}_{it['id']}.wav")
        if os.path.exists(wav):
            n_ok += 1
            continue
        mp4 = wav[:-4] + ".mp4"
        for src in it.get("srcs", []):
            try:
                fetch(src, mp4)
                break
            except Exception as e:
                print(f"  镜像失败：{e}", flush=True)
        else:
            print(f"[{i}] 所有镜像都失败（播放地址可能过期，重新跑 media 再试）", flush=True)
            continue
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-vn", "-ac", "1", "-ar", "44100", wav])
        os.remove(mp4)
        if r.returncode != 0 or not os.path.exists(wav):
            print(f"[{i}] 转换失败（可能下到的是无声画面流）", flush=True)
            continue
        n_ok += 1
        print(f"[{i}] ok {os.path.getsize(wav) / 1e6:.1f} MB", flush=True)
    print(f"完成：{n_ok} 条 -> {a.out}")
    if n_ok == 0:
        raise SystemExit("一条都没下载到。")


if __name__ == "__main__":
    main()
