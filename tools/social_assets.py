"""Collect Instagram/X post URLs and import accessible clips into a speaker's raw audio."""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


IG = "Instagram"
X = "X"
HOSTS = {
    "instagram.com": IG, "www.instagram.com": IG, "m.instagram.com": IG,
    "x.com": X, "www.x.com": X, "twitter.com": X, "www.twitter.com": X,
    "mobile.twitter.com": X, "mobile.x.com": X,
}
IG_POST = re.compile(r"/(?:[\w.]+/)?(p|reels?|tv)/([A-Za-z0-9_-]+)/?")
X_POST = re.compile(r"/([A-Za-z0-9_]+)/status/([0-9]+)(?:/video/[0-9]+)?/?")
X_SHORT = re.compile(r"/i/(?:web/)?status/([0-9]+)/?")


def social_url(url):
    try:
        parsed = urlsplit((url or "").strip())
        if (parsed.scheme in ("http", "https") and not parsed.username and not parsed.password
                and parsed.port in (None, 443)):
            platform = HOSTS.get(parsed.hostname)
            if platform:
                return platform, parsed.path
    except ValueError:
        pass
    raise ValueError("请输入 Instagram 或 X（Twitter）的完整网页链接。")


def post_url(url):
    platform, path = social_url(url)
    if platform == IG and (match := IG_POST.fullmatch(path)):
        kind, code = match.groups()
        kind = "reel" if kind == "reels" else kind
        return f"ig_{code}", f"https://www.instagram.com/{kind}/{code}/"
    if platform == X and ((match := X_POST.fullmatch(path)) or (match := X_SHORT.fullmatch(path))):
        code = match.groups()[-1]
        return f"x_{code}", f"https://x.com/i/web/status/{code}"
    raise ValueError("请选择帖子或 Reels 的链接。主页链接请填到「从主页收集链接」。")


def post_list(text):
    urls = re.findall(r"https?://[^\s<>\u3000]+", text or "", flags=re.IGNORECASE)
    if not urls:
        raise ValueError("请粘贴至少一条 Instagram 或 X 的完整帖子链接。")
    posts = {}
    for url in urls:
        key, canonical = post_url(url.rstrip("，。；！？、,;.)]}>\"'"))
        posts.setdefault(key, canonical)
    return posts


def profile_url(url):
    platform, path = social_url(url)
    parts = path.strip("/").split("/")
    user = parts[0]
    tail = parts[1:]
    if (platform == IG and re.fullmatch(r"[A-Za-z0-9_.]{1,30}", user)
            and user.lower() not in ("p", "reel", "reels", "tv", "explore", "accounts", "stories", "direct")
            and tail in ([], ["reels"])):
        suffix = "reels/" if tail else ""
        return platform, user.lower(), f"https://www.instagram.com/{user}/{suffix}"
    if (platform == X and re.fullmatch(r"[A-Za-z0-9_]{1,15}", user)
            and user.lower() not in ("i", "home", "search", "explore", "settings", "messages", "notifications")
            and tail in ([], ["media"])):
        return platform, user.lower(), f"https://x.com/{user}/media"
    raise ValueError("请粘贴 Instagram 或 X 的用户主页链接。")


def profile_browser_tabs(profile):
    try:
        port, browser = (profile / "DevToolsActivePort").read_text(encoding="ascii").splitlines()[:2]
        if not port.isdecimal() or not browser.startswith("/devtools/browser/"):
            return []
        endpoint = f"http://127.0.0.1:{int(port)}"
        with urllib.request.urlopen(f"{endpoint}/json/version", timeout=2) as response:
            if urlsplit(json.load(response)["webSocketDebuggerUrl"]).path != browser:
                return []
        with urllib.request.urlopen(f"{endpoint}/json/list", timeout=2) as response:
            return [tab for tab in json.load(response) if tab.get("type") == "page"]
    except (OSError, ValueError, KeyError):
        return []


def open_profile(profile, url):
    from browser_collect import EDGE

    _, _, canonical = profile_url(url)
    if not Path(EDGE).is_file():
        raise ValueError("没有找到 Microsoft Edge，请粘贴帖子链接直接导入。")
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([EDGE, "--remote-debugging-port=0", f"--user-data-dir={profile}", "--no-first-run",
                      "--no-default-browser-check", "--new-window", canonical])


def collect_profile(profile, url):
    from browser_collect import Tab
    from websocket import WebSocketException

    platform, user, _ = profile_url(url)
    for candidate in profile_browser_tabs(profile):
        try:
            candidate_platform, candidate_user, _ = profile_url(candidate["url"])
        except (KeyError, ValueError):
            continue
        if (candidate_platform, candidate_user) == (platform, user):
            break
    else:
        raise ValueError("独立 Edge 窗口里还没打开指定主页。请打开主页，等内容显示后再收集。")

    try:
        tab = Tab(candidate["webSocketDebuggerUrl"])
    except WebSocketException as error:
        raise ValueError("无法连接该 Edge 标签页。请重新打开主页再收集。") from error
    posts = {}
    stale = 0
    try:
        for _ in range(6):
            current = tab.js("location.href")
            if profile_url(current)[:2] != (platform, user):
                raise ValueError("请在 Edge 切回指定主页。")
            links = tab.js("Array.from(document.querySelectorAll('main a[href], [role=main] a[href]'), a => a.href)") or []
            size = len(posts)
            for link in links:
                try:
                    link_platform, path = social_url(link)
                    if link_platform != platform or (platform == X and path.strip("/").split("/")[0].lower() != user):
                        continue
                    key, canonical = post_url(link)
                except ValueError:
                    continue
                posts.setdefault(key, canonical)
            stale = stale + 1 if len(posts) == size else 0
            if stale >= 2 or len(posts) >= 50:
                break
            tab.js("window.scrollBy(0, window.innerHeight * 2)")
            time.sleep(1)
    except WebSocketException as error:
        raise ValueError("收集时 Edge 标签页已关闭或失去连接。") from error
    finally:
        tab.ws.close()
    if not posts:
        raise ValueError("主页没有显示可收集的帖子链接。可展开视频后复制帖子链接。")
    return list(posts.values())[:50]


def download_posts(posts, raw):
    from imageio_ffmpeg import get_ffmpeg_exe
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ModuleNotFoundError:
        raise SystemExit("当前工作室 Python 没有安装 yt-dlp，请按 README 补装。") from None

    # FengStudio sets NO_PROXY for localhost. On Windows, urllib then stops
    # falling back to the system proxy unless it is also present in the env.
    configured = {name.lower() for name in os.environ}
    if os.name == "nt" and not configured.intersection(("http_proxy", "https_proxy", "all_proxy")):
        system_proxy = urllib.request.getproxies_registry()
        for scheme in ("http", "https"):
            if value := system_proxy.get(scheme):
                os.environ[f"{scheme.upper()}_PROXY"] = value

    raw.mkdir(parents=True, exist_ok=True)
    archive = raw.parent / ".social-imports"
    archive.mkdir(exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    saved = reused = failed = 0
    for i, (key, url) in enumerate(posts.items(), 1):
        marker = archive / key
        print(f"[{i}/{len(posts)}] {url}", flush=True)
        if marker.is_file() and any(raw.glob(f"{key}_*.wav")):
            reused += 1
            continue
        marker.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="social-", dir=raw.parent) as temporary:
            folder = Path(temporary)
            options = {
                "format": "bestaudio/best", "outtmpl": str(folder / "%(id)s.%(ext)s"),
                "restrictfilenames": True, "quiet": True, "noprogress": True,
                "socket_timeout": 20, "retries": 1, "extractor_retries": 1,
                "allowed_extractors": ["Instagram", "Twitter", "TwitterAmplify"],
                "ffmpeg_location": ffmpeg,
            }
            download_ok = True
            try:
                with YoutubeDL(options) as downloader:
                    download_ok = downloader.download([url]) == 0
            except DownloadError:
                download_ok = False
            found_audio = False
            for file in sorted(folder.iterdir()):
                if not file.is_file() or file.suffix.lower() in (".part", ".ytdl", ".json"):
                    continue
                if not re.fullmatch(r"[A-Za-z0-9_-]+", file.stem):
                    continue
                wav = raw / f"{key}_{file.stem}.wav"
                if wav.is_file():
                    found_audio = True
                    continue
                part = folder / ".converted.wav"
                result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                                         "-i", str(file), "-map", "0:a:0", "-ac", "1", "-ar", "44100", str(part)],
                                        capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if not result.returncode and part.is_file():
                    part.replace(wav)
                    found_audio = True
                    saved += 1
                else:
                    download_ok = False
            if download_ok and found_audio:
                marker.touch()
            else:
                failed += 1
                print("此链接有内容无法导入。请检查是否有音轨、链接是否公开可访问，以及本机网络。", flush=True)
    print(f"导入完成：新增 {saved} 个 WAV，跳过 {reused} 条已导入链接，{failed} 条链接未完整导入。", flush=True)
    return 0 if saved or reused or not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(download_posts(post_list(args.links.read_text(encoding="utf-8")), args.raw))
