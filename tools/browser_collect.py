"""用一个可控的 Edge 窗口（用户自己扫码登录）收集抖音主页的视频，并读出每条视频的播放地址。

子命令：
  launch  --profile-dir DIR --urls URL [URL ...]   打开一个独立的 Edge 窗口（不影响用户平时的 Edge）
  status                                          检查抖音 / 小红书是否已登录
  collect --url 抖音主页 --out list.json             往下翻主页，收集全部视频链接
  media   --list list.json --out media.json         逐条读取视频详情（时长、标题、作者、可下载地址）
  close   --profile-dir DIR                          关闭窗口并删除这个窗口的数据（含登录信息）

注意：只支持抖音的批量下载。小红书主页可以 collect 出链接，但视频下载没做（小红书内容通常和抖音重复）。
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

import websocket

PORT = 9333
EDGE = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), r"Microsoft\Edge\Application\msedge.exe")


class Tab:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


def tabs():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as r:
        return [t for t in json.load(r) if t["type"] == "page"]


def tab_for(host):
    for t in tabs():
        if host in t["url"]:
            return Tab(t["webSocketDebuggerUrl"])
    raise SystemExit(f"Edge 窗口里没有打开 {host} 的标签页")


def cmd_launch(a):
    os.makedirs(a.profile_dir, exist_ok=True)
    subprocess.Popen([EDGE, f"--remote-debugging-port={PORT}", f"--user-data-dir={a.profile_dir}",
                      "--no-first-run", "--no-default-browser-check", "--new-window", *a.urls])
    for _ in range(30):
        try:
            tabs()
            print("Edge 已打开，请让用户在这个窗口里扫码登录抖音（和小红书）。")
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("Edge 调试端口没有起来")


def cmd_status(a):
    t = Tab(tabs()[0]["webSocketDebuggerUrl"])
    cookies = t.call("Network.getAllCookies")["cookies"]
    dy = any(c["name"] == "sessionid" and c["value"] for c in cookies if "douyin" in c["domain"])
    xhs = any(c["name"] == "web_session" and c["value"] for c in cookies if "xiaohongshu" in c["domain"])
    print(json.dumps({"douyin_logged_in": dy, "xhs_logged_in": xhs}, ensure_ascii=False))


COLLECT_JS = r"""
[...document.querySelectorAll('a[href*="/video/"]')].map(a => ({
  url: new URL(a.href, location.href).href.split('?')[0],
  title: (a.innerText || '').trim().slice(0, 80)
}))
"""


def cmd_collect(a):
    t = tab_for("douyin.com")
    t.call("Page.navigate", url=a.url)
    time.sleep(6)
    seen, stale = {}, 0
    for _ in range(a.rounds):
        before = len(seen)
        for it in t.js(COLLECT_JS) or []:
            seen.setdefault(it["url"], it)
        stale = stale + 1 if len(seen) == before else 0
        if stale >= 5:
            break
        t.js("window.scrollBy(0, document.body.scrollHeight); 1")
        time.sleep(2.5)
    json.dump(list(seen.values()), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"收集到 {len(seen)} 条链接 -> {a.out}（末尾可能混有推荐流里别人的视频，media 步骤会按作者过滤）")


DETAIL_JS = r"""
(async (id) => {
  const txt = await new Promise(res => {
    const q = new XMLHttpRequest();
    q.open('GET', '/aweme/v1/web/aweme/detail/?device_platform=webapp&aid=6383&aweme_id=' + id);
    q.onload = () => res(q.responseText); q.onerror = () => res(''); q.send();
  });
  const d = JSON.parse(txt || '{}').aweme_detail || {};
  const v = d.video || {};
  // dash 档位只有画面没有声音，必须用合流的 mp4（play_addr）
  const br = (v.bit_rate || []).filter(b => b.format === 'mp4').sort((a, b) => a.bit_rate - b.bit_rate);
  const pick = (v.play_addr && v.play_addr.url_list && v.play_addr.url_list.length) ? v.play_addr : ((br[0] && br[0].play_addr) || {});
  return { desc: d.desc || '', duration_ms: v.duration || d.duration || 0, srcs: (pick.url_list || []),
           is_image: !!(d.images && d.images.length), author_sec_uid: (d.author || {}).sec_uid || '',
           author: (d.author || {}).nickname || '' };
})
"""


def owner_of(items, sec_uid):
    """主页链接里带了 sec_uid 就用它；是 /user/self 或没带时，取出现次数最多的作者（主页上绝大多数是本人的视频）。"""
    if sec_uid and sec_uid != "self":
        return sec_uid
    from collections import Counter
    c = Counter(x.get("author_sec_uid") for x in items if x.get("author_sec_uid"))
    return c.most_common(1)[0][0] if c else ""


def cmd_media(a):
    items = json.load(open(a.list, encoding="utf-8"))
    t = tab_for("douyin.com")
    out = []
    for i, it in enumerate(items, 1):
        vid = it["url"].rstrip("/").split("/")[-1]
        info = t.js(f"{DETAIL_JS}('{vid}')") or {}
        info.update(id=vid, url=it["url"])
        out.append(info)
        print(f"[{i}/{len(items)}] {info.get('author', '')[:12]} {info.get('duration_ms', 0) / 1000:5.0f}s  {info.get('desc', '')[:30]}", flush=True)
        time.sleep(1.2)
    owner = owner_of(out, a.sec_uid)
    for x in out:
        x["is_owner"] = bool(owner) and x.get("author_sec_uid") == owner
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    mine = [x for x in out if x["is_owner"] and not x.get("is_image")]
    keep = [x for x in mine if x.get("duration_ms", 0) >= a.min_sec * 1000]
    name = next((x.get("author") for x in mine), "?")
    print(f"本人「{name}」的视频 {len(mine)} 条；其中 ≥{a.min_sec:g} 秒的 {len(keep)} 条，"
          f"合计 {sum(x['duration_ms'] for x in keep) / 60000:.1f} 分钟")
    if not keep:
        raise SystemExit("没有找到可下载的本人视频。请确认链接是要克隆的那个人的主页。")


def _uses_profile(cmdline, profile_dir):
    for arg in cmdline or []:
        if arg.startswith("--user-data-dir="):
            d = arg.split("=", 1)[1].strip('"')
            try:
                return os.path.exists(d) and os.path.samefile(d, profile_dir)
            except OSError:
                return False
    return False


def cmd_close(a):
    import psutil
    for _ in range(10):
        procs = []
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                if (p.info["name"] or "").lower() == "msedge.exe" and _uses_profile(p.info["cmdline"], a.profile_dir):
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        for p in procs:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(procs, timeout=3)
        shutil.rmtree(a.profile_dir, ignore_errors=True)
        if not procs and not os.path.exists(a.profile_dir):
            break
        time.sleep(1)
    print("已关闭窗口并删除登录数据：" if not os.path.exists(a.profile_dir) else "窗口已关，但部分文件被占用没删掉：", a.profile_dir)


def normalize_url(url):
    """补上 https://，去掉 ? 后面的搜索跟踪参数；App 里复制的 v.douyin.com 短链接先展开。"""
    url = (url or "").strip().strip('"')
    short = re.search(r"https?://v\.douyin\.com/[A-Za-z0-9_-]+/?", url) or re.search(r"v\.douyin\.com/[A-Za-z0-9_-]+/?", url)
    if short:
        s = short.group(0)
        s = s if s.startswith("http") else "https://" + s
        req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
        try:
            url = urllib.request.urlopen(req, timeout=20).geturl()
        except Exception as e:
            raise SystemExit(f"短链接打不开：{s}（{e}）")
    m = re.search(r"douyin\.com/user/([^/?#\s]+)", url)
    if not m:
        raise SystemExit(f"不是抖音主页链接：{url[:80]}（应该像 https://www.douyin.com/user/MS4wLjABAAAA...；视频链接不行，要主页）")
    return f"https://www.douyin.com/user/{m.group(1)}"


def sec_uid_from_url(url):
    m = re.search(r"/user/([^/?#]+)", url or "")
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch"); p.add_argument("--profile-dir", required=True); p.add_argument("--urls", nargs="+", required=True)
    sub.add_parser("status")
    p = sub.add_parser("collect"); p.add_argument("--url", required=True); p.add_argument("--out", required=True); p.add_argument("--rounds", type=int, default=80)
    p = sub.add_parser("media"); p.add_argument("--list", required=True); p.add_argument("--out", required=True)
    p.add_argument("--profile-url", help="抖音主页链接，用来只保留本人的视频"); p.add_argument("--sec-uid")
    p.add_argument("--min-sec", type=float, default=15, help="短于这个秒数的视频不算（默认 15）")
    p = sub.add_parser("close"); p.add_argument("--profile-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        a.url = normalize_url(a.url)
    if a.cmd == "launch":
        a.urls = [normalize_url(u) if "douyin.com" in u else u for u in a.urls]
    if a.cmd == "media" and not a.sec_uid:
        a.sec_uid = sec_uid_from_url(normalize_url(a.profile_url)) if a.profile_url else ""
    if a.cmd in ("status", "collect", "media"):
        try:
            tabs()
        except Exception:
            raise SystemExit("登录窗口没有打开（或已经被关掉）。请先点「1. 打开登录窗口（扫码）」，登录后再点「2. 开始下载」。")
    {"launch": cmd_launch, "status": cmd_status, "collect": cmd_collect, "media": cmd_media, "close": cmd_close}[a.cmd](a)


if __name__ == "__main__":
    main()
