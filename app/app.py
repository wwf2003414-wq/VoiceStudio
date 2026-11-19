"""本地配音工作室，基于 IndexTTS 2.5。

  启动：双击文件夹里的「启动本地配音工作室.bat」，浏览器打开 http://127.0.0.1:7860
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import paths  # noqa: E402

paths.setup_env()
_prob = paths.path_problem()
if _prob:
    print("\n" + _prob + "\n")
    sys.exit(1)
print("=" * 56 + "\n  本地配音工作室 · IndexTTS 2.5\n  本地地址 http://127.0.0.1:7860\n"
      "  第一次生成要加载模型，约 30-60 秒；用完直接关掉这个黑色窗口\n" + "=" * 56, flush=True)

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import engine  # noqa: E402
import trainer  # noqa: E402

ROOT = paths.ROOT
OUT = paths.OUT
OUT.mkdir(exist_ok=True)
PY_VOICE = trainer.PY_VOICE
ENG = engine.Engine()
GPU_LOCK = threading.Lock()  # 生成和训练不能同时占显卡

LANG_CHOICES = ["自动识别"] + list(engine.LANGS)
EMO_CHOICES = [engine.EMO_SAME, engine.EMO_REF]


# ------------------------------------------------------------------ 公共 --
def speakers():
    return trainer.load_speakers()


def speaker_choices():
    reg = speakers()
    return [(f"{v.get('label', k)}（{k}）", k) for k, v in reg.items()] + [("不用说话人模型（零样本克隆）", "__none__")]


def preset_choices(spk):
    reg = speakers()
    if spk in reg:
        return list(reg[spk].get("presets", {})) + ["自定义"]
    return ["自定义"]


def lora_of(spk):
    return None if spk in (None, "", "__none__") else speakers()[spk]["lora"]


def resolve_lang(choice, text):
    return engine.detect_lang(text) if choice == "自动识别" else engine.LANGS[choice]


def stamp(name):
    return time.strftime("%m%d_%H%M%S") + "_" + "".join(c for c in name if c.isalnum() or c in "_-")[:20]


def run_voice_env(args, timeout=1800):
    """跑辅助脚本（Demucs、转写、质检），用同一个 Python，环境变量指向本文件夹。"""
    env = paths.child_env()
    r = subprocess.run([str(PY_VOICE), *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise gr.Error("处理失败：" + (r.stderr or r.stdout)[-600:])
    return r.stdout


def quality_check(wav, text, lang):
    tf = OUT / "_qc_text.txt"
    tf.write_text(text, encoding="utf-8")
    out = run_voice_env([APP / "helpers" / "qc.py", wav, tf, "ZH" if lang == "ZH" else "EN"])
    line = next((l for l in out.splitlines() if l.startswith("QCJSON")), None)
    if not line:
        return "质检没有结果"
    r = json.loads(line[6:])
    msg = f"**质检**：转回文字比对，{r['unit']}错误率 **{r['error_rate'] * 100:.1f}%** → {r['verdict']}"
    if r["suspect_spans"]:
        msg += "\n\n可疑位置：" + " ／ ".join(f"…{s}…" for s in r["suspect_spans"][:6])
    return msg


def do_tts(text, lang, spk, ref, emo_method, emo_ref, emo_weight, speed, pause_ms, temperature, top_p, top_k,
           num_beams, rep_pen, steps, max_tokens, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("请先输入文案")
    if not ref or not os.path.isfile(ref):
        raise gr.Error("请先选择参考音频")
    if emo_method == engine.EMO_REF and not (emo_ref and os.path.isfile(emo_ref)):
        raise gr.Error("选了情感参考音频，但没有上传情感参考")
    if not GPU_LOCK.acquire(blocking=False):
        raise gr.Error("显卡正在训练或生成别的任务，请等它结束")
    try:
        progress(0, desc="加载模型（切换说话人时约 20–60 秒）…")
        audio, info = ENG.generate(text, lang, ref, lora_dir=lora_of(spk), emo_method=emo_method, emo_ref=emo_ref,
                                   emo_weight=emo_weight, speed=speed, pause_ms=pause_ms, temperature=temperature,
                                   top_p=top_p, top_k=top_k, num_beams=num_beams, repetition_penalty=rep_pen,
                                   diffusion_steps=steps, max_text_tokens=max_tokens, progress=progress)
    finally:
        GPU_LOCK.release()
    return audio, info


# ------------------------------------------------------------- 文本配音 --
def on_spk_change(spk):
    choices = preset_choices(spk)
    return gr.update(choices=choices, value=choices[0])


def on_preset(spk, preset):
    reg = speakers()
    p = reg.get(spk, {}).get("presets", {}).get(preset)
    if not p:
        return [gr.update()] * 7
    return (gr.update(value=p.get("ref") or None), gr.update(value=p.get("emo_method", engine.EMO_SAME)),
            gr.update(value=p.get("emo_ref") or None), gr.update(value=p.get("emo_weight", 0.6)),
            gr.update(value=p.get("speed", 0.97)), gr.update(value=p.get("pause_ms", 150)),
            gr.update(value=p.get("temperature", 0.75)))


def tab_tts(text, lang_choice, spk, ref, emo_method, emo_ref, emo_weight, speed, pause_ms, temperature, top_p,
            top_k, num_beams, rep_pen, steps, max_tokens, do_qc, progress=gr.Progress()):
    lang = resolve_lang(lang_choice, text or "")
    audio, info = do_tts(text, lang, spk, ref, emo_method, emo_ref, emo_weight, speed, pause_ms, temperature,
                         top_p, top_k, num_beams, rep_pen, steps, max_tokens, progress)
    path = OUT / f"{stamp(spk if spk != '__none__' else 'zeroshot')}.wav"
    sf.write(str(path), audio, engine.SR, subtype="PCM_16")
    params = dict(text=text, lang=lang, speaker=spk, ref=ref, emo_method=emo_method, emo_ref=emo_ref,
                  emo_weight=emo_weight, speed=speed, pause_ms=pause_ms, temperature=temperature, top_p=top_p,
                  top_k=top_k, num_beams=num_beams, repetition_penalty=rep_pen, diffusion_steps=steps,
                  max_text_tokens=max_tokens, **info)
    path.with_suffix(".json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    msg = (f"✅ 生成完成：{info['seconds']} 秒音频，{info['segments']} 段，用时 {info['gen_seconds']} 秒，"
           f"显存峰值 {info['peak_vram_gb']} GB\n\n文件：`{path}`")
    if do_qc and lang in ("ZH", "EN"):
        progress(0.95, desc="质检…")
        msg += "\n\n" + quality_check(str(path), text, lang)
    return str(path), msg


# --------------------------------------------------------- 真人开头 + AI --
def tab_head_prep(src, mode, cut):
    if not src:
        raise gr.Error("请先上传开头音频或完整音频")
    work = OUT / "_head" / stamp("head")
    m = "auto" if mode.startswith("完整") else "own"
    run_voice_env([APP / "helpers" / "head_prep.py", "prep", "--src", src, "--work", work, "--mode", m,
                   "--cut", cut])
    info = json.loads((work / "prep.json").read_text(encoding="utf-8"))
    lang_label = {v: k for k, v in engine.LANGS.items()}.get(info.get("lang"), "自动识别")
    note = (f"真人部分：**0–{info['cut']} 秒**（原音频 {info['duration']} 秒）。"
            + ("已自动对齐到一句话说完的地方。下面是自动转写的后段文案，请检查改正后再生成。" if m == "auto"
               else "请在下面填写要让 AI 接着念的文案。"))
    return str(work), info.get("text", ""), gr.update(value=lang_label if m == "auto" else "自动识别"), info["ref"], note


def tab_head_gen(work, text, lang_choice, spk, emo_method, emo_ref, emo_weight, speed, pause_ms, temperature,
                 do_qc, progress=gr.Progress()):
    if not work:
        raise gr.Error("请先点「第 1 步：处理开头音频」")
    info = json.loads((Path(work) / "prep.json").read_text(encoding="utf-8"))
    lang = resolve_lang(lang_choice, text or "")
    audio, gen_info = do_tts(text, lang, spk, info["ref"], emo_method, emo_ref, emo_weight, speed, pause_ms,
                             temperature, 0.8, 20, 3, 10.0, 50, 120, progress)
    ai = Path(work) / "ai.wav"
    sf.write(str(ai), audio, engine.SR, subtype="PCM_16")
    final_dir = OUT / f"{stamp('真人开头')}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final = final_dir / "成品_纯人声.wav"
    progress(0.95, desc="拼接…")
    out = run_voice_env([APP / "helpers" / "head_prep.py", "splice", "--work", work, "--ai", ai, "--out", final])
    r = json.loads(out.strip().splitlines()[-1])
    (final_dir / "后段文案.txt").write_text(text, encoding="utf-8")
    msg = (f"✅ 完成：真人 0–{r['head_s']} 秒 + AI {r['ai_s']} 秒，成品共 {round(r['head_s'] + r['gap_s'] + r['ai_s'], 1)} 秒"
           f"（AI 部分用时 {gen_info['gen_seconds']} 秒）\n\n文件夹：`{final_dir}`（含 mp3、分轨）")
    if do_qc and lang in ("ZH", "EN"):
        msg += "\n\n" + quality_check(str(Path(final_dir) / "分轨" / "B_AI人声.wav"), text, lang)
    return str(final), msg


# ------------------------------------------------------------- 训练 --
def train_status(spk):
    return trainer.status_md((spk or "").strip())


def train_upload(spk, files):
    spk = (spk or "").strip()
    if not trainer.NAME_RE.match(spk):
        raise gr.Error("名字不合法：英文字母开头，可含数字和下划线")
    n = trainer.save_uploads(spk, files)
    return f"已保存 {n} 个素材文件。\n\n" + trainer.status_md(spk)


def train_login(spk, url):
    spk = (spk or "").strip()
    if not trainer.NAME_RE.match(spk):
        raise gr.Error("名字不合法")
    if "douyin.com" not in (url or ""):
        raise gr.Error("请粘贴抖音主页链接（https://www.douyin.com/user/...，或 App 里复制的 v.douyin.com 短链接）")
    trainer.douyin_login(spk, url)
    return "已打开一个单独的 Edge 窗口，请在里面**扫码登录抖音**，登录好后点「2. 开始下载」。"


def _pipeline(spk, steps_list, *args):
    spk = (spk or "").strip()
    if not trainer.NAME_RE.match(spk):
        raise gr.Error("名字不合法")
    if not GPU_LOCK.acquire(blocking=False):
        raise gr.Error("显卡正在生成或训练别的任务，请等它结束")
    ENG.unload()  # 训练要独占显存
    try:
        for g in steps_list:
            for msg in g:
                yield msg
    except Exception as e:
        yield f"❌ {e}\n\n" + trainer.tail(trainer.paths(spk)["log"])
    finally:
        GPU_LOCK.release()


def train_download(spk, url):
    yield from _pipeline(spk, [trainer.douyin_collect((spk or "").strip(), url)])


def train_all(spk, label, minutes, lang_choice, epochs, batch):
    spk = (spk or "").strip()
    st = trainer.status(spk) if trainer.NAME_RE.match(spk) else {}
    if not st.get("素材"):
        raise gr.Error("还没有素材：先上传音频/视频，或者用抖音主页下载")
    lang = engine.LANGS.get(lang_choice, "ZH")
    gens = []
    if not st["去伴奏挑选"]:
        gens.append(trainer.step_prepare(spk, minutes))
    if not st["切片"]:
        gens.append(trainer.step_slice(spk))
    if not st["转录"]:
        gens.append(trainer.step_transcribe(spk))
    if not st["特征提取"]:
        gens.append(trainer.step_extract(spk, lang))
    gens.append(trainer.step_train(spk, epochs, batch))

    def finish():
        trainer.register(spk, label or spk)
        yield f"🎉 训练完成，已登记为说话人「{label or spk}」。到「文本配音」页点刷新就能选。\n\n" + trainer.status_md(spk)

    gens.append(finish())
    yield from _pipeline(spk, gens)


# ------------------------------------------------------------- 管理 --
def list_history():
    files = sorted(OUT.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:50]
    files += sorted(OUT.glob("*/成品_纯人声.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]
    return gr.update(choices=[str(f) for f in files], value=None)


def speaker_table():
    rows = []
    for k, v in speakers().items():
        ok = (Path(v["lora"]) / "adapter_config.json").is_file()
        rows.append([v.get("label", k), k, "✅" if ok else "❌ 找不到模型", " / ".join(v.get("presets", {})), v["lora"]])
    return rows


def free_vram():
    ENG.unload()
    return "已卸载模型，显存已释放。"


# ------------------------------------------------------------------ 界面 --
def build():
    reg = speakers()
    first = next(iter(reg), "__none__")
    with gr.Blocks(title="本地配音工作室", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 本地配音工作室\nIndexTTS 2.5 · 全部在本机运行，不需要云端工作台")
        with gr.Tab("文本配音"):
            with gr.Row():
                with gr.Column(scale=3):
                    text = gr.Textbox(label="文案", lines=9, placeholder="写法：数字写成汉字，句子多用逗号连起来，少用句号。")
                    with gr.Row():
                        lang = gr.Dropdown(LANG_CHOICES, value="自动识别", label="语言")
                        spk = gr.Dropdown(speaker_choices(), value=first, label="说话人模型")
                        preset = gr.Dropdown(preset_choices(first), value=preset_choices(first)[0], label="预设")
                        refresh = gr.Button("刷新说话人", size="sm")
                with gr.Column(scale=2):
                    ref = gr.Audio(label="音色参考音频（8–12 秒本人说话）", type="filepath")
                    emo_method = gr.Radio(EMO_CHOICES, value=engine.EMO_SAME, label="情感控制")
                    emo_ref = gr.Audio(label="情感参考音频（选「使用情感参考音频」时用）", type="filepath")
            with gr.Row():
                speed = gr.Slider(0.7, 1.4, 0.97, step=0.01, label="调速（数值越大越慢）")
                pause = gr.Slider(0, 600, 150, step=10, label="句间停顿 (ms)")
                emo_w = gr.Slider(0, 1, 0.6, step=0.05, label="情感权重（不建议超过 0.65）")
                temp = gr.Slider(0.3, 1.2, 0.75, step=0.05, label="temperature（0.7–0.8 最自然）")
            with gr.Accordion("高级参数", open=False):
                with gr.Row():
                    top_p = gr.Slider(0.1, 1, 0.8, step=0.05, label="top_p")
                    top_k = gr.Slider(0, 100, 20, step=1, label="top_k")
                    beams = gr.Slider(1, 5, 3, step=1, label="num_beams")
                    rep = gr.Slider(1, 15, 10, step=0.5, label="repetition_penalty")
                    steps = gr.Slider(10, 100, 50, step=5, label="迭代步数（50 最好；100 会变平）")
                    maxtok = gr.Slider(40, 300, 120, step=10, label="每段最大 token")
            qc = gr.Checkbox(True, label="生成后自动质检（转回文字和文案比对，中文/英文）")
            go = gr.Button("生成", variant="primary")
            out_audio = gr.Audio(label="结果", type="filepath")
            out_msg = gr.Markdown()
            spk.change(on_spk_change, spk, preset)
            preset.change(on_preset, [spk, preset], [ref, emo_method, emo_ref, emo_w, speed, pause, temp])
            demo.load(on_preset, [spk, preset], [ref, emo_method, emo_ref, emo_w, speed, pause, temp])
            go.click(tab_tts, [text, lang, spk, ref, emo_method, emo_ref, emo_w, speed, pause, temp, top_p, top_k,
                               beams, rep, steps, maxtok, qc], [out_audio, out_msg])

        with gr.Tab("真人开头 + AI"):
            gr.Markdown("**完整音频**：保留前 N 秒（自动对齐到句尾），后面的原文自动转写，改好后让 AI 重念。\n\n"
                        "**只有开头音频**：整段开头保留，后面念你写的文案。成品是纯人声（去掉背景音乐），另附分轨。")
            with gr.Row():
                h_src = gr.Audio(label="原音频 / 开头音频（mp3、wav、mp4 都行）", type="filepath")
                with gr.Column():
                    h_mode = gr.Radio(["完整音频：保留前 N 秒", "只有开头音频：后面念我的文案"],
                                      value="只有开头音频：后面念我的文案", label="用法")
                    h_cut = gr.Slider(3, 30, 10, step=1, label="保留前几秒（完整音频时用）")
                    h_prep = gr.Button("第 1 步：处理开头音频（去伴奏、找切点）")
            h_work = gr.Textbox(visible=False)
            h_ref = gr.Textbox(label="自动截取的参考音频（切点前 12 秒本人声音）", interactive=False)
            h_note = gr.Markdown()
            h_text = gr.Textbox(label="后段文案（AI 要念的内容）", lines=8)
            with gr.Row():
                h_lang = gr.Dropdown(LANG_CHOICES, value="自动识别", label="语言")
                h_spk = gr.Dropdown(speaker_choices(), value=first, label="说话人模型")
                h_emo = gr.Radio(EMO_CHOICES, value=engine.EMO_SAME, label="情感控制")
                h_emo_ref = gr.Audio(label="情感参考（可选）", type="filepath")
            with gr.Row():
                h_speed = gr.Slider(0.7, 1.4, 0.97, step=0.01, label="调速（越大越慢）")
                h_pause = gr.Slider(0, 600, 150, step=10, label="句间停顿 (ms)")
                h_emo_w = gr.Slider(0, 1, 0.6, step=0.05, label="情感权重")
                h_temp = gr.Slider(0.3, 1.2, 0.75, step=0.05, label="temperature")
            h_qc = gr.Checkbox(True, label="自动质检")
            h_go = gr.Button("第 2 步：生成并拼接", variant="primary")
            h_out = gr.Audio(label="成品", type="filepath")
            h_msg = gr.Markdown()
            h_prep.click(tab_head_prep, [h_src, h_mode, h_cut], [h_work, h_text, h_lang, h_ref, h_note])
            h_go.click(tab_head_gen, [h_work, h_text, h_lang, h_spk, h_emo, h_emo_ref, h_emo_w, h_speed, h_pause,
                                      h_temp, h_qc], [h_out, h_msg])
        # 两个页面的说话人下拉框一起刷新
        refresh.click(lambda: (gr.update(choices=speaker_choices()), gr.update(choices=speaker_choices())), None,
                      [spk, h_spk])

        with gr.Tab("训练说话人"):
            gr.Markdown("训练一个新的说话人模型（LoRA）。需要**本人授权**。素材要本人说话的声音，合计 30 分钟以上，60 分钟最好。\n\n"
                        "训练会独占显卡：期间不能生成配音。全程大约 1–3 小时，网页可以关，回来点「查看进度」。")
            with gr.Row():
                t_spk = gr.Textbox(label="说话人名字（英文/拼音，例如 xiaoli）")
                t_label = gr.Textbox(label="显示名（可中文）")
                t_check = gr.Button("查看进度", size="sm")
            with gr.Tab("方式 A：上传素材"):
                t_files = gr.File(label="上传本人说话的音频/视频（可多选）", file_count="multiple")
                t_up = gr.Button("保存素材")
            with gr.Tab("方式 B：从抖音主页下载"):
                t_url = gr.Textbox(label="抖音主页链接")
                with gr.Row():
                    t_login = gr.Button("1. 打开登录窗口（扫码）")
                    t_dl = gr.Button("2. 开始下载")
            with gr.Row():
                t_min = gr.Slider(20, 120, 60, step=5, label="训练素材挑多少分钟")
                t_lang = gr.Dropdown(list(engine.LANGS), value="中文", label="素材主要语言")
                t_ep = gr.Slider(1, 30, 10, step=1, label="训练轮数")
                t_bs = gr.Slider(1, 4, 2, step=1, label="batch（显存不足改 1）")
            t_go = gr.Button("开始训练（自动跑完：去伴奏挑选 → 切片 → 转录 → 特征提取 → 训练）", variant="primary")
            t_msg = gr.Markdown()
            t_check.click(train_status, t_spk, t_msg)
            t_up.click(train_upload, [t_spk, t_files], t_msg)
            t_login.click(train_login, [t_spk, t_url], t_msg)
            t_dl.click(train_download, [t_spk, t_url], t_msg)
            t_go.click(train_all, [t_spk, t_label, t_min, t_lang, t_ep, t_bs], t_msg).then(
                lambda: (gr.update(choices=speaker_choices()), gr.update(choices=speaker_choices())), None, [spk, h_spk])

        with gr.Tab("说话人和历史"):
            sp_tbl = gr.Dataframe(headers=["显示名", "名字", "模型", "预设", "模型位置"], value=speaker_table(),
                                  interactive=False)
            with gr.Row():
                sp_ref = gr.Button("刷新")
                sp_free = gr.Button("释放显存")
            sp_msg = gr.Markdown()
            hist = gr.Dropdown(label="最近生成的音频", choices=[])
            hist_audio = gr.Audio(type="filepath")
            sp_ref.click(lambda: (speaker_table(), list_history()), None, [sp_tbl, hist])
            sp_free.click(free_vram, None, sp_msg)
            hist.change(lambda p: p, hist, hist_audio)
            gr.Markdown(f"生成结果都保存在 `{OUT}`（每条旁边有同名 .json 记录参数）。说话人登记表：`{trainer.SPEAKERS}`")
    return demo


if __name__ == "__main__":
    demo = build()
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
                                                   inbrowser=not os.environ.get("NO_BROWSER"),
                                                   allowed_paths=[str(OUT), str(paths.SPEAKERS_DIR)])
