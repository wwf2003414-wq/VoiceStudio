"""FengStudio 本地配音工作室。

  启动：双击文件夹里的「启动本地配音工作室.bat」，浏览器打开 http://127.0.0.1:7860
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path
from uuid import uuid4

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import paths  # noqa: E402

paths.setup_env()
_prob = paths.path_problem()
if _prob:
    print("\n" + _prob + "\n")
    sys.exit(1)
PORT = int(os.environ.get("FENGSTUDIO_PORT", "7860"))
print("=" * 56 + f"\n  FengStudio · 本地配音工作室\n  本地地址 http://127.0.0.1:{PORT}\n"
      "  第一次生成要加载模型，约 30-60 秒；用完直接关掉这个黑色窗口\n" + "=" * 56, flush=True)

import gradio as gr  # noqa: E402
import soundfile as sf  # noqa: E402

import engine  # noqa: E402
import languages  # noqa: E402
import trainer  # noqa: E402
import dubbing  # noqa: E402
import social_assets  # noqa: E402

OUT = paths.OUT
OUT.mkdir(exist_ok=True)
PY_VOICE = trainer.PY_VOICE
ENG = engine.Engine()
GPU_LOCK = threading.Lock()  # 生成和训练不能同时占显卡

AUTO_LANGUAGE = "自动识别"
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


def speaker_reference(spk, preset="默认"):
    profile = speakers().get(spk, {})
    presets = profile.get("presets", {})
    ordered = [presets.get(preset, {}), presets.get("默认", {}), *presets.values()]
    ref = next((p["ref"] for p in ordered if p.get("ref") and Path(p["ref"]).is_file()), None)
    return ref, profile.get("adapters", {})


def resolve_lang(choice, text):
    return languages.CHOICES.get(choice) or languages.detect(text)


def stamp(name):
    return time.strftime("%m%d_%H%M%S") + "_" + "".join(c for c in name if c.isalnum() or c in "_-")[:20]


def output_id(name):
    return f"{stamp(name)}_{uuid4().hex[:8]}"


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
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        tf = Path(folder) / "script.txt"
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


@contextmanager
def generation_slot():
    if not GPU_LOCK.acquire(blocking=False):
        raise gr.Error("正在生成或训练，请等当前任务结束。")
    try:
        yield
    finally:
        GPU_LOCK.release()


def synthesize(model, text, lang, ref, transcript, adapter, speed, pause_ms, index_settings, progress):
    if adapter and not trainer.has_adapter(adapter):
        raise ValueError(f"找不到该说话人的 {model} LoRA，请刷新列表或重新训练。")
    if model != dubbing.INDEX:
        return dubbing.generate_external(ENG, model, text, lang, ref, transcript, speed, pause_ms,
                                         OUT / "_engine_logs", progress, adapter=adapter)

    emo_ref = index_settings["emo_ref"]
    if index_settings["emo_method"] == engine.EMO_REF and not (emo_ref and Path(emo_ref).is_file()):
        raise ValueError("请先上传 IndexTTS 的情感参考音频。")
    progress(0.05, desc="加载模型")
    audio, info = ENG.generate(text, lang, ref, lora_dir=adapter, speed=speed, pause_ms=pause_ms,
                               progress=progress, **index_settings)
    return audio, engine.SR, info


def generate_versions(selection, text, lang, ref, transcript, adapters, speed, pause_ms, index_settings,
                      finish, do_qc, progress):
    if not text or not text.strip():
        raise gr.Error("请先输入文案。")
    if not ref or not Path(ref).is_file():
        raise gr.Error("请先选择有效的参考音频。")
    allowed = dubbing.supported_languages(selection)
    if not allowed:
        raise gr.Error("请选择配音引擎。")
    if lang not in allowed:
        if selection == dubbing.COMPARE:
            raise gr.Error("三模型对比支持中文和英文。其他语言可选「按语言自动选择」。")
        raise gr.Error(f"{selection} 不支持{languages.NAMES[lang]}。可切换到 {dubbing.preferred_model(lang)}。")
    if selection == dubbing.COMPARE:
        models = dubbing.COMPARE_MODELS
    else:
        models = [dubbing.preferred_model(lang) if selection == dubbing.AUTO else selection]

    results = {}
    messages = []
    with generation_slot():
        for number, model in enumerate(models):
            def report(fraction, desc=None):
                progress((number + 0.88 * fraction) / len(models),
                         desc=f"{number + 1}/{len(models)} · {model} · {desc or '生成中'}")

            try:
                adapter = adapters.get(model)
                if adapter:
                    report(0, desc="加载说话人 LoRA")
                audio, rate, info = synthesize(model, text, lang, ref, transcript, adapter, speed, pause_ms,
                                               index_settings, report)
                output, qc_audio, message = finish(model, audio, rate, info)
                results[model] = output
                if adapter:
                    message = "已使用说话人 LoRA。 " + message
                if do_qc and lang in ("ZH", "EN"):
                    report(1, desc="转写质检")
                    try:
                        message += "\n\n" + quality_check(qc_audio, text, lang)
                    except Exception:
                        logging.exception("转写质检失败")
                        message += "\n\n转写质检失败，音频已保存。"
                messages.append(f"**{model}** · {message}")
            except Exception as error:
                if not isinstance(error, (ValueError, gr.Error)):
                    logging.exception("生成失败：%s", model)
                messages.append(f"**{model}** · 生成失败。{error}")
    progress(1, desc="处理完成")
    return (*[gr.update(value=results.get(model), visible=model in results) for model in dubbing.MODELS],
            "\n\n".join(messages))


# ------------------------------------------------------------- 文本配音 --
def on_spk_change(spk):
    choices = preset_choices(spk)
    return gr.update(choices=choices, value=choices[0])


def on_preset(spk, preset):
    reg = speakers()
    p = reg.get(spk, {}).get("presets", {}).get(preset)
    if not p:
        return [gr.update()] * 8
    method = p.get("emo_method", engine.EMO_SAME)
    uses_emotion_reference = method == engine.EMO_REF
    return (gr.update(value=p.get("ref") or None), gr.update(value=method),
            gr.update(value=p.get("emo_ref") or None, visible=uses_emotion_reference),
            gr.update(value=p.get("emo_weight", 0.6)),
            gr.update(value=p.get("speed", 0.97)), gr.update(value=p.get("pause_ms", 150)),
            gr.update(value=p.get("temperature", 0.75)),
            gr.update(open=True) if uses_emotion_reference else gr.update())


def tab_tts(selection, transcript, text, lang_choice, spk, ref, emo_method, emo_ref, emo_weight, speed,
            pause_ms, temperature, top_p, top_k, num_beams, rep_pen, steps, max_tokens, do_qc,
            progress=gr.Progress()):
    lang = resolve_lang(lang_choice, text or "")
    saved_ref, adapters = speaker_reference(spk)
    ref = ref or saved_ref
    index_settings = dict(emo_method=emo_method, emo_ref=emo_ref, emo_weight=emo_weight, temperature=temperature,
                          top_p=top_p, top_k=top_k, num_beams=num_beams, repetition_penalty=rep_pen,
                          diffusion_steps=steps, max_text_tokens=max_tokens)
    run_id = output_id("配音")

    def finish(model, audio, rate, info):
        path = OUT / f"{run_id}_{dubbing.MODEL_IDS[model]}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        params = dict(model=model, text=text, lang=lang, speaker=spk, ref=ref,
                      speed=speed, pause_ms=pause_ms, **info)
        if model == dubbing.INDEX:
            params.update(index_settings)
        dubbing.write_json(path.with_suffix(".json"), params)
        return str(path), str(path), f"{info['seconds']} 秒音频，耗时 {info['gen_seconds']} 秒。"

    return generate_versions(selection, text, lang, ref, transcript, adapters, speed, pause_ms,
                             index_settings, finish, do_qc, progress)


# --------------------------------------------------------- 真人开头 + AI --
def tab_head_prep(src, mode, cut):
    if not src:
        raise gr.Error("请先上传开头音频或完整音频")
    work = OUT / "_head" / output_id("head")
    m = "auto" if mode.startswith("完整") else "own"
    with generation_slot():
        ENG.unload()
        run_voice_env([APP / "helpers" / "head_prep.py", "prep", "--src", src, "--work", work,
                       "--mode", m, "--cut", cut])
    info = json.loads((work / "prep.json").read_text(encoding="utf-8"))
    lang_label = languages.NAMES.get(info.get("lang")) if m == "auto" else None
    note = (f"真人部分：**0–{info['cut']} 秒**（原音频 {info['duration']} 秒）。"
            + ("已自动对齐到一句话说完的地方。下面是自动转写的后段文案，请检查改正后再生成。" if m == "auto"
               else "请在下面填写要让 AI 接着念的文案。"))
    return str(work), info.get("text", ""), lang_label, info["ref"], note


def tab_head_gen(selection, transcript, work, text, lang_choice, spk, emo_method, emo_ref, emo_weight,
                 speed, pause_ms, temperature, do_qc, progress=gr.Progress()):
    if not work:
        raise gr.Error("请先点「第 1 步 · 处理开头音频」。")
    head = json.loads((Path(work) / "prep.json").read_text(encoding="utf-8"))
    lang = resolve_lang(lang_choice, text or "")
    _, adapters = speaker_reference(spk)
    index_settings = dict(emo_method=emo_method, emo_ref=emo_ref, emo_weight=emo_weight, temperature=temperature)
    run_id = output_id("真人开头")

    def finish(model, audio, rate, info):
        final_dir = OUT / f"{run_id}_{dubbing.MODEL_IDS[model]}"
        final_dir.mkdir()
        final = final_dir / "成品_纯人声.wav"
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            ai = Path(folder) / "tail.wav"
            sf.write(ai, audio, rate, subtype="PCM_16")
            result = run_voice_env([APP / "helpers" / "head_prep.py", "splice",
                                    "--work", work, "--ai", ai, "--out", final])
        timing = json.loads(result.strip().splitlines()[-1])
        (final_dir / "后段文案.txt").write_text(text, encoding="utf-8")
        ai_stem = final_dir / "分轨" / "B_AI人声.wav"
        summary = (f"真人开头 {timing['head_s']} 秒，AI 后段 {timing['ai_s']} 秒，生成耗时 {info['gen_seconds']} 秒。"
                   f"\n\n分轨和 MP3：`{final_dir}`")
        return str(final), str(ai_stem), summary

    return generate_versions(selection, text, lang, head["ref"], transcript, adapters, speed, pause_ms,
                             index_settings, finish, do_qc, progress)


# ------------------------------------------------------------- 训练 --
def train_status(spk, training_engine=dubbing.INDEX):
    try:
        key, name = trainer.resolve_speaker(spk)
    except ValueError as error:
        return str(error)
    return trainer.status_md(key, training_engine, name)


def train_upload(spk, files, training_engine=dubbing.INDEX):
    try:
        key, name = trainer.resolve_speaker(spk)
    except ValueError as error:
        return str(error)
    with generation_slot():
        n = trainer.save_uploads(key, files)
    note = f"已保存 {n} 个新素材文件。" if n else "没有新增素材。已导入的同名同内容文件会自动跳过。"
    return note + "\n\n" + trainer.status_md(key, training_engine, name)


def train_login(spk, url):
    try:
        key, _ = trainer.resolve_speaker(spk)
    except ValueError as error:
        return str(error)
    if "douyin.com" not in (url or ""):
        return "请粘贴抖音主页链接（https://www.douyin.com/user/...，或抖音 App 复制的主页短链接）。"
    trainer.douyin_login(key, url)
    return "已打开一个单独的 Edge 窗口，请在里面**扫码登录抖音**，登录好后点「2. 开始下载」。"


def _pipeline(spk, task):
    try:
        key, name = trainer.resolve_speaker(spk)
    except ValueError as error:
        yield str(error)
        return
    if not GPU_LOCK.acquire(blocking=False):
        yield "显卡正在处理另一项任务，请等它结束再试。"
        return
    try:
        ENG.unload()
        yield from task(key, name)
    except Exception as e:
        yield f"❌ {e}\n\n" + trainer.tail(trainer.paths(key)["log"])
    finally:
        GPU_LOCK.release()


def train_download(spk, url):
    yield from _pipeline(spk, lambda key, _: trainer.douyin_collect(key, url))


def open_training_profile(spk, url):
    try:
        key, _ = trainer.resolve_speaker(spk)
        social_assets.open_profile(trainer.paths(key)["social_edge"], url)
    except (OSError, ValueError) as error:
        return str(error)
    return "主页已在独立 Edge 窗口打开。内容显示后，回这里点击「收集帖子链接」。"


def collect_training_posts(spk, profile, links):
    try:
        key, _ = trainer.resolve_speaker(spk)
        previous = social_assets.post_list(links) if (links or "").strip() else {}
        visible = social_assets.collect_profile(trainer.paths(key)["social_edge"], profile)
        combined = {**previous, **social_assets.post_list("\n".join(visible))}
    except (OSError, ValueError, RuntimeError) as error:
        return gr.skip(), str(error)
    return "\n".join(combined.values()), f"新增 {len(combined) - len(previous)} 条链接。确认后点击「导入素材」。"


def train_social_import(spk, links):
    def run(key, _):
        try:
            posts = social_assets.post_list(links)
        except ValueError as error:
            yield str(error)
            return
        yield from trainer.social_import(key, posts)

    yield from _pipeline(spk, run)


def train_all(spk, label, files, minutes, lang_choice, epochs, batch, training_engine=dubbing.INDEX):
    lang = languages.CHOICES.get(lang_choice)

    def train(key, name):
        trainer.save_uploads(key, files)
        if not trainer.has_audio(trainer.paths(key)["raw"]):
            raise ValueError("当前说话人还没有素材。请先上传音视频或从社交媒体导入。")
        yield from trainer.train_speaker(key, (label or "").strip() or name, minutes, lang, epochs, batch, training_engine)
        yield f"{training_engine} LoRA 已登记。配音时选择这个引擎和说话人即可使用。\n\n" + trainer.status_md(key, training_engine, name)

    yield from _pipeline(spk, train)


# ------------------------------------------------------------- 管理 --
def list_history(selected):
    files = sorted(OUT.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:50]
    files += sorted(OUT.glob("*/成品_纯人声.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    choices = [(str(f.relative_to(OUT)), str(f)) for f in files]
    return gr.update(choices=choices, value=selected if any(path == selected for _, path in choices) else None)


def speaker_table():
    rows = []
    for k, v in speakers().items():
        adapters = v["adapters"]
        states = [f"{model}（{'可用' if trainer.has_adapter(adapter) else '找不到模型'}）"
                  for model, adapter in adapters.items()]
        rows.append([v.get("label", k), k, " / ".join(states) or "仅参考音频",
                     " / ".join(v.get("presets", {})), " / ".join(adapters.values())])
    return rows


def free_vram():
    with generation_slot():
        ENG.unload()
    return "已卸载模型，显存已释放。"


# ------------------------------------------------------------------ 界面 --
def page_heading(title, description):
    gr.Markdown(f"# {title}\n\n{description}", elem_classes="page-heading")


def card_heading(title, description=None):
    body = f"## {title}"
    if description:
        body += f"\n\n{description}"
    gr.Markdown(body, elem_classes="card-heading")


def emotion_reference_visibility(method):
    return gr.update(visible=method == engine.EMO_REF)


def refresh_speaker_inputs():
    choices = speaker_choices()
    return gr.update(choices=choices), gr.update(choices=choices)


def model_picker():
    return gr.Dropdown([dubbing.AUTO, *dubbing.MODELS, dubbing.COMPARE], value=dubbing.AUTO, label="配音引擎",
                       info="可按语言自动选引擎。原三模型的对比支持中文和英文。")


def engine_fields(selection, language, head=False):
    model = dubbing.preferred_model(languages.CHOICES.get(language, "ZH")) if selection == dubbing.AUTO else selection
    index = model in (dubbing.INDEX, dubbing.COMPARE)
    show_speaker = selection in (dubbing.AUTO, dubbing.COMPARE, *trainer.TRAIN_ENGINES)
    return (gr.update(visible=model != dubbing.INDEX), gr.update(visible=index),
            gr.update(visible=show_speaker if head else index))


def language_choices(selection):
    supported = dubbing.supported_languages(selection)
    return [AUTO_LANGUAGE, *(name for name, code in languages.CHOICES.items() if code in supported)]


def engine_language_fields(selection, language, detected=None, head=False):
    choices = language_choices(selection)
    if detected in choices:
        language = detected
    current = language if language in choices else AUTO_LANGUAGE
    return gr.update(choices=choices, value=current), *engine_fields(selection, current, head=head)


def training_engine_fields(selection, language):
    choices = language_choices(selection)[1:]
    current = language if language in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=current), 2 if selection == dubbing.INDEX else 1


def result_players(title, anchor):
    with gr.Column(elem_id=anchor, elem_classes="studio-card result-card"):
        card_heading(title)
        with gr.Row(elem_classes="model-results"):
            players = [
                gr.Audio(label=model, type="filepath", interactive=False, show_download_button=True,
                         visible=False, min_width=230, elem_classes="result-audio")
                for model in dubbing.MODELS
            ]
        status = gr.Markdown(elem_classes="studio-status")
    return [*players, status]


def bind_generation(button, status, run, inputs, results, anchor):
    label = button.value

    @wraps(run)
    def generate(*args):
        ready = gr.update(value=label, interactive=True)
        try:
            result = run(*args)
        except gr.Error as error:
            return (*[gr.skip() for _ in results], str(error.message), ready)

        count = sum(player.get("visible", False) for player in result[:-1])
        if not count:
            return (*result[:-1], "", result[-1], ready)
        summary = f"已生成 {count} 版 · [试听与下载](#{anchor})"
        if "· 生成失败。" in result[-1]:
            summary += "（部分引擎失败，成品区有详情）"
        return (*result, summary, ready)

    event = button.click(
        lambda: ("任务已提交，正在等待生成…", gr.update(value="生成中…", interactive=False)),
        outputs=[status, button], queue=False, show_progress="hidden", api_name=False,
    ).success(generate, inputs, [*results, status, button], show_progress_on=status)
    event.failure(lambda: ("生成失败，请重试。", gr.update(value=label, interactive=True)),
                  outputs=[status, button], queue=False,
                  show_progress="hidden", api_name=False)


def refresh_library(selected):
    return speaker_table(), list_history(selected)


def studio_theme():
    theme = gr.themes.Soft(
        primary_hue="violet",
        neutral_hue="slate",
        font=["Inter", "Microsoft YaHei UI", "PingFang SC", "sans-serif"],
        spacing_size="md",
        radius_size="md",
    ).set(
        body_background_fill="#f6f5fa",
        body_text_color="#211c36",
        block_background_fill="#ffffff",
        block_border_color="#e4e0ef",
        block_padding="0",
        block_label_padding="0",
        block_label_background_fill="transparent",
        block_label_text_color="#3f3756",
        block_label_margin="0 0 5px",
        block_label_text_size="13px",
        block_title_text_color="#3f3756",
        color_accent="#6241d8",
        input_background_fill="#faf9fd",
        input_border_color="#ded9eb",
        input_shadow="none",
        input_shadow_focus="none",
        slider_color="#6241d8",
        button_primary_background_fill="#6241d8",
        button_primary_background_fill_hover="#5434c7",
        button_primary_text_color="#ffffff",
    )
    # The studio uses one light palette even when Gradio follows a dark browser preference.
    for name in vars(theme):
        if name.endswith("_dark") and hasattr(theme, name[:-5]):
            setattr(theme, name, getattr(theme, name[:-5]))
    return theme


def build():
    reg = speakers()
    first = next(iter(reg), "__none__")
    with gr.Blocks(title="FengStudio · 本地配音工作室", theme=studio_theme(),
                   css_paths=APP / "studio.css", fill_width=True) as demo:
        gr.HTML(
            """
            <header class="studio-header">
              <div class="studio-brand">
                <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>
                <div><strong>FengStudio</strong><span>本地配音工作室</span></div>
              </div>
              <div class="studio-environment">
                <span class="studio-local">本机运行</span>
              </div>
            </header>
            """,
            elem_id="studio-masthead",
        )
        with gr.Tabs(elem_id="studio-navigation"):
            with gr.Tab("文本配音", id="tts", elem_id="studio-tab-tts"):
                page_heading("文本配音", "输入整段文案，选择一个引擎生成，或试听三种引擎的版本。")
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=7, min_width=350, elem_classes="workspace-column"):
                        with gr.Column(elem_classes=["studio-card", "script-card"]):
                            card_heading("配音文案", "用逗号控制语气，句末标点会用来分段。")
                            text = gr.Textbox(label="配音文案", show_label=False, lines=6, max_lines=16,
                                              placeholder="在这里输入要朗读的文案…", elem_id="studio-script")
                            lang = gr.Dropdown(language_choices(dubbing.AUTO), value=AUTO_LANGUAGE, label="文案语言",
                                               elem_classes="script-language")
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("生成设置", "语速倍率越大，读得越慢。")
                            with gr.Row(elem_classes="settings-row"):
                                speed = gr.Slider(0.7, 1.4, 0.97, step=0.01, label="语速倍率")
                                pause = gr.Slider(0, 600, 150, step=10, label="停顿 (ms)")
                            with gr.Accordion("IndexTTS 高级参数", open=False, elem_classes="studio-accordion") as sampling:
                                with gr.Row(elem_classes="settings-row"):
                                    temp = gr.Slider(0.3, 1.2, 0.75, step=0.05, label="采样温度")
                                    top_p = gr.Slider(0.1, 1, 0.8, step=0.05, label="top_p")
                                with gr.Row(elem_classes="settings-row"):
                                    top_k = gr.Slider(0, 100, 20, step=1, label="top_k")
                                    beams = gr.Slider(1, 5, 3, step=1, label="num_beams")
                                with gr.Row(elem_classes="settings-row"):
                                    rep = gr.Slider(1, 15, 10, step=0.5, label="repetition_penalty")
                                    steps = gr.Slider(10, 100, 50, step=5, label="扩散迭代步数")
                                maxtok = gr.Slider(40, 300, 120, step=10, label="每段最大 token")
                    with gr.Column(scale=5, min_width=300, elem_classes="workspace-column"):
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("声音设置")
                            model = model_picker()
                            spk = gr.Dropdown(speaker_choices(), value=first, label="说话人",
                                              info="有对应引擎的 LoRA 时自动加载，也可用保存的参考录音克隆。")
                            with gr.Row(elem_classes="field-action"):
                                preset = gr.Dropdown(preset_choices(first), value=preset_choices(first)[0],
                                                     label="声音预设")
                                refresh = gr.Button("刷新说话人", size="sm")
                            ref = gr.Audio(label="音色参考音频", type="filepath", elem_classes="reference-audio")
                            gr.Markdown("建议使用 8–12 秒干净人声，素材应来自本人或已获授权。",
                                        elem_classes="field-note")
                            ref_transcript = gr.Textbox(label="参考录音逐字稿（选填）", visible=False,
                                                       info="OmniVoice 留空时会用本机 Whisper 转写参考录音。")
                            with gr.Accordion("IndexTTS 情感控制", open=False, elem_classes="studio-accordion") as emotion:
                                emo_method = gr.Radio(EMO_CHOICES, value=engine.EMO_SAME, label="情感来源")
                                emo_ref = gr.Audio(label="情感参考音频", type="filepath", visible=False,
                                                   elem_classes="reference-audio")
                                emo_w = gr.Slider(0, 1, 0.6, step=0.05, label="情感权重")
                            with gr.Row(elem_classes="action-row"):
                                qc = gr.Checkbox(True, label="自动质检（中文 / 英文）")
                                go = gr.Button("生成配音", variant="primary")
                            tts_status = gr.Markdown(elem_classes="studio-generation-status")
                tts_results = result_players("配音成品", "tts-results")

                spk.change(on_spk_change, spk, preset)
                preset_fields = [ref, emo_method, emo_ref, emo_w, speed, pause, temp, emotion]
                preset.change(on_preset, [spk, preset], preset_fields)
                demo.load(on_preset, [spk, preset], preset_fields)
                emo_method.change(emotion_reference_visibility, emo_method, emo_ref, queue=False)
                tts_engine_fields = [ref_transcript, emotion, sampling]
                model.change(engine_language_fields, [model, lang], [lang, *tts_engine_fields], queue=False)
                lang.change(engine_fields, [model, lang], tts_engine_fields, queue=False)
                bind_generation(go, tts_status, tab_tts,
                                [model, ref_transcript, text, lang, spk, ref, emo_method, emo_ref, emo_w, speed,
                                 pause, temp, top_p, top_k, beams, rep, steps, maxtok, qc],
                                tts_results, "tts-results")

            with gr.Tab("真人开头 + AI", id="head", elem_id="studio-tab-head"):
                page_heading("真人开头 + AI", "保留真人开头，用 AI 接续后段文案，导出纯人声和分轨。")
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=5, min_width=300, elem_classes="workspace-column"):
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("处理真人开头", "完整音频会对齐句尾并自动转写后段。也可以上传一段完整保留的开头。")
                            h_src = gr.Audio(label="原音频或开头音频（MP3、WAV、MP4）", type="filepath",
                                             elem_classes="reference-audio")
                            h_mode = gr.Radio(["完整音频：保留前 N 秒", "只有开头音频：后面念我的文案"],
                                              value="只有开头音频：后面念我的文案", label="音频类型")
                            h_cut = gr.Slider(3, 30, 10, step=1, label="完整音频保留时长 (秒)", visible=False)
                            h_prep = gr.Button("第 1 步 · 处理开头音频", variant="primary")
                            h_work = gr.State()
                            h_detected = gr.State()
                            h_ref = gr.Textbox(label="自动提取的音色参考", interactive=False)
                            h_note = gr.Markdown(elem_classes="studio-status")
                    with gr.Column(scale=7, min_width=350, elem_classes="workspace-column"):
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("续写配音", "对比时三版成品共用同一段真人开头。语速倍率越大，读得越慢。")
                            h_model = model_picker()
                            h_text = gr.Textbox(label="后段文案", lines=7, max_lines=16,
                                                placeholder="填写或编辑 AI 要接着朗读的内容…",
                                                elem_classes="script-input")
                            with gr.Row(elem_classes="settings-row"):
                                h_lang = gr.Dropdown(language_choices(dubbing.AUTO), value=AUTO_LANGUAGE, label="文案语言")
                                h_spk = gr.Dropdown(speaker_choices(), value=first, label="说话人 LoRA")
                            h_transcript = gr.Textbox(label="提取的参考录音逐字稿（选填）", visible=False,
                                                      info="OmniVoice 留空时会用本机 Whisper 转写参考录音。")
                            with gr.Row(elem_classes="settings-row"):
                                h_speed = gr.Slider(0.7, 1.4, 0.97, step=0.01, label="语速倍率")
                                h_pause = gr.Slider(0, 600, 150, step=10, label="停顿 (ms)")
                            with gr.Accordion("IndexTTS 情感与采样", open=False, elem_classes="studio-accordion") as h_advanced:
                                h_emo = gr.Radio(EMO_CHOICES, value=engine.EMO_SAME, label="情感来源")
                                h_emo_ref = gr.Audio(label="情感参考音频", type="filepath", visible=False,
                                                     elem_classes="reference-audio")
                                with gr.Row(elem_classes="settings-row"):
                                    h_emo_w = gr.Slider(0, 1, 0.6, step=0.05, label="情感权重")
                                    h_temp = gr.Slider(0.3, 1.2, 0.75, step=0.05, label="采样温度")
                            with gr.Row(elem_classes="action-row"):
                                h_qc = gr.Checkbox(True, label="生成后自动质检（中文 / 英文）")
                                h_go = gr.Button("第 2 步 · 生成并拼接", variant="primary")
                            h_status = gr.Markdown(elem_classes="studio-generation-status")
                head_results = result_players("拼接成品", "head-results")

                head_engine_fields = [h_transcript, h_advanced, h_spk]
                head_language_change = partial(engine_language_fields, head=True)
                h_model.change(head_language_change, [h_model, h_lang], [h_lang, *head_engine_fields], queue=False)
                h_lang.change(partial(engine_fields, head=True), [h_model, h_lang], head_engine_fields, queue=False)

                h_mode.change(lambda mode: gr.update(visible=mode.startswith("完整")), h_mode, h_cut,
                              queue=False)
                h_emo.change(emotion_reference_visibility, h_emo, h_emo_ref, queue=False)
                gr.on([h_src.change, h_mode.input, h_cut.input], lambda: (None, "", ""), None,
                      [h_work, h_ref, h_note], queue=False, show_progress="hidden")
                h_prep.click(tab_head_prep, [h_src, h_mode, h_cut], [h_work, h_text, h_detected, h_ref, h_note]).success(
                    head_language_change, [h_model, h_lang, h_detected], [h_lang, *head_engine_fields],
                    queue=False, show_progress="hidden")
                bind_generation(h_go, h_status, tab_head_gen,
                                [h_model, h_transcript, h_work, h_text, h_lang, h_spk, h_emo, h_emo_ref, h_emo_w,
                                 h_speed, h_pause, h_temp, h_qc],
                                head_results, "head-results")

            refresh.click(refresh_speaker_inputs, None, [spk, h_spk])

            with gr.Tab("训练说话人", id="train", elem_id="studio-tab-train"):
                page_heading("训练说话人", "使用本人或已获授权的录音训练 LoRA，训练完成后即可在配音中选择。")
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=7, min_width=350, elem_classes="workspace-column"):
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("说话人与素材", "建议准备至少 30 分钟的人声素材，60 分钟更合适。")
                            with gr.Row(elem_classes="settings-row"):
                                t_spk = gr.Textbox(label="说话人名称", placeholder="例如 小李",
                                                   info="支持中文和英文。已有说话人也可填写原标识。")
                                t_label = gr.Textbox(label="配音显示名（可选）", placeholder="默认使用说话人名称")
                            with gr.Tabs(elem_classes="inner-tabs"):
                                with gr.Tab("上传历史素材"):
                                    t_files = gr.File(label="音频或视频（可多选）", file_count="multiple",
                                                      file_types=list(trainer.AUDIO_EXTS),
                                                      elem_classes="training-upload")
                                    t_up = gr.Button("仅保存素材")
                                    t_upload_msg = gr.Markdown(elem_classes="studio-status")
                                with gr.Tab("抖音主页"):
                                    t_url = gr.Textbox(label="抖音主页链接")
                                    with gr.Row():
                                        t_login = gr.Button("1. 打开登录窗口")
                                        t_dl = gr.Button("2. 开始下载")
                                    t_douyin_msg = gr.Markdown(elem_classes="studio-generation-status")
                                with gr.Tab("Instagram / X"):
                                    t_posts = gr.Textbox(label="帖子或 Reels 链接", lines=4,
                                                         placeholder="每行一条 Instagram Reels 或 X 视频帖子链接",
                                                         info="请使用本人或已获授权的素材。导入后和上传素材存放在一起。")
                                    with gr.Accordion("从主页收集链接", open=False, elem_classes="studio-accordion"):
                                        t_profile = gr.Textbox(label="Instagram 或 X 主页链接",
                                                               info="在独立 Edge 窗口打开主页。只收集页面已经显示的帖子链接。")
                                        with gr.Row():
                                            t_profile_open = gr.Button("打开主页")
                                            t_profile_collect = gr.Button("收集帖子链接")
                                    t_social_go = gr.Button("导入素材", variant="primary")
                                    t_social_msg = gr.Markdown(elem_classes="studio-generation-status")
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("训练参数")
                            t_engine = gr.Dropdown(trainer.TRAIN_ENGINES, value=dubbing.INDEX, label="训练引擎",
                                                   info="同一个说话人可复用素材，分别训练各引擎的 LoRA。")
                            with gr.Row(elem_classes="settings-row"):
                                t_min = gr.Slider(20, 120, 60, step=5, label="挑选素材时长 (分钟)")
                                t_lang = gr.Dropdown(list(engine.LANGS), value="中文", label="素材主要语言")
                            with gr.Row(elem_classes="settings-row"):
                                t_ep = gr.Slider(1, 30, 10, step=1, label="训练轮数")
                                t_bs = gr.Slider(1, 4, 2, step=1, label="批大小（显存不足时设为 1）")
                            t_go = gr.Button("保存素材并开始 LoRA 训练", variant="primary")
                    with gr.Column(scale=5, min_width=290, elem_classes="workspace-column"):
                        with gr.Column(elem_classes="studio-card"):
                            card_heading("训练进度", "训练会独占显卡，耗时取决于素材和引擎。可按说话人名称和引擎查询。")
                            t_check = gr.Button("查看进度")
                            t_msg = gr.Markdown(elem_classes="studio-status")

                t_engine.change(training_engine_fields, [t_engine, t_lang], [t_lang, t_bs], queue=False)
                t_check.click(train_status, [t_spk, t_engine], t_msg, queue=False)
                t_up.click(train_upload, [t_spk, t_files, t_engine], t_upload_msg)
                t_login.click(train_login, [t_spk, t_url], t_douyin_msg, show_progress_on=t_douyin_msg)
                t_dl.click(train_download, [t_spk, t_url], t_douyin_msg, show_progress_on=t_douyin_msg)
                t_profile_open.click(open_training_profile, [t_spk, t_profile], t_social_msg, queue=False)
                t_profile_collect.click(collect_training_posts, [t_spk, t_profile, t_posts], [t_posts, t_social_msg],
                                        show_progress_on=t_posts)
                t_social_go.click(train_social_import, [t_spk, t_posts], t_social_msg, show_progress_on=t_social_msg)
                t_go.click(train_all, [t_spk, t_label, t_files, t_min, t_lang, t_ep, t_bs, t_engine], t_msg).then(
                    refresh_speaker_inputs, None, [spk, h_spk])

            with gr.Tab("说话人和历史", id="library", elem_id="studio-tab-library") as library:
                page_heading("说话人和历史", "管理本机说话人模型，试听之前生成的音频。")
                with gr.Column(elem_classes="studio-card"):
                    card_heading("说话人模型")
                    sp_tbl = gr.Dataframe(headers=["显示名", "说话人标识", "模型", "预设", "模型位置"],
                                          value=speaker_table(), interactive=False)
                    with gr.Row(elem_classes="library-actions"):
                        sp_ref = gr.Button("刷新列表", size="sm")
                        sp_free = gr.Button("释放显存", size="sm")
                    sp_msg = gr.Markdown(elem_classes="studio-status")
                with gr.Column(elem_classes="studio-card"):
                    card_heading("最近生成")
                    hist = gr.Dropdown(label="选择音频", choices=[])
                    hist_audio = gr.Audio(label="历史音频", show_label=False, type="filepath", interactive=False,
                                          elem_classes="result-audio")
                    gr.Markdown(f"生成目录：`{OUT}`　·　说话人登记表：`{trainer.SPEAKERS}`",
                                elem_classes="field-note")
                library.select(refresh_library, hist, [sp_tbl, hist], queue=False, show_progress="hidden")
                sp_ref.click(refresh_library, hist, [sp_tbl, hist], queue=False, show_progress="hidden")
                sp_free.click(free_vram, None, sp_msg)
                hist.change(lambda p: p, hist, hist_audio)
    return demo


if __name__ == "__main__":
    demo = build()
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=PORT,
                                                   inbrowser=not os.environ.get("NO_BROWSER"),
                                                   allowed_paths=[str(OUT), str(paths.SPEAKERS_DIR)])
