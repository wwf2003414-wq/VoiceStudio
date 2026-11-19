"""本地 IndexTTS 2.5 推理：加载底模 + 说话人 LoRA，按句生成并拼接（复刻云端工作台的做法）。"""
import gc
import os
import re
import time
from pathlib import Path

import numpy as np

import paths

IX = paths.IX
CKPT = paths.CKPT
SR = 22050

LANGS = {"中文": "ZH", "英文": "EN", "日语": "JA", "西班牙语": "ES", "阿拉伯语": "AR"}
EMO_SAME = "与音色参考音频相同"
EMO_REF = "使用情感参考音频"


def detect_lang(text):
    cjk = len(re.findall(r"[一-鿿]", text))
    kana = len(re.findall(r"[぀-ヿ]", text))
    if kana > 5:
        return "JA"
    if re.search(r"[؀-ۿ]", text):
        return "AR"
    letters = len(re.findall(r"[A-Za-z]", text))
    if cjk >= letters / 4:
        return "ZH"
    if re.search(r"[ñáéíóú¿¡]", text.lower()):
        return "ES"
    return "EN"


def split_sentences(text):
    """按句末标点切段（和云端工作台一样：一句一段，句中的逗号不切）。"""
    text = "".join(line.strip() for line in text.splitlines())
    parts = re.findall(r"[^。！？!?.]+[。！？!?.]*", text)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 英文小数点 / 缩写里的 "." 不切：前一段太短就并回去
        if buf and len(re.sub(r"\W", "", buf)) < 4:
            buf += p
            continue
        if buf:
            out.append(buf)
        buf = p
    if buf:
        out.append(buf)
    return out or [text]


class Engine:
    """全局唯一：按需加载 / 切换说话人 / 卸载（训练前要卸载腾显存）。"""

    def __init__(self):
        self.tts = None
        self.key = None

    def unload(self):
        import torch
        self.tts = None
        self.key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure(self, lora_dir):
        key = str(lora_dir or "")
        if self.tts is not None and self.key == key:
            return
        self.unload()
        import torch
        from indextts.infer_v2_5 import IndexTTS2
        cwd = os.getcwd()
        os.chdir(IX)  # 部分模块按相对路径找 checkpoints
        try:
            self.tts = IndexTTS2(cfg_path=str(CKPT / "config.yaml"), model_dir=str(CKPT),
                                 use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                                 use_qwen_emo=False, lora_dir=str(lora_dir) if lora_dir else None)
        finally:
            os.chdir(cwd)
        # 8 GB 卡会被自动判成"低显存"，那个模式会在每个逗号处切开再插停顿，听起来像剪过的；关掉
        self.tts.low_vram = False
        self.key = key

    def generate(self, text, lang, ref, lora_dir=None, emo_method=EMO_SAME, emo_ref=None, emo_weight=0.6,
                 speed=0.97, pause_ms=150, fade_out_ms=30, temperature=0.75, top_p=0.8, top_k=20,
                 num_beams=3, repetition_penalty=10.0, diffusion_steps=50, max_text_tokens=120,
                 max_mel_tokens=1500, progress=None):
        import torch
        self.ensure(lora_dir)
        segs = split_sentences(text)
        gen = dict(do_sample=True, top_p=float(top_p), top_k=int(top_k) if int(top_k) > 0 else None,
                   temperature=float(temperature), length_penalty=0.0, num_beams=int(num_beams),
                   repetition_penalty=float(repetition_penalty), max_mel_tokens=int(max_mel_tokens),
                   diffusion_steps=int(diffusion_steps))
        wavs, t0 = [], time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for i, seg in enumerate(segs):
            if progress:
                progress(i / len(segs), desc=f"第 {i + 1}/{len(segs)} 段")
            r = self.tts.infer(spk_audio_prompt=str(ref), text=seg, output_path=None, lang=lang,
                               emo_audio_prompt=str(emo_ref) if (emo_method == EMO_REF and emo_ref) else None,
                               emo_alpha=float(emo_weight), emo_vector=None, use_emo_text=False, emo_text=None,
                               use_random=False, interval_silence=200, verbose=False,
                               max_text_tokens_per_segment=int(max_text_tokens),
                               duration_factor=float(speed), **dict(gen))
            if r is None:
                continue
            sr, w = r
            w = np.asarray(w)
            # 引擎返回的是 int16 整数（±32767），要换算成 ±1 的浮点
            w = w.astype(np.float32) / 32768.0 if np.issubdtype(w.dtype, np.integer) else w.astype(np.float32)
            w = w.mean(axis=1) if w.ndim > 1 else w
            n = min(len(w), int(SR * fade_out_ms / 1000))
            if n > 0:
                w[-n:] *= np.linspace(1, 0, n)
            wavs.append(w)
        if not wavs:
            raise RuntimeError("没有生成出音频")
        sil = np.zeros(int(SR * pause_ms / 1000), dtype=np.float32)
        out = []
        for i, w in enumerate(wavs):
            out.append(w)
            if i < len(wavs) - 1:
                out.append(sil)
        audio = np.concatenate(out)
        peak = float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
        return audio, {"segments": len(segs), "seconds": round(len(audio) / SR, 2),
                       "gen_seconds": round(time.time() - t0, 1), "peak_vram_gb": round(peak, 2)}
