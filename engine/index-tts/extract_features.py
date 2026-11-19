# extract_features.py
# 用法: uv run extract_features.py --metadata finetune_data/my_voice.lst --out data/feats --lang zh
import os
# ★ 必须在 import torch 之前设置：让 CUDA 缓存分配器启用“可伸缩段”，显著降低显存碎片与峰值。
#   本脚本在下面直接 import torch，如果等到 import infer_v2_5 时它内部再设置就已经太晚了
#   （torch 一旦初始化 CUDA 分配器，这个环境变量就不再生效）——这正是特征提取显存缓慢升高的元凶之一。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, json
import numpy as np
import torch, torchaudio
import torchaudio.compliance.kaldi as kaldi

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.tokenizer import lang_to_token


import gc

def free_unused_for_feat(tts):
    # 只在“合成”阶段用到的大模块，特征提取用不到，直接释放
    for name in ("s2mel", "bigvgan"):
        m = getattr(tts, name, None)
        if m is not None:
            try: m.to("cpu")
            except Exception: pass
            setattr(tts, name, None)
    # QwenEmotion 是一个完整 LLM（如果这个版本加载了它）
    qe = getattr(tts, "qwen_emo", None)
    if qe is not None:
        if getattr(qe, "model", None) is not None:
            try: qe.model.to("cpu")
            except Exception: pass
            qe.model = None
        tts.qwen_emo = None
    # 情绪/说话人先验矩阵，只有 emo_vector 推理路径才用
    for name in ("emo_matrix", "spk_matrix"):
        if getattr(tts, name, None) is not None:
            setattr(tts, name, None)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="每行: 音频绝对路径 <Tab> 文本（即 build_lst.py 的输出）")
    ap.add_argument("--out", default="data/feats")
    ap.add_argument("--model_dir", default="checkpoints")
    ap.add_argument("--cfg", default="checkpoints/config.yaml")
    ap.add_argument("--lang", default="zh", help="语言标签 zh/en/ja/es；方言按 §10，一般填 zh")
    ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--min_sec", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # 关 bf16 保证特征数值稳定；关自定义 CUDA 内核避免首次编译干扰
    tts = IndexTTS2(cfg_path=args.cfg, model_dir=args.model_dir,
                    use_bf16=False, use_cuda_kernel=False)

    free_unused_for_feat(tts)                       # 释放合成模块

    # ★ 关键修复：低显存/Windows 下这个 fork 会启用 "VRAM offload"，把 semantic_model 等编码器留在 CPU，
    #   只在推理主流程里按需临时搬到 GPU。本脚本直接调用 get_emb/quantize/campplus/merge_emovec，
    #   绕过了那套“按需搬运”，于是出现 weight(cpu) vs input(cuda) 的设备不一致报错。
    #   这里把特征提取真正用到的模块统一固定到同一设备，之后所有 .to(dev) 就都和权重一致。
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 每条样本都要跑的重活（w2v-bert 语义编码、量化、CAMPPlus）固定在 GPU 上，保证速度。
    for name in ("semantic_model", "semantic_codec", "campplus_model"):
        m = getattr(tts, name, None)
        if m is not None:
            m.to(dev)
    # ★ 显存优化：GPT 在特征提取里只用于每条样本算一次 emo_vec（merge_emovec），却是最占显存的模块之一。
    #   把它留在 CPU 上按需计算，可省下 GPT 的整块显存（这是把峰值压到 7GB 以下的关键）。
    #   下方 merge_emovec 会用 next(gpt.parameters()).device 自动跟随到 CPU，结果一致（仅 CPU/GPU
    #   浮点末位差异），速度略慢——符合“可接受慢一点”的取舍。
    if getattr(tts, "gpt", None) is not None:
        tts.gpt.to("cpu")
    # 归一化统计量若是普通张量属性(而非 register_buffer)，m.to(dev) 搬不动，这里单独搬一次，
    # 防止 get_emb 内部再报 mean/std 的设备不一致。（新版 get_emb 已会自动对齐设备，这里留作双保险。）
    for name in ("semantic_mean", "semantic_std"):
        t = getattr(tts, name, None)
        if isinstance(t, torch.Tensor):
            setattr(tts, name, t.to(dev))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpt = tts.gpt
    max_mel = gpt.max_mel_tokens
    max_txt = gpt.max_text_tokens
    lang_id = lang_to_token(args.lang)
    lang_prefix = f"<|{args.lang.lower()}|> "

    with open(args.metadata, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    index, kept, skipped = [], 0, 0
    for i, line in enumerate(lines):
        # 这些名字先占位，保证 finally 里清理时一定已定义（提前 skip 也不会 NameError）。
        feats = input_features = attention_mask = w2v_feat = wf = feat_fb = audio = audio_16k = None
        try:
            try:
                wav_path, text = line.split("\t", 1)
            except ValueError:
                print(f"[skip] 需要 路径<Tab>文本: {line[:50]}"); skipped += 1; continue
            text = text.strip()
            if not os.path.isfile(wav_path) or not text:
                print(f"[skip] 文件缺失或空文本: {wav_path}"); skipped += 1; continue

            # ---- 读音频：单声道 ----
            audio, sr = torchaudio.load(wav_path)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            dur = audio.shape[-1] / sr
            if dur < args.min_sec or dur > args.max_sec:
                print(f"[skip] 时长 {dur:.1f}s 越界: {wav_path}"); skipped += 1; continue

            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)  # (1, N)

            # ---- 1) w2v-bert 特征（按推理原样传 (1,N)）----
            feats = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = feats["input_features"].to(dev)
            attention_mask = feats["attention_mask"].to(dev)
            w2v_feat = tts.get_emb(input_features, attention_mask)          # (1, T, 1024) 已归一化

            # ---- 2) 语义 token（训练目标）----
            codes, _ = tts.semantic_codec.quantize(w2v_feat)
            codes = codes.squeeze(0) if codes.ndim == 2 else codes         # 守卫: (1,T)->(T,)
            if codes.ndim == 2:                                            # 万一 (n_q, T)，取主码本
                codes = codes[0]
            codes = codes.long().cpu().reshape(-1)
            if codes.numel() < 4 or codes.numel() > max_mel - 2:
                print(f"[skip] semantic 长度 {codes.numel()} 越界(>{max_mel-2}): {wav_path}")
                skipped += 1; continue

            # ---- 3) 情绪向量（推理默认路径 merge_emovec；参考=目标）----
            gpt_dev = next(gpt.parameters()).device
            wf = w2v_feat.to(gpt_dev)
            L = torch.tensor([wf.shape[-1]], device=gpt_dev)
            emo_vec = gpt.merge_emovec(wf, wf, L, L, alpha=1.0)            # (1, d)  # gpt 在 CPU 上时自动跟随到 CPU
            emo_vec = emo_vec.squeeze(0).float().cpu()

            # ---- 4) CAMPPlus 说话人向量(192)----
            feat_fb = kaldi.fbank(audio_16k.to(dev), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat_fb = feat_fb - feat_fb.mean(dim=0, keepdim=True)
            spk_emb = tts.campplus_model(feat_fb.unsqueeze(0)).squeeze(0).float().cpu()  # (192,)

            # ---- 5) 文本 token ----
            toks = tts.tokenizer.encode(lang_prefix + text, allowed_special="all")
            if len(toks) < 1 or len(toks) > max_txt - 2:
                print(f"[skip] 文本 token 数 {len(toks)} 越界(>{max_txt-2}): {text[:30]}")
                skipped += 1; continue
            text_ids = torch.LongTensor(toks)

            item_path = os.path.join(args.out, f"{i:06d}.pt")
            torch.save({"codes": codes, "text_ids": text_ids, "lang_id": int(lang_id),
                        "emo_vec": emo_vec, "spk_emb": spk_emb,
                        "wav": wav_path, "text": text}, item_path)
            index.append(item_path); kept += 1
            if kept % 20 == 0:
                print(f"  已处理 {kept} ...")
        finally:
            # ★ 每条样本处理完（无论成功或 skip）立即回收 GPU 显存：
            #   变长音频会让 CUDA 缓存分配器保留“见过的最大尺寸”分块并逐步碎片化，
            #   这正是特征提取显存从 7.8 → 8.4 → 9.0 缓慢升高的直接原因。
            #   把这几个 GPU 上的大中间量解引用后清缓存，峰值就被钉在“单条最大工作集”。
            feats = input_features = attention_mask = w2v_feat = wf = feat_fb = audio = audio_16k = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (i & 63) == 0:          # 每 64 条顺带做一次 Python GC，压制主机侧碎片
                gc.collect()

    n_val = max(1, len(index) // 20)
    with open(os.path.join(args.out, "train.json"), "w") as f:
        json.dump(index[:-n_val], f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "valid.json"), "w") as f:
        json.dump(index[-n_val:], f, ensure_ascii=False, indent=2)
    print(f"\n完成: 保留 {kept}, 跳过 {skipped}. train={len(index)-n_val} valid={n_val}\n特征目录: {args.out}")


if __name__ == "__main__":
    main()