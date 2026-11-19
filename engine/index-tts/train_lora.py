# train_lora.py
# 用法: .venv/bin/python -u train_lora.py --feats data/feats_yzy --out runs/exp1_yzy
#
# 省显存要点（都不影响效果）：
#   1) 梯度检查点(use_reentrant=False)：数学等价，用重算换显存 —— 省显存大头
#   2) 混合精度：冻结底座走半精度(bf16/fp16)，LoRA 主权重与 loss 保持 fp32
#   3) 自动按显卡选精度：支持 bf16 用 bf16(30系+)，不支持则退回 fp16(20系)
#   4) 训练前释放特征提取/合成用不到的大模型
# 想完全复现纯 fp32 数值可加 --precision fp32（仍有梯度检查点省显存）。
import os, gc, json, argparse, math, random
import torch
import shutil
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model


# ---------------- Dataset ----------------
class FeatDataset(Dataset):
    def __init__(self, list_json):
        with open(list_json) as f:
            self.items = json.load(f)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return torch.load(self.items[idx], map_location="cpu")


def collate(batch):
    B = len(batch)
    text_lens = torch.LongTensor([b["text_ids"].numel() for b in batch])
    code_lens = torch.LongTensor([b["codes"].numel() for b in batch])
    Lt, Lc = int(text_lens.max()), int(code_lens.max())
    text = torch.full((B, Lt), 1, dtype=torch.long)     # 用 stop_text_token=1 占位
    codes = torch.zeros((B, Lc), dtype=torch.long)      # 用 0 占位, forward 内会覆盖成 stop
    for i, b in enumerate(batch):
        text[i, :b["text_ids"].numel()] = b["text_ids"]
        codes[i, :b["codes"].numel()] = b["codes"]
    return dict(text=text, text_lens=text_lens, codes=codes, code_lens=code_lens,
                lang=torch.LongTensor([b["lang_id"] for b in batch]),
                emo=torch.stack([b["emo_vec"] for b in batch]),
                spk=torch.stack([b["spk_emb"] for b in batch]))


# --------- 训练 forward（对齐官方 forward + 补 lang_embedding + 算 loss）---------
def train_forward(gpt, batch, device):
    text_inputs = batch["text"].to(device)
    text_lengths = batch["text_lens"].to(device)
    mel_codes = batch["codes"].to(device)
    mel_code_lengths = batch["code_lens"].to(device)
    langs = batch["lang"].to(device)
    emo_vec = batch["emo"].to(device)              # (B, d)
    spk_emb = batch["spk"].to(device)              # (B, 192)

    spk_latent = gpt.spk_emb_proj(spk_emb).unsqueeze(1)                 # (B,1,d)

    text_inputs = gpt.set_text_padding(text_inputs, text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=gpt.stop_text_token)
    text_inputs, text_targets = gpt.build_aligned_inputs_and_targets(
        text_inputs, gpt.start_text_token, gpt.stop_text_token)
    text_emb = gpt.text_embedding(text_inputs) + gpt.text_pos_embedding(text_inputs)
    text_emb = text_emb + gpt.lang_embedding(langs).unsqueeze(1)        # 与推理一致

    mel_codes = gpt.set_mel_padding(mel_codes, mel_code_lengths)
    mel_codes = F.pad(mel_codes, (0, 1), value=gpt.stop_mel_token)
    mel_codes, mel_targets = gpt.build_aligned_inputs_and_targets(
        mel_codes, gpt.start_mel_token, gpt.stop_mel_token)
    mel_emb = gpt.mel_embedding(mel_codes) + gpt.mel_pos_embedding(mel_codes)

    # 说话人 + 情绪条件。把 emo_vec / pad 的 dtype 对齐到 spk_latent，
    # 否则在混合精度(autocast)下，get_logits 里对 conds/text_emb/mel_emb 做 cat 会因
    # dtype 不一致报错。数值本身不变（pad 全 0），对效果无影响。
    cond_spk = spk_latent + emo_vec.unsqueeze(1).to(spk_latent.dtype)   # (B,1,d)
    pad = torch.zeros(cond_spk.size(0), 2, cond_spk.size(2),
                      device=device, dtype=cond_spk.dtype)
    conds = torch.cat([cond_spk, pad], dim=1)                           # (B,3,d)

    # 双保险：保证送入 transformer 的输入带梯度，配合梯度检查点，
    # 杜绝“全冻结 embedding → LoRA 收不到梯度”的老问题。
    if gpt.training:
        for t in (conds, text_emb, mel_emb):
            if t.is_leaf and not t.requires_grad:
                t.requires_grad_(True)

    text_logits, mel_logits = gpt.get_logits(
        conds, text_emb, gpt.text_head, mel_emb, gpt.mel_head,
        get_attns=False, return_latent=False)

    loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
    loss_text = F.cross_entropy(text_logits, text_targets.long())
    return loss_mel + 0.01 * loss_text, loss_mel.detach(), loss_text.detach()


def resolve_precision(name):
    """返回 (amp_dtype, use_fp16)。amp_dtype=None 表示纯 fp32、不开混合精度。
       20 系(Turing)不支持 bf16 → 自动退回 fp16；30 系及以上 → bf16。"""
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if name == "auto":
        name = "bf16" if bf16_ok else "fp16"
    if name == "bf16" and not bf16_ok:
        print(">> 当前显卡不支持 bf16(如 RTX 20 系)，自动切换到 fp16")
        name = "fp16"
    if name == "fp32":
        return None, False
    return (torch.bfloat16 if name == "bf16" else torch.float16), (name == "fp16")


def build_gpt(args, amp_dtype):
    """用 IndexTTS2 构建并加载 gpt，再释放训练用不到的大模型。"""
    from indextts.infer_v2_5 import IndexTTS2
    tts = IndexTTS2(cfg_path=args.cfg, model_dir=args.model_dir,
                    use_bf16=False, use_cuda_kernel=False, use_accel=False)
    gpt = tts.gpt
    # 释放显存：训练只需 gpt 本体，特征已离线算好
    gpt.inference_model = None
    gpt.accel_engine = None
    for attr in ["semantic_model", "semantic_codec", "s2mel", "campplus_model",
                 "bigvgan", "qwen_emo", "spk_matrix", "emo_matrix"]:
        if hasattr(tts, attr):
            setattr(tts, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpt.gpt.config.use_cache = False

    # 冻结的底座转半精度：权重显存减半、前向激活也减半。
    # LoRA 主权重稍后升回 fp32、loss 也在 fp32 算，所以对训练效果几乎无影响。
    # amp_dtype=None（--precision fp32）时保持 fp32 不动。
    if amp_dtype is not None:
        gpt = gpt.to(amp_dtype)
    else:
        gpt = gpt.float()
    return gpt, tts.device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="data/feats")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--model_dir", default="checkpoints")
    ap.add_argument("--cfg", default="checkpoints/config.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto=按显卡自动选(30系+用bf16，20系用fp16)；fp32=纯全精度")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    # 注意：先清旧目录，再重建（原代码 makedirs 后紧跟 rmtree，会把刚建的目录删掉）
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)

    amp_dtype, use_fp16 = resolve_precision(args.precision)
    use_amp = amp_dtype is not None
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    prec_str = ("fp32(纯全精度)" if not use_amp
                else ("fp16 + GradScaler" if use_fp16 else "bf16"))
    print(f">> 设备: {dev_name} | 训练精度: {prec_str} | "
          f"梯度检查点: 开(use_reentrant=False)")

    gpt, device = build_gpt(args, amp_dtype)

    # 冻结全部
    for p in gpt.parameters():
        p.requires_grad = False
    # 给内部 GPT2 (Conv1D: c_attn/c_proj/c_fc) 加 LoRA
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout,
                          target_modules=["c_attn", "c_proj", "c_fc"], bias="none")
    gpt.gpt = get_peft_model(gpt.gpt, lora_cfg)

    # 只把可训练的 LoRA 参数升回 fp32（底座保持半精度）。
    # 这就是标准混合精度：底座/激活半精度省显存，LoRA 主权重 fp32 保稳。
    if use_amp:
        for p in gpt.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    # 重新打开梯度检查点（非重入实现）：数学上完全等价，只是用“重算”换显存，
    # 且在输入不带梯度时依然能把梯度回传给 LoRA。这是省显存的最大来源。
    gpt.gpt.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(gpt.gpt, "enable_input_require_grads"):
        gpt.gpt.enable_input_require_grads()

    gpt.gpt.print_trainable_parameters()

    trainable = [p for p in gpt.parameters() if p.requires_grad]
    assert len(trainable) > 0, "无可训练参数：检查 target_modules 是否匹配到 GPT2 层"
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    # 只有 fp16 需要 loss scaling；bf16/fp32 时 scaler 处于关闭状态（各调用均为直通）
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    train_dl = DataLoader(FeatDataset(os.path.join(args.feats, "train.json")),
                          batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=2, drop_last=True)
    val_dl = DataLoader(FeatDataset(os.path.join(args.feats, "valid.json")),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=max(total_steps, 1), pct_start=0.3)

    def autocast_ctx():
        return torch.autocast(device_type="cuda",
                              dtype=(amp_dtype or torch.float16),
                              enabled=use_amp and torch.cuda.is_available())

    gpt.train(); step = 0
    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        for it, batch in enumerate(train_dl):
            with autocast_ctx():
                loss, lm, lt = train_forward(gpt, batch, device)
            scaler.scale(loss / args.grad_accum).backward()
            if (it + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                # fp16 溢出时本步被跳过，此时不推进学习率调度，避免与优化器步数错位
                if scaler.get_scale() >= prev_scale:
                    sched.step()
                opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"ep{ep} step{step} loss={loss.item():.4f} "
                          f"mel={lm.item():.4f} text={lt.item():.4f} lr={sched.get_last_lr()[0]:.2e}")
        # 验证
        gpt.eval()
        with torch.no_grad(), autocast_ctx():
            vs = [train_forward(gpt, b, device)[1].item() for b in val_dl]
        print(f"[valid] epoch {ep} mel_loss={sum(vs)/max(len(vs),1):.4f}")
        gpt.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 存 LoRA（很小）
        #if ep % args.save_every == 0 or ep == args.epochs:
        if ep == args.epochs:
            #sd = os.path.join(args.out, f"lora_ep{ep}")
            sd = args.out
            gpt.gpt.save_pretrained(sd)
            print(f"saved LoRA -> {sd}")
    print("训练完成。")


if __name__ == "__main__":
    main()

