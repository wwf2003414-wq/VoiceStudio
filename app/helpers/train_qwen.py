"""Single-speaker LoRA using the Qwen3-TTS 12 Hz training input layout."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence


TARGETS = r"talker\.(model|code_predictor)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def check_runtime(device):
    from importlib import import_module

    for package in ("accelerate", "peft", "qwen_tts"):
        try:
            import_module(package)
        except ImportError as error:
            raise RuntimeError(f"Qwen3-TTS 环境缺少 {package}，详见 README。") from error
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("Qwen3-TTS LoRA 训练需要启用 CUDA。")
    torch.cuda.set_device(device if ":" in device else "cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen3-TTS LoRA 训练需要支持 BF16 的显卡。")


def encode_speech(model_path, device, records):
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(model_path / "speech_tokenizer"), device_map=device)
    encoded = []
    for start in range(0, len(records), 4):
        samples = records[start:start + 4]
        codes = tokenizer.encode([item["audio_path"] for item in samples]).audio_codes
        encoded.extend((item["text"], code.cpu()) for item, code in zip(samples, codes, strict=True) if code.numel())
        if start % 40 == 0:
            print(f"已编码 {min(len(records), start + 4)}/{len(records)} 段", flush=True)
    if not encoded:
        raise RuntimeError("Qwen3-TTS 没有提取到有效音频编码。")
    return encoded


def training_example(processor, config, text, codes):
    # Match the model's single-speaker prompt, with the reference embedding
    # inserted at position 6 and loss applied only to the audio continuation.
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ids = processor(text=prompt, return_tensors="pt")["input_ids"][0][:-5]
    codec = config.talker_config
    frames = len(codes)
    text_ids = torch.cat((
        ids[:3], torch.tensor([config.tts_pad_token_id] * 4 + [config.tts_bos_token_id]),
        ids[3:], torch.tensor([config.tts_eos_token_id]),
        torch.full((frames + 2,), config.tts_pad_token_id),
    ))
    first_frame = len(ids) + 7
    codec_ids = torch.zeros((len(text_ids), codes.shape[1]), dtype=torch.long)
    codec_ids[3:8, 0] = torch.tensor([
        codec.codec_nothink_id, codec.codec_think_bos_id, codec.codec_think_eos_id, 0, codec.codec_pad_id,
    ])
    codec_ids[8:first_frame - 1, 0] = codec.codec_pad_id
    codec_ids[first_frame - 1, 0] = codec.codec_bos_id
    codec_ids[first_frame:first_frame + frames] = codes
    codec_ids[-1, 0] = codec.codec_eos_token_id
    labels = torch.full_like(text_ids, -100)
    labels[first_frame:] = codec_ids[first_frame:, 0]
    return text_ids, codec_ids, labels


def collate(examples):
    texts, codecs, labels = zip(*examples, strict=True)
    lengths = torch.tensor([len(text) for text in texts])
    attention = torch.arange(lengths.max()) < lengths.unsqueeze(1)
    return (
        pad_sequence(texts, batch_first=True),
        pad_sequence(codecs, batch_first=True),
        pad_sequence(labels, batch_first=True, padding_value=-100),
        attention,
    )


def batch_loss(model, speaker, text_ids, codec_ids, labels, attention):
    talker = model.talker
    position = torch.arange(text_ids.shape[1], device=text_ids.device)
    codec_prompt = attention & (position >= 3) & (position != 6)
    code_frames = (labels != -100) & (labels != model.config.talker_config.codec_eos_token_id)
    text_embeddings = talker.model.text_embedding(text_ids) * attention.unsqueeze(-1)
    codec_embeddings = talker.model.codec_embedding(codec_ids[:, :, 0]) * codec_prompt.unsqueeze(-1)
    codec_embeddings[:, 6] = speaker
    inputs = text_embeddings + codec_embeddings
    for codebook, embedding in enumerate(talker.code_predictor.get_input_embeddings(), start=1):
        inputs = inputs + embedding(codec_ids[:, :, codebook]) * code_frames.unsqueeze(-1)
    prediction = talker(inputs_embeds=inputs[:, :-1], attention_mask=attention[:, :-1].long(),
                        labels=labels[:, 1:], output_hidden_states=True, use_cache=False)
    hidden = prediction.hidden_states[0][-1][code_frames[:, :-1]]
    _, sub_loss = talker.forward_sub_talker_finetune(codec_ids[code_frames], hidden)
    return prediction.loss + 0.3 * sub_loss


def train(args):
    check_runtime(args.device)
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from qwen_tts import Qwen3TTSModel

    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    encoded = encode_speech(args.model, args.device, records)
    torch.cuda.empty_cache()
    qwen = Qwen3TTSModel.from_pretrained(
        str(args.model), device_map=args.device, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    with torch.no_grad():
        prompt = qwen.create_voice_clone_prompt(ref_audio=str(args.reference), x_vector_only_mode=True)
        speaker = prompt[0].ref_spk_embedding.detach()

    model = qwen.model
    adapter = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, target_modules=TARGETS))
    model.talker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    examples = [training_example(qwen.processor, model.config, text, code) for text, code in encoded]
    loader = torch.utils.data.DataLoader(examples, batch_size=args.batch, shuffle=True, collate_fn=collate)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
    adapter, optimizer, loader = accelerator.prepare(adapter, optimizer, loader)
    adapter.train()
    for epoch in range(args.epochs):
        for batch_number, batch in enumerate(loader, start=1):
            with accelerator.accumulate(adapter):
                loss = batch_loss(model, speaker, *batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Qwen3-TTS 训练损失异常。")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if batch_number == 1 or batch_number % 10 == 0:
                print(f"第 {epoch + 1}/{args.epochs} 轮 · {batch_number}/{len(loader)} 批 · loss {loss.item():.4f}",
                      flush=True)
    accelerator.unwrap_model(adapter).save_pretrained(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.check:
        check_runtime(args.device)
    else:
        train(args)
