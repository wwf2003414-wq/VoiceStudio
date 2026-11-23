"""在独立的模型环境中批量合成，生成结束后随进程释放模型。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from languages import OMNI_CODES, QWEN_NAMES


def clone_qwen(request):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    device = request["device"]
    model = Qwen3TTSModel.from_pretrained(
        request["model"], device_map=device,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    adapter = request.get("adapter")
    if adapter:
        from peft import PeftModel

        model.model = PeftModel.from_pretrained(model.model, adapter).merge_and_unload()
    transcript = "" if adapter else request["transcript"]
    prompt = model.create_voice_clone_prompt(
        ref_audio=request["reference"], ref_text=transcript or None,
        x_vector_only_mode=not transcript,
    )
    for item in request["items"]:
        text = item["text"]
        wavs, rate = model.generate_voice_clone(
            text=text, voice_clone_prompt=prompt, language=QWEN_NAMES[item["lang"]],
            max_new_tokens=max(96, min(768, int(item["seconds"] * 24) + 48)),
            temperature=0.75, top_p=0.85, repetition_penalty=1.05,
        )
        sf.write(item["path"], wavs[0], rate, subtype="PCM_16")


def clone_omni(request):
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    device = request["device"]
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 12))
    model = OmniVoice.from_pretrained(
        request["model"], device_map=device,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    if request.get("adapter"):
        from omnivoice.utils.lora import load_lora_adapter

        model = load_lora_adapter(model, request["adapter"])
    prompt = model.create_voice_clone_prompt(ref_audio=request["reference"], ref_text=request["transcript"])
    for item in request["items"]:
        audio = model.generate(text=item["text"], language=OMNI_CODES[item["lang"]], voice_clone_prompt=prompt)
        sf.write(item["path"], audio[0], model.sampling_rate, subtype="PCM_16")


def clone_auk(request):
    import torch
    from auk.infer.infer_auk import AukInfer, get_gen_duration, save_audio

    model = Path(request["model"])
    device = request["device"]
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 12))
    engine = AukInfer(
        str(model / "config.yaml"), str(model / "auk_flash.safetensors"),
        qwen_path=request["encoder"], device=device,
        cpu_offload=device.startswith("cuda"),
    )
    reference = request["reference"]
    for item in request["items"]:
        text = item["text"]
        seconds = get_gen_duration(
            audio=reference, ref_text=request["transcript"] or None, gen_text=text,
            gen_seconds=None if request["transcript"] else item["seconds"],
        )
        instruction = f'Say the following with the same voice: "{text}"'
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction}, {"type": "audio", "audio": reference},
        ]}]
        audio, rate = engine.generate(messages, gen_seconds=max(1, min(28, seconds)), seed=item["seed"])
        save_audio(audio, rate, item["path"])


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    {"qwen": clone_qwen, "auk": clone_auk, "omni": clone_omni}[job["model_id"]](job)
