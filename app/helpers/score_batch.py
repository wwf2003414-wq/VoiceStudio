"""一次加载本地 Whisper，转写一批配音候选。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

paths.setup_env()
from make_audio import edit_ops, en_words, norm_text
import torch
import whisper


def local_whisper():
    checkpoint = paths.MODELS / "whisper" / "large-v3-turbo.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(str(checkpoint), device=device), device == "cuda"


def score_candidates(manifest):
    model, fp16 = local_whisper()
    scores = {}
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        lang = item["lang"]
        audio = Path(item["path"])
        try:
            transcript = model.transcribe(str(audio), language=lang.lower(), fp16=fp16,
                                          condition_on_previous_text=False)["text"].strip()
            tokens = (lambda text: list(norm_text(text))) if lang == "ZH" else en_words
            expected, actual = tokens(item["text"]), tokens(transcript)
            distance, _ = edit_ops(expected, actual)
            scores[audio.name] = {"transcript": transcript, "rate": round(distance / max(len(expected), 1), 3)}
        except Exception as error:
            print(f"{audio.name}: {error}", file=sys.stderr)
    manifest.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")


def transcribe_training(folder, metadata, lang):
    model, fp16 = local_whisper()
    with metadata.open("w", encoding="utf-8") as output:
        for number, audio in enumerate(sorted(folder.glob("*.wav")), 1):
            result = model.transcribe(str(audio), language=lang.lower(), fp16=fp16,
                                      condition_on_previous_text=False)
            transcript = " ".join(result["text"].split())
            if transcript:
                output.write(f"{audio.resolve()}\t{transcript}\n")
            if number % 10 == 0:
                print(f"已转写 {number} 个切片", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--train":
        transcribe_training(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "--ref":
        model, fp16 = local_whisper()
        result = model.transcribe(sys.argv[2], fp16=fp16, condition_on_previous_text=False)
        print("REFJSON " + json.dumps(result["text"].strip(), ensure_ascii=False))
    else:
        score_candidates(Path(sys.argv[1]))
