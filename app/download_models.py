"""Fetch the runtime assets from their publishers before running offline."""
import os

import paths


INDEX_FILES = (
    "LICENSE", "config.yaml", "codec.pth", "gpt.pth", "s2mel.pth",
    "feat1.pt", "feat2.pt", "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
)
DEMUCS_FILES = (
    "htdemucs_ft.yaml", "04573f0d.safetensors", "92cfc3b6.safetensors",
    "d12395a8.safetensors", "f7e0c4bc.safetensors",
)


def download():
    env = paths.env_vars()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env[name] = "0"
    os.environ.update(env)

    from huggingface_hub import snapshot_download
    from modelscope.hub.file_download import model_file_download
    from whisper import _MODELS, _download

    models = (
        ("IndexTeam/IndexTTS-2.5", "c39ce5ba981572cb187443877ff559dfb246ce63",
         paths.CKPT, INDEX_FILES),
        ("facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
         paths.HF_CACHE / "w2v-bert-2.0",
         ("config.json", "model.safetensors", "preprocessor_config.json")),
        ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
         paths.HF_CACHE, ("campplus_cn_common.bin",)),
        ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513",
         paths.HF_CACHE / "bigvgan", ("LICENSE", "config.json", "bigvgan_generator.pt")),
        ("funasr/paraformer-zh", "d7811ee3ac581fbcfdeb37c98c6ba674028433dc",
         paths.FUNASR["model"], ("am.mvn", "config.yaml", "configuration.json", "model.pt",
                                  "seg_dict", "tokens.json")),
        ("funasr/fsmn-vad", "df20e6b30c653645fa4ff125cacfcabd1020a669",
         paths.FUNASR["vad_model"], ("am.mvn", "config.yaml", "configuration.json", "model.pt")),
        ("funasr/ct-punc", "d0e55e2b8722a78b63705ff443d09c4f86e5d750",
         paths.FUNASR["punc_model"], ("config.yaml", "configuration.json", "model.pt", "tokens.json")),
        # Demucs resolves these files by repository name, using refs/main in the local Hub cache.
        ("adefossez/HTDemucs-ft", "main", None, DEMUCS_FILES),
    )
    for repo, revision, destination, files in models:
        print(f"Downloading {repo}", flush=True)
        snapshot_download(repo, revision=revision, local_dir=destination,
                          cache_dir=paths.HF_CACHE, allow_patterns=files, max_workers=2)

    print("Downloading CT-Punc tokenizer dictionaries", flush=True)
    for name in ("jieba_usr_dict", "jieba.c.dict"):
        model_file_download("iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                            file_path=name, revision="v2.0.4",
                            local_dir=str(paths.FUNASR["punc_model"]))

    print("Downloading OpenAI Whisper turbo", flush=True)
    _download(_MODELS["turbo"], str(paths.MODELS / "whisper"), in_memory=False)
    print("Model downloads complete.", flush=True)


if __name__ == "__main__":
    problem = paths.path_problem()
    if problem:
        raise SystemExit(problem)
    download()
