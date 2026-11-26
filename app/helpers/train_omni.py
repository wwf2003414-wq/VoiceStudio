"""Run upstream OmniVoice training with a Windows-safe data loader."""
import argparse
import json
from importlib import import_module
from operator import itemgetter
from pathlib import Path


def check_runtime(device):
    import torch
    try:
        import_module("omnivoice.training.builder")
        import_module("omnivoice.training.trainer")
        import_module("peft")
    except ImportError as error:
        raise RuntimeError('请在 OmniVoice 环境安装 "omnivoice[lora]==0.2.1"，详见 README。') from error

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("OmniVoice LoRA 训练需要启用 CUDA。")
    torch.cuda.set_device(device if ":" in device else "cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("OmniVoice LoRA 训练需要支持 BF16 的显卡。")


def train(config_path, data_path, output, device):
    check_runtime(device)
    import torch
    from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
    from omnivoice.training.config import TrainingConfig
    from omnivoice.training.trainer import OmniTrainer

    config = TrainingConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    config.data_config = str(data_path)
    config.output_dir = str(output)
    torch.manual_seed(config.seed)
    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    # The upstream SDPA loader uses an inline lambda. Windows spawns workers
    # and cannot pickle that lambda.
    train_loader.dataset.length_fn = itemgetter("length")
    OmniTrainer(model, config, train_loader, eval_loader, tokenizer).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check:
        check_runtime(args.device)
    else:
        train(args.config, args.data, args.output, args.device)
