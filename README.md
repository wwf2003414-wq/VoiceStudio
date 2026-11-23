# FengStudio · 本地配音工作室

FengStudio 在 Windows 上运行。文本配音和真人开头接续支持 30 种配音语言，可使用 IndexTTS 2.5、Qwen3-TTS、AuK-Flash 或 OmniVoice。工作室也支持三模型成品对比、说话人 LoRA 训练和本地转写质检。

## 安装和启动

建议使用 Windows 10 或 11（64 位）、NVIDIA 显卡（至少 8 GB 显存）、支持 CUDA 12.8 的显卡驱动，以及至少 30 GB 可用磁盘空间。首次安装需要访问 PyPI、PyTorch 下载站、Hugging Face、ModelScope 和 OpenAI Whisper 的权重下载站。

1. 克隆或下载此仓库，放到纯英文且无空格的路径，例如 `D:\FengStudio`。
2. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。安装后重新打开终端，使 `uv` 在 PATH 中可用。
3. 双击 `setup.bat`。脚本会创建 Python 3.11 环境、安装 CUDA 版 PyTorch 和其他依赖，再下载模型。下载中断时可重新运行。
4. 双击 `检查环境.bat`，然后双击 `启动 FengStudio.bat`，打开 [本机配音页面](http://127.0.0.1:7860)。

依赖和模型准备好后，配音在本机运行。从抖音下载训练素材需要网络和 Microsoft Edge，也可以从网页上传已有素材。

## 生成第一条配音

在「文本配音」输入文案，选择「不用说话人模型（零样本克隆）」，上传一段约 8–12 秒的本人或已获授权的干净人声，然后点击「生成」。首次生成需要加载模型。音频和生成参数保存在 `output/`。

### 整稿生成和三模型对比

在「文本配音」或「真人开头 + AI」选择配音引擎。选择「三模型对比」后，工作室按顺序为三个引擎各生成一版完整音频，结果可以分别试听和下载。中文和英文可用三个引擎对比。真人开头只处理一次，分别接上三个引擎生成的后段，每版保留自己的 WAV、MP3 和分轨。成品会出现在生成历史。

「文案语言」随引擎切换，只显示当前引擎支持的语言。IndexTTS 2.5 支持 5 种，Qwen3-TTS 支持 10 种，AuK-Flash 支持中文和英文，OmniVoice 支持本工作室提供的 30 种。三模型对比显示三个引擎都支持的语言。

选择「按语言自动选择」会显示全部 30 种。工作室优先用 IndexTTS 2.5，其次用 Qwen3-TTS，其他语言交给 OmniVoice。自动识别文案语言在短句和混合语言中可能有偏差，可以在「文案语言」手动指定。中文和英文成品支持转写质检。

### 用历史素材训练说话人

在「训练说话人」输入说话人标识，批量上传同一说话人的授权音频或视频，选择 **IndexTTS 2.5**、**Qwen3-TTS** 或 **OmniVoice**，再点「保存素材并开始 LoRA 训练」。同一标识可以复用历史素材。素材、语言和筛选时长相同时，可复用已经完成的人声提取、切片和转录。每个引擎单独保存适配器，重训成功后才更新对应的登记项。

在「文本配音」或「真人开头 + AI」选择引擎和说话人，系统会加载该引擎的 LoRA，并使用参考录音。AuK-Flash 使用参考录音克隆。各引擎的训练状态可在「说话人和历史」查看。训练产物位于 `speakers/`，登记表位于 `app/speakers.json`。

Qwen3-TTS 和 OmniVoice 训练需要支持 BF16 的 NVIDIA 显卡和完整的本机 Base 模型快照。请在两个引擎各自的独立 Python 环境安装依赖。使用默认环境时，可在 PowerShell 运行以下命令。

```powershell
uv pip install --python .qwen-venv/Scripts/python.exe 'qwen-tts==0.1.1' 'peft==0.21.2' accelerate
uv pip install --python .omni-venv/Scripts/python.exe 'omnivoice[lora]==0.2.1'
```

如果配置了 `FENG_QWEN_PYTHON` 或 `FENG_OMNI_PYTHON`，将命令中的 Python 路径替换为你的配置值。两个引擎默认批大小为 1。Qwen3-TTS 使用 12 Hz 单说话人训练格式，FengStudio 仅训练并保存 LoRA，推理时使用参考录音的说话人嵌入。OmniVoice 根据素材数、批大小和训练轮数估算训练步数，完成后只保留可加载的适配器。参见 [Qwen3-TTS 单说话人训练文档](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) 和 [OmniVoice LoRA 文档](https://github.com/k2-fsa/OmniVoice/blob/master/docs/lora_finetuning.md)。

### 配置可选引擎

在项目根目录新建 `fengstudio.local.cmd` 并用 `set "变量=绝对路径"` 写入本机配置。该文件不进 Git。

| 变量 | 用途 | 默认位置或值 |
| --- | --- | --- |
| `FENGSTUDIO_DATA` | 复用旧工作室的引擎、权重、说话人、素材和历史 | 当前项目目录 |
| `FENGSTUDIO_PYTHON` | 复用已有的 IndexTTS Python 环境 | `.venv\Scripts\python.exe` 或 `python\python.exe` |
| `FENGSTUDIO_PORT` | 启动端口 | `7860` |
| `FENG_QWEN_PYTHON` | 已安装 `qwen-tts`、PyTorch、`soundfile` 的环境 | `.qwen-venv\Scripts\python.exe` |
| `FENG_QWEN_MODEL` | 完整的 Qwen3-TTS 1.7B Base 本机快照，含 `speech_tokenizer` | 数据目录下的 `models/Qwen3-TTS-12Hz-1.7B-Base` |
| `FENG_QWEN_DEVICE` | Qwen3-TTS 推理设备 | `cuda:0` |
| `FENG_AUK_PYTHON` | 独立的 AuK Python 环境 | `.auk-venv\Scripts\python.exe` |
| `FENG_AUK_MODEL` | 放置 `auk_flash.safetensors`、`vae.safetensors` 与 `config.yaml` 的目录 | 数据目录下的 `models/AuK-Flash` |
| `FENG_AUK_ENCODER` | Qwen2.5-Omni-3B 本机快照 | 数据目录下的 `models/Qwen2.5-Omni-3B` |
| `FENG_AUK_DEVICE` | AuK-Flash 推理设备，显存充足时可用 `cuda:0` | `cpu` |
| `FENG_OMNI_PYTHON` | 已安装 `omnivoice` 和 PyTorch 的独立环境 | `.omni-venv\Scripts\python.exe` |
| `FENG_OMNI_MODEL` | 完整的 OmniVoice 本机快照，含 `audio_tokenizer` | 数据目录下的 `models/OmniVoice` |
| `FENG_OMNI_DEVICE` | OmniVoice 推理设备 | `cuda:0` |

Qwen3-TTS 使用 [官方 PyTorch 推理接口](https://github.com/QwenLM/Qwen3-TTS#voice-clone)。AuK-Flash 的 Python 环境和两个模型目录按 [AuK 安装说明](https://github.com/Tencent-Hunyuan/AuK#installation) 配置。CPU 推理较慢。官方测试中的 CUDA 卸载模式仍使用约 17 GiB 显存。OmniVoice 的 Python 环境按 [官方安装说明](https://github.com/k2-fsa/OmniVoice#installation) 配置，权重来自 [官方 OmniVoice 快照](https://huggingface.co/k2-fsa/OmniVoice)。参考录音的逐字稿留空时，FengStudio 使用本机 Whisper 转写。可选引擎在独立进程中读取本机权重。

整稿配音的参考音频处理与多模型比较受到 [Kaikou](https://github.com/syydaniel/kaikou) 启发。

## 发布内容和许可

仓库包含工作室源码、所需的 IndexTTS 2.5 引擎源码、Windows 启动脚本和模型下载脚本。公开版的登记表从空白开始。个人音色适配器、参考录音、训练素材和生成结果保存在用户本机。

- IndexTTS 引擎来源于 [上游项目](https://github.com/index-tts/index-tts)。引擎及其衍生部分遵循 [bilibili Model Use License Agreement](engine/index-tts/LICENSE)（[中文条款](engine/index-tts/LICENSE_ZH.txt)）。
- 底模从 [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) 下载。其他模型及其发布方列在 [模型下载脚本](app/download_models.py) 中，使用时遵循各自的许可。
- 请只使用本人或已获得明确授权的语音素材和说话人模型。

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.
