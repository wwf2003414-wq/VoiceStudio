# 本地配音工作室

VoiceStudio 是一个在 Windows 上运行的本地 AI 配音工具。它使用 IndexTTS 2.5 生成语音，支持参考音频配音、情感参考、真人开头接续、语音转写质检，以及说话人 LoRA 训练。

## 安装和启动

建议使用 Windows 10 或 11（64 位）、NVIDIA 显卡（至少 8 GB 显存）、支持 CUDA 12.8 的显卡驱动，以及至少 30 GB 可用磁盘空间。首次安装需要访问 PyPI、PyTorch 下载站、Hugging Face、ModelScope 和 OpenAI Whisper 的权重下载站。

1. 克隆或下载此仓库，放到纯英文且无空格的路径，例如 `D:\VoiceStudio`。
2. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。安装后重新打开终端，使 `uv` 在 PATH 中可用。
3. 双击 `setup.bat`。脚本会创建 Python 3.11 环境、安装 CUDA 版 PyTorch 和其他依赖，再下载模型。下载中断时可重新运行。
4. 双击 `检查环境.bat`，然后双击 `启动本地配音工作室.bat`，打开 [本机配音页面](http://127.0.0.1:7860)。

依赖和模型准备好后，配音在本机运行。从抖音下载训练素材需要网络和 Microsoft Edge，也可以从网页上传已有素材。

## 生成第一条配音

在「文本配音」输入文案，选择「不用说话人模型（零样本克隆）」，上传一段约 8–12 秒的本人或已获授权的干净人声，然后点击「生成」。首次生成需要加载模型。音频和生成参数保存在 `output/`。

「训练新说话人」可使用自己上传的素材训练 LoRA。训练产物写入 `speakers/`，模型登记写入 `app/speakers.json`。已有便携版的用户可以在本机导入自己有权使用的说话人目录和登记表。

## 发布内容和许可

仓库包含工作室源码、所需的 IndexTTS 2.5 引擎源码、Windows 启动脚本和模型下载脚本。公开版的登记表从空白开始。个人音色适配器、参考录音、训练素材和生成结果保存在用户本机。

- IndexTTS 引擎来源于 [上游项目](https://github.com/index-tts/index-tts)。引擎及其衍生部分遵循 [bilibili Model Use License Agreement](engine/index-tts/LICENSE)（[中文条款](engine/index-tts/LICENSE_ZH.txt)）。
- 底模从 [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) 下载。其他模型及其发布方列在 [模型下载脚本](app/download_models.py) 中，使用时遵循各自的许可。
- 请只使用本人或已获得明确授权的语音素材和说话人模型。

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.
