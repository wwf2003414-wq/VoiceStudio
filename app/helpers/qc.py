"""质检：把生成的音频转回文字，和文案比对。

  python qc.py 音频.wav 文案.txt 语言(ZH/EN)  → 打印 JSON
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

paths.setup_env()
import make_audio  # noqa: E402

wav, txt, lang = sys.argv[1], sys.argv[2], sys.argv[3]
text = "".join(l.strip() for l in Path(txt).read_text(encoding="utf-8-sig").splitlines())
r = make_audio.qc(wav, text, "ZH" if lang == "ZH" else "EN")
print("QCJSON" + json.dumps(r, ensure_ascii=False))
