"""Language names and codes used by FengStudio's dubbing and ASR flows."""
from functools import lru_cache

CHOICES = {
    "中文": "ZH", "英文": "EN", "日语": "JA", "韩语": "KO",
    "西班牙语": "ES", "法语": "FR", "德语": "DE", "意大利语": "IT",
    "葡萄牙语": "PT", "俄语": "RU", "阿拉伯语": "AR",
    "越南语": "VI", "泰语": "TH", "印尼语": "ID", "马来语": "MS", "菲律宾语": "TL",
    "印地语": "HI", "孟加拉语": "BN", "泰米尔语": "TA", "乌尔都语": "UR",
    "波斯语": "FA", "希伯来语": "HE", "土耳其语": "TR",
    "荷兰语": "NL", "波兰语": "PL", "瑞典语": "SV", "丹麦语": "DA",
    "捷克语": "CS", "希腊语": "EL", "乌克兰语": "UK",
}
NAMES = {code: name for name, code in CHOICES.items()}
WHISPER_CODES = {code.lower(): code for code in NAMES}
QWEN_NAMES = {
    "ZH": "Chinese", "EN": "English", "JA": "Japanese", "KO": "Korean",
    "ES": "Spanish", "FR": "French", "DE": "German", "IT": "Italian",
    "PT": "Portuguese", "RU": "Russian",
}
OMNI_CODES = {code: code.lower() for code in NAMES}
OMNI_CODES.update(AR="arb", TL="fil")


@lru_cache(maxsize=1)
def identifier():
    from langid.langid import LanguageIdentifier, model

    detector = LanguageIdentifier.from_modelstring(model)
    detector.set_languages(list(WHISPER_CODES))
    return detector


def detect(text):
    if not any(character.isalpha() for character in text):
        return "ZH"
    code, _ = identifier().classify(text)
    return WHISPER_CODES[code]
