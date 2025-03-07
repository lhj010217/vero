from langdetect import detect
from transformers import pipeline

class AutoMultiLangTranslator:
    """
    지원 언어: 한국어(ko), 중국어(zh), 일본어(ja), 프랑스어(fr), 스페인어(es)
    """
    def __init__(self):
        self.pipelines = {
            "ko": pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en"),
            "zh": pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en"),
            "ja": pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en"),
            "fr": pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"),
            "es": pipeline("translation", model="Helsinki-NLP/opus-mt-es-en"),
        }
    
    def translate(self, text: str) -> str:
        try:
            source_lang = detect(text)
        except Exception as e:
            raise ValueError(f"언어 감지 실패: {e}")
        
        if source_lang == "en":
            return text  # 영어일 경우 번역하지 않고 그대로 반환
        
        if source_lang not in self.pipelines:
            raise ValueError(
                f"지원하지 않는 언어 코드 '{source_lang}'. 지원 언어: {list(self.pipelines.keys())}"
            )
        
        result = self.pipelines[source_lang](text)
        return result[0]["translation_text"]