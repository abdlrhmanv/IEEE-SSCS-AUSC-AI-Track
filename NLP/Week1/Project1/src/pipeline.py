"""End-to-end Neurova NLP pipeline.

User text → language detector → language-specific sentiment model.

PDF display fields (mapped from ``analyze`` keys):

- User Text
- Language — Arabic or English
- Sentiment Classification
"""

from __future__ import annotations

from .Arabic_model import ArabicSentimentModel
from .English_model import EnglishSentimentModel
from .labels import ARABIC
from .language_model import LanguageClassifier
from .validation import EmptyTextError, NonLinguisticTextError, validate_text


class NeurovaNLPPipeline:
    def __init__(
        self,
        language_model: LanguageClassifier,
        arabic_model: ArabicSentimentModel,
        english_model: EnglishSentimentModel,
    ):
        self.language_model = language_model
        self.arabic_model = arabic_model
        self.english_model = english_model

    @classmethod
    def load(cls) -> "NeurovaNLPPipeline":
        return cls(
            language_model=LanguageClassifier.load(),
            arabic_model=ArabicSentimentModel.load(),
            english_model=EnglishSentimentModel.load(),
        )

    def analyze(self, text: str) -> dict[str, str]:
        text = validate_text(text)
        language = self.language_model.predict(text)
        if language == ARABIC:
            sentiment = self.arabic_model.predict(text)
        else:
            sentiment = self.english_model.predict(text)
        return {
            "User Text": text,
            "Language": language,
            "Sentiment Classification": sentiment,
        }


def predict(text: str) -> dict[str, str]:
    return NeurovaNLPPipeline.load().analyze(text)


__all__ = [
    "EmptyTextError",
    "NonLinguisticTextError",
    "NeurovaNLPPipeline",
    "predict",
]
