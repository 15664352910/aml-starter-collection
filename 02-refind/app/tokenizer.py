from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "that",
    "this",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "must",
    "please",
}


class PorterStemmer:
    """Compact Porter stemmer matching the original ReFind retrieval setup."""

    def __init__(self) -> None:
        self.vowels = set("aeiou")

    def _is_consonant(self, word: str, index: int) -> bool:
        if word[index] in self.vowels:
            return False
        if word[index] == "y":
            return index == 0 or not self._is_consonant(word, index - 1)
        return True

    def _measure(self, word: str) -> int:
        count = 0
        saw_vowel = False
        for index in range(len(word)):
            if not self._is_consonant(word, index):
                saw_vowel = True
            elif saw_vowel:
                count += 1
                saw_vowel = False
        return count

    def _has_vowel(self, word: str) -> bool:
        return any(not self._is_consonant(word, i) for i in range(len(word)))

    def stem(self, word: str) -> str:
        if len(word) <= 2 or not word.isascii():
            return word
        word = word.lower()
        if word.endswith(("sses", "ies")):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word.endswith("eed") and self._measure(word[:-3]) > 0:
            word = word[:-1]
        elif word.endswith("ed") and self._has_vowel(word[:-2]):
            word = word[:-2]
        elif word.endswith("ing") and self._has_vowel(word[:-3]):
            word = word[:-3]
        if word.endswith("y") and self._has_vowel(word[:-1]):
            word = word[:-1] + "i"
        return word


class Tokenizer:
    def __init__(self) -> None:
        self._pattern = re.compile(r"[^\W_]+", re.UNICODE)
        self._stemmer = PorterStemmer()

    def tokenize(self, text: str) -> list[str]:
        raw = self._pattern.findall(text.lower())
        return [self._stemmer.stem(token) for token in raw if token not in STOPWORDS]
