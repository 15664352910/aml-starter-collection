from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_RE = re.compile(r"[a-z0-9]+")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def tokenize(text: str) -> list[str]:
    """Shared tokenizer for BM25 (add indexing + search query).

    English: lowercase regex word tokens.
    Chinese / mixed: jieba cut, keep CJK and alphanumeric pieces.
    """
    if not text:
        return []

    if _has_cjk(text):
        import jieba

        tokens: list[str] = []
        for tok in jieba.cut(text):
            tok = tok.strip().lower()
            if not tok:
                continue
            if _CJK_RE.search(tok) or _EN_RE.fullmatch(tok):
                tokens.append(tok)
            else:
                tokens.extend(_EN_RE.findall(tok))
                tokens.extend(_CJK_RE.findall(tok))
        return tokens

    return _EN_RE.findall(text.lower())
