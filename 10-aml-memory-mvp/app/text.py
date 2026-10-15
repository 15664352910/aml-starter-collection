from __future__ import annotations

import re
import unicodedata


WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

CONCEPT_GROUPS = (
    {"prefer", "preference", "favorite", "favourite", "like", "love", "enjoy"},
    {"job", "career", "profession", "occupation", "work", "employment"},
    {"education", "educaton", "school", "study", "studies", "college", "university", "degree", "course", "training", "certification"},
    {"home", "live", "lives", "living", "reside", "residence", "move", "moved", "location", "city", "country"},
    {"trip", "travel", "journey", "vacation", "holiday", "roadtrip", "camping", "hiking", "outdoors"},
    {"park", "nature", "outdoors", "hiking", "camping", "theme", "amusement"},
    {"buy", "bought", "purchase", "purchased", "order", "ordered", "get", "got"},
    {"book", "books", "read", "reading", "author", "bookshelf", "novel"},
    {"music", "song", "songs", "listen", "listening", "artist", "band", "composer"},
    {"child", "children", "kid", "kids", "son", "daughter", "family", "parent"},
    {"friend", "friends", "relationship", "partner", "married", "single", "dating", "breakup"},
    {"health", "medical", "doctor", "physician", "medicine", "medication", "hospital", "clinic"},
    {"food", "meal", "eat", "eating", "restaurant", "dish", "cuisine"},
    {"sport", "sports", "exercise", "workout", "gym", "run", "running", "race"},
    {"movie", "film", "show", "series", "watch", "watched"},
    {"birthday", "born", "birth", "age", "old"},
    {"plan", "planning", "intend", "intention", "goal", "want", "decide", "decided"},
    {"change", "changed", "update", "updated", "correct", "correction", "instead", "now", "current", "latest"},
    {"before", "after", "earlier", "later", "first", "last", "sequence", "timeline"},
    {"religion", "religious", "faith", "church", "temple", "mosque", "spiritual"},
    {"politics", "political", "liberal", "conservative", "democrat", "republican", "rights", "activism"},
    {"lgbtq", "transgender", "trans", "gay", "lesbian", "bisexual", "queer", "identity", "ally", "pride"},
    {"art", "paint", "painting", "draw", "drawing", "pottery", "creative"},
    {"support", "help", "care", "counsel", "counseling", "therapy", "mental"},
    {"personality", "character", "trait", "traits", "describe", "kind", "caring", "thoughtful", "driven", "authentic"},
)

QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "that", "the", "their", "them", "they", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with", "would",
    "什么", "哪个", "哪些", "怎么", "如何", "是否", "我的", "他的", "她的", "他们",
    "这个", "那个", "我们", "你们", "多少", "为何", "为什么",
}

LATEST_MARKERS = {
    "latest", "last", "recent", "current", "currently", "now", "newest", "final",
    "most recent", "up to date", "as of", "最近", "最新", "最后", "当前", "现在", "最终",
}
EARLIEST_MARKERS = {
    "first", "earliest", "initial", "originally", "at first", "最早", "首先", "第一次", "起初", "原先",
}
SEQUENCE_MARKERS = {
    "before", "after", "earlier", "later", "timeline", "sequence", "order", "previous",
    "next", "subsequent", "之前", "之后", "此前", "后来", "时间线", "顺序", "先后",
}
UPDATE_MARKERS = {
    "actually", "instead", "changed", "updated", "correction", "corrected", "no longer",
    "rather than", "now", "currently", "其实", "改成", "改为", "更正", "更新", "不再", "现在",
    "而不是", "后来",
}


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _stem_english(token: str) -> set[str]:
    variants = {token}
    if len(token) > 5 and token.endswith("ing"):
        variants.add(token[:-3])
    if len(token) > 4 and token.endswith("ed"):
        variants.add(token[:-2])
    if len(token) > 4 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    return {item for item in variants if len(item) > 1}


def lexical_terms(value: str, limit: int = 96) -> list[str]:
    normalized = normalize_text(value)
    ordered: dict[str, None] = {}
    for match in WORD_RE.finditer(normalized):
        for variant in sorted(_stem_english(match.group(0))):
            ordered.setdefault(variant, None)
    for match in CJK_RE.finditer(normalized):
        run = match.group(0)
        if len(run) == 1:
            ordered.setdefault(run, None)
        else:
            for size in (2, 3):
                for index in range(max(0, len(run) - size + 1)):
                    ordered.setdefault(run[index : index + size], None)
    return list(ordered)[:limit]


def retrieval_terms(value: str, limit: int = 160) -> list[str]:
    """Add typo-tolerant word fragments and phrase tokens for FTS retrieval."""
    normalized = normalize_text(value)
    all_base_terms = lexical_terms(value, limit=max(512, limit * 8))
    filtered_terms = [term for term in all_base_terms if term not in QUESTION_STOPWORDS]
    source_terms = filtered_terms or all_base_terms
    base_limit = min(96, limit)
    if len(source_terms) <= base_limit:
        base_terms = source_terms
    else:
        # Long rule/reference questions often place the actual task at the end.
        # Preserve both ends plus exact-looking anchors instead of truncating a
        # query to its opening document boilerplate.
        balanced: dict[str, None] = {}
        for term in source_terms[:40] + source_terms[-40:]:
            balanced.setdefault(term, None)
        anchors = sorted(
            source_terms,
            key=lambda term: (any(character.isdigit() for character in term), len(term)),
            reverse=True,
        )
        for term in anchors:
            if len(balanced) >= base_limit:
                break
            balanced.setdefault(term, None)
        base_terms = list(balanced)[:base_limit]
    ordered: dict[str, None] = {term: None for term in base_terms}
    fragment_budget = min(40, max(0, limit - len(ordered)))
    fragments_added = 0
    for word in base_terms:
        if not WORD_RE.fullmatch(word) or len(word) < 6:
            continue
        for index in range(len(word) - 3):
            fragment = word[index : index + 4]
            if fragment not in ordered:
                ordered[fragment] = None
                fragments_added += 1
                if fragments_added >= fragment_budget:
                    break
        if fragments_added >= fragment_budget:
            break
    words = [match.group(0) for match in WORD_RE.finditer(normalized)]
    informative_words = [word for word in words if word not in QUESTION_STOPWORDS]
    pairs = list(zip(informative_words, informative_words[1:]))
    if len(pairs) > 24:
        pairs = pairs[:12] + pairs[-12:]
    for first, second in pairs:
        ordered.setdefault(f"{first}_{second}", None)
    return list(ordered)[:limit]


def informative_terms(value: str, limit: int = 64) -> list[str]:
    """Return query-bearing terms while dropping generic question scaffolding."""
    terms = lexical_terms(value, limit=max(limit * 3, 96))
    filtered = [term for term in terms if term not in QUESTION_STOPWORDS and len(term) > 1]
    return (filtered or terms)[:limit]


def semantic_terms(value: str, limit: int = 96) -> list[str]:
    """Expand common memory concepts without using benchmark-specific answers."""
    base = informative_terms(value, limit=limit)
    present = set(base)
    ordered: dict[str, None] = {term: None for term in base}
    for group in CONCEPT_GROUPS:
        if present & group:
            for term in sorted(group):
                ordered.setdefault(term, None)
    return list(ordered)[:limit]


def make_search_text(content: str, annotations: str = "") -> str:
    terms = retrieval_terms(f"{content}\n{annotations}")
    return " ".join(terms)


def fts_query(terms: list[str]) -> str:
    safe: list[str] = []
    for term in terms:
        cleaned = term.replace('"', '""').strip()
        if cleaned:
            safe.append(f'"{cleaned}"')
    return " OR ".join(safe)


def coverage_score(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = set(lexical_terms(content, limit=512))
    return len(query_terms & content_terms) / len(query_terms)


def exact_anchor_score(query: str, content: str) -> float:
    """Reward exact names, identifiers, quantities, and dates without guessing semantics."""
    query_anchors = {
        term for term in informative_terms(query, limit=64)
        if any(character.isdigit() for character in term) or len(term) >= 5
    }
    if not query_anchors:
        return 0.0
    content_terms = set(lexical_terms(content, limit=512))
    return len(query_anchors & content_terms) / len(query_anchors)


def temporal_intent(query: str) -> str:
    normalized = normalize_text(query)
    if any(marker in normalized for marker in EARLIEST_MARKERS):
        return "earliest"
    if any(marker in normalized for marker in LATEST_MARKERS):
        return "latest"
    if any(marker in normalized for marker in SEQUENCE_MARKERS):
        return "sequence"
    if any(marker in normalized for marker in {"when", "date", "time", "什么时候", "何时", "日期", "时间"}):
        return "point"
    return "none"


def contains_update_marker(content: str) -> bool:
    normalized = normalize_text(content)
    return any(marker in normalized for marker in UPDATE_MARKERS)
