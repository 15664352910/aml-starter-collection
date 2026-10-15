from app.text import (
    contains_update_marker,
    coverage_score,
    exact_anchor_score,
    fts_query,
    informative_terms,
    lexical_terms,
    retrieval_terms,
    semantic_terms,
    temporal_intent,
)


def test_cjk_terms_are_segmented() -> None:
    terms = lexical_terms("我喜欢茉莉花茶")
    assert "茉莉" in terms
    assert "花茶" in terms


def test_fts_query_quotes_terms() -> None:
    assert fts_query(["jasmine", "tea"]) == '"jasmine" OR "tea"'


def test_coverage_score_detects_overlap() -> None:
    assert coverage_score({"jasmine", "tea"}, "jasmine tea is preferred") == 1.0


def test_informative_terms_remove_question_scaffolding() -> None:
    terms = informative_terms("What is my current favorite drink?")
    assert "what" not in terms
    assert {"current", "favorite", "drink"} <= set(terms)


def test_temporal_intent_distinguishes_latest_and_earliest() -> None:
    assert temporal_intent("What is the latest preference?") == "latest"
    assert temporal_intent("What was the first preference?") == "earliest"
    assert temporal_intent("What happened after the move?") == "sequence"


def test_exact_anchors_and_update_markers() -> None:
    assert exact_anchor_score("Did it happen in 2024?", "The event happened in 2024.") > 0
    assert contains_update_marker("Actually, I changed it to jasmine tea.")


def test_retrieval_terms_tolerate_typos_and_compounds() -> None:
    assert set(retrieval_terms("bookshelf")) & set(retrieval_terms("books"))
    assert set(retrieval_terms("persue")) & set(retrieval_terms("pursue"))


def test_semantic_terms_expand_general_memory_concepts() -> None:
    terms = set(semantic_terms("What career did I choose after university?"))
    assert {"profession", "education", "degree"} <= terms
