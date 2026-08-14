"""Regression tests for #85524 — CJK and other no-space-script retrieval.

The holographic memory store's facts_fts virtual table now uses the trigram
tokenizer (language-agnostic substring search) plus a LIKE fallback for
short (1-2 char) terms. Optional language-aware tokenizers (jieba/fugashi/
konlpy/pythainlp) improve semantic segmentation when installed, and degrade
to overlapping n-grams otherwise.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


# ---------------------------------------------------------------------------
# _tokenize_query / _build_fts_query / _short_term_like — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_substrings",
    [
        # Chinese: with or without jieba, the query must yield searchable
        # terms that cover the original 2-char words (微信/文件/整理).
        ("微信文件整理", ["微信", "文件", "整理"]),
        # Japanese kana+kanji mixed run.
        ("こんにちは世界", ["こん", "世界"]),
        # Korean Hangul.
        ("안녕하세요", ["안녕"]),
        # Thai.
        ("ภาษาไทย", ["ภาษา"]),
        # English unchanged.
        ("homomorphic encryption", ["homomorphic", "encryption"]),
        # Mixed script + Latin (fact-like content).
        ("微信 NAS 下载", ["微信", "nas", "下载"]),
    ],
)
def test_tokenize_query_covers_short_terms(query, expected_substrings):
    """Every short (2-char) term of the original query must survive
    tokenization so the LIKE fallback can find it."""
    terms = FactRetriever._tokenize_query(query)
    for sub in expected_substrings:
        assert any(sub.lower() in t.lower() for t in terms), (
            f"{sub!r} not covered by tokens {terms!r} for query {query!r}"
        )


def test_build_fts_query_drops_short_terms():
    """trigram requires 3+ chars, so 1-2 char terms are excluded from the
    MATCH expression (they go through the LIKE fallback instead)."""
    tokens = ["微信", "文件", "homomorphic"]
    match = FactRetriever._build_fts_query(tokens)
    assert "微信" not in match
    assert "文件" not in match
    assert "homomorphic" in match


def test_short_term_like_detects_two_char_cjk():
    """A query whose meaningful terms are all 2-char (e.g. 微信) must produce
    LIKE patterns so the store can fall back to a scan."""
    patterns = FactRetriever._short_term_like("微信")
    assert patterns is not None
    assert any("微信" in p for p in patterns)


def test_short_term_like_none_for_long_terms():
    """Queries with terms >= 3 chars don't need the LIKE fallback."""
    assert FactRetriever._short_term_like("homomorphic encryption") is None


# ---------------------------------------------------------------------------
# Config gating — enabled_tokenizers controls which languages get semantic
# tokenization (and therefore which lazy installs can trigger)
# ---------------------------------------------------------------------------

def test_enabled_tokenizers_gate_jieba(tmp_path, monkeypatch):
    """With Chinese disabled in config, the retriever must NOT attempt to
    load jieba — it falls back to n-grams instead (no lazy install)."""
    db_path = tmp_path / "gated.db"
    store = MemoryStore(str(db_path))
    store.add_fact(content="微信文件整理 NAS 下载", category="chinese")
    try:
        monkeypatch.setattr(FactRetriever, "enabled_tokenizers", {"ja"})
        retriever = FactRetriever(
            store=store,
            enabled_tokenizers={"ja"},  # Chinese intentionally disabled
        )
        calls = []

        def spy(cls_, name):
            calls.append(name)
            return None

        monkeypatch.setattr(FactRetriever, "_load_optional_tokenizer", classmethod(spy))
        terms = retriever._tokenize_query("微信文件")
        assert calls == [], f"jieba attempted despite zh disabled: {calls}"
        # n-gram fallback still covers the query
        assert any("微信" in t for t in terms)
        # And the LIKE fallback path still finds the fact
        results = retriever.search("微信", limit=10)
        assert any("微信文件整理" in r["content"] for r in results)
    finally:
        store.close()


def test_enabled_tokenizers_allows_jieba(tmp_path, monkeypatch):
    """With Chinese enabled (default), jieba is loaded when present."""
    db_path = tmp_path / "allowed.db"
    store = MemoryStore(str(db_path))
    store.add_fact(content="微信文件整理 NAS 下载", category="chinese")
    try:
        # Reset class-level gating (previous test may have narrowed it).
        monkeypatch.setattr(FactRetriever, "enabled_tokenizers", {"zh", "ja", "ko", "th"})
        retriever = FactRetriever(store=store)  # default: all enabled
        if FactRetriever._try_import_tokenizer("jieba") is None:
            pytest.skip("jieba not installed")
        calls = []

        def spy(cls_, name):
            calls.append(name)
            real = FactRetriever._try_import_tokenizer(name)
            return real

        monkeypatch.setattr(FactRetriever, "_load_optional_tokenizer", classmethod(spy))
        retriever._tokenize_query("微信文件")
        assert "jieba" in calls, f"jieba not attempted: {calls}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Integration — full search against an in-memory trigram-backed store
# ---------------------------------------------------------------------------

@pytest.fixture
def multilingual_store(tmp_path):
    """Store seeded with facts in several scripts."""
    db_path = tmp_path / "multi_lang.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="微信文件整理 NAS 下载",
        category="chinese",
    )
    store.add_fact(
        content="こんにちは世界 日本語の研究",
        category="japanese",
    )
    store.add_fact(
        content="안녕하세요 한국어 테스트",
        category="korean",
    )
    store.add_fact(
        content="สวัสดี ภาษาไทย",
        category="thai",
    )
    store.add_fact(
        content="Harbin Institute of Technology 哈尔滨工业大学",
        category="mixed",
    )
    store.add_fact(
        content="BCP/BENC homomorphic encryption scheme",
        category="crypto",
    )
    yield store
    store.close()


def _hits(facts, query):
    """Return contents of facts whose id appears in search results."""
    ids = {f["fact_id"] for f in facts}
    matched = []
    for fact in facts:
        if fact["fact_id"] in ids:
            matched.append(fact["content"])
    return matched


def test_cjk_search_finds_chinese_fact(multilingual_store):
    results = multilingual_store.search_facts("微信", limit=10)
    assert any("微信文件整理" in r["content"] for r in results)


def test_cjk_search_finds_chinese_4char(multilingual_store):
    results = multilingual_store.search_facts("微信文件", limit=10)
    assert any("微信文件整理" in r["content"] for r in results)


def test_japanese_search(multilingual_store):
    results = multilingual_store.search_facts("世界", limit=10)
    assert any("こんにちは世界" in r["content"] for r in results)


def test_korean_search(multilingual_store):
    results = multilingual_store.search_facts("한국어", limit=10)
    assert any("한국어" in r["content"] for r in results)


def test_thai_search(multilingual_store):
    results = multilingual_store.search_facts("ภาษาไทย", limit=10)
    assert any("สวัสดี ภาษาไทย" in r["content"] for r in results)


def test_english_search(multilingual_store):
    results = multilingual_store.search_facts("homomorphic", limit=10)
    assert any("homomorphic encryption" in r["content"] for r in results)


def test_mixed_script_search(multilingual_store):
    results = multilingual_store.search_facts("哈尔滨", limit=10)
    assert any("哈尔滨工业大学" in r["content"] for r in results)


def test_search_via_retriever_finds_chinese(multilingual_store):
    """The full FactRetriever.search pipeline (FTS + Jaccard + HRR) must
    also surface CJK results — this is the path prefetch_all uses."""
    retriever = FactRetriever(store=multilingual_store)
    results = retriever.search("微信文件", limit=10)
    assert any("微信文件整理" in r["content"] for r in results)


# ---------------------------------------------------------------------------
# Tokenizer migration — old unicode61 DB gets rebuilt with trigram
# ---------------------------------------------------------------------------

def test_fts_tokenizer_migration_rebuilds_trigram(tmp_path):
    """A store created before the trigram change (unicode61 FTS) must be
    transparently migrated on init so existing facts stay searchable."""
    db_path = tmp_path / "legacy.db"
    # Build a store with the OLD schema (unicode61, no tokenize clause),
    # insert a fact, then close.
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE VIRTUAL TABLE facts_fts
            USING fts5(content, tags, content=facts, content_rowid=fact_id);
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE fact_entities (
            fact_id INTEGER REFERENCES facts(fact_id),
            entity_id INTEGER REFERENCES entities(entity_id),
            PRIMARY KEY (fact_id, entity_id)
        );
        CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid, content, tags)
            VALUES (new.fact_id, new.content, new.tags);
        END;
        """
    )
    conn.execute(
        "INSERT INTO facts(content) VALUES (?)", ("旧版存储的中文fact内容",)
    )
    conn.commit()
    conn.close()

    # Reopen with the new code path — migration should rebuild FTS as trigram.
    store = MemoryStore(str(db_path))
    try:
        row = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='facts_fts'"
        ).fetchone()
        assert row is not None and "trigram" in row[0]
        # The migrated index must actually match a Chinese query.
        results = store.search_facts("中文fact", limit=10)
        assert any("中文fact" in r["content"] for r in results)
    finally:
        store.close()
