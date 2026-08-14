"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    # Languages opted into for semantic tokenization (set per-process from
    # config `plugins.hermes-memory-store.tokenizers`). Class-level so the
    # classmethod tokenizer paths can read it; default enables all.
    enabled_tokenizers: set[str] = {"zh", "ja", "ko", "th"}

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
        enabled_tokenizers: set[str] | None = None,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim
        # Languages the user explicitly opted into for semantic tokenization
        # (config `plugins.hermes-memory-store.tokenizers: ["zh", ...]`).
        # Default: all supported, so out-of-the-box the provider still tries
        # each language's tokenizer; an empty set disables ALL lazy installs.
        # Stored on the class so the classmethod tokenizer paths see it.
        if enabled_tokenizers is not None:
            type(self).enabled_tokenizers = set(enabled_tokenizers)
        elif not getattr(type(self), "enabled_tokenizers", None):
            type(self).enabled_tokenizers = {"zh", "ja", "ko", "th"}

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        # The query vector is loop-invariant — encode it at most once, on
        # the first candidate that actually carries an HRR vector. Lazy on
        # purpose: migrated stores can have FTS candidates whose hrr_vector
        # was never backfilled (MemoryStore._init_db adds the column
        # without backfilling), and those must not pay for an encode
        # nothing will use. encode_text is deterministic (SHA-256 counter
        # blocks), so the hoisted vector is bit-identical to what the
        # per-candidate calls produced.
        query_vec = None
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"], dim=self.hrr_dim)
                if query_vec is None:
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Unbinds entity from memory bank to extract associated content.
        This is NOT keyword search — it uses algebraic structure to find facts
        where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as role-bound vector
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try category-specific bank first, then all facts
        if category:
            bank_name = f"cat:{category}"
            bank_row = conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            ).fetchone()
            if bank_row:
                bank_vec = hrr.bytes_to_phases(bank_row["vector"], dim=self.hrr_dim)
                extracted = hrr.unbind(bank_vec, probe_key)
                # Use extracted signal to score individual facts
                return self._score_facts_by_vector(
                    extracted, category=category, limit=limit
                )

        # Score against individual fact vectors directly
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            # Final fallback: keyword search
            return self.search(entity, category=category, limit=limit)

        # role_content is loop-invariant — encode it once (deterministic
        # SHA-256-based atom) instead of once per fact row.
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            # Unbind probe key from fact to see if entity is structurally present
            residual = hrr.unbind(fact_vec, probe_key)
            # Compare residual against content signal
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            sim = hrr.similarity(residual, content_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        # Both role atoms are loop-invariant — encode them once here
        # (deterministic SHA-256-based atoms) instead of twice per fact row.
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact

            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

        # For each entity, compute what the bank "remembers" about it
        # by unbinding entity+role from each fact vector
        entity_residuals = []
        for entity in entities:
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = hrr.bind(entity_vec, role_entity)
            entity_residuals.append(probe_key)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Score each fact by how much EACH entity is structurally present.
        # A fact scores high only if ALL entities have structural presence
        # (AND semantics via min, vs OR which would use mean/max).
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

            entity_scores = []
            for probe_key in entity_residuals:
                residual = hrr.unbind(fact_vec, probe_key)
                sim = hrr.similarity(residual, role_content)
                entity_scores.append(sim)

            min_sim = min(entity_scores)
            fact["score"] = (min_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"], dim=self.hrr_dim)
                v2 = hrr.bytes_to_phases(f2["hrr_vector"], dim=self.hrr_dim)
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        # FTS5 defaults to AND-between-tokens, which kills recall on
        # natural-language queries ("what happened with the deployment
        # rollback"). Sanitize: drop stopwords, OR-join content tokens, so
        # any significant term can match.
        params.append(self._sanitize_fts_query(query))

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        # Trigram only matches 3+ char terms; short CJK queries (e.g. 微信)
        # need a LIKE fallback so they stay searchable.
        like_patterns = self._short_term_like(query)
        if like_patterns:
            like_clauses = " OR ".join(
                f"(f.content LIKE ? OR f.tags LIKE ?)"
                for _ in like_patterns
            )
            like_params: list = []
            for pat in like_patterns:
                like_params.extend([pat, pat])
            like_where = [f"({like_clauses})"]
            if category:
                like_where.append("f.category = ?")
                like_params.append(category)
            like_where.append("f.trust_score >= ?")
            like_params.append(min_trust)
            like_params.append(limit)
            like_sql = f"""
                SELECT f.*, 0.0 as fts_rank_raw
                FROM facts f
                WHERE {" AND ".join(like_where)}
                ORDER BY f.trust_score DESC
                LIMIT ?
            """
            try:
                rows = conn.execute(like_sql, like_params).fetchall()
            except Exception:
                return []
            if not rows:
                return []
            return [
                {**dict(row), "fts_rank": 1.0} for row in rows
            ]

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    # Stopwords dropped before FTS5 OR-expansion. Short English function
    # words that carry no retrieval signal and force false-negative AND
    # matches when left in the query.
    _FTS_STOPWORDS = frozenset({
        "a", "about", "above", "after", "again", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being",
        "between", "both", "but", "by", "can", "could", "did", "do", "does",
        "doing", "don", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves",
    })

    # ── Optional language-aware tokenizers (lazy-imported, see LAZY_DEPS
    #    "memory.holographic.tokenizers"). Each is tried in order for the
    #    scripts it knows; when absent the caller falls back to n-gram
    #    trigram/LIKE matching which works for every script.
    _TOKENIZER_CACHE: dict[str, Callable[[str], list[str]] | None] = {}

    @classmethod
    def _try_import_tokenizer(cls, name: str) -> Callable[[str], list[str]] | None:
        """Best-effort import of an optional tokenizer; None when missing.

        Catches both missing packages (ImportError) and broken runtimes
        (e.g. konlpy requires a JVM, which may be absent) — anything that
        prevents the tokenizer from actually running degrades to None so the
        caller falls back to n-grams.
        """
        try:
            if name == "jieba":
                import jieba  # type: ignore[import-not-found]

                jieba.setLogLevel(60)
                return lambda text: list(jieba.cut_for_search(text))
            if name == "pythainlp":
                import pythainlp  # type: ignore[import-not-found]

                return lambda text: list(pythainlp.tokenize.word_tokenize(text))
            if name == "fugashi":
                import fugashi  # type: ignore[import-not-found]

                tagger = fugashi.Tagger()
                return lambda text: [t.surface for t in tagger(text)]
            if name == "konlpy":
                from konlpy.tag import Okt  # type: ignore[import-not-found]

                return Okt().morphs
        except Exception:
            return None
        return None

    # Optional tokenizer → lazy-deps feature name (see tools/lazy_deps.py).
    # Each language is a separate feature so a user only installs what their
    # queries actually use. konlpy (Korean) has no entry — it needs a JVM at
    # import time, so users install it manually.
    _TOKENIZER_FEATURES = {
        "jieba": "memory.holographic.tokenizer.jieba",
        "fugashi": "memory.holographic.tokenizer.fugashi",
        "pythainlp": "memory.holographic.tokenizer.pythainlp",
        "konlpy": None,
    }

    @classmethod
    def _load_optional_tokenizer(
        cls, name: str
    ) -> Callable[[str], list[str]] | None:
        """Return a callable tokenizer for *name* or None if not installed.

        Cache the result so import cost is paid once per process. When the
        package is missing, a lazy install is attempted via
        ``tools.lazy_deps.ensure`` for that language's feature (so a Chinese
        query never pulls the Japanese or Thai tokenizer); any failure —
        including lazy installs disabled by config — degrades to None
        (n-gram fallback) instead of raising.
        """
        if name in cls._TOKENIZER_CACHE:
            return cls._TOKENIZER_CACHE[name]
        tokenizer = cls._try_import_tokenizer(name)
        feature = cls._TOKENIZER_FEATURES.get(name)
        if tokenizer is None and feature is not None:
            # One lazy-install attempt per missing package, then fall back.
            try:
                from tools.lazy_deps import ensure as _lazy_ensure

                _lazy_ensure(feature, prompt=False)
                tokenizer = cls._try_import_tokenizer(name)
            except Exception:
                tokenizer = None
        cls._TOKENIZER_CACHE[name] = tokenizer
        return tokenizer

    # Unicode scripts that write without word separators; these get explicit
    # tokenizer treatment (jieba for Han, pythainlp for Thai, fugashi for
    # Japanese, konlpy for Hangul) with an n-gram fallback when the package
    # is not installed. FTS5's default unicode61 tokenizer treats an
    # unbroken CJK run as a single token, so a spaced query never matches.
    _RE_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
    _RE_HIRAGANA = re.compile(r"[\u3040-\u309f]+")
    _RE_KATAKANA = re.compile(r"[\u30a0-\u30ff]+")
    _RE_HANGUL = re.compile(r"[\uac00-\ud7af]+")
    _RE_THAI = re.compile(r"[\u0e00-\u0e7f]+")
    _RE_OTHER_NOSPACE = re.compile(
        r"[\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff\u0f00-\u0fff]+"
    )  # Lao, Myanmar, Khmer, Tibetan — no mature tokenizer, n-gram fallback

    @classmethod
    def _tokenize_query(cls, query: str) -> list[str]:
        """Language-aware tokenization of a natural-language query.

        Returns a list of search terms (not FTS5-syntax-quoted). English and
        other space-delimited scripts are split on whitespace; Han/Hiragana/
        Katakana/Hangul/Thai runs are handed to the optional installed
        tokenizer when available (jieba/fugashi/konlpy/pythainlp) and fall
        back to overlapping n-grams otherwise so any substring is searchable.
        """
        if not query:
            return []
        tokens: list[str] = []
        _FTS_SPECIAL = "\"()*^:-+"
        for raw in query.lower().split():
            cleaned = raw.strip(".,;:!?\"'()[]{}#@<>").translate(
                str.maketrans("", "", _FTS_SPECIAL)
            )
            if not cleaned:
                continue
            # English/digit tokens pass through verbatim.
            if not any(
                [
                    cls._RE_HAN.search(cleaned),
                    cls._RE_HIRAGANA.search(cleaned),
                    cls._RE_KATAKANA.search(cleaned),
                    cls._RE_HANGUL.search(cleaned),
                    cls._RE_THAI.search(cleaned),
                    cls._RE_OTHER_NOSPACE.search(cleaned),
                ]
            ):
                if len(cleaned) < 2:
                    continue
                if cleaned in cls._FTS_STOPWORDS:
                    continue
                tokens.append(cleaned)
                continue
            # No-space script(s): tokenize each run separately so one
            # tokenizer never sees another script it cannot handle.
            parts = re.split(
                f"({cls._RE_HAN.pattern}|{cls._RE_HIRAGANA.pattern}|"
                f"{cls._RE_KATAKANA.pattern}|{cls._RE_HANGUL.pattern}|"
                f"{cls._RE_THAI.pattern}|{cls._RE_OTHER_NOSPACE.pattern})",
                cleaned,
            )
            for part in parts:
                if not part:
                    continue
                if len(part) < 2:
                    continue
                script_tokens: list[str] = []
                if cls._RE_HAN.fullmatch(part):
                    tok = (
                        cls._load_optional_tokenizer("jieba")
                        if "zh" in cls.enabled_tokenizers
                        else None
                    )
                    if tok is not None:
                        script_tokens = [w for w in tok(part) if len(w) >= 1]
                elif cls._RE_HIRAGANA.fullmatch(part) or cls._RE_KATAKANA.fullmatch(part):
                    tok = (
                        cls._load_optional_tokenizer("fugashi")
                        if "ja" in cls.enabled_tokenizers
                        else None
                    )
                    if tok is not None:
                        script_tokens = [w for w in tok(part) if len(w) >= 1]
                elif cls._RE_HANGUL.fullmatch(part):
                    tok = (
                        cls._load_optional_tokenizer("konlpy")
                        if "ko" in cls.enabled_tokenizers
                        else None
                    )
                    if tok is not None:
                        script_tokens = [w for w in tok(part) if len(w) >= 1]
                elif cls._RE_THAI.fullmatch(part):
                    tok = (
                        cls._load_optional_tokenizer("pythainlp")
                        if "th" in cls.enabled_tokenizers
                        else None
                    )
                    if tok is not None:
                        script_tokens = [w for w in tok(part) if len(w) >= 1]
                if not script_tokens:
                    # Fallback: overlapping n-grams (n=2 preserves 2-char
                    # terms; trigram index still matches any 3+ substring).
                    script_tokens = [
                        part[i : i + 2] for i in range(len(part) - 1)
                    ]
                tokens.extend(script_tokens)
        return tokens

    @classmethod
    def _build_fts_query(cls, tokens: list[str]) -> str:
        """Join search tokens into an FTS5 MATCH expression.

        OR-joins every token so any single term can hit (trigram matches
        substrings of 3+ chars; LIKE covers shorter ones in the caller).
        """
        if not tokens:
            return ""
        return " OR ".join(f'"{t}"' for t in tokens if len(t) >= 3)

    @classmethod
    def _short_term_like(cls, query: str) -> list[str] | None:
        """Return LIKE patterns for queries whose meaningful terms are all
        shorter than 3 chars (trigram cannot match them), else None.

        Terms shorter than 3 chars (typically overlapping n-grams from the
        CJK fallback, e.g. 微信 or 한국/국어) cannot be AND-joined into a
        single ``%a%b%`` pattern because overlapping grams never appear
        consecutively in the source text (한국어 contains both 한국 and 국어
        but not as a run). The returned list holds one ``%term%`` pattern
        per short term; callers should OR them in SQL.
        """
        terms = [t for t in cls._tokenize_query(query) if len(t) < 3]
        if not terms:
            return None
        seen: set[str] = set()
        patterns = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                patterns.append(f"%{t}%")
        return patterns

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Convert a natural-language query to an FTS5-safe OR expression.

        The store's ``facts_fts`` virtual table uses the trigram tokenizer
        (language-agnostic substring search). This helper:
          - tokenizes the query with language-aware tokenizers
            (jieba/pythainlp/fugashi/konlpy when installed, n-gram fallback)
          - strips FTS5 special characters
          - OR-joins tokens of length >= 3 (trigram requires 3+ chars)

        Queries whose terms are all shorter than 3 chars cannot be matched
        by trigram; callers should detect that via :meth:`_short_term_like`
        and run a LIKE query instead. If nothing remains, falls back to the
        raw query so the caller sees zero results instead of a SQL error.
        """
        tokens = cls._tokenize_query(query)
        match = cls._build_fts_query(tokens)
        if not match:
            return query
        return match

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
