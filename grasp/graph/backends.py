"""Graph backends the sub-agent traverses: the full GRASP graph and its Figure 5 ablations."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
from neo4j import Driver
from pydantic import BaseModel, ConfigDict

from grasp.embeddings import Embedder
from grasp.graph.retrieval import RANK_VOTE, fulltext_query, retrieve_entities, scored_propositions

EVIDENCE_TOP_K = 5

GraphKind = Literal["proposition_passage", "flat_proposition_passage", "dense_passage", "proposition"]


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: int
    entity_name: str
    entity_type: str
    score: float
    degree: int


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    text: str


SKIP_SELECTION = Candidate(entity_id=-1, entity_name="(flat retrieval)", entity_type="", score=1.0, degree=0)


class GraphBackend(Protocol):
    def candidates(self, query: str, keywords: list[str], *, k: int, exclude_ids: list[int]) -> list[Candidate]: ...

    def evidence(self, node_ids: list[int], query: str, keywords: list[str], *, exclude_ids: list[int]) -> list[Evidence]: ...


class Neo4jBackend:
    def __init__(self, driver: Driver, embedder: Embedder, question_id: str | None = None, lexical_weight: float = 0.0) -> None:
        self.driver = driver
        self.embedder = embedder
        self.question_id = question_id
        self.lexical_weight = lexical_weight

    def candidates(self, query: str, keywords: list[str], *, k: int, exclude_ids: list[int]) -> list[Candidate]:
        rows = retrieve_entities(
            self.driver,
            self.embedder,
            query,
            keywords,
            k=k,
            exclude_ids=exclude_ids,
            question_id=self.question_id,
            lexical_weight=self.lexical_weight,
        )
        return [Candidate(**row) for row in rows]

    def fetch_passages(self, cypher: str, **params: Any) -> list[Evidence]:
        with self.driver.session() as session:
            rows = session.run(cypher, question_id=self.question_id, top_k=EVIDENCE_TOP_K, **params)
            return [Evidence(id=int(r["passage_id"]), text=r["text"] or "") for r in rows]

    def rank_voted_passages(self, query: str, keywords: list[str], entity_filter: str, n_candidates: int, **params: Any) -> list[Evidence]:
        keyword_query = fulltext_query(keywords)
        cypher = f"""
        {scored_propositions(bool(keyword_query), self.question_id, self.lexical_weight)}{entity_filter}{RANK_VOTE}
        WHERE NOT id(pass) IN $exclude_ids
        WITH pass, sum(rr_weight) AS rr_score, count(prop) AS prop_count
        ORDER BY rr_score DESC
        LIMIT $top_k
        RETURN id(pass) AS passage_id, pass.text AS text, prop_count, rr_score AS score
        """
        return self.fetch_passages(
            cypher, embedding=self.embedder.query(query), keywords=keyword_query, n_candidates=n_candidates, **params
        )


class GRASPBackend(Neo4jBackend):
    """GRASP: entities from hybrid proposition search; passages by RankVote over their propositions."""

    def evidence(self, node_ids: list[int], query: str, keywords: list[str], *, exclude_ids: list[int]) -> list[Evidence]:
        entity_filter = """
        MATCH (prop)-[:MENTIONS]->(e:Entity)
        WHERE id(e) IN $node_ids"""
        return self.rank_voted_passages(query, keywords, entity_filter, 50, node_ids=node_ids, exclude_ids=exclude_ids)


class FlatPropositionPassageBackend(Neo4jBackend):
    """Ablation "No entities": RankVote over all propositions, no entity selection."""

    def candidates(self, query: str, keywords: list[str], *, k: int, exclude_ids: list[int]) -> list[Candidate]:
        return [SKIP_SELECTION]

    def evidence(self, node_ids: list[int], query: str, keywords: list[str], *, exclude_ids: list[int]) -> list[Evidence]:
        return self.rank_voted_passages(query, keywords, "", 100, exclude_ids=exclude_ids)


class DensePassageBackend(Neo4jBackend):
    """Ablation "DPR (no graph)": cosine search over passage embeddings."""

    def candidates(self, query: str, keywords: list[str], *, k: int, exclude_ids: list[int]) -> list[Candidate]:
        return [SKIP_SELECTION]

    def evidence(self, node_ids: list[int], query: str, keywords: list[str], *, exclude_ids: list[int]) -> list[Evidence]:
        if self.question_id is not None:
            cypher = """
            MATCH (p:Passage)
            WHERE $question_id IN p.question_ids
              AND p.embedding IS NOT NULL
              AND NOT id(p) IN $exclude_ids
            WITH p, vector.similarity.cosine($embedding, p.embedding) AS score
            ORDER BY score DESC
            LIMIT $top_k
            RETURN id(p) AS passage_id, p.text AS text
            """
        else:
            cypher = """
            CALL db.index.vector.queryNodes('passage_embedding', $n_candidates, $embedding)
            YIELD node AS p, score
            WHERE NOT id(p) IN $exclude_ids
            WITH p, score
            ORDER BY score DESC
            LIMIT $top_k
            RETURN id(p) AS passage_id, p.text AS text
            """
        return self.fetch_passages(cypher, embedding=self.embedder.query(query), exclude_ids=exclude_ids, n_candidates=EVIDENCE_TOP_K * 2)


class PropositionBackend(Neo4jBackend):
    """Ablation "Propositions": return the selected entities' propositions, trimmed by adaptive-k."""

    def evidence(self, node_ids: list[int], query: str, keywords: list[str], *, exclude_ids: list[int]) -> list[Evidence]:
        seen = set(exclude_ids)
        found: list[Evidence] = []
        for node_id in node_ids:
            found.extend(self.entity_propositions(node_id, query, seen))
        return found

    def entity_propositions(self, entity_id: int, query: str, seen: set[int]) -> list[Evidence]:
        scope = "AND $question_id IN root.question_ids" if self.question_id is not None else ""
        cypher = f"""
            MATCH (e:Entity)
            WHERE id(e) = $entity_id

            MATCH (root:Proposition)-[:MENTIONS]->(e)
            WHERE NOT id(root) IN $exclude_ids
            {scope}

            CALL {{
                WITH root
                MATCH path = (root)-[:NEXT*0..10]-(p:Proposition)
                WHERE NOT id(p) IN $exclude_ids
                WITH p, min(length(path)) AS dist_to_root
                ORDER BY dist_to_root ASC
                LIMIT 3
                RETURN p
            }}

            RETURN DISTINCT
                id(p) AS id,
                p.text AS text,
                p.embedding AS embedding,
                e.name AS entity_name
            """
        with self.driver.session() as session:
            rows = session.run(cypher, entity_id=entity_id, exclude_ids=list(seen), question_id=self.question_id)
            props = [{"id": r["id"], "text": r["text"] or "", "embedding": r["embedding"]} for r in rows]
        if not props:
            return []
        if query and len(props) > 1 and all(p["embedding"] is not None for p in props):
            query_vector = np.array(self.embedder.query(query))
            matrix = np.array([list(p["embedding"]) for p in props])
            similarities = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)) @ (query_vector / np.linalg.norm(query_vector))
            ranked = np.argsort(-similarities)
            keep = set(ranked[: adaptive_k(similarities[ranked])].tolist())
            props = [p for i, p in enumerate(props) if i in keep]
        seen.update(int(p["id"]) for p in props)
        return [Evidence(id=int(p["id"]), text=p["text"]) for p in props]


def adaptive_k(scores: np.ndarray, min_k: int = 5, margin: float = 0.015) -> int:
    """Largest significant score gap, never below `min_k` or the near-tie margin."""
    n = len(scores)
    if n <= min_k:
        return n
    margin_k = int(np.sum(scores >= scores[0] - margin))
    gaps = np.abs(np.diff(scores))
    largest = int(np.argmax(gaps))
    gap_k = largest + 1 if gaps[largest] > gaps.mean() + gaps.std() else n
    return min(max(min_k, margin_k, gap_k), n)


BACKENDS: dict[str, type[Neo4jBackend]] = {
    "proposition_passage": GRASPBackend,
    "flat_proposition_passage": FlatPropositionPassageBackend,
    "dense_passage": DensePassageBackend,
    "proposition": PropositionBackend,
}
