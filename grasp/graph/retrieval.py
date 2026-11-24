"""Hybrid (dense + fulltext) retrieval over the proposition layer.

Propositions are scored by cosine similarity to the query plus
`lexical_weight * log(1 + fulltext score)` (Eq. 1's lambda). The QA runs use lambda = 0,
so fulltext keyword hits join the candidate pool with score 0; Table 2 used 0.2. Entities aggregate the scores of the
propositions that mention them (Eq. 1); passages aggregate them by RankVote (Eq. 2).
"""

from __future__ import annotations

from typing import Literal

from neo4j import Driver
from neo4j.exceptions import ServiceUnavailable, TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from grasp.embeddings import Embedder

EXACT_NAME_SCORE = 100.0
LUCENE_SPECIAL = r'+-&|!(){}[]^"~*?:\/'

neo4j_retry = retry(
    retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)


def escape_lucene(term: str) -> str:
    return "".join(f"\\{c}" if c in LUCENE_SPECIAL else c for c in term)


def fulltext_query(keywords: list[str]) -> str:
    return " OR ".join(escape_lucene(k) for k in keywords) if keywords else ""


def scored_propositions(hybrid: bool, question_id: str | None, lexical_weight: float = 0.0) -> str:
    """Cypher yielding `prop, score` rows for the query embedding (and keywords if hybrid)."""
    if question_id is not None:
        vector_branch = """
            MATCH (prop:Proposition)
            WHERE $question_id IN prop.question_ids AND prop.embedding IS NOT NULL
            WITH prop, vector.similarity.cosine($embedding, prop.embedding) AS vector_score
            ORDER BY vector_score DESC
            LIMIT $n_candidates"""
        scope = "\n            WHERE $question_id IN prop.question_ids"
    else:
        vector_branch = """
            CALL db.index.vector.queryNodes('proposition_embedding', $n_candidates, $embedding)
            YIELD node AS prop, score AS vector_score"""
        scope = ""

    if not hybrid:
        if question_id is not None:
            return vector_branch.replace("AS vector_score", "AS score").replace("BY vector_score", "BY score")
        return vector_branch.replace("score AS vector_score", "score") + "\n        WITH prop, score"

    if lexical_weight:
        score = f"""WITH prop, max(vector_score) AS vec_s, max(ft_score) AS ft_s
        WITH prop, coalesce(vec_s, 0.0) + log(1 + coalesce(ft_s, 0.0)) * {lexical_weight} AS score"""
    else:
        score = """WITH prop, max(vector_score) AS vec_s
        WITH prop, coalesce(vec_s, 0.0) AS score"""
    return f"""
        CALL () {{{vector_branch}
            RETURN prop, vector_score, null AS ft_score
            UNION
            CALL db.index.fulltext.queryNodes('prop_fulltext_index', $keywords, {{limit: $n_candidates}})
            YIELD node AS prop, score AS ft_score{scope}
            RETURN prop, null AS vector_score, ft_score
        }}
        {score}
        """


@neo4j_retry
def retrieve_entities(
    driver: Driver,
    embedder: Embedder,
    query: str,
    keywords: list[str],
    *,
    k: int = 5,
    n_candidates: int = 50,
    exclude_ids: list[int] | None = None,
    question_id: str | None = None,
    lexical_weight: float = 0.0,
) -> list[dict]:
    keyword_query = fulltext_query(keywords)
    name_scope = (
        "AND EXISTS { MATCH (prop2:Proposition)-[:MENTIONS]->(e) WHERE $question_id IN prop2.question_ids }"
        if question_id is not None
        else ""
    )
    cypher = f"""
    CALL {{
        {scored_propositions(bool(keyword_query), question_id, lexical_weight)}
        MATCH (prop)-[:MENTIONS]->(e:Entity)
        WHERE NOT id(e) IN $exclude_ids
        WITH e, sum(score) AS sum_score, count(prop) AS num_props
        CALL {{ WITH e MATCH (e)<-[:MENTIONS]-(anyprop) RETURN count(anyprop) AS degree }}
        WITH e, sum_score / sqrt(1.0 + degree) AS total_score, degree
        RETURN e AS entity, total_score, degree

        UNION

        UNWIND $keywords_list AS kw
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower(kw)
        {name_scope}
        CALL {{ WITH e MATCH (e)<-[:MENTIONS]-(anyprop) RETURN count(anyprop) AS degree }}
        RETURN e AS entity, {EXACT_NAME_SCORE} AS total_score, degree
    }}
    WITH entity, max(total_score) AS total_score, max(degree) AS degree
    ORDER BY total_score DESC
    LIMIT $k
    RETURN entity.name AS entity_name, entity.type AS entity_type,
           id(entity) AS entity_id, total_score, degree
    ORDER BY total_score DESC
    """
    with driver.session() as session:
        rows = session.run(
            cypher,
            embedding=embedder.query(query),
            keywords=keyword_query,
            keywords_list=keywords or [],
            n_candidates=n_candidates,
            k=k,
            exclude_ids=exclude_ids or [],
            question_id=question_id,
        )
        return [
            {
                "entity_id": int(r["entity_id"]),
                "entity_name": r["entity_name"],
                "entity_type": r["entity_type"] or "",
                "score": round(float(r["total_score"]), 4),
                "degree": int(r["degree"]),
            }
            for r in rows
        ]


RANK_VOTE = """
        WITH DISTINCT prop, score
        ORDER BY score DESC
        WITH collect(prop) AS ranked_props
        UNWIND range(0, size(ranked_props) - 1) AS k
        WITH ranked_props[k] AS prop, 1.0 / (k + 1) AS rr_weight
        MATCH (pass:Passage)-[:CONTAINS]->(prop)"""

MAJORITY_VOTE = """
        WITH DISTINCT prop, score
        MATCH (pass:Passage)-[:CONTAINS]->(prop)"""


@neo4j_retry
def retrieve_passages(
    driver: Driver,
    embedder: Embedder,
    query: str,
    keywords: list[str],
    *,
    scoring: Literal["rr", "count"] = "rr",
    n_candidates: int = 100,
    question_id: str | None = None,
    lexical_weight: float = 0.0,
) -> list[dict]:
    """Top-10 passages by RankVote (`rr`) or majority vote (`count`) over all propositions."""
    keyword_query = fulltext_query(keywords)
    if scoring == "rr":
        vote = f"""{RANK_VOTE}
        WITH pass.text AS text, sum(rr_weight) AS rr_score, count(prop) AS prop_count
        ORDER BY rr_score DESC
        LIMIT 10
        RETURN text, prop_count, rr_score AS score"""
    else:
        vote = f"""{MAJORITY_VOTE}
        WITH pass.text AS text, count(prop) AS prop_count, avg(score) AS avg_score
        ORDER BY prop_count DESC, avg_score DESC
        LIMIT 10
        RETURN text, prop_count, avg_score AS score"""
    cypher = scored_propositions(bool(keyword_query), question_id, lexical_weight) + vote
    with driver.session() as session:
        rows = session.run(
            cypher,
            embedding=embedder.query(query),
            keywords=keyword_query,
            n_candidates=n_candidates,
            question_id=question_id,
        )
        return [{"text": r["text"], "prop_count": int(r["prop_count"]), "score": round(float(r["score"]), 4)} for r in rows]
