"""Neo4j storage for the three-layer graph: Passage -CONTAINS-> Proposition -MENTIONS-> Entity."""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
from neo4j import Driver, GraphDatabase, ManagedTransaction
from pydantic import BaseModel, Field
from ulid import ulid

from grasp.embeddings import Embedder
from grasp.tokens import TokenUsage

TYPE_MERGE_THRESHOLD = 0.7


def connect(uri: str | None = None) -> Driver:
    return GraphDatabase.driver(
        uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
    )


def ensure_schema(driver: Driver, embedding_dim: int) -> None:
    statements = [
        "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT proposition_id IF NOT EXISTS FOR (p:Proposition) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
        "CREATE VECTOR INDEX proposition_embedding IF NOT EXISTS FOR (p:Proposition) ON (p.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dim}, `vector.similarity_function`: 'cosine'}}}}",
        "CREATE FULLTEXT INDEX prop_fulltext_index IF NOT EXISTS FOR (p:Proposition) ON EACH [p.text]",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement).consume()


def ensure_passage_vector_index(driver: Driver, embedding_dim: int) -> None:
    with driver.session() as session:
        session.run(
            "CREATE VECTOR INDEX passage_embedding IF NOT EXISTS FOR (p:Passage) ON (p.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dim}, `vector.similarity_function`: 'cosine'}}}}"
        ).consume()


def passage_exists(driver: Driver, text: str) -> bool:
    with driver.session() as session:
        record = session.run("MATCH (p:Passage) WHERE p.text = $text RETURN count(p) AS count", text=text.strip()).single()
        return record["count"] > 0


def question_graph_exists(driver: Driver, question_id: str) -> bool:
    with driver.session() as session:
        record = session.run(
            "MATCH (n) WHERE $qid IN coalesce(n.question_ids, []) RETURN count(n) > 0 AS exists",
            qid=question_id,
        ).single()
        return bool(record["exists"]) if record is not None else False


def indexed_question(driver: Driver, question_id: str) -> tuple[TokenUsage, float] | None:
    """What a completed per-question graph cost to build, or None if it was never finished."""
    with driver.session() as session:
        record = session.run("MATCH (m:IndexedQuestion {question_id: $qid}) RETURN properties(m) AS m", qid=question_id).single()
    return (TokenUsage.from_dict(record["m"]), record["m"]["seconds"]) if record else None


def mark_indexed(driver: Driver, question_id: str, usage: TokenUsage, seconds: float) -> None:
    with driver.session() as session:
        session.run(
            "MERGE (m:IndexedQuestion {question_id: $qid}) SET m += $usage, m.seconds = $seconds",
            qid=question_id,
            usage=usage.to_dict(),
            seconds=seconds,
        ).consume()


def delete_question_graph(driver: Driver, question_id: str) -> None:
    with driver.session() as session:
        for match in ("MATCH (n {question_id: $qid}) RETURN n", "MATCH (n) WHERE $qid IN coalesce(n.question_ids, []) RETURN n"):
            session.run(
                "CALL apoc.periodic.iterate($match, 'DETACH DELETE n', {batchSize: 10000, parallel: false, params: {qid: $qid}})",
                match=match,
                qid=question_id,
            ).consume()


def clear_database(driver: Driver) -> None:
    with driver.session() as session:
        session.run("CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DETACH DELETE n', {batchSize: 10000, parallel: false})").consume()


def proposition_id(text: str, question_id: str | None) -> str:
    key = text.strip().lower()
    if question_id:
        key = f"{key}::{question_id}"
    return hashlib.md5(key.encode()).hexdigest()


def normalize_entity_name(name: str) -> str:
    words = []
    for word in str(name).strip().split():
        core = word.strip(".,;:'\"()-")
        if core.isupper() and 1 <= len(core) <= 5:
            words.append(word)
        else:
            words.append(word[0].upper() + word[1:].lower() if word else word)
    return " ".join(words)


def type_label(entity_type: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", entity_type.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_") or "Unknown"
    return f"`{sanitized}`"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0.0 else 0.0


class Extraction(BaseModel):
    propositions: list[str]
    entities: list[list[tuple[str, str]]] = Field(default_factory=list)


class GraphWriter:
    """Writes one passage with its propositions and typed entities.

    Entities sharing a normalized name merge when their type embeddings are similar.
    """

    def __init__(self, driver: Driver, embedder: Embedder) -> None:
        self.driver = driver
        self.embedder = embedder
        self._type_vectors: dict[str, list[float]] = {}

    def write(self, text: str, index: int, extraction: Extraction, question_id: str | None) -> None:
        new_propositions: list[tuple[str, str]] = []

        def work(tx: ManagedTransaction) -> None:
            new_propositions.clear()
            self.write_passage(tx, text, index, extraction, question_id, new_propositions)

        with self.driver.session() as session:
            session.execute_write(work)
        if new_propositions:
            vectors = self.embedder.documents([text for _, text in new_propositions])
            with self.driver.session() as session:
                for (prop_id, _), vector in zip(new_propositions, vectors):
                    session.run("MATCH (p:Proposition {id: $prop_id}) SET p.embedding = $embedding", prop_id=prop_id, embedding=vector)

    def write_passage(
        self,
        tx: ManagedTransaction,
        text: str,
        index: int,
        extraction: Extraction,
        question_id: str | None,
        new_propositions: list[tuple[str, str]],
    ) -> None:
        passage_id = ulid()
        if question_id:
            tx.run(
                "MERGE (p:Passage {id: $passage_id}) SET p.text = $text, p.index = $index, p.question_ids = [$question_id]",
                passage_id=passage_id,
                text=text,
                index=index,
                question_id=question_id,
            ).consume()
        else:
            tx.run(
                "MERGE (p:Passage {id: $passage_id}) SET p.text = $text, p.index = $index",
                passage_id=passage_id,
                text=text,
                index=index,
            ).consume()

        previous: str | None = None
        for prop_text, entities in zip(extraction.propositions, extraction.entities):
            prop_id = proposition_id(prop_text, question_id)
            scoped = ", prop.question_ids = [$question_id]" if question_id else ""
            created = tx.run(
                f"""
                MATCH (passage:Passage {{id: $passage_id}})
                MERGE (prop:Proposition {{id: $prop_id}})
                ON CREATE SET prop.text = $text{scoped}, prop.__new = true
                MERGE (passage)-[:CONTAINS]->(prop)
                WITH prop WHERE prop.__new = true
                REMOVE prop.__new
                RETURN prop.id AS id, prop.text AS text
                """,
                passage_id=passage_id,
                prop_id=prop_id,
                text=prop_text,
                question_id=question_id,
            )
            new_propositions.extend((record["id"], record["text"]) for record in created)
            if previous is not None:
                tx.run(
                    "MATCH (prev:Proposition {id: $prev_id}) MATCH (curr:Proposition {id: $curr_id}) MERGE (prev)-[:NEXT]->(curr)",
                    prev_id=previous,
                    curr_id=prop_id,
                ).consume()
            previous = prop_id
            for name, entity_type in entities:
                self.link_entity(tx, prop_id, name, entity_type, question_id)

    def link_entity(self, tx: ManagedTransaction, prop_id: str, raw_name: str, entity_type: str, question_id: str | None) -> None:
        if not raw_name or not entity_type:
            return
        name = normalize_entity_name(raw_name)
        if not name:
            return
        existing = self.similar_entity(tx, name, entity_type, question_id)
        if existing is not None:
            tx.run(
                """
                MATCH (prop:Proposition {id: $prop_id})
                MATCH (e:Entity {entity_key: $existing_key})
                MERGE (prop)-[:MENTIONS]->(e)
                SET e.types = CASE
                    WHEN $new_type IN coalesce(e.types, [])
                    THEN coalesce(e.types, [e.type])
                    ELSE coalesce(e.types, [e.type]) + [$new_type]
                END
                """,
                prop_id=prop_id,
                existing_key=existing,
                new_type=entity_type,
            ).consume()
            tx.run(f"MATCH (e:Entity {{entity_key: $ek}}) SET e:{type_label(entity_type)}", ek=existing).consume()
            return

        key = f"{name}::{entity_type}::{question_id}" if question_id else f"{name}::{entity_type}"
        scoped = ", e.question_id = $question_id" if question_id else ""
        created = tx.run(
            f"""
            MATCH (prop:Proposition {{id: $prop_id}})
            MERGE (e:Entity {{entity_key: $entity_key}})
            ON CREATE SET e.name = $entity_name, e.type = $entity_type, e.types = [$entity_type]{scoped}, e.__new = true
            MERGE (prop)-[:MENTIONS]->(e)
            WITH e WHERE e.__new = true
            REMOVE e.__new
            RETURN e.type AS type
            """,
            prop_id=prop_id,
            entity_key=key,
            entity_name=name,
            entity_type=entity_type,
            question_id=question_id,
        )
        for record in created:
            tx.run(f"MATCH (e:Entity {{entity_key: $entity_key}}) SET e:{type_label(record['type'])}", entity_key=key).consume()

    def similar_entity(self, tx: ManagedTransaction, name: str, entity_type: str, question_id: str | None) -> str | None:
        if question_id:
            rows = tx.run(
                "MATCH (e:Entity {name: $name, question_id: $question_id}) RETURN e.entity_key AS key, e.type AS type, e.types AS types",
                name=name,
                question_id=question_id,
            )
        else:
            rows = tx.run("MATCH (e:Entity {name: $name}) RETURN e.entity_key AS key, e.type AS type, e.types AS types", name=name)
        candidates = [dict(row) for row in rows]
        if not candidates:
            return None
        new_vector = self.type_vector(entity_type)
        best_key, best_score = None, 0.0
        for candidate in candidates:
            for known_type in candidate["types"] or [candidate["type"]]:
                score = cosine_similarity(new_vector, self.type_vector(known_type))
                if score > best_score:
                    best_key, best_score = candidate["key"], score
        return best_key if best_score >= TYPE_MERGE_THRESHOLD else None

    def type_vector(self, entity_type: str) -> list[float]:
        if entity_type not in self._type_vectors:
            self._type_vectors[entity_type] = self.embedder.documents([entity_type])[0]
        return self._type_vectors[entity_type]
