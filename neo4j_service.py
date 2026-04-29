"""
Neo4j Graph Database Service for Enhanced Graph RAG
Handles connection, storage, and querying of knowledge graphs
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from entity_canonicalizer import canonical_id as _canonical_id

import litellm

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)


class CypherGenerationError(Exception):
    """Raised when LLM-generated Cypher cannot be validated after one retry."""


class Neo4jService:
    """Service for interacting with Neo4j graph database."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = None):
        """Initialize Neo4j connection."""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database  # None uses the default database
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                keep_alive=True
            )
            # Test connection
            session_kwargs = {"database": self.database} if self.database else {}
            with self.driver.session(**session_kwargs) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _session(self):
        """Return a driver session, targeting the configured database if set."""
        kwargs = {"database": self.database} if self.database else {}
        return self.driver.session(**kwargs)

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_graph(self) -> bool:
        """Clear all nodes and relationships from the graph."""
        try:
            with self._session() as session:
                # Delete all relationships first, then nodes
                session.run("MATCH ()-[r]-() DELETE r")
                session.run("MATCH (n) DELETE n")
                logger.info("Graph cleared successfully")
                return True
        except Neo4jError as e:
            logger.error(f"Error clearing graph: {e}")
            return False
    
    def create_indexes(self):
        """Create necessary indexes for performance."""
        indexes = [
            "CREATE INDEX entity_id_idx           IF NOT EXISTS FOR (e:Entity)   ON (e.id)",
            "CREATE INDEX entity_name_idx         IF NOT EXISTS FOR (e:Entity)   ON (e.name)",
            "CREATE INDEX entity_type_idx         IF NOT EXISTS FOR (e:Entity)   ON (e.type)",
            "CREATE INDEX entity_corpus_idx       IF NOT EXISTS FOR (e:Entity)   ON (e.benchmark_corpus)",
            "CREATE INDEX document_name_idx       IF NOT EXISTS FOR (d:Document) ON (d.name)",
            "CREATE INDEX document_corpus_idx     IF NOT EXISTS FOR (d:Document) ON (d.benchmark_corpus)",
        ]
        
        try:
            with self._session() as session:
                for index in indexes:
                    session.run(index)
            logger.info("Indexes created successfully")
        except Neo4jError as e:
            logger.error(f"Error creating indexes: {e}")
    
    _BATCH_SIZE = 500  # UNWIND batch size — safe for AuraDB free tier memory

    def store_entities_and_relationships(self,
                                         entities: List[Dict[str, Any]],
                                         relationships: List[Dict[str, Any]],
                                         document_name: str = None,
                                         benchmark_corpus: str = "",
                                         doc_metadata: Dict[str, Any] = None) -> Dict[str, int]:
        """
        Store entities and relationships using UNWIND batches.
        1 network round-trip per 500 entities instead of 1 per entity — ~100x faster on cloud Neo4j.

        doc_metadata — optional dict with keys: title, source, category, published_at, url.
        benchmark_corpus — tag stored on all Entity nodes for namespace isolation.
        """
        stats = {"entities_created": 0, "relationships_created": 0, "errors": 0}
        ts = datetime.now().isoformat()
        doc_metadata = doc_metadata or {}

        try:
            with self._session() as session:
                # ── Document node (single call) with full metadata ────────────
                if document_name:
                    session.run(
                        """
                        MERGE (d:Document {name: $doc_name})
                        ON CREATE SET d.created_at  = $ts,
                                      d.title        = $title,
                                      d.source       = $source,
                                      d.category     = $category,
                                      d.published_at = $published_at,
                                      d.url          = $url,
                                      d.benchmark_corpus = $benchmark_corpus
                        ON MATCH  SET d.updated_at  = $ts,
                                      d.title        = $title,
                                      d.source       = $source,
                                      d.category     = $category,
                                      d.published_at = $published_at,
                                      d.url          = $url,
                                      d.benchmark_corpus = $benchmark_corpus
                        """,
                        doc_name=document_name,
                        ts=ts,
                        title=doc_metadata.get("title", document_name),
                        source=doc_metadata.get("source", ""),
                        category=doc_metadata.get("category", ""),
                        published_at=doc_metadata.get("published_at", ""),
                        url=doc_metadata.get("url", ""),
                        benchmark_corpus=benchmark_corpus,
                    )

                # ── Entities — UNWIND in batches ─────────────────────────────
                entity_rows = []
                for e in entities:
                    eid = e.get("id") or _canonical_id(
                        e.get("name", ""), e.get("type", "UNKNOWN")
                    )
                    entity_rows.append({
                        "id":              eid,
                        "name":            e.get("name", ""),
                        "type":            e.get("type", "UNKNOWN"),
                        "aliases":         json.dumps(e.get("aliases", [e.get("name", "")])),
                        "description":     e.get("description", ""),
                        "source_chunks":   json.dumps(e.get("source_chunks", [])),
                        "source_texts":    json.dumps(e.get("source_texts", [])),
                        "merged_from":     e.get("merged_from", 1),
                        "benchmark_corpus": benchmark_corpus,
                        "ts":              ts,
                        "doc_name":        document_name or "",
                    })

                for i in range(0, len(entity_rows), self._BATCH_SIZE):
                    batch = entity_rows[i:i + self._BATCH_SIZE]
                    try:
                        result = session.run(
                            """
                            UNWIND $rows AS row
                            MERGE (e:Entity {id: row.id})
                            SET e.name             = row.name,
                                e.type             = row.type,
                                e.aliases          = row.aliases,
                                e.description      = row.description,
                                e.source_chunks    = row.source_chunks,
                                e.source_texts     = row.source_texts,
                                e.merged_from      = row.merged_from,
                                e.benchmark_corpus = row.benchmark_corpus,
                                e.updated_at       = row.ts
                            WITH e, row
                            WHERE row.doc_name <> ''
                            MATCH (d:Document {name: row.doc_name})
                            MERGE (e)-[:MENTIONED_IN]->(d)
                            """,
                            rows=batch,
                        )
                        result.consume()
                        stats["entities_created"] += len(batch)
                    except Exception as ex:
                        logger.error(f"Entity batch {i//self._BATCH_SIZE + 1} failed: {ex}")
                        stats["errors"] += len(batch)

                # ── Relationships — UNWIND in batches ────────────────────────
                # Build name→id lookup from the entities we just stored so we can
                # resolve relationship endpoints even when source_type is unknown.
                name_to_id = {e.get("name", ""): e.get("id") or _canonical_id(e.get("name",""), e.get("type","UNKNOWN"))
                              for e in entities if e.get("name")}

                rel_rows = []
                for r in relationships:
                    source = r.get("source", "")
                    target = r.get("target", "")
                    if not source or not target:
                        continue
                    edge_kind = r.get("relationship", "")
                    rel_rows.append({
                        "source_id":        name_to_id.get(source) or _canonical_id(source, r.get("source_type", "UNKNOWN")),
                        "target_id":        name_to_id.get(target) or _canonical_id(target, r.get("target_type", "UNKNOWN")),
                        "source_name":      source,
                        "target_name":      target,
                        "relationship":     edge_kind,
                        "edge_kind":        edge_kind,
                        "description":      r.get("description", ""),
                        "source_chunk":     r.get("source_chunk", ""),
                        "source_text":      r.get("source_text", ""),
                        "provenance":       r.get("provenance", "semantic_llm"),
                        "benchmark_corpus": benchmark_corpus,
                        "document_name":    document_name or "",
                        "ts":               ts,
                    })

                for i in range(0, len(rel_rows), self._BATCH_SIZE):
                    batch = rel_rows[i:i + self._BATCH_SIZE]
                    try:
                        result = session.run(
                            """
                            UNWIND $rows AS row
                            MATCH (s:Entity {id: row.source_id})
                            MATCH (t:Entity {id: row.target_id})
                            MERGE (s)-[r:RELATES_TO {relationship: row.relationship,
                                                      provenance:   row.provenance}]->(t)
                            SET r.edge_kind        = row.edge_kind,
                                r.description      = row.description,
                                r.source_chunk     = row.source_chunk,
                                r.source_text      = row.source_text,
                                r.source_name      = row.source_name,
                                r.target_name      = row.target_name,
                                r.benchmark_corpus = row.benchmark_corpus,
                                r.document_name    = row.document_name,
                                r.updated_at       = row.ts
                            """,
                            rows=batch,
                        )
                        result.consume()
                        stats["relationships_created"] += len(batch)
                    except Exception as ex:
                        logger.error(f"Relationship batch {i//self._BATCH_SIZE + 1} failed: {ex}")
                        stats["errors"] += len(batch)

        except Neo4jError as e:
            logger.error(f"Database error during storage: {e}")
            stats["errors"] += 1

        logger.info(
            f"Neo4j storage complete: {stats['entities_created']} entities, "
            f"{stats['relationships_created']} relationships, {stats['errors']} errors"
        )
        return stats

    def _store_entity(self, session, entity: Dict[str, Any], document_name: str = None):
        """Single-entity write — kept for compatibility. Prefer store_entities_and_relationships for bulk."""
        eid = entity.get("id") or _canonical_id(
            entity.get("name", ""), entity.get("type", "UNKNOWN")
        )
        ts = datetime.now().isoformat()
        session.run(
            """
            MERGE (e:Entity {id: $id})
            SET e.name=$name, e.type=$type, e.aliases=$aliases,
                e.description=$description, e.source_chunks=$sc,
                e.source_texts=$st, e.merged_from=$mf, e.updated_at=$ts
            """,
            id=eid, name=entity.get("name",""), type=entity.get("type","UNKNOWN"),
            aliases=json.dumps(entity.get("aliases",[entity.get("name","")])),
            description=entity.get("description",""),
            sc=json.dumps(entity.get("source_chunks",[])),
            st=json.dumps(entity.get("source_texts",[])),
            mf=entity.get("merged_from",1), ts=ts,
        )
        if document_name:
            session.run(
                "MATCH (e:Entity {id:$eid}),(d:Document {name:$dn}) MERGE (e)-[:MENTIONED_IN]->(d)",
                eid=eid, dn=document_name,
            )

    def _store_relationship(self, session, relationship: Dict[str, Any], document_name: str = None):
        """Single-relationship write — kept for compatibility."""
        source = relationship.get("source", "")
        target = relationship.get("target", "")
        if not source or not target:
            return
        session.run(
            """
            MATCH (s:Entity {id: $sid})
            MATCH (t:Entity {id: $tid})
            MERGE (s)-[r:RELATES_TO {relationship: $rel}]->(t)
            SET r.description=$desc, r.source_chunk=$sc, r.source_text=$st, r.updated_at=$ts
            """,
            sid=_canonical_id(source, relationship.get("source_type","UNKNOWN")),
            tid=_canonical_id(target, relationship.get("target_type","UNKNOWN")),
            rel=relationship.get("relationship",""),
            desc=relationship.get("description",""),
            sc=relationship.get("source_chunk",""),
            st=relationship.get("source_text",""),
            ts=datetime.now().isoformat(),
        )
    
    def _validate_cypher(self, cypher: str) -> Tuple[bool, Optional[str]]:
        """Run EXPLAIN on cypher in a read-only tx. Returns (valid, error_or_None)."""
        try:
            with self._session() as session:
                session.run(f"EXPLAIN {cypher}").consume()
            return True, None
        except Neo4jError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _call_cypher_llm(self, prompt: str) -> str:
        """Single LLM call to gemini-2.5-flash-lite; returns stripped Cypher text."""
        resp = litellm.completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        cypher = resp.choices[0].message.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        return cypher.splitlines()[0].strip()

    def generate_cypher_from_question(self, question: str, llm_service=None) -> Tuple[str, str]:
        """Generate a Cypher query from a natural-language question.

        Uses gemini/gemini-2.0-flash directly via LiteLLM — a cheap, non-thinking
        model that reliably outputs a single short Cypher statement without
        truncation or verbose explanation.
        """
        schema_info = self.get_schema_info()

        # Strict single-task prompt — model must output ONLY the Cypher query.
        prompt = (
            "You are a Neo4j Cypher query generator.\n"
            "Your ONLY job is to output a single valid Cypher query. "
            "Do NOT include any explanation, markdown fences, comments, or extra text. "
            "Output the Cypher query on one line and nothing else.\n\n"
            "DATABASE SCHEMA:\n"
            f"{schema_info}\n\n"
            "RULES:\n"
            "- Use toLower() + CONTAINS for all string comparisons (case-insensitive).\n"
            "- Use MATCH … WHERE … RETURN … LIMIT 20.\n"
            "- For relationship queries use: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity).\n"
            "- Always RETURN relevant node/relationship properties.\n\n"
            "EXAMPLES:\n"
            "Question: Who founded TechCorp?\n"
            "Query: MATCH (e:Entity)-[r:RELATES_TO]->(e2:Entity) WHERE toLower(r.relationship) CONTAINS 'found' OR toLower(r.relationship) CONTAINS 'co-found' OR toLower(r.relationship) CONTAINS 'ceo' RETURN e.name, e.type, e.description, r.relationship, e2.name LIMIT 20\n\n"
            "Question: What are TechCorp's revenue figures?\n"
            "Query: MATCH (e:Entity) WHERE toLower(e.description) CONTAINS 'revenue' OR toLower(e.description) CONTAINS 'million' RETURN e.name, e.type, e.description LIMIT 20\n\n"
            "Question: What partnerships does TechCorp have?\n"
            "Query: MATCH (e:Entity)-[r:RELATES_TO]->(e2:Entity) WHERE toLower(r.relationship) CONTAINS 'partner' OR toLower(r.description) CONTAINS 'partner' RETURN e.name, r.relationship, r.description, e2.name LIMIT 20\n\n"
            "Question: What are the risk factors?\n"
            "Query: MATCH (e:Entity) WHERE toLower(e.description) CONTAINS 'risk' OR toLower(e.description) CONTAINS 'threat' OR toLower(e.description) CONTAINS 'challenge' RETURN e.name, e.type, e.description LIMIT 20\n\n"
            "Question: What products or technologies does TechCorp have?\n"
            "Query: MATCH (e:Entity) WHERE e.type IN ['PRODUCT', 'TECHNOLOGY'] OR toLower(e.description) CONTAINS 'product' OR toLower(e.description) CONTAINS 'platform' RETURN e.name, e.type, e.description LIMIT 20\n\n"
            f"Question: {question}\n"
            "Query:"
        )

        try:
            cypher_query = self._call_cypher_llm(prompt)

            if not any(kw in cypher_query.upper() for kw in ("MATCH", "RETURN")):
                logger.warning(f"LLM returned unexpected Cypher output: {cypher_query!r}")
                return "", f"LLM did not return a valid Cypher query: {cypher_query!r}"

            # Validate with EXPLAIN before executing
            valid, err = self._validate_cypher(cypher_query)
            if not valid:
                logger.warning(f"Cypher validation failed ({err}); attempting self-correction")
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"The query you previously generated failed validation with error:\n{err}\n"
                    "Return ONLY a corrected Cypher query, no prose."
                )
                cypher_query = self._call_cypher_llm(retry_prompt)
                valid2, err2 = self._validate_cypher(cypher_query)
                if not valid2:
                    raise CypherGenerationError(
                        f"Cypher generation failed after retry. Last error: {err2}"
                    )

            logger.info(f"LLM-generated Cypher (validated): {cypher_query}")
            return cypher_query, ""

        except CypherGenerationError:
            raise
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            return "", f"Error generating query: {str(e)}"
    
    def execute_cypher_query(self, cypher_query: str) -> Tuple[List[Dict], str]:
        """Execute Cypher query and return results - FIXED."""
        try:
            with self._session() as session:
                result = session.run(cypher_query)
                records = []
                
                # Process all records
                for record in result:
                    record_dict = {}
                    
                    # Convert Neo4j Record to dictionary
                    for key in record.keys():
                        value = record[key]
                        
                        # Handle different Neo4j types
                        if value is None:
                            record_dict[key] = None
                        elif isinstance(value, (str, int, float, bool)):
                            # Primitive types
                            record_dict[key] = value
                        elif isinstance(value, list):
                            # Handle lists (like collected relationships)
                            list_values = []
                            for item in value:
                                if hasattr(item, '__dict__'):
                                    # Neo4j Node or Relationship
                                    list_values.append(dict(item))
                                else:
                                    list_values.append(item)
                            record_dict[key] = list_values
                        elif hasattr(value, '__dict__'):
                            # Neo4j Node or Relationship object
                            record_dict[key] = dict(value)
                        else:
                            # Fallback to string representation
                            record_dict[key] = str(value)
                    
                    records.append(record_dict)
                
                # Log the results for debugging
                if records:
                    logger.info(f"Query returned {len(records)} records")
                    logger.debug(f"First record keys: {list(records[0].keys()) if records else 'No records'}")
                else:
                    logger.info("Query returned 0 records")
                    
                return records, ""
                
        except Neo4jError as e:
            error_msg = f"Cypher query error: {str(e)}"
            logger.error(f"{error_msg}\nQuery was: {cypher_query}")
            return [], error_msg
        except Exception as e:
            error_msg = f"Unexpected error executing query: {str(e)}"
            logger.error(f"{error_msg}\nQuery was: {cypher_query}")
            return [], error_msg
    
    def get_schema_info(self) -> str:
        """Get database schema information for LLM context."""
        try:
            with self._session() as session:
                # Get node labels and their properties
                node_info = session.run("""
                    CALL db.labels() YIELD label
                    CALL db.propertyKeys() YIELD propertyKey
                    RETURN collect(DISTINCT label) as labels, 
                           collect(DISTINCT propertyKey) as properties
                """).single()
                
                # Get relationship types
                rel_info = session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    RETURN collect(relationshipType) as relationship_types
                """).single()
                
                # Sample some data to understand structure
                sample_data = session.run("""
                    MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
                    RETURN n.name as source_name, n.type as source_type,
                           r.relationship as rel_type,
                           m.name as target_name, m.type as target_type
                    LIMIT 5
                """).data()
                
                schema_text = f"""
NODE LABELS: {', '.join(node_info['labels'])}
RELATIONSHIP TYPES: {', '.join(rel_info['relationship_types'])}
PROPERTIES: {', '.join(node_info['properties'])}

SAMPLE DATA STRUCTURE:
"""
                for sample in sample_data:
                    schema_text += f"({sample['source_name']}:{sample['source_type']})-[:{sample['rel_type']}]->({sample['target_name']}:{sample['target_type']})\n"
                
                return schema_text
                
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return "Schema information unavailable"
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the graph.

        Uses separate aggregations — the old pattern
        MATCH (n:Entity) OPTIONAL MATCH ()-[r]->() RETURN count(n), count(r)
        is a Cartesian product that inflates relationship_count by node_count.
        """
        try:
            with self._session() as session:
                node_count = session.run(
                    "MATCH (n:Entity) RETURN count(n) AS c"
                ).single()["c"]

                rel_row = session.run(
                    """
                    MATCH ()-[r:RELATES_TO]->()
                    RETURN count(r) AS total,
                           sum(CASE WHEN r.provenance = 'co_occurrence' THEN 1 ELSE 0 END) AS co_occ,
                           sum(CASE WHEN coalesce(r.provenance, 'semantic_llm') <> 'co_occurrence'
                                    THEN 1 ELSE 0 END) AS semantic
                    """
                ).single()

                return {
                    "nodes":                   node_count,
                    "relationships":           rel_row["total"],
                    "semantic_relationships":  rel_row["semantic"],
                    "cooccurrence_edges":      rel_row["co_occ"],
                }
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"nodes": 0, "relationships": 0, "semantic_relationships": 0, "cooccurrence_edges": 0}
    
    def search_entities_by_name(self, name_query: str, limit: int = 10) -> List[Dict]:
        """Search entities by name similarity."""
        try:
            with self._session() as session:
                results = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($query)
                       OR toLower(e.description) CONTAINS toLower($query)
                    RETURN e.name as name, e.type as type, e.description as description
                    LIMIT $limit
                """, query=name_query, limit=limit)

                return [dict(record) for record in results]

        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []

    def _execute_parameterized(self, cypher: str, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        """Execute a parameterized Cypher query and return (records, error_or_empty)."""
        try:
            with self._session() as session:
                result = session.run(cypher, **params)
                records = []
                for record in result:
                    row = {}
                    for key in record.keys():
                        value = record[key]
                        if value is None:
                            row[key] = None
                        elif isinstance(value, (str, int, float, bool)):
                            row[key] = value
                        elif isinstance(value, list):
                            row[key] = [dict(item) if hasattr(item, "__dict__") else item for item in value]
                        elif hasattr(value, "__dict__"):
                            row[key] = dict(value)
                        else:
                            row[key] = str(value)
                    records.append(row)
                return records, ""
        except Neo4jError as e:
            return [], str(e)
        except Exception as e:
            return [], str(e)

    def traverse_neighbors(
        self,
        seed_names: List[str],
        depth: int = 2,
        limit: int = 30,
        benchmark_corpus: str = "",
    ) -> List[Dict[str, Any]]:
        """Traverse entity neighbors from seed entity names using Cypher.

        Depth 1: direct RELATES_TO neighbors.
        Depth 2: also includes second-hop neighbors (hop-1 names used as new seeds).

        benchmark_corpus — when non-empty, filters seed and neighbor entities AND
                           relationships to that corpus tag, preventing cross-benchmark
                           contamination (e.g. AMD/Boeing data leaking into multihop results).

        Returns list of dicts with keys:
            seed_name, seed_type, relationship, provenance, rel_benchmark_corpus,
            rel_description, source_text, neighbor_name, neighbor_type,
            neighbor_description, neighbor_source_texts
        """
        if not seed_names:
            return []

        # $benchmark_corpus = '' disables filtering (WHERE clause becomes always-true)
        cypher_hop = """
        MATCH (seed:Entity)
        WHERE seed.name IN $seed_names
          AND ($benchmark_corpus = '' OR seed.benchmark_corpus = $benchmark_corpus)
        MATCH (seed)-[r:RELATES_TO]-(neighbor:Entity)
        WHERE neighbor.name <> seed.name
          AND ($benchmark_corpus = '' OR neighbor.benchmark_corpus = $benchmark_corpus)
        RETURN seed.name              AS seed_name,
               seed.type              AS seed_type,
               r.relationship         AS relationship,
               r.provenance           AS provenance,
               r.benchmark_corpus     AS rel_benchmark_corpus,
               r.description          AS rel_description,
               r.source_text          AS source_text,
               neighbor.name          AS neighbor_name,
               neighbor.type          AS neighbor_type,
               neighbor.description   AS neighbor_description,
               neighbor.source_texts  AS neighbor_source_texts
        LIMIT $limit
        """

        params = {"seed_names": seed_names, "limit": limit, "benchmark_corpus": benchmark_corpus}
        rows, err = self._execute_parameterized(cypher_hop, params)
        if err:
            logger.warning(f"Neo4j traverse_neighbors depth-1 error: {err}")
            return []

        if depth >= 2 and rows:
            hop1_names = list({r["neighbor_name"] for r in rows if r.get("neighbor_name")})
            remaining = max(0, limit - len(rows))
            if hop1_names and remaining > 0:
                rows2, err2 = self._execute_parameterized(
                    cypher_hop,
                    {"seed_names": hop1_names, "limit": remaining, "benchmark_corpus": benchmark_corpus},
                )
                if not err2:
                    rows = rows + rows2

        logger.debug(f"traverse_neighbors: {len(seed_names)} seeds → {len(rows)} triplets (depth={depth})")
        return rows

# Factory function for easy initialization
def create_neo4j_service(config: Dict[str, str]) -> Optional[Neo4jService]:
    """Create Neo4j service from configuration."""
    try:
        service = Neo4jService(
            uri=config.get("NEO4J_URI", config.get("neo4j_uri", "")),
            username=config.get("NEO4J_USERNAME", config.get("neo4j_username", "neo4j")),
            password=config.get("NEO4J_PASSWORD", config.get("neo4j_password", "")),
            database=config.get("NEO4J_DATABASE", config.get("neo4j_database", None))
        )
        service.create_indexes()
        return service
    except Exception as e:
        logger.error(f"Failed to create Neo4j service: {e}")
        return None