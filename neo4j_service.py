"""
Neo4j Graph Database Service for Enhanced Graph RAG
Handles connection, storage, and querying of knowledge graphs
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

import litellm

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)

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
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_name_idx IF NOT EXISTS FOR (d:Document) ON (d.name)",
        ]
        
        try:
            with self._session() as session:
                for index in indexes:
                    session.run(index)
            logger.info("Indexes created successfully")
        except Neo4jError as e:
            logger.error(f"Error creating indexes: {e}")
    
    def store_entities_and_relationships(self, 
                                       entities: List[Dict[str, Any]], 
                                       relationships: List[Dict[str, Any]],
                                       document_name: str = None) -> Dict[str, int]:
        """Store entities and relationships in Neo4j."""
        stats = {"entities_created": 0, "relationships_created": 0, "errors": 0}
        
        try:
            with self._session() as session:
                # Create document node if provided
                if document_name:
                    session.run("""
                        MERGE (d:Document {name: $doc_name})
                        ON CREATE SET d.created_at = $timestamp
                        ON MATCH SET d.updated_at = $timestamp
                        """, doc_name=document_name, timestamp=datetime.now().isoformat())
                
                # Store entities
                for entity in entities:
                    try:
                        self._store_entity(session, entity, document_name)
                        stats["entities_created"] += 1
                    except Exception as e:
                        logger.error(f"Error storing entity {entity.get('name', 'unknown')}: {e}")
                        stats["errors"] += 1
                
                # Store relationships
                for rel in relationships:
                    try:
                        self._store_relationship(session, rel, document_name)
                        stats["relationships_created"] += 1
                    except Exception as e:
                        logger.error(f"Error storing relationship {rel.get('source', '')} -> {rel.get('target', '')}: {e}")
                        stats["errors"] += 1
                        
        except Neo4jError as e:
            logger.error(f"Database error during storage: {e}")
            stats["errors"] += 1
        
        logger.info(f"Storage complete: {stats}")
        return stats
    
    def _store_entity(self, session, entity: Dict[str, Any], document_name: str = None):
        """Store a single entity in Neo4j, MERGing on stable canonical id."""
        from entity_canonicalizer import canonical_id as _canonical_id
        entity_id = entity.get("id") or _canonical_id(
            entity.get("name", ""), entity.get("type", "UNKNOWN")
        )
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.aliases = $aliases,
            e.description = $description,
            e.source_chunks = $source_chunks,
            e.source_texts = $source_texts,
            e.merged_from = $merged_from,
            e.updated_at = $timestamp
        """

        params = {
            "id": entity_id,
            "name": entity.get("name", ""),
            "type": entity.get("type", "UNKNOWN"),
            "aliases": json.dumps(entity.get("aliases", [entity.get("name", "")])),
            "description": entity.get("description", ""),
            "source_chunks": json.dumps(entity.get("source_chunks", [])),
            "source_texts": json.dumps(entity.get("source_texts", [])),
            "merged_from": entity.get("merged_from", 1),
            "timestamp": datetime.now().isoformat(),
        }

        session.run(query, **params)

        # Link to document if provided
        if document_name:
            session.run("""
                MATCH (e:Entity {id: $entity_id}), (d:Document {name: $doc_name})
                MERGE (e)-[:EXTRACTED_FROM]->(d)
                """, entity_id=entity_id, doc_name=document_name)
    
    def _store_relationship(self, session, relationship: Dict[str, Any], document_name: str = None):
        """Store a single relationship in Neo4j."""
        query = """
        MATCH (source:Entity {name: $source_name})
        MATCH (target:Entity {name: $target_name})
        MERGE (source)-[r:RELATES_TO]->(target)
        SET r.relationship = $relationship,
            r.description = $description,
            r.source_chunk = $source_chunk,
            r.source_text = $source_text,
            r.updated_at = $timestamp
        """
        
        params = {
            "source_name": relationship.get("source", ""),
            "target_name": relationship.get("target", ""),
            "relationship": relationship.get("relationship", ""),
            "description": relationship.get("description", ""),
            "source_chunk": relationship.get("source_chunk", ""),
            "source_text": relationship.get("source_text", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        session.run(query, **params)
    
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
            resp = litellm.completion(
                model="gemini/gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            cypher_query = resp.choices[0].message.content.strip()

            # Strip any markdown fences the model may still add
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            # Take only the first line if the model output multiple lines
            cypher_query = cypher_query.splitlines()[0].strip()

            if not any(kw in cypher_query.upper() for kw in ("MATCH", "RETURN")):
                logger.warning(f"LLM returned unexpected Cypher output: {cypher_query!r}")
                return "", f"LLM did not return a valid Cypher query: {cypher_query!r}"

            logger.info(f"LLM-generated Cypher: {cypher_query}")
            return cypher_query, ""

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
        """Get basic statistics about the graph."""
        try:
            with self._session() as session:
                stats = session.run("""
                    MATCH (n:Entity) 
                    OPTIONAL MATCH ()-[r:RELATES_TO]->()
                    RETURN count(DISTINCT n) as node_count, 
                           count(r) as relationship_count
                """).single()
                
                return {
                    "nodes": stats["node_count"],
                    "relationships": stats["relationship_count"]
                }
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"nodes": 0, "relationships": 0}
    
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