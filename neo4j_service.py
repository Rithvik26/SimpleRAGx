"""
Neo4j Graph Database Service for Enhanced Graph RAG
Handles connection, storage, and querying of knowledge graphs
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)

class Neo4jService:
    """Service for interacting with Neo4j graph database."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.uri = uri
        self.username = username  
        self.password = password
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
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_graph(self) -> bool:
        """Clear all nodes and relationships from the graph."""
        try:
            with self.driver.session() as session:
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
            with self.driver.session() as session:
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
            with self.driver.session() as session:
                # Create document node if provided
                if document_name:
                    session.run("""
                        MERGE (d:Document {name: $doc_name, created_at: $timestamp})
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
        """Store a single entity in Neo4j."""
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.description = $description,
            e.source_chunks = $source_chunks,
            e.source_texts = $source_texts,
            e.merged_from = $merged_from,
            e.updated_at = $timestamp
        """
        
        params = {
            "name": entity.get("name", ""),
            "type": entity.get("type", "UNKNOWN"),
            "description": entity.get("description", ""),
            "source_chunks": json.dumps(entity.get("source_chunks", [])),
            "source_texts": json.dumps(entity.get("source_texts", [])),
            "merged_from": entity.get("merged_from", 1),
            "timestamp": datetime.now().isoformat()
        }
        
        session.run(query, **params)
        
        # Link to document if provided
        if document_name:
            session.run("""
                MATCH (e:Entity {name: $entity_name}), (d:Document {name: $doc_name})
                MERGE (e)-[:EXTRACTED_FROM]->(d)
                """, entity_name=entity.get("name"), doc_name=document_name)
    
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
    
    def generate_cypher_from_question(self, question: str, llm_service) -> Tuple[str, str]:
        """Generate Cypher query from natural language question using LLM - CASE INSENSITIVE VERSION."""
        schema_info = self.get_schema_info()
        
        prompt = f"""
    You are a Neo4j Cypher query generator. Given a natural language question and database schema, generate a valid Cypher query.

    DATABASE SCHEMA:
    {schema_info}

    IMPORTANT RULES:
    1. ALWAYS use case-insensitive matching with toLower() for name comparisons
    2. Use CONTAINS for partial matching instead of exact equality when searching names
    3. Use MATCH, WHERE, RETURN patterns
    4. Limit results to 20 unless specifically asked for more
    5. For entity searches, match on name or description using toLower()
    6. For relationship queries, traverse the graph appropriately
    7. Always include node properties in RETURN when relevant

    EXAMPLES:
    Question: "What is TechCorp about?"
    Cypher: MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('TechCorp') RETURN e.name, e.type, e.description LIMIT 20

    Question: "What entities are related to TechCorp?"
    Cypher: MATCH (e1:Entity)-[r:RELATES_TO]-(e2:Entity) WHERE toLower(e1.name) CONTAINS toLower('TechCorp') RETURN e1.name, r.relationship, e2.name, e2.type LIMIT 20

    Question: "Show me all companies and their relationships"
    Cypher: MATCH (e1:Entity {{type: 'ORGANIZATION'}})-[r:RELATES_TO]-(e2:Entity) RETURN e1.name, r.relationship, e2.name, e2.type LIMIT 20

    Question: "Tell me about Microsoft"
    Cypher: MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('Microsoft') RETURN e.name, e.type, e.description LIMIT 20

    QUESTION: {question}

    Generate only the Cypher query, no explanation. Use case-insensitive matching:
    """
        
        try:
            # Use generate_answer method
            response = llm_service.generate_answer(prompt, [])
            
            # Alternative: use internal method if available
            if hasattr(llm_service, '_generate_with_llm'):
                response = llm_service._generate_with_llm(prompt)
            
            cypher_query = response.strip()
            
            # Clean up the response (remove any markdown formatting)
            if cypher_query.startswith("```"):
                lines = cypher_query.split("\n")
                if lines[0].startswith("```") and lines[-1].startswith("```"):
                    cypher_query = "\n".join(lines[1:-1])
            
            # Additional cleanup
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
            # Validate basic Cypher syntax
            if not any(keyword in cypher_query.upper() for keyword in ["MATCH", "RETURN"]):
                logger.warning(f"Generated query might be invalid: {cypher_query}")
                
                # Fallback: Generate a simple case-insensitive query ourselves
                # Extract the main entity from the question
                import re
                entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)
                if entities:
                    entity_name = entities[0]
                    cypher_query = f"""
                    MATCH (e:Entity) 
                    WHERE toLower(e.name) CONTAINS toLower('{entity_name}')
                    RETURN e.name, e.type, e.description
                    LIMIT 20
                    """.strip()
                    logger.info(f"Used fallback query for entity: {entity_name}")
            
            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query, ""
            
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            
            # Last resort: try to extract entity name and create basic query
            import re
            entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)
            if entities:
                entity_name = entities[0]
                fallback_query = f"""
                MATCH (e:Entity) 
                WHERE toLower(e.name) CONTAINS toLower('{entity_name}')
                RETURN e.name, e.type, e.description
                LIMIT 20
                """.strip()
                logger.info(f"Using emergency fallback query for: {entity_name}")
                return fallback_query, ""
            
            return "", f"Error generating query: {str(e)}"
    
    def execute_cypher_query(self, cypher_query: str) -> Tuple[List[Dict], str]:
        """Execute Cypher query and return results - FIXED."""
        try:
            with self.driver.session() as session:
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
            with self.driver.session() as session:
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
            with self.driver.session() as session:
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
            with self.driver.session() as session:
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
            uri=config["NEO4J_URI"],
            username=config["NEO4J_USERNAME"], 
            password=config["NEO4J_PASSWORD"]
        )
        service.create_indexes()
        return service
    except Exception as e:
        logger.error(f"Failed to create Neo4j service: {e}")
        return None