"""
Graph RAG service for managing knowledge graphs and graph-based retrieval
"""

import logging
import time
import networkx as nx
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
from extensions import ProgressTracker
from graph_extractor import GraphExtractor
from neo4j_service import Neo4jService  # Add this import

logger = logging.getLogger(__name__)

class GraphRAGService:
    """Manages the knowledge graph and provides graph-based retrieval with enhanced entity merging."""
    
    def __init__(self, config):
        self.config = config
        self.graph = nx.Graph()
        self.entity_embeddings = {}
        self.relationship_embeddings = {}
        self.embedding_service = None
        self.vector_db_service = None
        self.graph_extractor = GraphExtractor(config)
        
        # Configuration parameters
        self.entity_similarity_threshold = config.get("entity_similarity_threshold", 0.8)
        self.graph_reasoning_depth = config.get("graph_reasoning_depth", 2)
        
        logger.info("GraphRAGService initialized")
    
    def set_services(self, embedding_service, vector_db_service):
        """Set the embedding and vector DB services."""
        self.embedding_service = embedding_service
        self.vector_db_service = vector_db_service
        logger.debug("Services set for GraphRAGService")
    
    def process_document_for_graph(self, chunks: List[Dict[str, Any]], 
                                 progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
        """Process document chunks to extract and build knowledge graph structure."""
        if not chunks:
            logger.warning("No chunks provided for graph processing")
            return {"entities": [], "relationships": [], "graph_stats": {"nodes": 0, "edges": 0}}
        
        all_entities = []
        all_relationships = []
        
        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks for graph extraction")
        
        if progress_tracker:
            progress_tracker.update(0, total_chunks, status="graph_extraction", 
                                   message="Extracting entities and relationships")
        
        # Phase 1: Extract entities and relationships from all chunks
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"chunk_{i}_{int(time.time())}"
                
                logger.debug(f"Extracting from chunk {i+1}/{total_chunks}")
                
                # Extract entities and relationships from this chunk
                graph_data = self.graph_extractor.extract_entities_and_relationships(
                    chunk["text"], chunk_id
                )
                
                # Add source metadata to entities and relationships
                for entity in graph_data.get("entities", []):
                    entity["source_text"] = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    entity["metadata"] = chunk.get("metadata", {})
                    entity["chunk_index"] = i
                
                for rel in graph_data.get("relationships", []):
                    rel["source_text"] = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    rel["metadata"] = chunk.get("metadata", {})
                    rel["chunk_index"] = i
                
                all_entities.extend(graph_data.get("entities", []))
                all_relationships.extend(graph_data.get("relationships", []))
                
                if progress_tracker:
                    progress_tracker.update(i + 1, total_chunks, 
                                           message=f"Extracted from chunk {i + 1} of {total_chunks}")
                
                # Small delay to respect API limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
        
        logger.info(f"Raw extraction complete: {len(all_entities)} entities, {len(all_relationships)} relationships")
        
        # Phase 2: Merge similar entities
        if progress_tracker:
            progress_tracker.update(total_chunks, total_chunks + 1, status="entity_merging", 
                                   message="Merging similar entities")
        
        merged_entities = self._merge_similar_entities(all_entities)
        logger.info(f"After merging: {len(merged_entities)} unique entities")
        
        # Phase 3: Filter and validate relationships
        validated_relationships = self._validate_relationships(all_relationships, merged_entities)
        logger.info(f"After validation: {len(validated_relationships)} valid relationships")
        
        # Phase 4: Build NetworkX graph
        if progress_tracker:
            progress_tracker.update(total_chunks + 1, total_chunks + 2, status="graph_building", 
                                   message="Building knowledge graph")
        
        self._build_graph(merged_entities, validated_relationships)
        
        # Phase 5: Generate embeddings and store in vector DB
        if progress_tracker:
            progress_tracker.update(total_chunks + 2, total_chunks + 3, status="graph_embedding", 
                                   message="Generating graph embeddings")
        
        self._generate_and_store_graph_embeddings(merged_entities, validated_relationships, progress_tracker)
        
        graph_stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "raw_entities": len(all_entities),
            "merged_entities": len(merged_entities),
            "raw_relationships": len(all_relationships),
            "valid_relationships": len(validated_relationships)
        }
        
        logger.info(f"Graph processing complete: {graph_stats}")
        
        return {
            "entities": merged_entities,
            "relationships": validated_relationships,
            "graph_stats": graph_stats
        }
    
    def _merge_similar_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge entities that likely refer to the same thing using advanced similarity."""
        if not entities:
            return []
        
        logger.debug(f"Starting entity merging with {len(entities)} entities")
        
        # Group entities by type for more efficient processing
        entities_by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            entities_by_type[entity_type].append(entity)
        
        merged_entities = []
        
        for entity_type, type_entities in entities_by_type.items():
            logger.debug(f"Merging {len(type_entities)} entities of type {entity_type}")
            
            # For each type, group by normalized name
            groups = defaultdict(list)
            
            for entity in type_entities:
                normalized_name = self._normalize_entity_name(entity["name"])
                groups[normalized_name].append(entity)
            
            # Merge entities within each group
            for normalized_name, group in groups.items():
                if len(group) == 1:
                    # Single entity, no merging needed
                    merged_entities.append(group[0])
                else:
                    # Multiple entities, merge them
                    merged_entity = self._merge_entity_group(group)
                    merged_entities.append(merged_entity)
        
        logger.debug(f"Entity merging complete: {len(entities)} -> {len(merged_entities)}")
        return merged_entities
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for similarity comparison."""
        if not name:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = name.lower().strip()
        
        # Remove common suffixes and prefixes
        suffixes_to_remove = [" inc", " corp", " corporation", " company", " ltd", " llc", " co"]
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["the ", "dr ", "mr ", "ms ", "mrs "]
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Remove special characters and extra spaces
        import re
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _merge_entity_group(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of similar entities into a single entity."""
        if not entities:
            return {}
        
        if len(entities) == 1:
            return entities[0]
        
        # Use the first entity as the base
        merged_entity = entities[0].copy()
        
        # Collect all names and pick the most common or longest
        all_names = [e["name"] for e in entities]
        # Pick the longest name as it's likely more descriptive
        merged_entity["name"] = max(all_names, key=len)
        
        # Merge descriptions
        descriptions = [e.get("description", "") for e in entities if e.get("description")]
        if descriptions:
            # Use the longest description
            merged_entity["description"] = max(descriptions, key=len)
        
        # Combine source chunks and texts
        all_source_chunks = []
        all_source_texts = []
        all_metadata = []
        
        for entity in entities:
            if "source_chunk" in entity:
                all_source_chunks.append(entity["source_chunk"])
            if "source_text" in entity:
                all_source_texts.append(entity["source_text"])
            if "metadata" in entity:
                all_metadata.append(entity["metadata"])
            
            # Also handle lists of sources
            if "source_chunks" in entity:
                all_source_chunks.extend(entity["source_chunks"])
            if "source_texts" in entity:
                all_source_texts.extend(entity["source_texts"])
        
        # Store unique sources
        merged_entity["source_chunks"] = list(set(all_source_chunks))
        merged_entity["source_texts"] = list(set(all_source_texts))
        merged_entity["merged_from"] = len(entities)  # Track how many entities were merged
        
        return merged_entity
    
    def _validate_relationships(self, relationships: List[Dict[str, Any]], 
                              valid_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate relationships ensuring both entities exist in the entity list."""
        if not relationships or not valid_entities:
            return []
        
        # Create a mapping from entity names to entities for quick lookup
        entity_names = set()
        entity_name_mapping = {}
        
        for entity in valid_entities:
            entity_name = entity["name"]
            entity_names.add(entity_name)
            # Also add normalized version for fuzzy matching
            normalized_name = self._normalize_entity_name(entity_name)
            entity_name_mapping[normalized_name] = entity_name
        
        validated_relationships = []
        
        for rel in relationships:
            try:
                source = rel.get("source", "")
                target = rel.get("target", "")
                
                if not source or not target:
                    continue
                
                # Check direct matches first
                source_exists = source in entity_names
                target_exists = target in entity_names
                
                # If direct match fails, try normalized matching
                if not source_exists:
                    normalized_source = self._normalize_entity_name(source)
                    if normalized_source in entity_name_mapping:
                        source = entity_name_mapping[normalized_source]
                        source_exists = True
                
                if not target_exists:
                    normalized_target = self._normalize_entity_name(target)
                    if normalized_target in entity_name_mapping:
                        target = entity_name_mapping[normalized_target]
                        target_exists = True
                
                # Only include relationship if both entities exist
                if source_exists and target_exists and source != target:
                    # Update the relationship with corrected entity names
                    validated_rel = rel.copy()
                    validated_rel["source"] = source
                    validated_rel["target"] = target
                    validated_relationships.append(validated_rel)
                
            except Exception as e:
                logger.debug(f"Error validating relationship: {e}")
                continue
        
        logger.debug(f"Relationship validation: {len(relationships)} -> {len(validated_relationships)}")
        return validated_relationships
    
    def _build_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """Build NetworkX graph from entities and relationships."""
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes (entities)
        for entity in entities:
            entity_name = entity["name"]
            self.graph.add_node(
                entity_name,
                type=entity.get("type", "UNKNOWN"),
                description=entity.get("description", ""),
                source_chunks=entity.get("source_chunks", []),
                source_texts=entity.get("source_texts", []),
                merged_from=entity.get("merged_from", 1)
            )
        
        # Add edges (relationships)
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            
            # Only add edge if both entities exist in the graph
            if source in self.graph.nodes and target in self.graph.nodes:
                # If edge already exists, combine the relationship descriptions
                if self.graph.has_edge(source, target):
                    existing_data = self.graph.edges[source, target]
                    existing_rel = existing_data.get("relationship", "")
                    new_rel = rel.get("relationship", "")
                    combined_rel = f"{existing_rel}; {new_rel}" if existing_rel else new_rel
                    
                    existing_desc = existing_data.get("description", "")
                    new_desc = rel.get("description", "")
                    combined_desc = f"{existing_desc}; {new_desc}" if existing_desc else new_desc
                    
                    self.graph.edges[source, target]["relationship"] = combined_rel
                    self.graph.edges[source, target]["description"] = combined_desc
                else:
                    # Add new edge
                    self.graph.add_edge(
                        source,
                        target,
                        relationship=rel.get("relationship", ""),
                        description=rel.get("description", ""),
                        source_chunk=rel.get("source_chunk", ""),
                        source_text=rel.get("source_text", "")
                    )
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _generate_and_store_graph_embeddings(self, entities: List[Dict[str, Any]], 
                                           relationships: List[Dict[str, Any]], 
                                           progress_tracker: Optional[ProgressTracker] = None):
        """Generate embeddings for entities and relationships and store in vector DB."""
        if not self.embedding_service or not self.vector_db_service:
            logger.warning("Embedding or Vector DB service not available for graph embeddings")
            return
        
        # Prepare documents for vector storage
        graph_documents = []
        graph_embeddings = []
        
        total_items = len(entities) + len(relationships)
        current_item = 0
        
        if progress_tracker:
            progress_tracker.update(0, total_items, status="graph_embedding", 
                                  message="Generating graph embeddings")
        
        # Process entities
        logger.debug(f"Processing {len(entities)} entities for embedding")
        for entity in entities:
            try:
                # Create rich text representation for embedding
                entity_text = self._create_entity_embedding_text(entity)
                
                # Generate embedding with retry logic
                embedding = self._get_embedding_with_retry(entity_text)
                
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for entity: {entity['name']}")
                    continue
                
                # Prepare document for vector storage
                entity_doc = {
                    "text": entity_text,
                    "metadata": {
                        "type": "entity",
                        "entity_name": entity["name"],
                        "entity_type": entity.get("type", "UNKNOWN"),
                        "description": entity.get("description", ""),
                        "source_chunks": entity.get("source_chunks", []),
                        "graph_element": True,
                        "merged_from": entity.get("merged_from", 1)
                    }
                }
                
                graph_documents.append(entity_doc)
                graph_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing entity {entity.get('name', 'unknown')}: {e}")
                continue
            
            current_item += 1
            if progress_tracker:
                progress_tracker.update(current_item, total_items, 
                                      message=f"Processed entity {current_item} of {total_items}")
        
        # Process relationships
        logger.debug(f"Processing {len(relationships)} relationships for embedding")
        for rel in relationships:
            try:
                # Create rich text representation for embedding
                rel_text = self._create_relationship_embedding_text(rel)
                
                # Generate embedding with retry logic
                embedding = self._get_embedding_with_retry(rel_text)
                
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for relationship: {rel['source']} -> {rel['target']}")
                    continue
                
                # Prepare document for vector storage
                rel_doc = {
                    "text": rel_text,
                    "metadata": {
                        "type": "relationship",
                        "source": rel["source"],
                        "target": rel["target"],
                        "relationship": rel.get("relationship", ""),
                        "description": rel.get("description", ""),
                        "source_chunk": rel.get("source_chunk", ""),
                        "graph_element": True
                    }
                }
                
                graph_documents.append(rel_doc)
                graph_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing relationship {rel.get('source', 'unknown')} -> {rel.get('target', 'unknown')}: {e}")
                continue
            
            current_item += 1
            if progress_tracker:
                progress_tracker.update(current_item, total_items, 
                                      message=f"Processed relationship {current_item} of {total_items}")
        
        # Store in vector database (graph collection)
        if graph_documents and graph_embeddings:
            logger.info(f"Storing {len(graph_documents)} graph elements in vector DB")
            self._store_graph_in_vector_db(graph_documents, graph_embeddings, progress_tracker)
        else:
            logger.warning("No graph elements to store in vector database")
    
    def _create_entity_embedding_text(self, entity: Dict[str, Any]) -> str:
        """Create rich text representation for entity embedding."""
        name = entity.get("name", "")
        entity_type = entity.get("type", "")
        description = entity.get("description", "")
        
        # Create context-rich text
        text_parts = [f"Entity: {name}"]
        
        if entity_type:
            text_parts.append(f"Type: {entity_type}")
        
        if description:
            text_parts.append(f"Description: {description}")
        
        # Add source context if available
        source_texts = entity.get("source_texts", [])
        if source_texts:
            # Use first source text for context
            context = source_texts[0][:200] + "..." if len(source_texts[0]) > 200 else source_texts[0]
            text_parts.append(f"Context: {context}")
        
        return " | ".join(text_parts)
    
    def _create_relationship_embedding_text(self, relationship: Dict[str, Any]) -> str:
        """Create rich text representation for relationship embedding."""
        source = relationship.get("source", "")
        rel_type = relationship.get("relationship", "")
        target = relationship.get("target", "")
        description = relationship.get("description", "")
        
        # Create context-rich text
        text_parts = [f"Relationship: {source} {rel_type} {target}"]
        
        if description:
            text_parts.append(f"Description: {description}")
        
        # Add source context if available
        source_text = relationship.get("source_text", "")
        if source_text:
            context = source_text[:200] + "..." if len(source_text) > 200 else source_text
            text_parts.append(f"Context: {context}")
        
        return " | ".join(text_parts)
    
    def _get_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Get embedding with retry logic."""
        for attempt in range(max_retries):
            try:
                embedding = self.embedding_service.get_embedding(text)
                return embedding
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for embedding: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts")
                    return None
                time.sleep(1)  # Wait before retry
        return None
    
    def _store_graph_in_vector_db(self, docs: List[Dict[str, Any]], 
                                embeddings: List[List[float]], 
                                progress_tracker: Optional[ProgressTracker] = None):
        """Store graph elements in a separate vector collection."""
        try:
            collection_name = self.config["graph_collection_name"]
            
            logger.info(f"Storing {len(docs)} graph elements in collection: {collection_name}")
            
            # Use the vector DB service to insert documents
            self.vector_db_service.insert_documents(
                docs, 
                embeddings, 
                progress_tracker=progress_tracker,
                collection_name=collection_name
            )
            
            logger.info(f"Successfully stored {len(docs)} graph elements in vector DB")
            
        except Exception as e:
            logger.error(f"Error storing graph in vector DB: {str(e)}")
            raise
    
    def search_graph(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph using semantic similarity."""
        if not self.embedding_service or not self.vector_db_service:
            logger.warning("Services not available for graph search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.get_embedding(query)
            
            # Search in graph collection
            collection_name = self.config["graph_collection_name"]
            
            results = self.vector_db_service.search_similar(
                query_embedding,
                top_k=top_k,
                collection_name=collection_name
            )
            
            logger.debug(f"Graph search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching graph: {str(e)}")
            return []
    
    def get_entity_neighborhood(self, entity_name: str, depth: int = None) -> Dict[str, Any]:
        """Get the neighborhood of an entity in the graph."""
        if depth is None:
            depth = self.graph_reasoning_depth
        
        if entity_name not in self.graph.nodes:
            logger.debug(f"Entity not found in graph: {entity_name}")
            return {"entities": [], "relationships": [], "center_entity": entity_name}
        
        # Get subgraph within specified depth
        subgraph_nodes = set([entity_name])
        current_nodes = {entity_name}
        
        for level in range(depth):
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(self.graph.neighbors(node))
                next_nodes.update(neighbors)
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            if not next_nodes:  # No more neighbors to explore
                break
        
        # Extract subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Format entities and relationships
        entities = []
        for node in subgraph.nodes():
            node_data = self.graph.nodes[node]
            entities.append({
                "name": node,
                "type": node_data.get("type", ""),
                "description": node_data.get("description", ""),
                "distance_from_center": 0 if node == entity_name else 1  # Could be improved with actual distance calculation
            })
        
        relationships = []
        for edge in subgraph.edges():
            edge_data = self.graph.edges[edge]
            relationships.append({
                "source": edge[0],
                "target": edge[1],
                "relationship": edge_data.get("relationship", ""),
                "description": edge_data.get("description", "")
            })
        
        return {
            "entities": entities,
            "relationships": relationships,
            "center_entity": entity_name,
            "depth": depth,
            "total_nodes": len(entities),
            "total_edges": len(relationships)
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {},
            "connected_components": nx.number_connected_components(self.graph),
            "average_degree": 0
        }
        
        if self.graph.number_of_nodes() > 0:
            # Calculate average degree
            degrees = [d for n, d in self.graph.degree()]
            stats["average_degree"] = sum(degrees) / len(degrees) if degrees else 0
            
            # Count node types
            for node in self.graph.nodes():
                node_type = self.graph.nodes[node].get("type", "UNKNOWN")
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        return stats