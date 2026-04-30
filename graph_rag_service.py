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
        _diag_limit = 10  # emit detailed diagnostics for first N chunks
        logger.info(f"Processing {total_chunks} chunks for graph extraction")

        if progress_tracker:
            progress_tracker.update(0, total_chunks, status="graph_extraction",
                                   message="Extracting entities and relationships")

        # Phase 1: Extract entities and relationships — parallel across all chunks
        chunk_texts = [c["text"] for c in chunks]

        def _progress(done, total, msg=""):
            if progress_tracker:
                progress_tracker.update(done, total, message=msg)

        parallel_result = self.graph_extractor.extract_from_multiple_chunks(
            chunk_texts,
            progress_callback=_progress,
            max_workers=8,
        )

        # Attach per-chunk source metadata (chunk index, source text, chunk metadata)
        # extract_from_multiple_chunks returns flat entity/rel lists tagged chunk_0..chunk_N
        chunk_id_to_index = {f"chunk_{i}": i for i in range(total_chunks)}

        for entity in parallel_result.get("entities", []):
            # source_chunks list already set by extractor; derive chunk_index from first entry
            chunk_id = (entity.get("source_chunks") or ["chunk_0"])[0]
            i = chunk_id_to_index.get(chunk_id, 0)
            chunk = chunks[i] if i < total_chunks else chunks[0]
            entity["source_text"] = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            entity["metadata"]    = chunk.get("metadata", {})
            entity["chunk_index"] = i

        for rel in parallel_result.get("relationships", []):
            chunk_id = rel.get("source_chunk", "chunk_0")
            i = chunk_id_to_index.get(chunk_id, 0)
            chunk = chunks[i] if i < total_chunks else chunks[0]
            rel["source_text"] = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            rel["metadata"]    = chunk.get("metadata", {})
            rel["chunk_index"] = i

        all_entities      = parallel_result.get("entities", [])
        all_relationships = parallel_result.get("relationships", [])

        # Diagnostics for first _diag_limit chunks
        seen_chunks: set = set()
        for entity in all_entities:
            i = entity.get("chunk_index", 0)
            if i < _diag_limit and i not in seen_chunks:
                seen_chunks.add(i)
                chunk_ents = [e["name"] for e in all_entities if e.get("chunk_index") == i]
                chunk_rels = [r for r in all_relationships if r.get("chunk_index") == i]
                text_preview = chunks[i]["text"][:200].replace("\n", " ") if i < total_chunks else ""
                logger.info(
                    f"[DIAG chunk {i}] text={text_preview!r} | "
                    f"entities({len(chunk_ents)})={chunk_ents} | raw_rels={len(chunk_rels)}"
                )
                for r in chunk_rels:
                    logger.info(f"  [DIAG rel] {r.get('source')} --{r.get('relationship')}--> {r.get('target')}")
        
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

        # Tag every LLM-extracted relationship with provenance before co-occurrence phase
        for r in validated_relationships:
            r.setdefault("provenance", "semantic_llm")

        # Phase 3.5: Co-occurrence fallback — entities sharing a chunk get a typed weak edge.
        # These are provenance-tagged as co_occurrence so they are never confused with LLM
        # semantic relationships.
        existing_pairs = {(r["source"], r["target"]) for r in validated_relationships}
        cooccur_relationships = self._build_cooccurrence_edges(all_entities, merged_entities, existing_pairs)
        logger.info(f"Co-occurrence fallback: {len(cooccur_relationships)} CO_OCCURS_WITH edges")
        all_final_relationships = validated_relationships + cooccur_relationships

        # Phase 4: Build NetworkX graph
        if progress_tracker:
            progress_tracker.update(total_chunks + 1, total_chunks + 2, status="graph_building", 
                                   message="Building knowledge graph")
        
        self._build_graph(merged_entities, all_final_relationships)

        # Phase 5: Generate embeddings and store in vector DB
        if progress_tracker:
            progress_tracker.update(total_chunks + 2, total_chunks + 3, status="graph_embedding",
                                   message="Generating graph embeddings")

        self._generate_and_store_graph_embeddings(merged_entities, all_final_relationships, progress_tracker)

        graph_stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "raw_entities": len(all_entities),
            "merged_entities": len(merged_entities),
            "raw_relationships": len(all_relationships),
            "valid_relationships": len(validated_relationships),
            "cooccurrence_edges": len(cooccur_relationships),
            "total_edges": len(all_final_relationships),
            "rejection_stats": getattr(self, "_last_rejection_stats", {}),
        }

        logger.info(f"Graph processing complete: {graph_stats}")

        return {
            "entities": merged_entities,
            "relationships": all_final_relationships,
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

            # Group by canonical ID — stable across re-ingests
            groups = defaultdict(list)
            for entity in type_entities:
                key = entity.get("id") or self._normalize_entity_name(entity["name"])
                groups[key].append(entity)
            
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

        # Collect all surface names; longest wins as canonical display name
        all_names = [e["name"] for e in entities]
        merged_entity["name"] = max(all_names, key=len)

        # Preserve canonical ID from the first entity (stable across merges)
        # and collect all surface forms as aliases
        all_aliases: list = []
        for e in entities:
            all_aliases.extend(e.get("aliases", [e["name"]]))
        merged_entity["aliases"] = list(dict.fromkeys(all_aliases))  # dedup, order-stable
        
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
        """Validate relationships ensuring both entities exist, resolving via aliases."""
        if not relationships or not valid_entities:
            return []

        # Build alias → canonical and normalized-alias → canonical from ALL aliases in merged entities.
        # Pre-merge, the LLM used original entity names; after merging the canonical name may differ.
        alias_to_canonical: Dict[str, str] = {}
        norm_to_canonical: Dict[str, str] = {}
        for entity in valid_entities:
            canonical = entity["name"]
            all_aliases = [canonical] + entity.get("aliases", [])
            for alias in all_aliases:
                alias_to_canonical[alias] = canonical
                norm_key = self._normalize_entity_name(alias)
                if norm_key:
                    norm_to_canonical[norm_key] = canonical

        def _resolve(name: str) -> Optional[str]:
            if name in alias_to_canonical:
                return alias_to_canonical[name]
            norm = self._normalize_entity_name(name)
            return norm_to_canonical.get(norm)

        validated_relationships = []
        rejection_stats: Dict[str, int] = {
            "no_endpoint": 0, "missing_source": 0, "missing_target": 0, "self_loop": 0
        }

        for rel in relationships:
            source_raw = rel.get("source", "")
            target_raw = rel.get("target", "")

            if not source_raw or not target_raw:
                rejection_stats["no_endpoint"] += 1
                continue

            resolved_source = _resolve(source_raw)
            resolved_target = _resolve(target_raw)

            if not resolved_source:
                rejection_stats["missing_source"] += 1
                logger.debug(f"Rejected (missing source): '{source_raw}' → '{target_raw}'")
                continue
            if not resolved_target:
                rejection_stats["missing_target"] += 1
                logger.debug(f"Rejected (missing target): '{source_raw}' → '{target_raw}'")
                continue
            if resolved_source == resolved_target:
                rejection_stats["self_loop"] += 1
                continue

            validated_rel = rel.copy()
            validated_rel["source"] = resolved_source
            validated_rel["target"] = resolved_target
            validated_relationships.append(validated_rel)

        logger.info(
            f"Relationship validation: {len(relationships)} raw → {len(validated_relationships)} valid | "
            f"rejections: {rejection_stats}"
        )
        self._last_rejection_stats = rejection_stats
        return validated_relationships

    def _build_cooccurrence_edges(
        self,
        all_entities: List[Dict[str, Any]],
        merged_entities: List[Dict[str, Any]],
        existing_pairs: Set[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        """Return CO_OCCURS_WITH edges for entity pairs sharing a chunk with no existing link.

        Edges are provenance=co_occurrence — never confused with semantic relationships.
        Caps prevent weak edges from drowning semantic ones:
          max_cooccurrence_per_chunk  — max CO_OCCURS_WITH edges emitted per chunk
          max_cooccurrence_edges_total — max total across the whole document
        """
        max_per_chunk = self.config.get("max_cooccurrence_per_chunk", 10)
        max_total     = self.config.get("max_cooccurrence_edges_total", 100)

        alias_to_canonical: Dict[str, str] = {}
        for entity in merged_entities:
            canonical = entity["name"]
            for alias in [canonical] + entity.get("aliases", []):
                alias_to_canonical[alias] = canonical

        # Group resolved canonical names by chunk index
        chunk_entity_map: Dict[int, Set[str]] = defaultdict(set)
        for e in all_entities:
            canonical = alias_to_canonical.get(e["name"], e["name"])
            chunk_entity_map[e.get("chunk_index", -1)].add(canonical)

        cooccur: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set(existing_pairs)
        # Also seed reverse direction so we don't add (B,A) when (A,B) exists
        seen.update((b, a) for a, b in existing_pairs)

        for chunk_idx, canonicals in chunk_entity_map.items():
            if len(cooccur) >= max_total:
                break
            canon_list = sorted(canonicals)
            chunk_count = 0
            for i in range(len(canon_list)):
                if chunk_count >= max_per_chunk or len(cooccur) >= max_total:
                    break
                for j in range(i + 1, len(canon_list)):
                    if chunk_count >= max_per_chunk or len(cooccur) >= max_total:
                        break
                    a, b = canon_list[i], canon_list[j]
                    if (a, b) in seen or (b, a) in seen:
                        continue
                    seen.add((a, b))
                    cooccur.append({
                        "source":       a,
                        "target":       b,
                        "relationship": "CO_OCCURS_WITH",
                        "description":  f"Co-occurs in chunk {chunk_idx}",
                        "source_chunk": f"chunk_{chunk_idx}",
                        "source_text":  "",
                        "provenance":   "co_occurrence",
                    })
                    chunk_count += 1

        return cooccur

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

        # Build all texts and docs in one pass — no per-item API calls
        texts: List[str] = []
        graph_documents: List[Dict[str, Any]] = []

        for entity in entities:
            text = self._create_entity_embedding_text(entity)
            texts.append(text)
            graph_documents.append({
                "text": text,
                "metadata": {
                    "type": "entity",
                    "entity_name": entity["name"],
                    "entity_type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "source_chunks": entity.get("source_chunks", []),
                    "graph_element": True,
                    "merged_from": entity.get("merged_from", 1)
                }
            })

        for rel in relationships:
            text = self._create_relationship_embedding_text(rel)
            texts.append(text)
            graph_documents.append({
                "text": text,
                "metadata": {
                    "type": "relationship",
                    "source": rel["source"],
                    "target": rel["target"],
                    "relationship": rel.get("relationship", ""),
                    "description": rel.get("description", ""),
                    "source_chunk": rel.get("source_chunk", ""),
                    "graph_element": True
                }
            })

        if not texts:
            logger.warning("No graph elements to store in vector database")
            return

        if progress_tracker:
            progress_tracker.update(0, len(texts), status="graph_embedding",
                                    message="Generating graph embeddings")

        # Single batch call — uses batchEmbedContents (50 texts/request)
        graph_embeddings = self.embedding_service.get_embeddings_batch(texts, progress_tracker=progress_tracker)

        if len(graph_embeddings) != len(graph_documents):
            logger.error(f"Embedding count mismatch: {len(graph_embeddings)} vs {len(graph_documents)} docs")
            return

        logger.info(f"Storing {len(graph_documents)} graph elements in vector DB")
        self._store_graph_in_vector_db(graph_documents, graph_embeddings, progress_tracker)
    
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
    
    def search_graph(self, query: str, top_k: int = 10, neo4j_service=None) -> List[Dict[str, Any]]:
        """Search knowledge graph: Qdrant vector search for seeds, Neo4j Cypher traversal for neighborhoods.

        NetworkX is NOT used for traversal. If neo4j_service is None, returns vector-only
        results with a warning — production graph mode requires Neo4j.
        """
        if not self.embedding_service or not self.vector_db_service:
            logger.warning("Services not available for graph search")
            return []

        try:
            # Step 1: Qdrant vector search — find semantically similar entities / relationships
            query_embedding = self.embedding_service.get_embedding(query)
            collection_name = self.config["graph_collection_name"]

            vector_results = self.vector_db_service.search_similar(
                query_embedding,
                top_k=top_k,
                collection_name=collection_name,
            )

            logger.debug(f"Graph vector search: {len(vector_results)} results")

            if not vector_results:
                return []

            # Step 2: Extract seed entity names from vector results
            seed_names: List[str] = []
            for r in vector_results:
                if r["metadata"].get("type") == "entity":
                    name = r["metadata"].get("entity_name")
                    if name:
                        seed_names.append(name)

            logger.debug(f"Graph seeds: {seed_names[:5]}")

            # Step 3: Neo4j Cypher traversal from seeds
            traversal_results: List[Dict[str, Any]] = []

            if neo4j_service is None:
                logger.warning(
                    "graph mode: Neo4j not configured — returning vector-only graph results. "
                    "Production graph mode requires Neo4j for topology traversal."
                )
            elif seed_names:
                depth            = self.config.get("graph_reasoning_depth", 2)
                benchmark_corpus = self.config.get("benchmark_corpus_tag", "")
                rows = neo4j_service.traverse_neighbors(
                    seed_names[:10], depth=depth, limit=max(top_k * 5, 50),
                    benchmark_corpus=benchmark_corpus,
                )

                seen_triplets: Set[str] = set()
                for row in rows:
                    source = row.get("seed_name", "")
                    rel    = row.get("relationship", "")
                    target = row.get("neighbor_name", "")
                    key    = f"{source}|{rel}|{target}"
                    if key in seen_triplets or not source or not target:
                        continue
                    seen_triplets.add(key)

                    rel_desc   = row.get("rel_description", "")
                    tgt_desc   = row.get("neighbor_description", "")
                    src_text   = row.get("source_text", "")
                    tgt_type   = row.get("neighbor_type", "")
                    provenance = row.get("provenance") or "semantic_llm"

                    parts = [f"Relationship: {source} → {rel} → {target}"]
                    if rel_desc:
                        parts.append(f"Description: {rel_desc}")
                    if tgt_desc:
                        parts.append(f"Neighbor: {target} ({tgt_type}) — {tgt_desc}")

                    traversal_results.append({
                        "text": " | ".join(parts),
                        "metadata": {
                            "type":                  "relationship",
                            "source":                source,
                            "source_type":           row.get("seed_type", ""),
                            "relationship":          rel,
                            "rel_description":       rel_desc,
                            "target":                target,
                            "target_type":           tgt_type,
                            "target_description":    tgt_desc,
                            "source_text":           src_text,
                            "neighbor_source_texts": row.get("neighbor_source_texts", ""),
                            "discovery_method":      "neo4j_traversal",
                            "seed_entity":           source,
                            "provenance":            provenance,
                            "rel_benchmark_corpus":  row.get("rel_benchmark_corpus", ""),
                            "graph_element":         True,
                        },
                        "score": None,
                    })

                logger.info(
                    f"Neo4j traversal: {len(rows)} rows → {len(traversal_results)} unique triplets"
                )

            # Step 4: Deduplicate and combine
            seen_rels: Set[str] = set()
            for r in vector_results:
                if r["metadata"].get("type") == "relationship":
                    k = (f"{r['metadata'].get('source')}|"
                         f"{r['metadata'].get('relationship')}|"
                         f"{r['metadata'].get('target')}")
                    seen_rels.add(k)

            unique_traversal = [
                r for r in traversal_results
                if f"{r['metadata']['source']}|{r['metadata']['relationship']}|{r['metadata']['target']}"
                not in seen_rels
            ]

            combined = vector_results + unique_traversal
            combined.sort(key=lambda x: (x.get("score") or 0), reverse=True)
            final = combined[:top_k * 2]

            logger.info(
                f"Graph search: {len(vector_results)} vector + {len(unique_traversal)} traversal"
                f" → {len(final)} combined"
            )
            return final

        except Exception as e:
            logger.error(f"Error searching graph: {e}")
            return []
    
    def get_entity_neighborhood(self, entity_name: str, depth: int = None) -> Dict[str, Any]:
        """Get the neighborhood of an entity in the graph with accurate distance tracking."""
        if depth is None:
            depth = self.graph_reasoning_depth
        
        if entity_name not in self.graph.nodes:
            logger.debug(f"Entity not found in graph: {entity_name}")
            return {"entities": [], "relationships": [], "center_entity": entity_name}
        
        # Track nodes with their distance from center
        node_distances = {entity_name: 0}
        current_nodes = {entity_name}
        
        for level in range(1, depth + 1):
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in node_distances:
                        node_distances[neighbor] = level
                        next_nodes.add(neighbor)
            current_nodes = next_nodes
            
            if not next_nodes:  # No more neighbors to explore
                break
        
        # Extract subgraph
        subgraph = self.graph.subgraph(node_distances.keys())
        
        # Format entities with accurate distance
        entities = []
        for node in subgraph.nodes():
            node_data = self.graph.nodes[node]
            entities.append({
                "name": node,
                "type": node_data.get("type", ""),
                "description": node_data.get("description", ""),
                "distance_from_center": node_distances.get(node, 0)
            })
        
        # Sort entities by distance (closer first)
        entities.sort(key=lambda x: x['distance_from_center'])
        
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