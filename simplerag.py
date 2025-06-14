"""
Enhanced SimpleRAG - A Retrieval-Augmented Generation System with Graph RAG Support

This system now supports two RAG modes:
1. Normal RAG - Traditional semantic search with chunks
2. Graph RAG - Entity and relationship extraction with knowledge graph reasoning

Both modes use the same Gemini API key and Qdrant vector database.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import networkx as nx
from collections import defaultdict

# Document processing
import PyPDF2
import docx
import tiktoken
from bs4 import BeautifulSoup

# Vector DB and embeddings
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import requests

# LLM integration
import anthropic

# Import extensions
from extensions import RateLimiter, EmbeddingCache, rate_limited, cached_embedding, ProgressTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('SimpleRAG')

# Enhanced configuration with Graph RAG options
DEFAULT_CONFIG = {
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    "claude_api_key": os.environ.get("CLAUDE_API_KEY", ""),
    "qdrant_url": os.environ.get("QDRANT_URL", "https://3cbcacc0-1fe5-42a1-8be0-81515a21771b.us-west-2-0.aws.cloud.qdrant.io"),
    "qdrant_api_key": os.environ.get("QDRANT_API_KEY", ""),
    "collection_name": os.environ.get("QDRANT_COLLECTION", "simple_rag_docs"),
    "graph_collection_name": os.environ.get("QDRANT_GRAPH_COLLECTION", "simple_rag_graph"),
    "embedding_dimension": 768,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "preferred_llm": "claude",
    "rag_mode": "normal",
    "rate_limit": 60,
    "enable_cache": True,
    "cache_dir": None,
    # Graph RAG specific settings
    "max_entities_per_chunk": 10,
    "relationship_extraction_prompt": "extract_relationships",
    "graph_reasoning_depth": 2,
    "entity_similarity_threshold": 0.8
}

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.expanduser("~/.simplerag/config.json"))

def ensure_config_exists():
    """Make sure config file exists, create with defaults if not."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Created default configuration at {CONFIG_PATH}")
        return False
    return True

def load_config():
    """Load configuration from disk."""
    ensure_config_exists()
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Override with environment variables if set
    env_overrides = {
        "gemini_api_key": "GEMINI_API_KEY",
        "claude_api_key": "CLAUDE_API_KEY", 
        "qdrant_api_key": "QDRANT_API_KEY",
        "qdrant_url": "QDRANT_URL"
    }
    
    for config_key, env_key in env_overrides.items():
        if os.environ.get(env_key):
            config[config_key] = os.environ.get(env_key)
    
    # Add default values for new config options
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    
    return config

class GraphExtractor:
    """Extracts entities and relationships from text using Gemini API."""
    
    def __init__(self, config):
        self.api_key = config["gemini_api_key"]
        self.max_entities_per_chunk = config.get("max_entities_per_chunk", 20)
        self.rate_limiter = RateLimiter(calls_per_minute=config.get("rate_limit", 60))
    
    @rate_limited(RateLimiter(calls_per_minute=30))  # More conservative for generation
    def extract_entities_and_relationships(self, text: str, chunk_id: str = None) -> Dict[str, Any]:
        """Extract entities and relationships from text using Gemini."""
    
        extraction_prompt = f"""
You are an expert knowledge graph extractor. Extract entities and relationships from the following text.

TEXT:
{text[:2000]}  # Limit text length for API

INSTRUCTIONS:
1. Extract up to {self.max_entities_per_chunk} most important entities
2. For each entity, provide: name, type (PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, etc.), description
3. Extract relationships between entities as triplets (entity1, relationship, entity2)
4. Focus on meaningful, factual relationships
5. Return the result as valid JSON

OUTPUT FORMAT:
{{
    "entities": [
        {{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}},
        ...
    ],
    "relationships": [
        {{"source": "Entity1", "relationship": "relationship_type", "target": "Entity2", "description": "relationship description"}},
        ...
    ]
}}

Respond with only the JSON, no other text.
"""

    # FIXED: Use the correct Gemini API endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        data = {
            "contents": [{
                "parts": [{"text": extraction_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Lower temperature for more consistent JSON
                "maxOutputTokens": 1024,  # Reduced for more reliable responses
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    generated_text = candidate["content"]["parts"][0]["text"]
                    
                    # Clean up the JSON - remove markdown formatting if present
                    json_text = generated_text.strip()
                    if json_text.startswith("```json"):
                        json_text = json_text[7:]
                    elif json_text.startswith("```"):
                        json_text = json_text[3:]
                    if json_text.endswith("```"):
                        json_text = json_text[:-3]
                    json_text = json_text.strip()
                    
                    # Parse the JSON
                    try:
                        graph_data = json.loads(json_text)
                        
                        # Validate the structure
                        if not isinstance(graph_data, dict):
                            raise ValueError("Response is not a JSON object")
                        
                        # Ensure required keys exist
                        if "entities" not in graph_data:
                            graph_data["entities"] = []
                        if "relationships" not in graph_data:
                            graph_data["relationships"] = []
                        
                        # Add chunk metadata to entities and relationships
                        if chunk_id:
                            for entity in graph_data.get("entities", []):
                                entity["source_chunk"] = chunk_id
                            for rel in graph_data.get("relationships", []):
                                rel["source_chunk"] = chunk_id
                        
                        logger.info(f"Successfully extracted {len(graph_data['entities'])} entities and {len(graph_data['relationships'])} relationships")
                        return graph_data
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from Gemini response: {e}")
                        logger.debug(f"Raw response: {json_text}")
                        return {"entities": [], "relationships": []}
                else:
                    logger.warning("No content in Gemini candidate")
                    return {"entities": [], "relationships": []}
            else:
                logger.warning("No candidates in Gemini response")
                return {"entities": [], "relationships": []}
                
        except requests.exceptions.Timeout:
            logger.error("Timeout when calling Gemini API")
            return {"entities": [], "relationships": []}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when calling Gemini API: {str(e)}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {str(e)}")
            return {"entities": [], "relationships": []}

class GraphRAGService:
    """Manages the knowledge graph and provides graph-based retrieval."""
    
    def __init__(self, config):
        self.config = config
        self.graph = nx.Graph()
        self.entity_embeddings = {}
        self.relationship_embeddings = {}
        self.embedding_service = None  # Will be set by SimpleRAG
        self.vector_db_service = None  # Will be set by SimpleRAG
        self.graph_extractor = GraphExtractor(config)
        
    def set_services(self, embedding_service, vector_db_service):
        """Set the embedding and vector DB services."""
        self.embedding_service = embedding_service
        self.vector_db_service = vector_db_service
    
    def process_document_for_graph(self, chunks: List[Dict[str, Any]], progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
        """Process document chunks to extract graph structure."""
        all_entities = []
        all_relationships = []
        
        total_chunks = len(chunks)
        if progress_tracker:
            progress_tracker.update(0, total_chunks, status="graph_extraction", 
                                   message="Extracting entities and relationships")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}_{int(time.time())}"
            
            # Extract entities and relationships from this chunk
            graph_data = self.graph_extractor.extract_entities_and_relationships(
                chunk["text"], chunk_id
            )
            
            # Add source text to entities and relationships
            for entity in graph_data.get("entities", []):
                entity["source_text"] = chunk["text"][:200] + "..."
                entity["metadata"] = chunk["metadata"]
            
            for rel in graph_data.get("relationships", []):
                rel["source_text"] = chunk["text"][:200] + "..."
                rel["metadata"] = chunk["metadata"]
            
            all_entities.extend(graph_data.get("entities", []))
            all_relationships.extend(graph_data.get("relationships", []))
            
            if progress_tracker:
                progress_tracker.update(i + 1, total_chunks, 
                                       message=f"Processed chunk {i + 1} of {total_chunks}")
        
        # Merge similar entities
        merged_entities = self._merge_similar_entities(all_entities)
        
        # Build graph
        self._build_graph(merged_entities, all_relationships)
        
        # Generate embeddings for entities and relationships
        self._generate_graph_embeddings(merged_entities, all_relationships, progress_tracker)
        
        return {
            "entities": merged_entities,
            "relationships": all_relationships,
            "graph_stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            }
        }
    
    def _merge_similar_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge entities that are likely referring to the same thing."""
        merged = []
        entity_groups = defaultdict(list)
        
        # Group entities by type and similar names
        for entity in entities:
            # Create a normalized key for grouping
            name_key = entity["name"].lower().strip()
            type_key = entity["type"]
            group_key = f"{type_key}:{name_key}"
            entity_groups[group_key].append(entity)
        
        # Merge entities in each group
        for group_key, group_entities in entity_groups.items():
            if len(group_entities) == 1:
                merged.extend(group_entities)
            else:
                # Merge multiple entities
                primary_entity = group_entities[0].copy()
                
                # Combine descriptions
                descriptions = [e.get("description", "") for e in group_entities]
                primary_entity["description"] = " | ".join(filter(None, descriptions))
                
                # Combine source chunks
                source_chunks = [e.get("source_chunk", "") for e in group_entities]
                primary_entity["source_chunks"] = list(filter(None, source_chunks))
                
                # Combine source texts
                source_texts = [e.get("source_text", "") for e in group_entities]
                primary_entity["source_texts"] = list(filter(None, source_texts))
                
                merged.append(primary_entity)
        
        return merged
    
    def _build_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """Build NetworkX graph from entities and relationships."""
        # Add nodes (entities)
        for entity in entities:
            self.graph.add_node(
                entity["name"],
                type=entity["type"],
                description=entity.get("description", ""),
                source_chunks=entity.get("source_chunks", []),
                source_texts=entity.get("source_texts", [])
            )
        
        # Add edges (relationships)
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            
            # Only add edge if both entities exist in the graph
            if source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(
                    source,
                    target,
                    relationship=rel["relationship"],
                    description=rel.get("description", ""),
                    source_chunk=rel.get("source_chunk", ""),
                    source_text=rel.get("source_text", "")
                )
    
    def _generate_graph_embeddings(self, entities: List[Dict[str, Any]], 
                                relationships: List[Dict[str, Any]], 
                                progress_tracker: Optional[ProgressTracker] = None):
            """Generate embeddings for entities and relationships and store in vector DB."""
            if not self.embedding_service or not self.vector_db_service:
                logger.warning("Embedding or Vector DB service not available for graph embeddings")
                return

            # Prepare entity documents for vector storage
            entity_docs = []
            entity_embeddings = []
            
            total_items = len(entities) + len(relationships)
            current_item = 0
            
            if progress_tracker:
                progress_tracker.update(0, total_items, status="graph_embedding", 
                                    message="Generating graph embeddings")
            
            # Process entities with error handling
            for entity in entities:
                try:
                    # Create a rich text representation for embedding
                    entity_text = f"Entity: {entity['name']} (Type: {entity['type']}) - {entity.get('description', '')}"
                    
                    # Generate embedding with retry logic
                    max_retries = 3
                    embedding = None
                    for attempt in range(max_retries):
                        try:
                            embedding = self.embedding_service.get_embedding(entity_text)
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt + 1} failed for entity embedding: {e}")
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to generate embedding for entity: {entity['name']}")
                                continue
                            time.sleep(1)  # Wait before retry
                    
                    if embedding is None:
                        continue  # Skip this entity if we can't get embedding
                    
                    # Prepare document for vector storage
                    entity_doc = {
                        "text": entity_text,
                        "metadata": {
                            "type": "entity",
                            "entity_name": entity["name"],
                            "entity_type": entity["type"],
                            "description": entity.get("description", ""),
                            "source_chunks": entity.get("source_chunks", []),
                            "graph_element": True
                        }
                    }
                    
                    entity_docs.append(entity_doc)
                    entity_embeddings.append(embedding)
                    
                except Exception as e:
                    logger.error(f"Error processing entity {entity.get('name', 'unknown')}: {e}")
                    continue
                
                current_item += 1
                if progress_tracker:
                    progress_tracker.update(current_item, total_items, 
                                        message=f"Processed entity {current_item} of {total_items}")
            
            # Process relationships with error handling
            for rel in relationships:
                try:
                    # Create a rich text representation for embedding
                    rel_text = f"Relationship: {rel['source']} {rel['relationship']} {rel['target']} - {rel.get('description', '')}"
                    
                    # Generate embedding with retry logic
                    max_retries = 3
                    embedding = None
                    for attempt in range(max_retries):
                        try:
                            embedding = self.embedding_service.get_embedding(rel_text)
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt + 1} failed for relationship embedding: {e}")
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to generate embedding for relationship: {rel['source']} -> {rel['target']}")
                                continue
                            time.sleep(1)  # Wait before retry
                    
                    if embedding is None:
                        continue  # Skip this relationship if we can't get embedding
                    
                    # Prepare document for vector storage
                    rel_doc = {
                        "text": rel_text,
                        "metadata": {
                            "type": "relationship",
                            "source": rel["source"],
                            "target": rel["target"],
                            "relationship": rel["relationship"],
                            "description": rel.get("description", ""),
                            "source_chunk": rel.get("source_chunk", ""),
                            "graph_element": True
                        }
                    }
                    
                    entity_docs.append(rel_doc)
                    entity_embeddings.append(embedding)
                    
                except Exception as e:
                    logger.error(f"Error processing relationship {rel.get('source', 'unknown')} -> {rel.get('target', 'unknown')}: {e}")
                    continue
                
                current_item += 1
                if progress_tracker:
                    progress_tracker.update(current_item, total_items, 
                                        message=f"Processed relationship {current_item} of {total_items}")
            
            # Store in vector database (graph collection) only if we have data
            if entity_docs and entity_embeddings:
                self._store_graph_in_vector_db(entity_docs, entity_embeddings, progress_tracker)
            else:
                logger.warning("No graph elements to store in vector database")
    
    def _store_graph_in_vector_db(self, docs: List[Dict[str, Any]], 
                             embeddings: List[List[float]], 
                             progress_tracker: Optional[ProgressTracker] = None):
        """Store graph elements in a separate vector collection."""
        # Create/ensure graph collection exists
        collection_name = self.config["graph_collection_name"]
        
        try:
            # Create collection with same configuration as main collection
            self.vector_db_service.ensure_collection_exists(collection_name)
            
            # Prepare points for insertion
            points = []
            for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
                points.append(
                    models.PointStruct(
                        id=i + int(time.time() * 1000),
                        vector=embedding,
                        payload={
                            "text": doc["text"],
                            "metadata": doc["metadata"]
                        }
                    )
                )
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                try:
                    self.vector_db_service.client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    
                    if progress_tracker:
                        progress = min(i + batch_size, len(points))
                        progress_tracker.update(progress, len(points), 
                                            message=f"Stored graph batch {progress} of {len(points)}")
                        
                    logger.info(f"Inserted graph batch of {len(batch)} elements")
                except Exception as e:
                    logger.error(f"Error inserting graph batch {i}: {str(e)}")
                    raise
            
            logger.info(f"Successfully stored {len(docs)} graph elements in vector DB")
            
        except Exception as e:
            logger.error(f"Error storing graph in vector DB: {str(e)}")
            raise
    
    def search_graph(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph using semantic similarity."""
        if not self.embedding_service:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.get_embedding(query)
            
            # Search in graph collection
            collection_name = self.config["graph_collection_name"]
            
            search_result = self.vector_db_service.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            results = []
            for scored_point in search_result:
                results.append({
                    "text": scored_point.payload["text"],
                    "metadata": scored_point.payload["metadata"],
                    "score": scored_point.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching graph: {str(e)}")
            return []
    
    def get_entity_neighborhood(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get the neighborhood of an entity in the graph."""
        if entity_name not in self.graph.nodes:
            return {"entities": [], "relationships": []}
        
        # Get subgraph within specified depth
        subgraph_nodes = set([entity_name])
        current_nodes = {entity_name}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(self.graph.neighbors(node))
                next_nodes.update(neighbors)
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        # Extract subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Format entities and relationships
        entities = []
        for node in subgraph.nodes():
            node_data = self.graph.nodes[node]
            entities.append({
                "name": node,
                "type": node_data.get("type", ""),
                "description": node_data.get("description", "")
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
            "center_entity": entity_name
        }

class EnhancedDocumentProcessor:
    """Enhanced document processor that supports both normal and graph RAG modes."""
    
    def __init__(self, config):
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.rag_mode = config.get("rag_mode", "normal")
    
    def extract_text_from_file(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from various file formats."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if progress_tracker:
            progress_tracker.update(0, 100, status="extracting", 
                                   message=f"Extracting text from {path.name}",
                                   current_file=path.name)
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path, progress_tracker)
        elif extension == '.txt':
            return self._extract_from_txt(file_path, progress_tracker)
        elif extension == '.docx':
            return self._extract_from_docx(file_path, progress_tracker)
        elif extension == '.html' or extension == '.htm':
            return self._extract_from_html(file_path, progress_tracker)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_from_pdf(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text += page.extract_text() + "\n"
                
                if progress_tracker and total_pages > 0:
                    progress_tracker.update(i + 1, total_pages, 
                                           message=f"Extracting page {i + 1} of {total_pages}")
        return text
    
    def _extract_from_txt(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
            if progress_tracker:
                progress_tracker.update(1, 1, status="complete", 
                                       message="Text file extraction complete")
            return text
    
    def _extract_from_docx(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        paragraphs = doc.paragraphs
        total_paragraphs = len(paragraphs)
        
        text_parts = []
        for i, paragraph in enumerate(paragraphs):
            text_parts.append(paragraph.text)
            
            if progress_tracker and total_paragraphs > 0:
                progress_tracker.update(i + 1, total_paragraphs, 
                                       message=f"Extracting paragraph {i + 1} of {total_paragraphs}")
        
        return "\n".join(text_parts)
    
    def _extract_from_html(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            html_content = file.read()
            
            if progress_tracker:
                progress_tracker.update(1, 2, message="Parsing HTML content")
                
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
                
            if progress_tracker:
                progress_tracker.update(2, 2, message="Extracting text content")
                
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    
    def chunk_text(self, text: str, metadata: dict, progress_tracker: Optional[ProgressTracker] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for embedding."""
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        if progress_tracker:
            progress_tracker.update(0, 100, status="chunking", 
                                   message="Splitting text into chunks")
        
        # For graph RAG, use slightly larger chunks to capture more context
        chunk_size = self.chunk_size
        if self.rag_mode == "graph":
            chunk_size = int(self.chunk_size * 1.5)  # 50% larger chunks for graph mode
        
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        total_sentences = len(sentences)
        
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_text"] = current_chunk[:100] + "..."
                chunk_metadata["rag_mode"] = self.rag_mode
                chunks.append({
                    "text": current_chunk,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_word_count = min(len(words), self.chunk_overlap // 10)
                current_chunk = " ".join(words[-overlap_word_count:]) + " " + sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    
            if progress_tracker and total_sentences > 0:
                progress_tracker.update(i + 1, total_sentences, 
                                       message=f"Processing sentence {i + 1} of {total_sentences}")
        
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_text"] = current_chunk[:100] + "..."
            chunk_metadata["rag_mode"] = self.rag_mode
            chunks.append({
                "text": current_chunk,
                "metadata": chunk_metadata
            })
        
        if progress_tracker:
            progress_tracker.update(total_sentences, total_sentences, status="chunking_complete",
                                   message=f"Created {len(chunks)} chunks from document")
        
        logger.info(f"Created {len(chunks)} chunks from document (mode: {self.rag_mode})")
        return chunks

class EmbeddingService:
    """Handles creation of embeddings using Gemini API."""
    
    def __init__(self, config):
        self.api_key = config["gemini_api_key"]
        self.embedding_dimension = config["embedding_dimension"]
        
        self.rate_limiter = RateLimiter(calls_per_minute=config.get("rate_limit", 60))
        
        self.enable_cache = config.get("enable_cache", True)
        if self.enable_cache:
            self.cache = EmbeddingCache(cache_dir=config.get("cache_dir"))
    
    @rate_limited(RateLimiter(calls_per_minute=60))
    def _get_embedding_from_api(self, text: str) -> List[float]:
        """Generate embedding vector for text using Gemini API."""
        # FIXED: Use the correct embedding endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        data = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text[:8000]}]},  # Limit text length
            "taskType": "RETRIEVAL_DOCUMENT"
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            embedding = result.get("embedding", {}).get("values", [])
            
            if not embedding:
                raise ValueError("No embedding values returned from API")
                
            return embedding
        except requests.exceptions.Timeout:
            logger.error("Timeout when calling Gemini embedding API")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when calling Gemini embedding API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if enabled."""
        if self.enable_cache and hasattr(self, 'cache'):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                return cached_embedding
        
        embedding = self._get_embedding_from_api(text)
        
        if self.enable_cache and hasattr(self, 'cache'):
            self.cache.set(text, embedding)
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], progress_tracker: Optional[ProgressTracker] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts with progress tracking."""
        embeddings = []
        total_texts = len(texts)
        
        if progress_tracker:
            progress_tracker.update(0, total_texts, status="embedding", 
                                   message="Generating embeddings")
        
        for i, text in enumerate(texts):
            try:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                
                if progress_tracker:
                    progress_tracker.update(i + 1, total_texts, 
                                           message=f"Generated embedding {i + 1} of {total_texts}")
                
                # Add a small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                # Create a zero vector as fallback
                embeddings.append([0.0] * self.embedding_dimension)
                
                if progress_tracker:
                    progress_tracker.update(i + 1, total_texts, 
                                           message=f"Error with embedding {i + 1}: {str(e)}")
        
        return embeddings

# Replace the VectorDBService class in simplerag.py with this fixed version

class VectorDBService:
    """Handles interactions with Qdrant vector database."""
    
    def __init__(self, config):
        self.qdrant_url = config["qdrant_url"]
        self.qdrant_api_key = config["qdrant_api_key"]
        self.collection_name = config["collection_name"]
        self.graph_collection_name = config["graph_collection_name"]
        self.embedding_dimension = config["embedding_dimension"]
        
        # Initialize client with proper error handling
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client with comprehensive error handling."""
        try:
            # Validate configuration
            if not self.qdrant_url:
                raise ValueError("Qdrant URL not configured")
            if not self.qdrant_api_key:
                raise ValueError("Qdrant API key not configured")
            
            logger.info(f"Initializing Qdrant client with URL: {self.qdrant_url}")
            
            # For Qdrant Cloud
            if "cloud.qdrant.io" in self.qdrant_url:
                # Extract host from URL
                host = self.qdrant_url.replace("https://", "").replace("http://", "").split(':')[0]
                
                logger.info(f"Using Qdrant Cloud host: {host}")
                
                self.client = qdrant_client.QdrantClient(
                    host=host,
                    api_key=self.qdrant_api_key,
                    timeout=60,
                    https=True,
                    port=443
                )
            else:
                # For self-hosted Qdrant
                logger.info(f"Using self-hosted Qdrant: {self.qdrant_url}")
                self.client = qdrant_client.QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=60
                )
            
            # Test the connection
            self._test_connection()
            logger.info("Qdrant client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            self.client = None
            # Don't raise the error, let the app continue but mark as unavailable
    
    def _test_connection(self):
        """Test the Qdrant connection."""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        try:
            # Simple test - get collections
            collections = self.client.get_collections()
            logger.info(f"Connection test successful. Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise RuntimeError(f"Qdrant connection test failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the vector database service is available."""
        if not self.client:
            return False
        
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def ensure_collection_exists(self, collection_name: str = None):
        """Create collection if it doesn't exist."""
        if not self.client:
            raise RuntimeError("Qdrant client not available")
        
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Successfully created collection: {collection_name}")
                return True
            else:
                logger.info(f"Collection already exists: {collection_name}")
            return False
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def insert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], 
                         progress_tracker: Optional[ProgressTracker] = None, 
                         collection_name: str = None):
        """Insert documents with their embeddings into Qdrant."""
        if not self.client:
            raise RuntimeError("Qdrant client not available")
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        if collection_name is None:
            collection_name = self.collection_name
        
        if progress_tracker:
            progress_tracker.update(0, len(documents), status="storing", 
                                   message="Storing documents in vector database")
        
        try:
            self.ensure_collection_exists(collection_name)
            
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append(
                    models.PointStruct(
                        id=i + int(time.time() * 1000),
                        vector=embedding,
                        payload={
                            "text": doc["text"],
                            "metadata": doc["metadata"]
                        }
                    )
                )
            
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    
                    if progress_tracker:
                        progress = min(i + batch_size, len(points))
                        progress_tracker.update(progress, len(points), 
                                               message=f"Stored {progress} of {len(points)} chunks")
                    
                    logger.info(f"Inserted batch of {len(batch)} documents into {collection_name}")
                except Exception as e:
                    logger.error(f"Error inserting batch {i}: {str(e)}")
                    raise
            
            if progress_tracker:
                progress_tracker.update(len(documents), len(documents), status="complete", 
                                       message=f"Successfully stored {len(documents)} chunks")
            
            logger.info(f"Successfully inserted {len(documents)} documents into {collection_name}")
            
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            if progress_tracker:
                progress_tracker.update(len(documents), len(documents), status="error", 
                                       message=f"Error storing documents: {str(e)}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      filter_condition=None, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        if not self.client:
            raise RuntimeError("Qdrant client not available")
        
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            self.ensure_collection_exists(collection_name)
            
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_embedding,
                "limit": top_k
            }
            
            if filter_condition:
                search_params["query_filter"] = filter_condition
            
            search_result = self.client.search(**search_params)
            
            results = []
            for scored_point in search_result:
                results.append({
                    "text": scored_point.payload["text"],
                    "metadata": scored_point.payload["metadata"],
                    "score": scored_point.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {str(e)}")
            return []

class LLMService:
    """Handles interactions with LLM APIs (Claude)."""
    
    def __init__(self, config):
        self.preferred_llm = config["preferred_llm"]
        self.claude_api_key = config.get("claude_api_key", "")
        self.rag_mode = config.get("rag_mode", "normal")
        
        if self.preferred_llm == "claude" and self.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
            except TypeError:
                import httpx
                http_client = httpx.Client()
                self.claude_client = anthropic.Anthropic(
                    api_key=self.claude_api_key,
                    http_client=http_client
                )
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]], 
                       graph_context: Dict[str, Any] = None, 
                       progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate an answer based on the query and retrieved contexts."""
        if progress_tracker:
            progress_tracker.update(70, 100, status="generating", 
                                  message="Generating answer with LLM")
        
        if self.rag_mode == "graph" and graph_context:
            return self._generate_graph_rag_answer(query, contexts, graph_context, progress_tracker)
        else:
            return self._generate_normal_rag_answer(query, contexts, progress_tracker)
    
    def _generate_normal_rag_answer(self, query: str, contexts: List[Dict[str, Any]], 
                                   progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using traditional RAG approach."""
        context_text = "\n\n---\n\n".join([
            f"Document: {ctx['metadata'].get('filename', 'Unknown')}\n{ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""
You are a helpful assistant that answers questions based on the provided document context.

CONTEXT:
{context_text}

USER QUESTION:
{query}

Please answer the question based only on the provided context. If the answer cannot be determined from the context, please state that clearly. Include relevant citations to the documents when possible.

ANSWER:
"""
        
        return self._generate_with_llm(prompt, progress_tracker)
    
    def _generate_graph_rag_answer(self, query: str, contexts: List[Dict[str, Any]], 
                                  graph_context: Dict[str, Any], 
                                  progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Graph RAG approach with entity and relationship context."""
        
        # Prepare document context
        document_context = "\n\n---\n\n".join([
            f"Document: {ctx['metadata'].get('filename', 'Unknown')}\n{ctx['text']}"
            for ctx in contexts if ctx['metadata'].get('type') != 'entity' and ctx['metadata'].get('type') != 'relationship'
        ])
        
        # Prepare graph context
        entities_context = ""
        relationships_context = ""
        
        if graph_context.get('entities'):
            entities_list = []
            for ctx in contexts:
                if ctx['metadata'].get('type') == 'entity':
                    entities_list.append(f"- {ctx['text']}")
            if entities_list:
                entities_context = f"RELEVANT ENTITIES:\n" + "\n".join(entities_list)
        
        if graph_context.get('relationships'):
            relationships_list = []
            for ctx in contexts:
                if ctx['metadata'].get('type') == 'relationship':
                    relationships_list.append(f"- {ctx['text']}")
            if relationships_list:
                relationships_context = f"RELEVANT RELATIONSHIPS:\n" + "\n".join(relationships_list)
        
        prompt = f"""
You are an advanced AI assistant that answers questions using both document content and knowledge graph information (entities and relationships).

DOCUMENT CONTEXT:
{document_context}

{entities_context}

{relationships_context}

USER QUESTION:
{query}

Please provide a comprehensive answer that:
1. Uses information from both the document context and the knowledge graph
2. Explains how entities and relationships are relevant to the question
3. Provides citations to the source documents
4. If the answer cannot be fully determined, clearly state what information is missing

ANSWER:
"""
        
        return self._generate_with_llm(prompt, progress_tracker)
    
    def _generate_with_llm(self, prompt: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using the configured LLM."""
        if self.preferred_llm == "claude" and hasattr(self, "claude_client"):
            return self._generate_with_claude(prompt, progress_tracker)
        elif self.preferred_llm == "raw":
            return self._generate_raw_response(prompt)
        else:
            raise ValueError(f"LLM {self.preferred_llm} not properly configured")
    
    def _generate_with_claude(self, prompt: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Claude API."""
        try:
            if progress_tracker:
                progress_tracker.update(80, 100, status="querying", 
                                      message="Querying Claude API")
                
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            if progress_tracker:
                progress_tracker.update(95, 100, status="formatting", 
                                      message="Response received, formatting answer")
                
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response with Claude: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _generate_raw_response(self, prompt: str) -> str:
        """Generate a raw response without LLM processing."""
        return f"Raw mode - Context provided in prompt:\n\n{prompt}"

class EnhancedSimpleRAG:
    """Enhanced SimpleRAG with both Normal and Graph RAG capabilities."""
    
   # Replace the __init__ method in EnhancedSimpleRAG class with this fixed version

    def __init__(self):
        self.config = load_config()
        self.rag_mode = self.config.get("rag_mode", "normal")
        
        # Initialize services with comprehensive error handling
        self.document_processor = None
        self.embedding_service = None
        self.vector_db_service = None
        self.llm_service = None
        self.graph_rag_service = None
        
        # Track initialization errors
        self.initialization_errors = []
        
        try:
            # Initialize document processor (should always work)
            self.document_processor = EnhancedDocumentProcessor(self.config)
            logger.info("Document processor initialized")
        except Exception as e:
            error_msg = f"Failed to initialize document processor: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        try:
            # Initialize embedding service
            if not self.config.get("gemini_api_key"):
                raise ValueError("Gemini API key not configured")
            
            self.embedding_service = EmbeddingService(self.config)
            logger.info("Embedding service initialized")
        except Exception as e:
            error_msg = f"Failed to initialize embedding service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        try:
            # Initialize vector database service
            if not self.config.get("qdrant_url") or not self.config.get("qdrant_api_key"):
                raise ValueError("Qdrant URL or API key not configured")
            
            self.vector_db_service = VectorDBService(self.config)
            
            # Check if it's actually available
            if not self.vector_db_service.is_available():
                raise RuntimeError("Qdrant service not available - check connection")
            
            logger.info("Vector database service initialized")
        except Exception as e:
            error_msg = f"Failed to initialize vector database service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            self.vector_db_service = None
        
        try:
            # Initialize LLM service
            self.llm_service = LLMService(self.config)
            logger.info("LLM service initialized")
        except Exception as e:
            error_msg = f"Failed to initialize LLM service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        try:
            # Initialize Graph RAG service only if other services are available
            if self.embedding_service and self.vector_db_service:
                self.graph_rag_service = GraphRAGService(self.config)
                self.graph_rag_service.set_services(self.embedding_service, self.vector_db_service)
                logger.info("Graph RAG service initialized")
            else:
                logger.warning("Skipping Graph RAG service initialization - dependencies not available")
        except Exception as e:
            error_msg = f"Failed to initialize Graph RAG service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        # Log final status
        if self.initialization_errors:
            logger.warning(f"SimpleRAG initialized with {len(self.initialization_errors)} errors")
            for error in self.initialization_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("Enhanced SimpleRAG initialized successfully")

    def is_ready(self) -> bool:
        """Check if SimpleRAG is ready for use."""
        return (self.embedding_service is not None and 
                self.vector_db_service is not None and 
                self.vector_db_service.is_available())

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all services."""
        return {
            "ready": self.is_ready(),
            "services": {
                "document_processor": self.document_processor is not None,
                "embedding_service": self.embedding_service is not None,
                "vector_db_service": self.vector_db_service is not None and self.vector_db_service.is_available(),
                "llm_service": self.llm_service is not None,
                "graph_rag_service": self.graph_rag_service is not None
            },
            "errors": self.initialization_errors,
            "rag_mode": self.rag_mode
        }
    
    def set_rag_mode(self, mode: str):
        """Switch between 'normal' and 'graph' RAG modes."""
        if mode not in ["normal", "graph"]:
            raise ValueError("RAG mode must be 'normal' or 'graph'")
        
        self.rag_mode = mode
        self.config["rag_mode"] = mode
        
        # Update processor mode
        self.document_processor.rag_mode = mode
        self.llm_service.rag_mode = mode
        
        logger.info(f"RAG mode set to: {mode}")
    
    def index_document(self, file_path: str, session_id=None) -> bool:
        """Process and index a document using the configured RAG mode."""
        try:
            progress_tracker = None
            if session_id:
                progress_tracker = ProgressTracker(session_id, "index_document")
                progress_tracker.update(0, 100, status="starting", 
                                      message=f"Starting document indexing in {self.rag_mode} mode")
            
            # Extract text from document
            logger.info(f"Extracting text from {file_path}")
            text = self.document_processor.extract_text_from_file(file_path, progress_tracker)
            
            # Create metadata
            filename = os.path.basename(file_path)
            metadata = {
                "filename": filename,
                "path": file_path,
                "created_at": time.time(),
                "file_type": os.path.splitext(filename)[1][1:].lower(),
                "rag_mode": self.rag_mode
            }
            
            if session_id:
                metadata["session_id"] = session_id
            
            # Chunk the document
            chunks = self.document_processor.chunk_text(text, metadata, progress_tracker)
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                if progress_tracker:
                    progress_tracker.update(100, 100, status="error", 
                                          message="No content could be extracted from document")
                return False
            
            # Process based on RAG mode
            if self.rag_mode == "graph":
                return self._index_document_graph_mode(chunks, progress_tracker)
            else:
                return self._index_document_normal_mode(chunks, progress_tracker)
                
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            if progress_tracker:
                progress_tracker.update(0, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return False
    
    def _index_document_normal_mode(self, chunks: List[Dict[str, Any]], 
                               progress_tracker: Optional[ProgressTracker] = None) -> bool:
        """Index document using normal RAG mode."""
        if progress_tracker:
            progress_tracker.update(30, 100, status="embedding", 
                                message="Generating embeddings for chunks")
        
        # Generate embeddings for chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        try:
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts, progress_tracker)
            
            # Verify we have the same number of embeddings as chunks
            if len(embeddings) != len(chunks):
                logger.error(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
                return False
            
            # Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in vector database")
            self.vector_db_service.insert_documents(chunks, embeddings, progress_tracker)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message="Document indexed successfully in normal mode")
            
            return True
        
        except Exception as e:
            logger.error(f"Error in normal mode indexing: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                    message=f"Error: {str(e)}")
            return False
    
    # Fix 1: Update _index_document_graph_mode in simplerag.py

    def _index_document_graph_mode(self, chunks: List[Dict[str, Any]], 
                            progress_tracker: Optional[ProgressTracker] = None) -> bool:
        """Index document using graph RAG mode with proper error handling."""
        try:
            # First, do normal chunking and embedding
            if progress_tracker:
                progress_tracker.update(20, 100, status="embedding", 
                                    message="Generating embeddings for chunks")
            
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts)
            
            # Verify we have the same number of embeddings as chunks
            if len(embeddings) != len(chunks):
                logger.error(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
                return False
            
            # Store chunks in normal collection EXPLICITLY
            logger.info(f"Storing {len(chunks)} chunks in normal collection: {self.config['collection_name']}")
            self.vector_db_service.insert_documents(
                chunks, 
                embeddings, 
                progress_tracker=progress_tracker,
                collection_name=self.config["collection_name"]  # EXPLICIT collection name
            )
            
            # Verify chunks were stored
            try:
                test_search = self.vector_db_service.search_similar(
                    embeddings[0], 
                    top_k=1, 
                    collection_name=self.config["collection_name"]
                )
                if not test_search:
                    logger.error("Failed to verify chunk storage in normal collection")
                    return False
                else:
                    logger.info(f"Verified {len(test_search)} chunks stored in normal collection")
            except Exception as e:
                logger.error(f"Error verifying chunk storage: {e}")
                return False
            
            # Extract graph structure
            if progress_tracker:
                progress_tracker.update(50, 100, status="graph_processing", 
                                    message="Extracting knowledge graph")
            
            graph_data = self.graph_rag_service.process_document_for_graph(chunks, progress_tracker)
            
            # Verify graph storage
            try:
                if graph_data and graph_data.get('graph_stats', {}).get('nodes', 0) > 0:
                    # Test graph collection
                    query_embedding = self.embedding_service.get_embedding("test query")
                    graph_results = self.graph_rag_service.search_graph("test", top_k=1)
                    logger.info(f"Verified graph collection has {len(graph_results)} elements")
                else:
                    logger.warning("No graph data was extracted")
            except Exception as e:
                logger.error(f"Error verifying graph storage: {e}")
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message=f"Document indexed in graph mode - {graph_data['graph_stats']['nodes']} entities, {graph_data['graph_stats']['edges']} relationships")
            
            logger.info(f"Graph mode indexing complete: {graph_data['graph_stats']}")
            return True
            
        except Exception as e:
            logger.error(f"Error in graph mode indexing: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                    message=f"Error: {str(e)}")
            return False
    
    def query(self, question: str, session_id=None) -> str:
        """Query indexed documents using the configured RAG mode."""
        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting", 
                                  message=f"Processing query in {self.rag_mode} mode")
        
        try:
            if self.rag_mode == "graph":
                return self._query_graph_mode(question, progress_tracker)
            else:
                return self._query_normal_mode(question, progress_tracker)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return f"Error processing your query: {str(e)}"
    
    def _query_normal_mode(self, question: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Query using normal RAG mode."""
        # Generate embedding for the query
        if progress_tracker:
            progress_tracker.update(10, 100, status="embedding", 
                                  message="Generating query embedding")
            
        query_embedding = self.embedding_service.get_embedding(question)
        
        # Retrieve similar contexts
        if progress_tracker:
            progress_tracker.update(30, 100, status="searching", 
                                  message="Searching for relevant documents")
            
        contexts = self.vector_db_service.search_similar(
            query_embedding,
            top_k=self.config["top_k"]
        )
        
        if not contexts:
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message="No relevant information found")
            return "I couldn't find any relevant information to answer your question."
        
        # Generate answer
        if hasattr(self, "llm_service") and self.llm_service:
            answer = self.llm_service.generate_answer(question, contexts, progress_tracker=progress_tracker)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message="Answer generated successfully")
            return answer
        else:
            # Raw mode
            results = []
            for i, ctx in enumerate(contexts):
                results.append(f"--- Result {i+1} (Score: {ctx['score']:.2f}) ---\n")
                results.append(f"Source: {ctx['metadata'].get('filename', 'Unknown')}\n")
                results.append(f"{ctx['text']}\n\n")
            return "\n".join(results)
    
    
    def _query_graph_mode(self, question: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Query using graph RAG mode with enhanced debugging."""
        # Generate embedding for the query
        if progress_tracker:
            progress_tracker.update(10, 100, status="embedding", 
                                message="Generating query embedding")
            
        query_embedding = self.embedding_service.get_embedding(question)
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # Search both normal and graph collections with debugging
        if progress_tracker:
            progress_tracker.update(20, 100, status="searching", 
                                message="Searching documents and knowledge graph")
        
        # Get document contexts with explicit collection name
        logger.info(f"Searching normal collection: {self.config['collection_name']}")
        doc_contexts = self.vector_db_service.search_similar(
            query_embedding,
            top_k=self.config["top_k"] // 2,  # Use half for documents
            collection_name=self.config["collection_name"]
        )
        logger.info(f"Found {len(doc_contexts)} document contexts")
        
        # Debug: Log what we found in document search
        if doc_contexts:
            for i, ctx in enumerate(doc_contexts[:2]):  # Log first 2 results
                filename = ctx['metadata'].get('filename', 'Unknown')
                text_preview = ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
                logger.info(f"Doc result {i+1}: {filename} - {text_preview}")
        else:
            logger.warning("No document contexts found in normal collection")
        
        # Get graph contexts (entities and relationships)
        logger.info(f"Searching graph collection: {self.config['graph_collection_name']}")
        graph_contexts = self.graph_rag_service.search_graph(
            question,
            top_k=self.config["top_k"] // 2  # Use half for graph elements
        )
        logger.info(f"Found {len(graph_contexts)} graph contexts")
        
        # Debug: Log what we found in graph search
        if graph_contexts:
            for i, ctx in enumerate(graph_contexts[:2]):  # Log first 2 results
                element_type = ctx['metadata'].get('type', 'unknown')
                text_preview = ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
                logger.info(f"Graph result {i+1} ({element_type}): {text_preview}")
        else:
            logger.warning("No graph contexts found in graph collection")
        
        # Combine contexts
        all_contexts = doc_contexts + graph_contexts
        logger.info(f"Total contexts for answer generation: {len(all_contexts)}")
        
        if not all_contexts:
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message="No relevant information found")
            logger.warning("No contexts found in either collection")
            return "I couldn't find any relevant information to answer your question. This might indicate:\n1. No documents have been indexed yet\n2. The indexed documents don't contain relevant information\n3. There may be an issue with the vector database connection\n\nPlease check the admin panel to verify your collections contain data."
        
        # Prepare graph context summary
        graph_context = {
            "entities": [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'entity'],
            "relationships": [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'relationship']
        }
        
        logger.info(f"Graph context summary: {len(graph_context['entities'])} entities, {len(graph_context['relationships'])} relationships")
        
        # Generate answer using graph-enhanced prompt
        if hasattr(self, "llm_service") and self.llm_service:
            answer = self.llm_service.generate_answer(
                question, 
                all_contexts, 
                graph_context=graph_context,
                progress_tracker=progress_tracker
            )
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message="Graph RAG answer generated successfully")
            return answer
        else:
            # Raw mode for graph - show what we found for debugging
            results = ["=== GRAPH RAG DEBUG RESULTS ===\n"]
            
            results.append("DOCUMENT CONTEXTS:")
            if doc_contexts:
                for i, ctx in enumerate(doc_contexts):
                    results.append(f"--- Document {i+1} (Score: {ctx['score']:.2f}) ---")
                    results.append(f"Source: {ctx['metadata'].get('filename', 'Unknown')}")
                    results.append(f"Type: {ctx['metadata'].get('type', 'document_chunk')}")
                    results.append(f"{ctx['text'][:300]}...\n")
            else:
                results.append("No document contexts found!\n")
            
            results.append("\nGRAPH CONTEXTS:")
            if graph_contexts:
                for i, ctx in enumerate(graph_contexts):
                    ctx_type = ctx['metadata'].get('type', 'unknown')
                    results.append(f"--- {ctx_type.title()} {i+1} (Score: {ctx['score']:.2f}) ---")
                    results.append(f"{ctx['text']}\n")
            else:
                results.append("No graph contexts found!\n")
            
            # Add diagnostic information
            results.append("\n=== DIAGNOSTIC INFO ===")
            results.append(f"Query: {question}")
            results.append(f"Normal collection: {self.config['collection_name']}")
            results.append(f"Graph collection: {self.config['graph_collection_name']}")
            results.append(f"Total contexts found: {len(all_contexts)}")
            
            return "\n".join(results)

# For backward compatibility, alias the enhanced class
SimpleRAG = EnhancedSimpleRAG

def main():
    """Enhanced command line interface with Graph RAG support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced SimpleRAG with Graph RAG support")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Config command (enhanced)
    config_parser = subparsers.add_parser("config", help="Configure API keys and settings")
    config_parser.add_argument("--gemini-key", help="Set Gemini API key")
    config_parser.add_argument("--claude-key", help="Set Claude API key")
    config_parser.add_argument("--qdrant-key", help="Set Qdrant API key")
    config_parser.add_argument("--qdrant-url", help="Set Qdrant URL")
    config_parser.add_argument("--preferred-llm", choices=["claude", "raw"], help="Set preferred LLM")
    config_parser.add_argument("--rag-mode", choices=["normal", "graph"], help="Set RAG mode")
    config_parser.add_argument("--chunk-size", type=int, help="Set chunk size")
    config_parser.add_argument("--chunk-overlap", type=int, help="Set chunk overlap")
    config_parser.add_argument("--top-k", type=int, help="Set number of results to retrieve")
    
    # Index command (enhanced)
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", help="Path to document to index")
    index_parser.add_argument("--mode", choices=["normal", "graph"], help="RAG mode for this document")
    
    # Query command (enhanced)
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--mode", choices=["normal", "graph"], help="RAG mode for this query")
    
    # New: Mode command
    mode_parser = subparsers.add_parser("mode", help="Get or set RAG mode")
    mode_parser.add_argument("rag_mode", nargs="?", choices=["normal", "graph"], help="RAG mode to set")
    
    args = parser.parse_args()
    
    if args.command == "config":
        config = load_config()
        modified = False
        
        config_updates = {
            "gemini_api_key": args.gemini_key,
            "claude_api_key": args.claude_key, 
            "qdrant_api_key": args.qdrant_key,
            "qdrant_url": args.qdrant_url,
            "preferred_llm": args.preferred_llm,
            "rag_mode": args.rag_mode,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k
        }
        
        for key, value in config_updates.items():
            if value is not None:
                config[key] = value
                modified = True
        
        if modified:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print("Configuration updated successfully")
        else:
            print("Current configuration:")
            for key, value in config.items():
                if key.endswith("_api_key") and value:
                    print(f"{key}: {'*' * 10}")
                else:
                    print(f"{key}: {value}")
    
    elif args.command == "mode":
        simple_rag = SimpleRAG()
        
        if args.rag_mode:
            simple_rag.set_rag_mode(args.rag_mode)
            print(f"RAG mode set to: {args.rag_mode}")
        else:
            print(f"Current RAG mode: {simple_rag.rag_mode}")
    
    elif args.command == "index":
        simple_rag = SimpleRAG()
        
        if args.mode:
            simple_rag.set_rag_mode(args.mode)
        
        print(f"Indexing in {simple_rag.rag_mode} mode...")
        success = simple_rag.index_document(args.file_path)
        
        if success:
            print(f"Document indexed successfully: {args.file_path}")
        else:
            print(f"Failed to index document: {args.file_path}")
            sys.exit(1)
    
    elif args.command == "query":
        simple_rag = SimpleRAG()
        
        if args.mode:
            simple_rag.set_rag_mode(args.mode)
        
        print(f"Querying in {simple_rag.rag_mode} mode...")
        answer = simple_rag.query(args.question)
        print("\nAnswer:")
        print("-------")
        print(answer)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()