"""
SimpleRAGx - Main orchestrator class combining all services
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import all the modular services
from config import ConfigManager, get_config_manager
from embedding_service import EmbeddingService
from vector_db_service import VectorDBService
from graph_rag_service import GraphRAGService
from document_processor import DocumentProcessor
from llm_service import LLMService, _clean_source_name, GEMINI_MODEL
from extensions import ProgressTracker
from domain_config import get_domain
from metadata_extractor import MetadataExtractor
from reranker_service import RerankerService
from query_planner import QueryPlanner, rrf_merge
try:
    from agentic_service import AgenticRAGService
    _AGENTIC_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(f"AgenticRAGService unavailable (LangChain version mismatch): {_e}")
    AgenticRAGService = None
    _AGENTIC_AVAILABLE = False
from neo4j_service import Neo4jService
from pageindex_service import PageIndexService

logger = logging.getLogger(__name__)

class EnhancedSimpleRAG:
    """SimpleRAGx with both Normal and Graph RAG capabilities."""
    
    def __init__(self, config_manager: ConfigManager = None):
        """Initialize SimpleRAG with comprehensive error handling and service validation."""
        import os
        os.environ.update(os.environ)
        # Configuration
        self.config_manager = config_manager or get_config_manager()
        #self.config_manager._apply_env_overrides(self.config_manager.config)

        self.config = self.config_manager.get_all()
         # Log configuration status for debugging
        logger.info(f"Gemini API Key present: {'Yes' if self.config.get('gemini_api_key') else 'No'}")
        logger.info(f"Qdrant API Key present: {'Yes' if self.config.get('qdrant_api_key') else 'No'}")
        logger.info(f"Qdrant URL: {self.config.get('qdrant_url', 'Not set')}")
        
        # Current RAG mode
        self.rag_mode = self.config.get("rag_mode", "normal")

        # Domain configuration (plug-and-play)
        active_domain_name = self.config.get("active_domain", "vc_financial")
        self.active_domain = get_domain(active_domain_name)
        logger.info(f"Active domain: {active_domain_name} — {self.active_domain['name']}")

        # Service instances
        self.document_processor = None
        self.embedding_service = None
        self.vector_db_service = None
        self.llm_service = None
        self.graph_rag_service = None
        self.pageindex_service = None
        self.metadata_extractor = None
        self.reranker = None
        self.query_planner = None

        # Track initialization status
        self.initialization_errors = []
        self.initialization_warnings = []

        # Initialize all services
        self._initialize_services()
        self._initialize_neo4j_service()
        self._initialize_pageindex_service()

        # 6. Agentic AI Service (after all other services are ready)
        try:
            if self.is_ready() and AgenticRAGService is not None and self.config.get("enable_agentic_ai", True):
                self.agentic_service = AgenticRAGService(self.config, self)
                logger.info(" Agentic AI service initialized")
            else:
                self.agentic_service = None
                self.initialization_warnings.append("Agentic AI service skipped - basic services not ready")
        except Exception as e:
            error_msg = f"Failed to initialize Agentic AI service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            self.agentic_service = None
        
        # Log final status
        self._log_initialization_status()
    
    def _initialize_services(self):
        """Initialize all services with comprehensive error handling."""
        logger.info("Initializing SimpleRAGx services...")
        
        # 1. Document Processor (should always work)
        try:
            self.document_processor = DocumentProcessor(self.config)
            logger.info(". Document processor initialized")
        except Exception as e:
            error_msg = f"Failed to initialize document processor: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        # 2. Embedding Service - MAKE CONDITIONAL
        if self.config.get("gemini_api_key"):  # Only if key exists
            try:
                self.embedding_service = EmbeddingService(self.config)
                # Remove the test embedding call that fails without valid key
                logger.info(". Embedding service initialized")
            except Exception as e:
                error_msg = f"Failed to initialize embedding service: {str(e)}"
                logger.error(error_msg)
                self.initialization_errors.append(error_msg)
        else:
            logger.info("Ã¢Å¡Â  Embedding service skipped - Gemini API key not configured")
        
        # 3. Vector Database Service - MAKE CONDITIONAL AND NON-BLOCKING
        if self.config.get("qdrant_url") and self.config.get("qdrant_api_key"):
            try:
                self.vector_db_service = VectorDBService(self.config)
                
                # Don't fail if connection doesn't work immediately
                if self.vector_db_service.is_connected:
                    logger.info(". Vector database service initialized and connected")
                else:
                    logger.warning("Ã¢Å¡Â  Vector database service initialized but not connected")
                    self.initialization_warnings.append(f"Vector DB connection issue: {self.vector_db_service.last_error}")
                
            except Exception as e:
                error_msg = f"Failed to initialize vector database service: {str(e)}"
                logger.error(error_msg)
                self.initialization_warnings.append(error_msg)  # Change to warning instead of error
                self.vector_db_service = None
        else:
            logger.info("Ã¢Å¡Â  Vector database service skipped - credentials not configured")
        
        # 4. LLM Service
        try:
            self.llm_service = LLMService(self.config)
            if not self.llm_service.is_available():
                self.initialization_warnings.append("Gemini API key not configured — LLM answers unavailable")
            logger.info(". LLM service initialized")
        except Exception as e:
            error_msg = f"Failed to initialize LLM service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        # 5. Graph RAG Service (only if other services are available)
        try:
            if self.embedding_service and self.vector_db_service:
                self.graph_rag_service = GraphRAGService(self.config)
                self.graph_rag_service.set_services(self.embedding_service, self.vector_db_service)
                logger.info(". Graph RAG service initialized")
            else:
                self.initialization_warnings.append("Graph RAG service skipped - dependencies not available")
        except Exception as e:
            error_msg = f"Failed to initialize Graph RAG service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)

        # 6. Domain Metadata Extractor
        try:
            gemini_key = self.config.get("gemini_api_key", "")
            if gemini_key and self.config.get("enable_metadata_extraction", True):
                self.metadata_extractor = MetadataExtractor(gemini_key, self.active_domain)
                logger.info(f". Metadata extractor initialized for domain: {self.active_domain['name']}")
            else:
                logger.info("Metadata extractor skipped (no Gemini key or extraction disabled)")
        except Exception as e:
            logger.warning(f"Metadata extractor init failed (non-fatal): {e}")

        # 7. Reranker
        try:
            gemini_key = self.config.get("gemini_api_key", "")
            if gemini_key and self.config.get("enable_reranking", True):
                self.reranker = RerankerService(gemini_key)
                logger.info(". Reranker initialized (Gemini-based)")
            else:
                logger.info("Reranker skipped (no Gemini key or reranking disabled)")
        except Exception as e:
            logger.warning(f"Reranker init failed (non-fatal): {e}")

        # 8. Query Planner (decomposition + HyDE)
        try:
            gemini_key = self.config.get("gemini_api_key", "")
            if gemini_key and self.config.get("enable_query_planning", True):
                self.query_planner = QueryPlanner(gemini_key)
                logger.info(". Query planner initialized (decomposition + HyDE)")
            else:
                logger.info("Query planner skipped (no Gemini key or planning disabled)")
        except Exception as e:
            logger.warning(f"Query planner init failed (non-fatal): {e}")
    
    def is_agentic_ready(self) -> bool:
        """Check if Agentic AI functionality is ready."""
        return (self.agentic_service is not None and 
                self.agentic_service.is_available())

    def query_agentic(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """Query using the agentic approach with autonomous tool selection."""
        if not self.is_agentic_ready():
            return {
                "answer": "Agentic AI not available. Please check Claude API configuration.",
                "reasoning_steps": [],
                "tools_used": [],
                "success": False
            }
        
        return self.agentic_service.process_agentic_query(question, session_id)



    def _log_initialization_status(self):
        """Log the final initialization status."""
        if self.initialization_errors:
            logger.warning(f"SimpleRAG initialized with {len(self.initialization_errors)} errors")
            for error in self.initialization_errors:
                logger.warning(f"  ERROR: {error}")
        
        if self.initialization_warnings:
            logger.info(f"SimpleRAG has {len(self.initialization_warnings)} warnings")
            for warning in self.initialization_warnings:
                logger.info(f"  WARNING: {warning}")
        
        if not self.initialization_errors:
            logger.info(". SimpleRAGx initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if SimpleRAG is ready for basic operations with connection retry."""
        # Check basic service initialization
        if not (self.embedding_service and self.vector_db_service and self.document_processor):
            return False
        
        # Check vector DB availability - retry connection if needed
        if not self.vector_db_service.is_available():
            logger.warning("Vector DB not available, attempting to reconnect...")
            # Try to reconnect
            if self.vector_db_service.retry_connection():
                logger.info("Successfully reconnected to Vector DB")
                return True
            else:
                logger.error("Could not reconnect to Vector DB")
                return False
        
        return True
    
    def is_graph_ready(self) -> bool:
        """Check if Graph RAG functionality is ready."""
        return (self.is_ready() and 
                self.graph_rag_service is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all services."""
        status = {
            "ready": self.is_ready(),
            "graph_ready": self.is_graph_ready(),
            "agentic_ready": self.is_agentic_ready(),
            "rag_mode": self.rag_mode,
            "services": {
                "document_processor": self.document_processor is not None,
                "embedding_service": self.embedding_service is not None,
                "vector_db_service": self.vector_db_service is not None and self.vector_db_service.is_available(),
                "llm_service": self.llm_service is not None,
                "graph_rag_service": self.graph_rag_service is not None,
                "agentic_service": self.agentic_service is not None,
                "pageindex_service": self.is_pageindex_ready(),
            },
            "initialization_errors": self.initialization_errors,
            "initialization_warnings": self.initialization_warnings
        }
        
        # Add service-specific status
        if self.vector_db_service:
            status["vector_db_status"] = self.vector_db_service.get_status()
        
        if self.embedding_service:
            status["embedding_stats"] = self.embedding_service.get_embedding_stats()
        
        if self.llm_service:
            status["llm_stats"] = self.llm_service.get_usage_stats()
        
        if self.graph_rag_service:
            status["graph_stats"] = self.graph_rag_service.get_graph_stats()
        if self.agentic_service:
            status["agentic_stats"] = self.agentic_service.get_agentic_stats()
        return status
    
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a file for processing."""
        validation = {
            "valid": False,
            "file_exists": False,
            "supported_format": False,
            "estimated_processing": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation["errors"].append(f"File not found: {file_path}")
                return validation
            
            validation["file_exists"] = True
            
            # Check if format is supported
            if not self.document_processor.is_supported_file(file_path):
                supported_formats = ", ".join(self.document_processor.get_supported_formats().keys())
                validation["errors"].append(f"Unsupported file format. Supported: {supported_formats}")
                return validation
            
            validation["supported_format"] = True
            
            # Get processing estimates
            validation["estimated_processing"] = self.document_processor.estimate_processing_time(file_path)
            
            # Check file size warnings
            file_size_mb = validation["estimated_processing"].get("file_size_mb", 0)
            if file_size_mb > 50:
                validation["warnings"].append(f"Large file ({file_size_mb:.1f}MB) - processing may take longer")
            
            if self.rag_mode == "graph" and file_size_mb > 10:
                validation["warnings"].append("Graph RAG mode with large files requires significant processing time")
            
            validation["valid"] = True
            
        except Exception as e:
            validation["errors"].append(f"File validation error: {str(e)}")
        
        return validation
    
    def index_document(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None,
                       extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Index a document with OPTIMIZED parallel batch embedding (5x faster).

        extra_metadata — optional dict merged into every chunk's metadata after chunking.
        Useful for injecting corpus-level tags (title, source, benchmark_corpus, etc.)
        without requiring the document processor to understand those fields.
        """
        logger.info(f"Starting document indexing for: {file_path}")
        start_time = time.time()
        
        if progress_tracker:
            progress_tracker.update(0, 100, status="processing", 
                                message="Processing document")
        
        try:
            # 1. Validate file
            if progress_tracker:
                progress_tracker.update(5, 100, status="validating",
                                    message="Validating file")

            validation_result = self.validate_file(file_path)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid file: {validation_result.get('error', 'Unknown error')}")

            # 1b. Skip if already indexed (content hash check)
            import hashlib as _hl
            file_hash = _hl.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
            if self.vector_db_service:
                try:
                    hits, _ = self.vector_db_service.client.scroll(
                        collection_name=self.config["collection_name"],
                        scroll_filter={"must": [{"key": "metadata.doc_hash", "match": {"value": file_hash}}]},
                        limit=1,
                        with_payload=False,
                        with_vectors=False,
                    )
                    if hits:
                        logger.info(f"Doc already indexed (hash={file_hash}), skipping: {file_path}")
                        return {"success": True, "chunks_indexed": 0, "entities_extracted": 0, "skipped": True}
                except Exception:
                    pass  # collection may not exist yet — proceed normally
            
            # 2. Process document into chunks
            if progress_tracker:
                progress_tracker.update(10, 100, status="chunking", 
                                    message="Chunking document")
            
            chunks = self.document_processor.process_document(file_path)

            if not chunks:
                raise ValueError("No chunks generated from document")

            logger.info(f"Generated {len(chunks)} chunks from document")

            # 2a. Merge caller-supplied extra_metadata into every chunk
            for chunk in chunks:
                chunk.setdefault("metadata", {})["doc_hash"] = file_hash
            if extra_metadata:
                for chunk in chunks:
                    chunk["metadata"].update(extra_metadata)

            # 2b. Domain metadata extraction — one LLM call per document, merged into all chunks
            if self.metadata_extractor and chunks:
                if progress_tracker:
                    progress_tracker.update(15, 100, status="metadata",
                                        message="Extracting domain metadata")
                sample_text = " ".join(c['text'] for c in chunks[:8])
                domain_meta = self.metadata_extractor.extract(sample_text, Path(file_path).name)
                if domain_meta:
                    domain_meta["domain"] = self.config.get("active_domain", "vc_financial")
                    for chunk in chunks:
                        chunk['metadata'].update(domain_meta)
                    logger.info(f"Domain metadata merged into chunks: {list(domain_meta.keys())}")

            # 3. Generate embeddings in PARALLEL BATCHES (5x faster)
            if progress_tracker:
                progress_tracker.update(30, 100, status="embedding", 
                                    message=f"Generating embeddings for {len(chunks)} chunks (parallel)")
            
            # Ã¢Å“â€¦ SPEED OPTIMIZATION: Extract all texts for batch processing
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Ã¢Å“â€¦ Process in optimal batch sizes
            batch_size = 50  # Process 50 chunks at a time
            all_embeddings = []
            
            for batch_start in range(0, len(chunk_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_texts))
                batch_texts = chunk_texts[batch_start:batch_end]
                
                logger.info(f"Processing embedding batch {batch_start//batch_size + 1} "
                        f"({batch_start+1}-{batch_end}/{len(chunk_texts)})")
                
                # Ã¢Å“â€¦ Get embeddings in parallel (5x faster than sequential)
                batch_embeddings = self.embedding_service.get_embeddings_batch(
                    batch_texts,
                    max_workers=5,  # Parallel API calls
                    progress_tracker=progress_tracker
                )
                
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                if progress_tracker:
                    embed_progress = 30 + int((batch_end / len(chunk_texts)) * 40)
                    progress_tracker.update(embed_progress, 100, status="embedding",
                                        message=f"Embedded {batch_end}/{len(chunk_texts)} chunks")
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings in parallel")
            
            if len(all_embeddings) != len(chunks):
                raise ValueError(f"Embedding count mismatch: {len(all_embeddings)} != {len(chunks)}")
            
            # 4. Store in vector database
            if progress_tracker:
                progress_tracker.update(70, 100, status="storing", 
                                    message="Storing in vector database")
            
            # Normal RAG: Store in single collection
            if self.rag_mode == "normal":
                collection_name = self.config.get("collection_name", "documents")
                
                # Prepare documents in the format expected by insert_documents
                # insert_documents expects: documents (with 'text' and 'metadata') and embeddings separately
                documents_to_insert = []
                embeddings_to_insert = []
                
                for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                    documents_to_insert.append({
                        "text": chunk['text'],
                        "metadata": chunk['metadata']
                    })
                    embeddings_to_insert.append(embedding)
                
                # Insert using the correct method
                self.vector_db_service.insert_documents(
                    documents=documents_to_insert,
                    embeddings=embeddings_to_insert,
                    collection_name=collection_name,
                    progress_tracker=progress_tracker
                )
                
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete",
                                        message=f"Indexed {len(chunks)} chunks successfully")
                
                elapsed = time.time() - start_time
                logger.info(f"Normal RAG indexing completed in {elapsed:.2f}s")
                
                return {
                    "success": True,
                    "chunks_indexed": len(chunks),
                    "collection": collection_name,
                    "mode": "normal",
                    "time_elapsed": round(elapsed, 2)
                }
            
            # Graph RAG: Store in both collections + extract entities
            elif self.rag_mode == "graph":
                if progress_tracker:
                    progress_tracker.update(75, 100, status="graph_extraction",
                                        message="Extracting entities and relationships")
                
                # Store document chunks using correct format
                doc_collection = self.config.get("collection_name", "documents")
                
                # Prepare documents and embeddings in the format expected by insert_documents
                documents_to_insert = []
                embeddings_to_insert = []
                
                for chunk, embedding in zip(chunks, all_embeddings):
                    documents_to_insert.append({
                        "text": chunk['text'],
                        "metadata": chunk['metadata']
                    })
                    embeddings_to_insert.append(embedding)
                
                self.vector_db_service.insert_documents(
                    documents=documents_to_insert,
                    embeddings=embeddings_to_insert,
                    collection_name=doc_collection,
                    progress_tracker=progress_tracker
                )
                
                # Extract and store graph data
                if progress_tracker:
                    progress_tracker.update(85, 100, status="graph_building",
                                        message="Building knowledge graph")
                
                graph_result = self.graph_rag_service.process_document_for_graph(
                    chunks,
                    progress_tracker
                )

                # Also push to Neo4j if configured — no re-extraction needed
                neo4j_stats = {}
                if self.neo4j_service:
                    try:
                        if progress_tracker:
                            progress_tracker.update(92, 100, status="neo4j",
                                                message="Storing graph in Neo4j")
                        import os as _os
                        file_name = _os.path.basename(file_path)
                        # Pull benchmark_corpus and document metadata from the first chunk
                        _first_meta = chunks[0].get("metadata", {}) if chunks else {}
                        _bm_corpus  = _first_meta.get("benchmark_corpus", "")
                        _doc_meta   = {k: _first_meta.get(k, "") for k in
                                       ("title", "source", "category", "published_at", "url")}
                        neo4j_stats = self.neo4j_service.store_entities_and_relationships(
                            graph_result["entities"],
                            graph_result["relationships"],
                            document_name=file_name,
                            benchmark_corpus=_bm_corpus,
                            doc_metadata=_doc_meta,
                        )
                        self.neo4j_service.create_indexes()
                        logger.info(
                            f"Neo4j: {neo4j_stats.get('entities_created',0)} entities, "
                            f"{neo4j_stats.get('relationships_created',0)} relationships"
                        )
                    except Exception as e:
                        logger.warning(f"Neo4j storage skipped: {e}")

                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete",
                                        message="Graph RAG indexing complete")

                elapsed = time.time() - start_time
                logger.info(f"Graph RAG indexing completed in {elapsed:.2f}s")

                graph_stats = graph_result.get("graph_stats", {})

                return {
                    "success": True,
                    "chunks_indexed": len(chunks),
                    "entities_extracted": graph_stats.get("merged_entities", 0),
                    "relationships_extracted": graph_stats.get("valid_relationships", 0),
                    "collections": [doc_collection, self.config.get("graph_collection_name", "graph_entities")],
                    "mode": "graph",
                    "time_elapsed": round(elapsed, 2),
                    "graph_stats": graph_stats,
                    "neo4j_stats": neo4j_stats,
                }
            
            # neo4j mode: alias for graph — graph mode now writes to both Qdrant and Neo4j
            elif self.rag_mode == "neo4j":
                self.rag_mode = "graph"
                return self.index_document(file_path, progress_tracker)

            # Hybrid mode: vector store + graph collection + Neo4j
            elif self.rag_mode == "hybrid_neo4j":
                if not self.neo4j_service:
                    raise RuntimeError("Neo4j service not configured for hybrid mode.")

                if progress_tracker:
                    progress_tracker.update(70, 100, status="storing",
                                        message="Storing document chunks in vector DB")

                doc_collection = self.config.get("collection_name", "documents")
                documents_to_insert = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
                self.vector_db_service.insert_documents(
                    documents=documents_to_insert,
                    embeddings=all_embeddings,
                    collection_name=doc_collection,
                    progress_tracker=progress_tracker
                )

                if progress_tracker:
                    progress_tracker.update(80, 100, status="graph_extraction",
                                        message="Building knowledge graph and storing in Neo4j")

                graph_result = self.graph_rag_service.process_document_for_graph(chunks, progress_tracker)
                graph_stats = graph_result.get("graph_stats", {})

                # Also store entities in Neo4j
                import os as _os
                file_name = _os.path.basename(file_path)
                neo4j_stats = self.neo4j_service.store_entities_and_relationships(
                    graph_result.get("entities", []),
                    graph_result.get("relationships", []),
                    document_name=file_name
                )
                self.neo4j_service.create_indexes()

                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete",
                                        message="Hybrid indexing complete")

                elapsed = time.time() - start_time
                return {
                    "success": True,
                    "chunks_indexed": len(chunks),
                    "entities_extracted": graph_stats.get("merged_entities", 0),
                    "relationships_extracted": graph_stats.get("valid_relationships", 0),
                    "neo4j_entities": neo4j_stats.get("entities_created", 0),
                    "collections": [doc_collection, self.config.get("graph_collection_name")],
                    "mode": "hybrid_neo4j",
                    "time_elapsed": round(elapsed, 2),
                    "graph_stats": graph_stats
                }

            else:
                raise ValueError(f"Unknown RAG mode: {self.rag_mode}")

        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error",
                                    message=f"Indexing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunks_indexed": 0
            }

    
    def _index_document_normal_mode(self, text: str, metadata: Dict[str, Any], 
                                   progress_tracker: Optional[ProgressTracker] = None) -> bool:
        """Index document using normal RAG mode."""
        try:
            logger.info("Processing document in normal RAG mode")
            
            # Step 1: Chunk the document
            if progress_tracker:
                progress_tracker.update(20, 100, status="chunking", 
                                      message="Creating text chunks")
            
            chunks = self.document_processor.chunk_text(text, metadata, progress_tracker)
            
            if not chunks:
                logger.error("No chunks created from document")
                return False
            
            # Validate chunks
            chunk_validation = self.document_processor.validate_chunks(chunks)
            if not chunk_validation["valid"]:
                logger.error(f"Chunk validation failed: {chunk_validation['error']}")
                return False
            
            logger.info(f"Created {len(chunks)} chunks (avg size: {chunk_validation['average_chunk_size']:.0f} chars)")
            
            # Step 2: Generate embeddings
            if progress_tracker:
                progress_tracker.update(40, 100, status="embedding", 
                                      message="Generating embeddings for chunks")
            
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts, progress_tracker)
            
            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
                return False
            
            # Step 3: Store in vector database
            if progress_tracker:
                progress_tracker.update(70, 100, status="storing", 
                                      message="Storing chunks in vector database")
            
            self.vector_db_service.insert_documents(
                chunks, 
                embeddings, 
                progress_tracker=progress_tracker,
                collection_name=self.config["collection_name"]
            )
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message="Document indexed successfully in normal mode")
            
            logger.info("Document successfully indexed in normal RAG mode")
            return True
            
        except Exception as e:
            logger.error(f"Error in normal mode indexing: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return False
    
    def _index_document_graph_mode(self, text: str, metadata: Dict[str, Any], 
                                  progress_tracker: Optional[ProgressTracker] = None) -> bool:
        """Index document using graph RAG mode."""
        if not self.is_graph_ready():
            logger.error("Graph RAG mode not available")
            return False
        
        try:
            logger.info("Processing document in graph RAG mode")
            
            # Step 1: Create chunks for both normal and graph processing
            if progress_tracker:
                progress_tracker.update(10, 100, status="chunking", 
                                      message="Creating text chunks")
            
            chunks = self.document_processor.chunk_text(text, metadata, progress_tracker)
            
            if not chunks:
                logger.error("No chunks created from document")
                return False
            
            logger.info(f"Created {len(chunks)} chunks for graph processing")
            
            # Step 2: Store chunks in normal collection (for hybrid search)
            if progress_tracker:
                progress_tracker.update(20, 100, status="storing_chunks", 
                                      message="Storing document chunks")
            
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts)
            
            self.vector_db_service.insert_documents(
                chunks, 
                embeddings, 
                collection_name=self.config["collection_name"]
            )
            
            logger.info("Document chunks stored in normal collection")
            
            # Step 3: Extract and store graph elements
            if progress_tracker:
                progress_tracker.update(40, 100, status="graph_processing", 
                                      message="Extracting knowledge graph")
            
            graph_data = self.graph_rag_service.process_document_for_graph(chunks, progress_tracker)
            
            # Step 4: Log results
            graph_stats = graph_data.get("graph_stats", {})
            entities_count = len(graph_data.get("entities", []))
            relationships_count = len(graph_data.get("relationships", []))
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message=f"Graph RAG complete: {entities_count} entities, {relationships_count} relationships")
            
            logger.info(f"Graph RAG indexing complete: {graph_stats}")
            return True
            
        except Exception as e:
            logger.error(f"Error in graph mode indexing: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return False
    
    def query(self, question: str, session_id: str = None) -> str:
        """Query indexed documents using the configured RAG mode."""
        if not self.is_ready():
            return "SimpleRAG is not ready. Please check your configuration and ensure services are properly initialized."
        
        if not question or not question.strip():
            return "Please provide a valid question."
        
        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting", 
                                  message=f"Processing query in {self.rag_mode} mode")
        
        try:
            logger.info(f"Processing query in {self.rag_mode} mode: {question[:100]}...")
            
            if self.rag_mode == "pageindex" and self.is_pageindex_ready():
                result = self.pageindex_service.query(question, progress_tracker=progress_tracker)
                # Return just the answer string for backward-compat with callers that
                # expect a plain string.  Callers that want full citations should call
                # query_pageindex() directly.
                return result.get("answer", "PageIndex returned no answer.")
            elif self.rag_mode == "graph" and self.is_graph_ready():
                return self._query_graph_mode(question, progress_tracker)
            elif self.rag_mode == "neo4j" and self.is_neo4j_ready():
                return self.query_neo4j(question, session_id)
            elif self.rag_mode == "hybrid_neo4j" and self.is_graph_ready() and self.is_neo4j_ready():
                return self._query_hybrid_neo4j_mode(question, progress_tracker)
            else:
                return self._query_normal_mode(question, progress_tracker)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    def query_with_filters(self, question: str, filters: Dict[str, Any],
                           session_id: str = None) -> str:
        """
        Query with explicit domain metadata pre-filters.

        Filters are applied before semantic search so only chunks that match
        the filter conditions are considered. Useful for domain-specific retrieval.

        Example (vc_financial domain):
            rag.query_with_filters("What is the ARR?", {"sector": "fintech", "stage": "series_a"})
        """
        if not self.is_ready():
            return "SimpleRAG is not ready. Please check your configuration."

        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "query_filtered")
            progress_tracker.update(0, 100, status="starting",
                                    message=f"Filtered query: {list(filters.keys())}")
        try:
            return self._query_normal_mode(question, progress_tracker, filters=filters)
        except Exception as e:
            logger.error(f"Error in filtered query: {e}")
            return f"Error processing your query: {e}"

    def _query_normal_mode(self, question: str, progress_tracker: Optional[ProgressTracker] = None,
                           filters: Dict[str, Any] = None) -> str:
        """Query using normal RAG mode, with optional metadata pre-filters."""
        try:
            top_k = self.config["top_k"]
            # Retrieve a larger pool when reranking is active so reranker has good candidates
            retrieval_k = top_k * 4 if self.reranker else top_k

            # Step 1: Query planning (HyDE + decomposition) or single embedding
            if progress_tracker:
                progress_tracker.update(15, 100, status="planning", message="Planning retrieval strategy")

            if self.query_planner:
                plan = self.query_planner.plan(question)
            else:
                plan = {"strategy": "simple", "sub_queries": [question], "hyde_docs": []}

            # Step 2: Embed sub-queries + HyDE docs, retrieve for each, RRF-merge
            if progress_tracker:
                progress_tracker.update(30, 100, status="searching", message="Retrieving relevant documents")

            result_lists = []
            queries_to_embed = list(plan["sub_queries"])
            hyde_docs = plan.get("hyde_docs", [])
            # Pair sub-queries with their hyde docs (embed hyde if non-empty, else query)
            for i, q in enumerate(queries_to_embed):
                hyde = hyde_docs[i] if i < len(hyde_docs) and hyde_docs[i] else None
                embed_text = hyde if hyde else q
                emb = self.embedding_service.get_embedding(embed_text)
                results = self.vector_db_service.search_similar(
                    emb,
                    top_k=retrieval_k,
                    collection_name=self.config["collection_name"],
                    filters=filters,
                )
                result_lists.append(results)

            contexts = rrf_merge(result_lists) if len(result_lists) > 1 else (result_lists[0] if result_lists else [])

            logger.info(
                f"Retrieved {len(contexts)} contexts "
                f"(strategy={plan['strategy']}, sub_queries={len(plan['sub_queries'])})"
            )

            if not contexts:
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", message="No relevant information found")
                return "I couldn't find any relevant information to answer your question. Please ensure documents have been indexed."

            # Step 3: Rerank — reorder the larger pool down to top_k
            if self.reranker and len(contexts) > top_k:
                if progress_tracker:
                    progress_tracker.update(55, 100, status="reranking", message="Reranking results")
                contexts = self.reranker.rerank(question, contexts, top_k)
            else:
                contexts = contexts[:top_k]

            # Step 4: Generate answer
            if progress_tracker:
                progress_tracker.update(65, 100, status="generating", message="Generating answer")

            if self.llm_service and self.llm_service.is_available():
                answer = self.llm_service.generate_answer(question, contexts, rag_mode="normal", progress_tracker=progress_tracker)
            else:
                answer = self._format_raw_results(contexts)

            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", message="Answer generated successfully")

            return answer

        except Exception as e:
            logger.error(f"Error in normal mode query: {str(e)}")
            return f"Error processing your query: {str(e)}"
    
    def _query_graph_mode(self, question: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Query using graph RAG mode with hybrid search."""
        try:
            # Step 1: Generate query embedding
            if progress_tracker:
                progress_tracker.update(10, 100, status="embedding", 
                                      message="Generating query embedding")
            
            query_embedding = self.embedding_service.get_embedding(question)
            
            # Step 2: Search both collections
            if progress_tracker:
                progress_tracker.update(20, 100, status="searching", 
                                      message="Searching documents and knowledge graph")
            
            top_k = self.config["top_k"]
            retrieval_k = top_k * 2 if self.reranker else top_k // 2

            # Search document chunks
            doc_contexts = self.vector_db_service.search_similar(
                query_embedding,
                top_k=retrieval_k,
                collection_name=self.config["collection_name"]
            )

            # Search graph elements (Neo4j traversal when available)
            if not self.neo4j_service:
                logger.warning(
                    "graph mode: Neo4j not configured — topology traversal disabled. "
                    "Configure NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD for full graph mode."
                )
            graph_contexts = self.graph_rag_service.search_graph(
                question,
                top_k=top_k // 2,
                neo4j_service=self.neo4j_service,
            )

            # Rerank doc_contexts before combining with graph contexts
            if self.reranker and len(doc_contexts) > top_k // 2:
                if progress_tracker:
                    progress_tracker.update(40, 100, status="reranking", message="Reranking document results")
                doc_contexts = self.reranker.rerank(question, doc_contexts, top_k // 2)

            all_contexts = doc_contexts + graph_contexts

            logger.info(f"Found {len(doc_contexts)} doc contexts and {len(graph_contexts)} graph contexts")

            if not all_contexts:
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete",
                                          message="No relevant information found")
                return "I couldn't find any relevant information to answer your question in either the documents or knowledge graph."

            # Step 3: Prepare graph context for enhanced prompting
            if progress_tracker:
                progress_tracker.update(55, 100, status="analyzing",
                                      message="Analyzing graph relationships")

            graph_context = {
                "entities": [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'entity'],
                "relationships": [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'relationship']
            }

            # Step 4: Generate enhanced answer
            if progress_tracker:
                progress_tracker.update(70, 100, status="generating",
                                      message="Generating graph-enhanced answer")
            
            if self.llm_service and self.llm_service.is_available():
                answer = self.llm_service.generate_answer(
                    question,
                    all_contexts,
                    graph_context=graph_context,
                    rag_mode="graph",
                    progress_tracker=progress_tracker
                )
            else:
                # Fallback to enhanced raw results
                answer = self._format_graph_raw_results(doc_contexts, graph_contexts)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message="Graph RAG answer generated successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in graph mode query: {str(e)}")
            return f"Error processing your graph query: {str(e)}"
    
    def _format_raw_results(self, contexts: List[Dict[str, Any]]) -> str:
        """Format raw search results when LLM is not available."""
        if not contexts:
            return "No relevant results found."
        
        results = ["=== SEARCH RESULTS (Raw Mode) ===\n"]
        
        for i, ctx in enumerate(contexts):
            filename = _clean_source_name(ctx['metadata'])
            score = ctx.get('score', 0)
            text = ctx['text']
            
            results.append(f"Result {i+1} (Score: {score:.3f})")
            results.append(f"Source: {filename}")
            results.append(f"Content: {text}")
            results.append("-" * 50)
        
        return "\n".join(results)
    
    def _format_graph_raw_results(self, doc_contexts: List[Dict[str, Any]], 
                                 graph_contexts: List[Dict[str, Any]]) -> str:
        """Format raw graph search results when LLM is not available."""
        results = ["=== GRAPH RAG RESULTS (Raw Mode) ===\n"]
        
        if doc_contexts:
            results.append("DOCUMENT CONTEXTS:")
            for i, ctx in enumerate(doc_contexts):
                filename = _clean_source_name(ctx['metadata'])
                score = ctx.get('score', 0)
                results.append(f"  Doc {i+1} ({filename}, Score: {score:.3f}): {ctx['text'][:200]}...")
        
        if graph_contexts:
            results.append("\nGRAPH CONTEXTS:")
            entities = [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'entity']
            relationships = [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'relationship']
            
            if entities:
                results.append("  Entities:")
                for ctx in entities:
                    entity_name = ctx['metadata'].get('entity_name', 'Unknown')
                    entity_type = ctx['metadata'].get('entity_type', 'Unknown')
                    score = ctx.get('score', 0)
                    results.append(f"    - {entity_name} ({entity_type}, Score: {score:.3f})")
            
            if relationships:
                results.append("  Relationships:")
                for ctx in relationships:
                    source = ctx['metadata'].get('source', ctx['metadata'].get('filename', ctx['metadata'].get('doc_name', 'Unknown')))
                    target = ctx['metadata'].get('target', 'Unknown')
                    rel_type = ctx['metadata'].get('relationship', 'unknown')
                    score = ctx.get('score', 0)
                    results.append(f"    - {source} Ã¢â€ â€™ {rel_type} Ã¢â€ â€™ {target} (Score: {score:.3f})")
        
        return "\n".join(results)
    
    def get_collections_info(self) -> Dict[str, Any]:
        """Get information about vector database collections."""
        if not self.vector_db_service:
            return {"error": "Vector database service not available"}
        
        try:
            collections = self.vector_db_service.list_collections()
            
            # Add type information
            for collection in collections:
                if collection["name"] == self.config["collection_name"]:
                    collection["type"] = "normal_rag"
                elif collection["name"] == self.config["graph_collection_name"]:
                    collection["type"] = "graph_rag"
                else:
                    collection["type"] = "other"
            
            return {
                "collections": collections,
                "normal_collection": self.config["collection_name"],
                "graph_collection": self.config["graph_collection_name"]
            }
        except Exception as e:
            return {"error": f"Failed to get collections info: {str(e)}"}

    
    def _initialize_neo4j_service(self):
        """Initialize Neo4j service if credentials are configured."""
        if (self.config.get("neo4j_uri") and 
            self.config.get("neo4j_username") and 
            self.config.get("neo4j_password") and
            self.config.get("neo4j_enabled", False)):
            try:
                from neo4j_service import Neo4jService
                self.neo4j_service = Neo4jService(
                    uri=self.config["neo4j_uri"],
                    username=self.config["neo4j_username"],
                    password=self.config["neo4j_password"],
                    database=self.config.get("neo4j_database") or None
                )
                logger.info(". Neo4j service initialized")
            except Exception as e:
                error_msg = f"Failed to initialize Neo4j service: {str(e)}"
                logger.error(error_msg)
                self.initialization_warnings.append(error_msg)
                self.neo4j_service = None
        else:
            logger.info("Ã¢Å¡Â  Neo4j service skipped - credentials not configured or not enabled")
            self.neo4j_service = None

    def is_neo4j_ready(self) -> bool:
        """Check if Neo4j service is ready for operations."""
        return self.neo4j_service is not None

    def _index_document_neo4j_mode(self, text: str, metadata: Dict[str, Any], 
                                progress_tracker: Optional[ProgressTracker] = None) -> bool:
        """Index document using Neo4j-only mode (no vector embeddings)."""
        try:
            logger.info("Processing document in Neo4j-only mode")
            
            if not self.neo4j_service:
                logger.error("Neo4j service not available")
                if progress_tracker:
                    progress_tracker.update(100, 100, status="error", 
                                        message="Neo4j service not configured")
                return False
            
            # Step 1: Create chunks for entity extraction
            if progress_tracker:
                progress_tracker.update(10, 100, status="chunking", 
                                    message="Creating text chunks for entity extraction")
            
            chunks = self.document_processor.chunk_text(text, metadata, progress_tracker)
            
            if not chunks:
                logger.error("No chunks created from document")
                return False
            
            logger.info(f"Created {len(chunks)} chunks for Neo4j processing")
            
            # Step 2: Extract entities and relationships using graph extractor
            if progress_tracker:
                progress_tracker.update(30, 100, status="extracting", 
                                    message="Extracting entities and relationships")
            
            all_entities = []
            all_relationships = []
            
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = f"chunk_{i}_{int(time.time())}"
                    
                    # Use the graph extractor to get entities and relationships
                    graph_data = self.graph_rag_service.graph_extractor.extract_entities_and_relationships(
                        chunk["text"], chunk_id
                    )
                    
                    # Add metadata to entities and relationships
                    for entity in graph_data.get("entities", []):
                        entity["metadata"] = metadata
                        entity["chunk_index"] = i
                    
                    for rel in graph_data.get("relationships", []):
                        rel["metadata"] = metadata
                        rel["chunk_index"] = i
                    
                    all_entities.extend(graph_data.get("entities", []))
                    all_relationships.extend(graph_data.get("relationships", []))
                    
                    if progress_tracker:
                        progress = 30 + int((i + 1) / total_chunks * 40)  # 30-70%
                        progress_tracker.update(progress, 100, 
                                            message=f"Extracted from chunk {i + 1} of {total_chunks}")
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships")
            
            # Step 3: Merge similar entities
            if progress_tracker:
                progress_tracker.update(70, 100, status="merging", 
                                    message="Merging similar entities")
            
            merged_entities = self.graph_rag_service._merge_similar_entities(all_entities)
            validated_relationships = self.graph_rag_service._validate_relationships(all_relationships, merged_entities)
            
            logger.info(f"After processing: {len(merged_entities)} entities, {len(validated_relationships)} relationships")
            
            # Step 4: Store in Neo4j
            if progress_tracker:
                progress_tracker.update(85, 100, status="storing", 
                                    message="Storing in Neo4j database")
            
            document_name = metadata.get("filename", "unknown_document")
            stats = self.neo4j_service.store_entities_and_relationships(
                merged_entities, 
                validated_relationships,
                document_name
            )
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message=f"Neo4j storage complete: {stats['entities_created']} entities, {stats['relationships_created']} relationships")
            
            logger.info(f"Neo4j indexing complete: {stats}")
            return stats["errors"] == 0
            
        except Exception as e:
            logger.error(f"Error in Neo4j mode indexing: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                    message=f"Error: {str(e)}")
            return False

    def query_neo4j(self, question: str, session_id: str = None) -> str:
        """Query Neo4j database directly using Cypher queries."""
        if not self.neo4j_service:
            return "Neo4j service not available. Please check your Neo4j configuration."
        
        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "neo4j_query")
            progress_tracker.update(0, 100, status="starting", 
                                message="Processing Neo4j query")
        
        try:
            # Step 1: Generate Cypher query from natural language
            if progress_tracker:
                progress_tracker.update(30, 100, status="generating", 
                                    message="Generating Cypher query")
            
            cypher_query, error = self.neo4j_service.generate_cypher_from_question(
                question, self.llm_service
            )
            
            if error:
                return f"Error generating query: {error}"
            
            logger.info(f"Generated Cypher query: {cypher_query}")
            
            # Step 2: Execute the query
            if progress_tracker:
                progress_tracker.update(60, 100, status="executing", 
                                    message="Executing query on Neo4j")
            
            results, error = self.neo4j_service.execute_cypher_query(cypher_query)
            
            if error:
                return f"Error executing query: {error}"
            
            # Step 3: Format results
            if progress_tracker:
                progress_tracker.update(80, 100, status="formatting", 
                                    message="Formatting results")
            
            # Build context — match benchmark pattern (empty results still go to LLM)
            context = json.dumps(results, indent=2) if results else "(no matching entities found in graph)"

            # Use LLM to generate a natural language answer if available
            if self.llm_service and self.llm_service.is_available():
                prompt = f"""Answer the following question based ONLY on the provided Neo4j graph query results. Be concise and precise.

Question: {question}

Neo4j Query Results:
{context}

Answer:"""
                answer = self.llm_service._generate_with_gemini(prompt)
            else:
                # Format raw results
                if not results:
                    answer = "No results found in the Neo4j database for your query."
                else:
                    answer = self._format_neo4j_results(results, cypher_query)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message="Query complete")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error querying Neo4j: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                    message=f"Error: {str(e)}")
            return f"Error querying Neo4j database: {str(e)}"

    def _format_neo4j_results(self, results: List[Dict], cypher_query: str) -> str:
        """Format Neo4j query results for display."""
        formatted = ["=== Neo4j Query Results ===\n"]
        formatted.append(f"Query: {cypher_query}\n")
        formatted.append(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            formatted.append(f"\nResult {i}:")
            for key, value in result.items():
                if isinstance(value, dict):
                    formatted.append(f"  {key}:")
                    for k, v in value.items():
                        formatted.append(f"    {k}: {v}")
                else:
                    formatted.append(f"  {key}: {value}")
        
        return "\n".join(formatted)
    def _query_hybrid_neo4j_mode(self, question: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Query using Graph RAG combined with Neo4j Cypher queries for enhanced relationship understanding."""
        try:
            # Step 1: Get Graph RAG results first (includes both document and graph vector search)
            if progress_tracker:
                progress_tracker.update(10, 100, status="graph_search", 
                                    message="Searching documents and knowledge graph")
            
            # Use existing graph mode query to get vector-based results
            query_embedding = self.embedding_service.get_embedding(question)
            
            # Search document chunks
            doc_contexts = self.vector_db_service.search_similar(
                query_embedding,
                top_k=self.config["top_k"] // 3,  # Reduce to make room for Neo4j results
                collection_name=self.config["collection_name"]
            )
            
            # Search graph elements (Neo4j traversal when available)
            graph_contexts = self.graph_rag_service.search_graph(
                question,
                top_k=self.config["top_k"] // 3,
                neo4j_service=self.neo4j_service,
            )

            # Step 2: Query Neo4j if available
            neo4j_contexts = []
            if self.neo4j_service:
                if progress_tracker:
                    progress_tracker.update(40, 100, status="neo4j_query", 
                                        message="Generating and executing Neo4j Cypher query")
                
                try:
                    # Generate Cypher query from natural language
                    cypher_query, error = self.neo4j_service.generate_cypher_from_question(
                        question, self.llm_service
                    )
                    
                    if not error and cypher_query:
                        logger.info(f"Generated Cypher query: {cypher_query}")
                        
                        # Execute the query
                        results, exec_error = self.neo4j_service.execute_cypher_query(cypher_query)
                        
                        if not exec_error and results:
                            # Convert Neo4j results to context format (no fake scores)
                            for i, result in enumerate(results[:self.config["top_k"] // 3]):
                                neo4j_context = {
                                    "text": self._format_neo4j_result_as_text(result),
                                    "metadata": {
                                        "type": "neo4j_result",
                                        "source": "Neo4j Graph Database",
                                        "cypher_query": cypher_query,
                                        "result_index": i,
                                    },
                                    "score": None,  # Neo4j results don't have a meaningful similarity score
                                }
                                neo4j_contexts.append(neo4j_context)
                            
                            logger.info(f"Retrieved {len(neo4j_contexts)} contexts from Neo4j")
                except Exception as e:
                    logger.warning(f"Neo4j query failed, continuing with Graph RAG only: {e}")
            
            # Step 3: Combine and deduplicate contexts by text hash
            raw_contexts = doc_contexts + graph_contexts + neo4j_contexts
            seen_hashes: set = set()
            all_contexts = []
            for ctx in raw_contexts:
                text_hash = hash(ctx.get("text", "")[:200])
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    all_contexts.append(ctx)

            logger.info(
                f"Hybrid mode: {len(doc_contexts)} doc + {len(graph_contexts)} graph + "
                f"{len(neo4j_contexts)} neo4j → {len(all_contexts)} unique contexts "
                f"(deduped {len(raw_contexts) - len(all_contexts)})"
            )
            
            if not all_contexts:
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", 
                                        message="No relevant information found")
                return "I couldn't find any relevant information to answer your question in documents, knowledge graph, or Neo4j database."
            
            # Step 4: Prepare enhanced context for LLM
            if progress_tracker:
                progress_tracker.update(70, 100, status="generating", 
                                    message="Generating comprehensive answer from all sources")
            
            # Separate contexts by type for better prompting
            graph_context = {
                "entities": [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'entity'],
                "relationships": [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'relationship'],
                "neo4j_results": neo4j_contexts
            }
            
            # Step 5: Generate enhanced answer with all contexts
            if self.llm_service and self.llm_service.is_available():
                answer = self.llm_service.generate_hybrid_neo4j_answer(
                    question, 
                    all_contexts, 
                    graph_context=graph_context,
                    progress_tracker=progress_tracker
                )
            else:
                # Fallback to formatted raw results
                answer = self._format_hybrid_raw_results(doc_contexts, graph_contexts, neo4j_contexts)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                    message="Hybrid Graph + Neo4j answer generated successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in hybrid Neo4j mode query: {str(e)}")
            return f"Error processing your hybrid query: {str(e)}"

    def _format_neo4j_result_as_text(self, result: Dict) -> str:
        """Convert Neo4j result to readable text format."""
        text_parts = []
        
        for key, value in result.items():
            if isinstance(value, dict):
                # Node or relationship properties
                text_parts.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)

    def _format_hybrid_raw_results(self, doc_contexts, graph_contexts, neo4j_contexts) -> str:
        """Format raw results when LLM is not available for hybrid mode."""
        results = ["=== HYBRID GRAPH + NEO4J RESULTS (Raw Mode) ===\n"]
        
        if doc_contexts:
            results.append("DOCUMENT CONTEXTS:")
            for i, ctx in enumerate(doc_contexts[:3]):
                filename = _clean_source_name(ctx['metadata'])
                score = ctx.get('score', 0)
                results.append(f"  Doc {i+1} ({filename}, Score: {score:.3f}): {ctx['text'][:200]}...")
        
        if graph_contexts:
            results.append("\nGRAPH CONTEXTS:")
            entities = [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'entity']
            relationships = [ctx for ctx in graph_contexts if ctx['metadata'].get('type') == 'relationship']
            
            if entities:
                results.append("  Entities:")
                for ctx in entities[:5]:
                    entity_name = ctx['metadata'].get('entity_name', 'Unknown')
                    entity_type = ctx['metadata'].get('entity_type', 'Unknown')
                    score = ctx.get('score', 0)
                    results.append(f"    - {entity_name} ({entity_type}, Score: {score:.3f})")
            
            if relationships:
                results.append("  Relationships:")
                for ctx in relationships[:5]:
                    source = ctx['metadata'].get('source', ctx['metadata'].get('filename', ctx['metadata'].get('doc_name', 'Unknown')))
                    target = ctx['metadata'].get('target', 'Unknown')
                    rel_type = ctx['metadata'].get('relationship', 'unknown')
                    score = ctx.get('score', 0)
                    results.append(f"    - {source} Ã¢â€ â€™ {rel_type} Ã¢â€ â€™ {target} (Score: {score:.3f})")
        
        if neo4j_contexts:
            results.append("\nNEO4J GRAPH DATABASE RESULTS:")
            for i, ctx in enumerate(neo4j_contexts):
                results.append(f"  Result {i+1}: {ctx['text'][:300]}...")
        
        return "\n".join(results)
    # ── PageIndex service ─────────────────────────────────────────────────────

    def _initialize_pageindex_service(self):
        """Initialise PageIndexService if dependencies and an LLM key are present."""
        if not self.config.get("pageindex_enabled", True):
            logger.info("PageIndex service disabled in config")
            return
        try:
            svc = PageIndexService(self.config)
            if svc.is_ready():
                self.pageindex_service = svc
                logger.info("PageIndex service initialised (model=%s)", svc.model)
            else:
                self.initialization_warnings.append(
                    "PageIndex service not ready — install pageindex package "
                    "and set gemini_api_key"
                )
        except Exception as e:
            self.initialization_warnings.append(f"PageIndex init warning: {e}")

    def is_pageindex_ready(self) -> bool:
        """Return True if PageIndex indexing and querying are available."""
        return self.pageindex_service is not None and self.pageindex_service.is_ready()

    def index_document_pageindex(
        self,
        file_path: str,
        progress_tracker=None,
    ) -> Dict[str, Any]:
        """Build a PageIndex tree for *file_path* (vectorless indexing)."""
        if not self.is_pageindex_ready():
            return {
                "success": False,
                "error":   "PageIndex service not available. "
                           "Install the pageindex package and set an LLM API key.",
            }
        return self.pageindex_service.index_document(file_path, progress_tracker)

    def query_pageindex(
        self,
        question:  str,
        doc_id:    str = None,
        session_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run a PageIndex agentic query.

        Returns a dict with keys: success, answer, citations, tree_path,
        tool_calls, doc_id, doc_name, error.
        """
        if not self.is_pageindex_ready():
            return PageIndexService._error_result(
                "PageIndex service not available. "
                "Install the pageindex package and set an LLM API key."
            )

        progress_tracker = None
        if session_id:
            from extensions import ProgressTracker
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting",
                                    message="PageIndex agentic query starting…")

        return self.pageindex_service.query(question, doc_id, progress_tracker)

    # ── query_debug: structured query result for eval/benchmarks ─────────────

    def query_debug(self, question: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Like query() but returns a structured dict exposing retrieved contexts.
        Non-breaking: the existing query() API is unchanged.

        Returns:
            answer            str   — generated answer
            mode              str   — rag_mode used
            model             str   — LLM model string
            collection_names  list  — Qdrant collection(s) searched
            contexts_doc      list  — retrieved document chunk dicts {text, metadata, score}
            contexts_graph    list  — retrieved graph element dicts (graph/hybrid modes)
            entities_used     list  — graph contexts where type=="entity"
            relationships_used list — graph contexts where type=="relationship"
            cypher_query      str   — Cypher string (neo4j/hybrid modes only)
            cypher_rows       list  — Neo4j result rows (neo4j/hybrid modes only)
            latency_ms        float — total wall-clock ms
        """
        import time as _time
        if not self.is_ready():
            return {
                "answer": "SimpleRAG is not ready.",
                "mode": self.rag_mode, "model": GEMINI_MODEL,
                "collection_names": [], "contexts_doc": [], "contexts_graph": [],
                "entities_used": [], "relationships_used": [],
                "cypher_query": "", "cypher_rows": [], "latency_ms": 0,
            }

        t0 = _time.time()
        result: Dict[str, Any] = {
            "mode": self.rag_mode,
            "model": GEMINI_MODEL,
            "collection_names": [self.config.get("collection_name", "documents")],
            "contexts_doc": [],
            "contexts_graph": [],
            "entities_used": [],
            "relationships_used": [],
            "cypher_query": "",
            "cypher_rows": [],
            "answer": "",
        }

        try:
            if self.rag_mode == "normal":
                answer, doc_ctxs = self._query_normal_debug(question, filters=filters)
                result["answer"] = answer
                result["contexts_doc"] = doc_ctxs

            elif self.rag_mode == "graph" and self.is_graph_ready():
                answer, doc_ctxs, graph_ctxs = self._query_graph_debug(question)
                result["answer"] = answer
                result["contexts_doc"] = doc_ctxs
                result["contexts_graph"] = graph_ctxs
                result["entities_used"] = [c for c in graph_ctxs if c.get("metadata", {}).get("type") == "entity"]
                result["relationships_used"] = [c for c in graph_ctxs if c.get("metadata", {}).get("type") == "relationship"]
                result["collection_names"] = [
                    self.config.get("collection_name", "documents"),
                    self.config.get("graph_collection_name", "graph_entities"),
                ]

            elif self.rag_mode == "neo4j" and self.is_neo4j_ready():
                # Pure Neo4j: generate Cypher → execute → LLM answer. No Qdrant/vector retrieval.
                # Mirrors production query_neo4j() exactly.
                answer, cypher_q, cypher_rows = self._query_neo4j_debug(question)
                result["answer"] = answer
                result["cypher_query"] = cypher_q
                result["cypher_rows"] = cypher_rows
                # Production neo4j mode has no doc or graph vector contexts.

            elif self.rag_mode == "hybrid_neo4j" and self.is_neo4j_ready():
                answer, doc_ctxs, graph_ctxs, cypher_q, cypher_rows = self._query_hybrid_neo4j_debug(question)
                result["answer"] = answer
                result["contexts_doc"] = doc_ctxs
                result["contexts_graph"] = graph_ctxs
                result["cypher_query"] = cypher_q
                result["cypher_rows"] = cypher_rows
                result["entities_used"] = [c for c in graph_ctxs if c.get("metadata", {}).get("type") == "entity"]
                result["relationships_used"] = [c for c in graph_ctxs if c.get("metadata", {}).get("type") == "relationship"]

            else:
                # pageindex or fallback
                result["answer"] = self.query(question)

        except Exception as e:
            logger.error(f"query_debug error: {e}")
            result["answer"] = f"Error: {e}"

        result["latency_ms"] = round((_time.time() - t0) * 1000, 1)
        return result

    def _query_normal_debug(self, question: str, filters: Dict[str, Any] = None):
        """Normal RAG retrieval — returns (answer, doc_contexts)."""
        top_k = self.config["top_k"]
        retrieval_k = top_k * 4 if self.reranker else top_k

        if self.query_planner:
            plan = self.query_planner.plan(question)
        else:
            plan = {"strategy": "simple", "sub_queries": [question], "hyde_docs": []}

        result_lists = []
        for i, q in enumerate(plan["sub_queries"]):
            hyde_docs = plan.get("hyde_docs", [])
            hyde = hyde_docs[i] if i < len(hyde_docs) and hyde_docs[i] else None
            emb = self.embedding_service.get_embedding(hyde if hyde else q)
            result_lists.append(self.vector_db_service.search_similar(
                emb, top_k=retrieval_k,
                collection_name=self.config["collection_name"],
                filters=filters,
            ))

        contexts = rrf_merge(result_lists) if len(result_lists) > 1 else (result_lists[0] if result_lists else [])

        if not contexts:
            return "No relevant information found.", []

        if self.reranker and len(contexts) > top_k:
            contexts = self.reranker.rerank(question, contexts, top_k)
        else:
            contexts = contexts[:top_k]

        if self.llm_service and self.llm_service.is_available():
            answer = self.llm_service.generate_answer(question, contexts, rag_mode="normal")
        else:
            answer = self._format_raw_results(contexts)

        return answer, contexts

    def _query_graph_debug(self, question: str):
        """Graph RAG retrieval — returns (answer, doc_contexts, graph_contexts)."""
        query_embedding = self.embedding_service.get_embedding(question)
        top_k = self.config["top_k"]
        retrieval_k = top_k * 2 if self.reranker else top_k // 2

        doc_contexts = self.vector_db_service.search_similar(
            query_embedding, top_k=retrieval_k,
            collection_name=self.config["collection_name"]
        )
        graph_contexts = self.graph_rag_service.search_graph(
            question, top_k=top_k // 2, neo4j_service=self.neo4j_service
        )

        if self.reranker and len(doc_contexts) > top_k // 2:
            doc_contexts = self.reranker.rerank(question, doc_contexts, top_k // 2)

        all_contexts = doc_contexts + graph_contexts
        if not all_contexts:
            return "No relevant information found.", [], []

        graph_context = {
            "entities": [c for c in graph_contexts if c["metadata"].get("type") == "entity"],
            "relationships": [c for c in graph_contexts if c["metadata"].get("type") == "relationship"],
        }
        if self.llm_service and self.llm_service.is_available():
            answer = self.llm_service.generate_answer(
                question, all_contexts, graph_context=graph_context, rag_mode="graph"
            )
        else:
            answer = self._format_graph_raw_results(doc_contexts, graph_contexts)

        return answer, doc_contexts, graph_contexts

    def _query_hybrid_neo4j_debug(self, question: str):
        """Hybrid Neo4j retrieval — returns (answer, doc_ctxs, graph_ctxs, cypher_q, cypher_rows)."""
        query_embedding = self.embedding_service.get_embedding(question)
        top_k = self.config["top_k"]

        doc_contexts = self.vector_db_service.search_similar(
            query_embedding, top_k=top_k // 3,
            collection_name=self.config["collection_name"]
        )
        graph_contexts = (
            self.graph_rag_service.search_graph(
                question, top_k=top_k // 3, neo4j_service=self.neo4j_service
            )
            if self.is_graph_ready() else []
        )

        cypher_query = ""
        cypher_rows = []
        neo4j_contexts = []
        if self.neo4j_service:
            cypher_query, err = self.neo4j_service.generate_cypher_from_question(question, self.llm_service)
            if not err and cypher_query:
                cypher_rows, exec_err = self.neo4j_service.execute_cypher_query(cypher_query)
                if not exec_err:
                    for i, row in enumerate(cypher_rows[:top_k // 3]):
                        neo4j_contexts.append({
                            "text": self._format_neo4j_result_as_text(row),
                            "metadata": {"type": "neo4j_result", "source": "Neo4j", "cypher_query": cypher_query},
                            "score": None,
                        })

        all_contexts = doc_contexts + graph_contexts + neo4j_contexts
        if not all_contexts:
            return "No relevant information found.", [], [], cypher_query, cypher_rows

        graph_context = {
            "entities": [c for c in graph_contexts if c["metadata"].get("type") == "entity"],
            "relationships": [c for c in graph_contexts if c["metadata"].get("type") == "relationship"],
            "neo4j_results": neo4j_contexts,
        }
        if self.llm_service and self.llm_service.is_available():
            answer = self.llm_service.generate_hybrid_neo4j_answer(question, all_contexts, graph_context=graph_context)
        else:
            answer = self._format_hybrid_raw_results(doc_contexts, graph_contexts, neo4j_contexts)

        return answer, doc_contexts, graph_contexts, cypher_query, cypher_rows

    def _query_neo4j_debug(self, question: str):
        """Pure Neo4j mode — mirrors production query_neo4j() exactly.

        Returns (answer, cypher_query, cypher_rows).
        There are no Qdrant/vector/NetworkX contexts in this mode — the
        production path does not retrieve them either.
        """
        cypher_query = ""
        cypher_rows = []

        if not self.neo4j_service:
            return "Neo4j service not available.", "", []

        cypher_query, err = self.neo4j_service.generate_cypher_from_question(
            question, self.llm_service
        )
        if err:
            return f"Error generating Cypher: {err}", cypher_query, []

        cypher_rows, exec_err = self.neo4j_service.execute_cypher_query(cypher_query)
        if exec_err:
            return f"Error executing Cypher: {exec_err}", cypher_query, []

        context = json.dumps(cypher_rows, indent=2) if cypher_rows else "(no matching entities found in graph)"

        if self.llm_service and self.llm_service.is_available():
            prompt = (
                f"Answer the following question based ONLY on the provided Neo4j graph query results. "
                f"Be concise and precise.\n\nQuestion: {question}\n\n"
                f"Neo4j Query Results:\n{context}\n\nAnswer:"
            )
            answer = self.llm_service._generate_with_gemini(prompt)
        else:
            answer = self._format_neo4j_results(cypher_rows, cypher_query)

        return answer, cypher_query, cypher_rows

    # ── set_rag_mode: extend to include 'pageindex' ───────────────────────────

    def set_rag_mode(self, mode: str):
        """Switch between 'normal', 'graph', 'neo4j', 'hybrid_neo4j', 'pageindex'."""
        valid = {"normal", "graph", "neo4j", "hybrid_neo4j", "pageindex"}
        if mode not in valid:
            raise ValueError(f"RAG mode must be one of {sorted(valid)}")

        if mode == "graph" and not self.is_graph_ready():
            raise RuntimeError("Graph RAG mode not available — check service initialisation")
        if mode == "neo4j" and not self.is_neo4j_ready():
            raise RuntimeError("Neo4j mode not available — check Neo4j configuration")
        if mode == "hybrid_neo4j":
            if not self.is_graph_ready():
                raise RuntimeError("Graph RAG not available for hybrid mode")
            if not self.is_neo4j_ready():
                raise RuntimeError("Neo4j not available for hybrid mode")
        if mode == "pageindex" and not self.is_pageindex_ready():
            raise RuntimeError(
                "PageIndex mode not available — install the pageindex package "
                "and set gemini_api_key"
            )

        old_mode = self.rag_mode
        self.rag_mode = mode
        self.config_manager.set("rag_mode", mode)
        self.config_manager.save()

        if self.document_processor:
            self.document_processor.rag_mode = mode
        if self.llm_service:
            self.llm_service.rag_mode = mode

        logger.info("RAG mode changed from %s to %s", old_mode, mode)


# Backward compatibility alias
SimpleRAG = EnhancedSimpleRAG