"""
Enhanced SimpleRAG - Main orchestrator class combining all services
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional

# Import all the modular services
from config import ConfigManager, get_config_manager
from embedding_service import EmbeddingService
from vector_db_service import VectorDBService
from graph_rag_service import GraphRAGService
from document_processor import DocumentProcessor
from llm_service import LLMService
from extensions import ProgressTracker

logger = logging.getLogger(__name__)

class EnhancedSimpleRAG:
    """Enhanced SimpleRAG with both Normal and Graph RAG capabilities."""
    
    def __init__(self, config_manager: ConfigManager = None):
        """Initialize SimpleRAG with comprehensive error handling and service validation."""
        # Configuration
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_all()
        
        # Current RAG mode
        self.rag_mode = self.config.get("rag_mode", "normal")
        
        # Service instances
        self.document_processor = None
        self.embedding_service = None
        self.vector_db_service = None
        self.llm_service = None
        self.graph_rag_service = None
        
        # Track initialization status
        self.initialization_errors = []
        self.initialization_warnings = []
        
        # Initialize all services
        self._initialize_services()
        
        # Log final status
        self._log_initialization_status()
    
    def _initialize_services(self):
        """Initialize all services with comprehensive error handling."""
        logger.info("Initializing Enhanced SimpleRAG services...")
        
        # 1. Document Processor (should always work)
        try:
            self.document_processor = DocumentProcessor(self.config)
            logger.info("✓ Document processor initialized")
        except Exception as e:
            error_msg = f"Failed to initialize document processor: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        # 2. Embedding Service
        try:
            if not self.config.get("gemini_api_key"):
                raise ValueError("Gemini API key not configured")
            
            self.embedding_service = EmbeddingService(self.config)
            
            # Test the service
            test_embedding = self.embedding_service.get_embedding("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding service test failed")
            
            logger.info("✓ Embedding service initialized and tested")
        except Exception as e:
            error_msg = f"Failed to initialize embedding service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        # 3. Vector Database Service
        try:
            if not self.config.get("qdrant_url") or not self.config.get("qdrant_api_key"):
                raise ValueError("Qdrant URL or API key not configured")
            
            self.vector_db_service = VectorDBService(self.config)
            
            # Test the service
            if not self.vector_db_service.is_available():
                raise RuntimeError(f"Qdrant service not available: {self.vector_db_service.last_error}")
            
            logger.info("✓ Vector database service initialized and tested")
        except Exception as e:
            error_msg = f"Failed to initialize vector database service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            self.vector_db_service = None
        
        # 4. LLM Service
        try:
            self.llm_service = LLMService(self.config)
            
            if self.config.get("preferred_llm") == "claude":
                # Test Claude connection if configured
                if self.config.get("claude_api_key"):
                    test_result = self.llm_service.test_connection()
                    if not test_result.get("service_available"):
                        self.initialization_warnings.append(f"Claude API test failed: {test_result.get('error')}")
                else:
                    self.initialization_warnings.append("Claude API key not configured")
            
            logger.info("✓ LLM service initialized")
        except Exception as e:
            error_msg = f"Failed to initialize LLM service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
        
        # 5. Graph RAG Service (only if other services are available)
        try:
            if self.embedding_service and self.vector_db_service:
                self.graph_rag_service = GraphRAGService(self.config)
                self.graph_rag_service.set_services(self.embedding_service, self.vector_db_service)
                logger.info("✓ Graph RAG service initialized")
            else:
                self.initialization_warnings.append("Graph RAG service skipped - dependencies not available")
        except Exception as e:
            error_msg = f"Failed to initialize Graph RAG service: {str(e)}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
    
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
            logger.info("✓ Enhanced SimpleRAG initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if SimpleRAG is ready for basic operations."""
        return (self.embedding_service is not None and 
                self.vector_db_service is not None and 
                self.vector_db_service.is_available() and
                self.document_processor is not None)
    
    def is_graph_ready(self) -> bool:
        """Check if Graph RAG functionality is ready."""
        return (self.is_ready() and 
                self.graph_rag_service is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all services."""
        status = {
            "ready": self.is_ready(),
            "graph_ready": self.is_graph_ready(),
            "rag_mode": self.rag_mode,
            "services": {
                "document_processor": self.document_processor is not None,
                "embedding_service": self.embedding_service is not None,
                "vector_db_service": self.vector_db_service is not None and self.vector_db_service.is_available(),
                "llm_service": self.llm_service is not None,
                "graph_rag_service": self.graph_rag_service is not None
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
        
        return status
    
    def set_rag_mode(self, mode: str):
        """Switch between 'normal' and 'graph' RAG modes."""
        if mode not in ["normal", "graph"]:
            raise ValueError("RAG mode must be 'normal' or 'graph'")
        
        if mode == "graph" and not self.is_graph_ready():
            raise RuntimeError("Graph RAG mode not available - check service initialization")
        
        old_mode = self.rag_mode
        self.rag_mode = mode
        
        # Update configuration
        self.config_manager.set("rag_mode", mode)
        self.config_manager.save()
        
        # Update services
        if self.document_processor:
            self.document_processor.rag_mode = mode
        if self.llm_service:
            self.llm_service.rag_mode = mode
        
        logger.info(f"RAG mode changed from {old_mode} to {mode}")
    
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
    
    def index_document(self, file_path: str, session_id: str = None) -> bool:
        """Process and index a document using the configured RAG mode."""
        if not self.is_ready():
            logger.error("SimpleRAG not ready for indexing")
            return False
        
        # Validate file first
        validation = self.validate_file(file_path)
        if not validation["valid"]:
            logger.error(f"File validation failed: {validation['errors']}")
            return False
        
        try:
            progress_tracker = None
            if session_id:
                progress_tracker = ProgressTracker(session_id, "index_document")
                progress_tracker.update(0, 100, status="starting", 
                                      message=f"Starting document indexing in {self.rag_mode} mode")
            
            logger.info(f"Starting document indexing: {file_path} (mode: {self.rag_mode})")
            
            # Step 1: Extract text from document
            text = self.document_processor.extract_text_from_file(file_path, progress_tracker)
            
            if not text or not text.strip():
                logger.error("No text extracted from document")
                if progress_tracker:
                    progress_tracker.update(100, 100, status="error", 
                                          message="No text could be extracted from document")
                return False
            
            logger.info(f"Extracted {len(text)} characters from document")
            
            # Step 2: Create metadata
            filename = os.path.basename(file_path)
            metadata = {
                "filename": filename,
                "path": file_path,
                "created_at": time.time(),
                "file_type": os.path.splitext(filename)[1][1:].lower(),
                "rag_mode": self.rag_mode,
                "text_length": len(text)
            }
            
            if session_id:
                metadata["session_id"] = session_id
            
            # Step 3: Process based on RAG mode
            if self.rag_mode == "graph":
                return self._index_document_graph_mode(text, metadata, progress_tracker)
            else:
                return self._index_document_normal_mode(text, metadata, progress_tracker)
                
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return False
    
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
            
            if self.rag_mode == "graph" and self.is_graph_ready():
                return self._query_graph_mode(question, progress_tracker)
            else:
                return self._query_normal_mode(question, progress_tracker)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    def _query_normal_mode(self, question: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Query using normal RAG mode."""
        try:
            # Step 1: Generate query embedding
            if progress_tracker:
                progress_tracker.update(20, 100, status="embedding", 
                                      message="Generating query embedding")
            
            query_embedding = self.embedding_service.get_embedding(question)
            
            # Step 2: Search for similar chunks
            if progress_tracker:
                progress_tracker.update(40, 100, status="searching", 
                                      message="Searching for relevant documents")
            
            contexts = self.vector_db_service.search_similar(
                query_embedding,
                top_k=self.config["top_k"],
                collection_name=self.config["collection_name"]
            )
            
            logger.info(f"Found {len(contexts)} relevant contexts")
            
            if not contexts:
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", 
                                          message="No relevant information found")
                return "I couldn't find any relevant information to answer your question. Please ensure documents have been indexed."
            
            # Step 3: Generate answer
            if progress_tracker:
                progress_tracker.update(60, 100, status="generating", 
                                      message="Generating answer")
            
            if self.llm_service and self.llm_service.is_available():
                answer = self.llm_service.generate_answer(question, contexts, progress_tracker=progress_tracker)
            else:
                # Fallback to raw results
                answer = self._format_raw_results(contexts)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message="Answer generated successfully")
            
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
            
            # Search document chunks
            doc_contexts = self.vector_db_service.search_similar(
                query_embedding,
                top_k=self.config["top_k"] // 2,
                collection_name=self.config["collection_name"]
            )
            
            # Search graph elements
            graph_contexts = self.graph_rag_service.search_graph(
                question,
                top_k=self.config["top_k"] // 2
            )
            
            # Combine contexts
            all_contexts = doc_contexts + graph_contexts
            
            logger.info(f"Found {len(doc_contexts)} document contexts and {len(graph_contexts)} graph contexts")
            
            if not all_contexts:
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", 
                                          message="No relevant information found")
                return "I couldn't find any relevant information to answer your question in either the documents or knowledge graph."
            
            # Step 3: Prepare graph context for enhanced prompting
            if progress_tracker:
                progress_tracker.update(50, 100, status="analyzing", 
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
            filename = ctx['metadata'].get('filename', 'Unknown')
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
                filename = ctx['metadata'].get('filename', 'Unknown')
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
                    source = ctx['metadata'].get('source', 'Unknown')
                    target = ctx['metadata'].get('target', 'Unknown')
                    rel_type = ctx['metadata'].get('relationship', 'unknown')
                    score = ctx.get('score', 0)
                    results.append(f"    - {source} → {rel_type} → {target} (Score: {score:.3f})")
        
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


# Backward compatibility alias
SimpleRAG = EnhancedSimpleRAG