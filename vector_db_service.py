"""
Vector database service using Qdrant with comprehensive error handling
"""

import logging
import time
from typing import List, Dict, Any, Optional
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from extensions import ProgressTracker

logger = logging.getLogger(__name__)

class VectorDBService:
    """Handles interactions with Qdrant vector database with improved reliability."""
    
    def __init__(self, config):
        self.qdrant_url = config["qdrant_url"]
        self.qdrant_api_key = config["qdrant_api_key"]
        self.collection_name = config["collection_name"]
        self.graph_collection_name = config["graph_collection_name"]
        self.embedding_dimension = config["embedding_dimension"]
        
        # Connection state
        self.client = None
        self.is_connected = False
        self.last_error = None
        
        # Initialize client
        self._initialize_client()
    def retry_connection(self):
        """Retry connection to Qdrant with user's current credentials."""
        logger.info("Retrying Qdrant connection...")
        self._initialize_client()
        return self.is_connected
    def _initialize_client(self):
        """Initialize Qdrant client with comprehensive error handling."""
        try:
            # Validate configuration
            if not self.qdrant_url or not self.qdrant_api_key:
                logger.info("Qdrant credentials not configured - service will be unavailable")
                self.client = None
                self.is_connected = False
                self.last_error = "Credentials not configured"
                return
            
            logger.info(f"Initializing Qdrant client with URL: {self.qdrant_url}")
            
            # Try normal connection first
            try:
                if "cloud.qdrant.io" in self.qdrant_url:
                    host = self.qdrant_url.replace("https://", "").replace("http://", "").split(':')[0]
                    
                    self.client = qdrant_client.QdrantClient(
                        host=host,
                        api_key=self.qdrant_api_key,
                        timeout=60,
                        https=True,
                        port=443
                    )
                else:
                    self.client = qdrant_client.QdrantClient(
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        timeout=60
                    )
                
                # Test the connection
                self._test_connection()
                self.is_connected = True
                self.last_error = None
                logger.info("Qdrant client initialized successfully")
                
            except Exception as primary_error:
                logger.warning(f"Primary connection failed: {primary_error}")
                
                # Try alternative connection methods
                if self._try_alternative_connection():
                    logger.info("âœ“ Connected using alternative method")
                else:
                    raise primary_error  # Re-raise if all methods fail
            
        except Exception as e:
            error_msg = f"Failed to initialize Qdrant client: {str(e)}"
            logger.error(error_msg)
            self.client = None
            self.is_connected = False
            self.last_error = error_msg
            # Don't raise - allow graceful degradation
    def _try_alternative_connection(self):
        """Try alternative connection methods for SSL issues."""
        if not self.qdrant_url or not self.qdrant_api_key:
            return False
        
        try:
            # Method 1: Try without SSL verification
            if "https://" in self.qdrant_url:
                logger.info("Trying connection without SSL verification...")
                host = self.qdrant_url.replace("https://", "").replace("http://", "").split(':')[0]
                
                self.client = qdrant_client.QdrantClient(
                    host=host,
                    api_key=self.qdrant_api_key,
                    timeout=60,
                    https=True,
                    port=443,
                    verify=False  # Disable SSL verification
                )
                
                # Test this connection
                self.client.get_collections()
                self.is_connected = True
                self.last_error = None
                logger.info("âœ“ Connected with SSL verification disabled")
                return True
                
        except Exception as e:
            logger.warning(f"Alternative connection method 1 failed: {e}")
        
        try:
            # Method 2: Try with REST API instead of gRPC
            logger.info("Trying REST API connection...")
            import requests
            
            # Simple REST API test
            headers = {"api-key": self.qdrant_api_key}
            response = requests.get(f"{self.qdrant_url}/collections", headers=headers, timeout=10, verify=False)
            
            if response.status_code == 200:
                # If REST works, try initializing client with prefer_grpc=False
                self.client = qdrant_client.QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=60,
                    prefer_grpc=False,  # Use REST instead of gRPC
                    verify=False
                )
                
                self.client.get_collections()
                self.is_connected = True
                self.last_error = None
                logger.info("âœ“ Connected using REST API")
                return True
                
        except Exception as e:
            logger.warning(f"Alternative connection method 2 failed: {e}")
        
        return False
    def _test_connection(self):
        """Test the Qdrant connection with detailed error reporting."""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        try:
            # Test with a simple collections call
            collections_response = self.client.get_collections()
            collection_count = len(collections_response.collections)
            logger.info(f"Connection test successful. Found {collection_count} collections.")
            return True
            
        except Exception as e:
            error_msg = f"Qdrant connection test failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def is_available(self) -> bool:
        """Check if the vector database service is available."""
        if not self.is_connected or not self.client:
            return False
        
        try:
            # Quick health check
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Vector DB availability check failed: {str(e)}")
            self.is_connected = False
            self.last_error = str(e)
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the vector database service."""
        status = {
            "connected": self.is_connected,
            "client_initialized": self.client is not None,
            "url": self.qdrant_url,
            "collections": {
                "normal": self.collection_name,
                "graph": self.graph_collection_name
            }
        }
        
        if self.last_error:
            status["last_error"] = self.last_error
        
        if self.is_available():
            try:
                collections = self.client.get_collections()
                status["total_collections"] = len(collections.collections)
                status["collection_names"] = [c.name for c in collections.collections]
            except Exception as e:
                status["status_error"] = str(e)
        
        return status
    
    def ensure_collection_exists(self, collection_name: str = None) -> bool:
        """Create collection if it doesn't exist. Returns True if created, False if already existed."""
        if not self.is_available():
            raise RuntimeError(f"Vector database not available. Last error: {self.last_error}")
        
        if collection_name is None:
            collection_name = self.collection_name
        
        try:
            # Get existing collections
            collections_response = self.client.get_collections()
            existing_collections = [c.name for c in collections_response.collections]
            
            if collection_name in existing_collections:
                logger.info(f"Collection already exists: {collection_name}")
                return False
            
            # Create new collection
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            
            # Verify creation
            time.sleep(1)  # Give it a moment
            collections_response = self.client.get_collections()
            existing_collections = [c.name for c in collections_response.collections]
            
            if collection_name in existing_collections:
                logger.info(f"Successfully created collection: {collection_name}")
                return True
            else:
                raise RuntimeError(f"Collection creation verification failed for: {collection_name}")
                
        except Exception as e:
            error_msg = f"Error ensuring collection {collection_name} exists: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get detailed information about a collection."""
        if not self.is_available():
            raise RuntimeError("Vector database not available")
        
        if collection_name is None:
            collection_name = self.collection_name
        
        try:
            info = self.client.get_collection(collection_name)
            
            # Extract information safely
            collection_info = {
                "name": collection_name,
                "vectors_count": getattr(info, 'vectors_count', 0) or 0,
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0) or 0,
                "points_count": getattr(info, 'points_count', 0) or 0,
                "config": {"distance": "cosine", "size": self.embedding_dimension}
            }
            
            # Try to extract more detailed config
            try:
                if hasattr(info, 'config') and info.config:
                    if hasattr(info.config, 'params') and info.config.params:
                        if hasattr(info.config.params, 'vectors'):
                            vectors_config = info.config.params.vectors
                            if hasattr(vectors_config, 'distance'):
                                distance = vectors_config.distance
                                collection_info["config"]["distance"] = distance.value if hasattr(distance, 'value') else str(distance)
                            if hasattr(vectors_config, 'size'):
                                collection_info["config"]["size"] = vectors_config.size
            except Exception as config_error:
                logger.debug(f"Could not extract detailed config: {config_error}")
            
            return collection_info
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {str(e)}")
            raise
    
    def insert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], 
                         progress_tracker: Optional[ProgressTracker] = None, 
                         collection_name: str = None):
        """Insert documents with their embeddings into Qdrant with comprehensive error handling."""
        if not self.is_available():
            raise RuntimeError(f"Vector database not available. Last error: {self.last_error}")
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
        
        if collection_name is None:
            collection_name = self.collection_name
        
        logger.info(f"Inserting {len(documents)} documents into collection: {collection_name}")
        
        if progress_tracker:
            progress_tracker.update(0, len(documents), status="storing", 
                                   message=f"Storing documents in collection: {collection_name}")
        
        try:
            # Ensure collection exists
            self.ensure_collection_exists(collection_name)
            
            # Prepare points for insertion
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Validate embedding
                if not embedding or len(embedding) != self.embedding_dimension:
                    logger.warning(f"Invalid embedding at index {i}, skipping document")
                    continue
                
                # Create point
                point = models.PointStruct(
                    id=i + int(time.time() * 1000),  # Unique ID
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {})
                    }
                )
                points.append(point)
            
            if not points:
                raise ValueError("No valid documents to insert")
            
            # Insert in batches for better reliability
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(points) + batch_size - 1) // batch_size
                
                try:
                    logger.debug(f"Inserting batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    
                    total_inserted += len(batch)
                    
                    if progress_tracker:
                        progress_tracker.update(
                            total_inserted, len(points), 
                            message=f"Stored {total_inserted} of {len(points)} documents"
                        )
                    
                    logger.debug(f"Successfully inserted batch {batch_num}")
                    
                    # Small delay between batches
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error inserting batch {batch_num}: {str(e)}")
                    raise RuntimeError(f"Failed to insert batch {batch_num}: {str(e)}")
            
            # Verify insertion
            try:
                collection_info = self.get_collection_info(collection_name)
                final_count = collection_info["points_count"]
                logger.info(f"Insertion complete. Collection {collection_name} now has {final_count} points")
            except Exception as e:
                logger.warning(f"Could not verify insertion: {e}")
            
            if progress_tracker:
                progress_tracker.update(len(documents), len(documents), status="complete", 
                                       message=f"Successfully stored {total_inserted} documents")
            
            logger.info(f"Successfully inserted {total_inserted} documents into {collection_name}")
            
        except Exception as e:
            error_msg = f"Error inserting documents into {collection_name}: {str(e)}"
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.update(len(documents), len(documents), status="error", 
                                       message=f"Error: {str(e)}")
            raise RuntimeError(error_msg)
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      filter_condition=None, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity with error handling."""
        if not self.is_available():
            logger.error(f"Vector database not available for search. Last error: {self.last_error}")
            return []
        
        if collection_name is None:
            collection_name = self.collection_name
        
        if not query_embedding or len(query_embedding) != self.embedding_dimension:
            logger.error(f"Invalid query embedding: expected {self.embedding_dimension} dimensions, got {len(query_embedding) if query_embedding else 0}")
            return []
        
        try:
            # Ensure collection exists
            self.ensure_collection_exists(collection_name)
            
            logger.debug(f"Searching in collection: {collection_name} with top_k={top_k}")
            
            # Use query_points for qdrant-client >= 1.7 (current standard API)
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=filter_condition,
                with_payload=True
            ).points
            
            # Process results
            results = []
            for scored_point in search_result:
                try:
                    result = {
                        "text": scored_point.payload.get("text", ""),
                        "metadata": scored_point.payload.get("metadata", {}),
                        "score": float(scored_point.score)
                    }
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            logger.debug(f"Search completed: found {len(results)} results in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        if not self.is_available():
            raise RuntimeError("Vector database not available")
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            raise
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their information."""
        if not self.is_available():
            raise RuntimeError("Vector database not available")
        
        try:
            collections_response = self.client.get_collections()
            collections_info = []
            
            for collection in collections_response.collections:
                try:
                    info = self.get_collection_info(collection.name)
                    collections_info.append(info)
                except Exception as e:
                    logger.warning(f"Could not get info for collection {collection.name}: {e}")
                    # Add basic info
                    collections_info.append({
                        "name": collection.name,
                        "vectors_count": 0,
                        "indexed_vectors_count": 0,
                        "points_count": 0,
                        "config": {"distance": "unknown", "size": 0},
                        "error": str(e)
                    })
            
            return collections_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise