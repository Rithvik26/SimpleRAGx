"""
Embedding service for SimpleRAG using Gemini API
"""

import logging
import time
import requests
from typing import List, Optional
from extensions import RateLimiter, EmbeddingCache, rate_limited, ProgressTracker

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Handles creation of embeddings using Gemini API with improved error handling."""
    
    def __init__(self, config):
        self.api_key = config["gemini_api_key"]
        self.embedding_dimension = config["embedding_dimension"]
        self.rate_limiter = RateLimiter(calls_per_minute=config.get("rate_limit", 60))
        
        # Initialize cache if enabled
        self.enable_cache = config.get("enable_cache", True)
        if self.enable_cache:
            cache_dir = config.get("cache_dir")
            self.cache = EmbeddingCache(cache_dir=cache_dir)
            logger.info("Embedding cache enabled")
        else:
            logger.info("Embedding cache disabled")
        
        
    
    def _prepare_text_for_embedding(self, text: str) -> str:
        """Prepare text for embedding by cleaning and truncating."""
        # Clean the text
        cleaned_text = text.strip()
        
        # Truncate to maximum length (Gemini embedding has limits)
        max_length = 8000  # Conservative limit for Gemini
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]
            logger.debug(f"Truncated text from {len(text)} to {len(cleaned_text)} characters")
        
        return cleaned_text
    
    @rate_limited(RateLimiter(calls_per_minute=60))
    def _get_embedding_from_api(self, text: str, retry_count: int = 3) -> List[float]:
        """Generate embedding vector for text using Gemini API with retry logic."""
        prepared_text = self._prepare_text_for_embedding(text)
        
        # Use the correct Gemini embedding endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        data = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": prepared_text}]},
            "taskType": "RETRIEVAL_DOCUMENT"
        }
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                logger.debug(f"Embedding API call attempt {attempt + 1}")
                response = requests.post(
                    url, 
                    headers=headers, 
                    params=params, 
                    json=data, 
                    timeout=30
                )
                
                # Check for HTTP errors
                if response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Extract embedding from response
                embedding = result.get("embedding", {}).get("values", [])
                
                if not embedding:
                    raise ValueError("No embedding values returned from API")
                
                # Validate embedding dimension
                if len(embedding) != self.embedding_dimension:
                    logger.warning(f"Expected {self.embedding_dimension} dimensions, got {len(embedding)}")
                
                logger.debug(f"Successfully generated embedding with {len(embedding)} dimensions")
                return embedding
                
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout on attempt {attempt + 1}: {str(e)}"
                logger.warning(last_error)
                if attempt < retry_count - 1:
                    time.sleep(1)
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request error on attempt {attempt + 1}: {str(e)}"
                logger.warning(last_error)
                if attempt < retry_count - 1:
                    time.sleep(1)
                    
            except (KeyError, ValueError) as e:
                last_error = f"Response parsing error on attempt {attempt + 1}: {str(e)}"
                logger.warning(last_error)
                if attempt < retry_count - 1:
                    time.sleep(1)
        
        # All attempts failed
        error_msg = f"Failed to generate embedding after {retry_count} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if enabled."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dimension
        
        # Check cache first if enabled
        if self.enable_cache and hasattr(self, 'cache'):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                logger.debug("Retrieved embedding from cache")
                return cached_embedding
        
        try:
            # Generate new embedding
            embedding = self._get_embedding_from_api(text)
            
            # Store in cache if enabled
            if self.enable_cache and hasattr(self, 'cache'):
                self.cache.set(text, embedding)
                logger.debug("Stored embedding in cache")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            logger.warning("Returning zero vector as fallback")
            return [0.0] * self.embedding_dimension
    
    def get_embeddings_batch(self, texts: List[str], 
                           progress_tracker: Optional[ProgressTracker] = None,
                           batch_delay: float = 0.1) -> List[List[float]]:
        """Generate embeddings for multiple texts with progress tracking and rate limiting."""
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        embeddings = []
        total_texts = len(texts)
        failed_count = 0
        
        if progress_tracker:
            progress_tracker.update(0, total_texts, status="embedding", 
                                   message="Generating embeddings")
        
        logger.info(f"Starting batch embedding for {total_texts} texts")
        
        for i, text in enumerate(texts):
            try:
                if not text or not text.strip():
                    logger.warning(f"Empty text at index {i}, using zero vector")
                    embeddings.append([0.0] * self.embedding_dimension)
                    continue
                
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                
                if progress_tracker:
                    progress_tracker.update(
                        i + 1, total_texts, 
                        message=f"Generated embedding {i + 1} of {total_texts}"
                    )
                
                # Add delay to respect rate limits
                if batch_delay > 0:
                    time.sleep(batch_delay)
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating embedding for text {i}: {str(e)}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.embedding_dimension)
                
                if progress_tracker:
                    progress_tracker.update(
                        i + 1, total_texts, 
                        message=f"Error with embedding {i + 1}: {str(e)[:50]}..."
                    )
        
        success_count = total_texts - failed_count
        logger.info(f"Batch embedding completed: {success_count}/{total_texts} successful")
        
        if failed_count > 0:
            logger.warning(f"{failed_count} embeddings failed and were replaced with zero vectors")
        
        return embeddings
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding is properly formatted."""
        if not isinstance(embedding, list):
            return False
        
        if len(embedding) != self.embedding_dimension:
            return False
        
        # Check if all values are numbers and not all zero
        try:
            float_values = [float(x) for x in embedding]
            is_all_zero = all(abs(x) < 1e-10 for x in float_values)
            return not is_all_zero  # Valid if not all zeros
        except (ValueError, TypeError):
            return False
    
    def get_embedding_stats(self) -> dict:
        """Get statistics about embedding usage."""
        stats = {
            "cache_enabled": self.enable_cache,
            "embedding_dimension": self.embedding_dimension,
            "api_key_configured": bool(self.api_key)
        }
        
        if self.enable_cache and hasattr(self, 'cache'):
            try:
                cache_files = list(self.cache.cache_dir.glob("*.json"))
                stats["cached_embeddings"] = len(cache_files)
            except Exception:
                stats["cached_embeddings"] = "unknown"
        
        return stats
    
    def clear_cache(self) -> bool:
        """Clear the embedding cache."""
        if not self.enable_cache or not hasattr(self, 'cache'):
            logger.warning("Cache not enabled, nothing to clear")
            return False
        
        try:
            import shutil
            if self.cache.cache_dir.exists():
                shutil.rmtree(self.cache.cache_dir)
                self.cache.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Embedding cache cleared successfully")
                return True
            else:
                logger.info("Cache directory doesn't exist")
                return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False