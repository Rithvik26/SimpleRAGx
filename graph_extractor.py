"""
Graph extractor for extracting entities and relationships from text using Gemini API
"""

import json
import logging
import time
import requests
from typing import Dict, Any, List, Optional
from extensions import RateLimiter, rate_limited

logger = logging.getLogger(__name__)

class GraphExtractor:
    """Extracts entities and relationships from text using Gemini API with improved error handling."""
    
    def __init__(self, config):
        self.api_key = config["gemini_api_key"]
        self.max_entities_per_chunk = config.get("max_entities_per_chunk", 20)
        self.max_chunk_length = config.get("max_chunk_length_for_graph", 2000)
        self.extraction_timeout = config.get("graph_extraction_timeout", 45)
        self.rate_limiter = RateLimiter(calls_per_minute=config.get("rate_limit", 30))
        
        if not self.api_key:
            raise ValueError("Gemini API key is required for graph extraction")
        
        logger.info(f"GraphExtractor initialized with max_entities={self.max_entities_per_chunk}")
    
    def _prepare_text_for_extraction(self, text: str) -> str:
        """Prepare text for entity extraction by cleaning and truncating."""
        # Clean and normalize the text
        cleaned_text = text.strip()
        
        # Remove excessive whitespace
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Truncate if too long
        if len(cleaned_text) > self.max_chunk_length:
            # Try to truncate at sentence boundary
            sentences = cleaned_text[:self.max_chunk_length].split('. ')
            if len(sentences) > 1:
                cleaned_text = '. '.join(sentences[:-1]) + '.'
            else:
                cleaned_text = cleaned_text[:self.max_chunk_length]
            
            logger.debug(f"Truncated text from {len(text)} to {len(cleaned_text)} characters")
        
        return cleaned_text
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a well-structured prompt for entity and relationship extraction."""
        return f"""
You are an expert knowledge graph extractor. Extract entities and relationships from the following text.

TEXT TO ANALYZE:
{text}

INSTRUCTIONS:
1. Extract the {self.max_entities_per_chunk} most important entities from the text
2. For each entity, provide:
   - name: The entity name (be precise and consistent)
   - type: One of these types only: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, PRODUCT, TECHNOLOGY, DATE, NUMBER
   - description: A brief, factual description (2-3 words max)

3. Extract meaningful relationships between entities as triplets:
   - source: Source entity name (must match an entity name above)
   - relationship: Relationship type (use simple verbs like: leads, owns, develops, located_in, part_of, works_for, created, founded)
   - target: Target entity name (must match an entity name above)
   - description: Brief description of the relationship (optional)

4. Only include relationships where BOTH entities are in your entity list
5. Focus on factual, explicit relationships mentioned in the text
6. Return valid JSON only, no additional text

REQUIRED JSON FORMAT:
{{
    "entities": [
        {{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "brief description"}},
        ...
    ],
    "relationships": [
        {{"source": "Entity1", "relationship": "relationship_type", "target": "Entity2", "description": "optional description"}},
        ...
    ]
}}

Return only the JSON, no other text:"""
    
    @rate_limited(RateLimiter(calls_per_minute=30))
    def _call_gemini_api(self, prompt: str, retry_count: int = 3) -> Dict[str, Any]:
        """Call Gemini API for content generation with retry logic."""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for consistent JSON
                "maxOutputTokens": 2048,
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
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                logger.debug(f"Gemini API call attempt {attempt + 1}")
                response = requests.post(
                    url, 
                    headers=headers, 
                    params=params, 
                    json=data, 
                    timeout=self.extraction_timeout
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated text
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        generated_text = candidate["content"]["parts"][0]["text"]
                        return self._parse_extraction_response(generated_text)
                    else:
                        raise ValueError("No content in API response")
                else:
                    raise ValueError("No candidates in API response")
                
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout on attempt {attempt + 1}: {str(e)}"
                logger.warning(last_error)
                if attempt < retry_count - 1:
                    time.sleep(2)
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request error on attempt {attempt + 1}: {str(e)}"
                logger.warning(last_error)
                if attempt < retry_count - 1:
                    time.sleep(2)
                    
            except Exception as e:
                last_error = f"Processing error on attempt {attempt + 1}: {str(e)}"
                logger.warning(last_error)
                if attempt < retry_count - 1:
                    time.sleep(1)
        
        # All attempts failed
        logger.error(f"Failed to extract entities after {retry_count} attempts. Last error: {last_error}")
        return {"entities": [], "relationships": []}
    
    def _parse_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from Gemini API with robust error handling."""
        try:
            # Clean up the response text
            json_text = response_text.strip()
            
            # Remove markdown formatting if present
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            elif json_text.startswith("```"):
                json_text = json_text[3:]
            
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            json_text = json_text.strip()
            
            # Try to find JSON in the response if it's embedded in other text
            if not json_text.startswith('{'):
                # Look for JSON object in the text
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_text = json_text[start_idx:end_idx + 1]
            
            # Parse the JSON
            graph_data = json.loads(json_text)
            
            # Validate the structure
            if not isinstance(graph_data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Ensure required keys exist with defaults
            if "entities" not in graph_data:
                graph_data["entities"] = []
            if "relationships" not in graph_data:
                graph_data["relationships"] = []
            
            # Validate entities
            validated_entities = []
            for entity in graph_data.get("entities", []):
                if self._validate_entity(entity):
                    validated_entities.append(entity)
            
            # Validate relationships
            entity_names = {e["name"] for e in validated_entities}
            validated_relationships = []
            for rel in graph_data.get("relationships", []):
                if self._validate_relationship(rel, entity_names):
                    validated_relationships.append(rel)
            
            result = {
                "entities": validated_entities,
                "relationships": validated_relationships
            }
            
            logger.debug(f"Parsed {len(result['entities'])} entities and {len(result['relationships'])} relationships")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Raw response: {response_text[:200]}...")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.warning(f"Error processing extraction response: {e}")
            return {"entities": [], "relationships": []}
    
    def _validate_entity(self, entity: Dict[str, Any]) -> bool:
        """Validate that an entity has the required structure."""
        required_fields = ["name", "type", "description"]
        
        # Check all required fields are present
        for field in required_fields:
            if field not in entity:
                logger.debug(f"Entity missing required field: {field}")
                return False
        
        # Validate entity name
        if not entity["name"] or not isinstance(entity["name"], str):
            logger.debug("Entity has invalid name")
            return False
        
        # Validate entity type
        valid_types = ["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY", "DATE", "NUMBER"]
        if entity["type"] not in valid_types:
            logger.debug(f"Entity has invalid type: {entity['type']}")
            return False
        
        # Validate description
        if not isinstance(entity["description"], str):
            logger.debug("Entity has invalid description")
            return False
        
        return True
    
    def _validate_relationship(self, relationship: Dict[str, Any], valid_entities: set) -> bool:
        """Validate that a relationship has the required structure and references valid entities."""
        required_fields = ["source", "relationship", "target"]
        
        # Check all required fields are present
        for field in required_fields:
            if field not in relationship:
                logger.debug(f"Relationship missing required field: {field}")
                return False
        
        # Validate source and target entities exist
        if relationship["source"] not in valid_entities:
            logger.debug(f"Relationship source entity not found: {relationship['source']}")
            return False
        
        if relationship["target"] not in valid_entities:
            logger.debug(f"Relationship target entity not found: {relationship['target']}")
            return False
        
        # Validate relationship type
        if not relationship["relationship"] or not isinstance(relationship["relationship"], str):
            logger.debug("Relationship has invalid relationship type")
            return False
        
        # Ensure source and target are different
        if relationship["source"] == relationship["target"]:
            logger.debug("Relationship source and target are the same")
            return False
        
        return True
    
    def extract_entities_and_relationships(self, text: str, chunk_id: str = None) -> Dict[str, Any]:
        """Extract entities and relationships from text using Gemini API."""
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return {"entities": [], "relationships": []}
        
        try:
            # Prepare text for extraction
            prepared_text = self._prepare_text_for_extraction(text)
            
            if len(prepared_text) < 50:  # Too short to extract meaningful entities
                logger.debug("Text too short for entity extraction")
                return {"entities": [], "relationships": []}
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(prepared_text)
            
            # Call API
            graph_data = self._call_gemini_api(prompt)
            
            # Add metadata to entities and relationships
            if chunk_id:
                for entity in graph_data.get("entities", []):
                    entity["source_chunk"] = chunk_id
                    entity["source_text"] = text[:200] + "..." if len(text) > 200 else text
                
                for rel in graph_data.get("relationships", []):
                    rel["source_chunk"] = chunk_id
                    rel["source_text"] = text[:200] + "..." if len(text) > 200 else text
            
            entities_count = len(graph_data.get("entities", []))
            relationships_count = len(graph_data.get("relationships", []))
            
            logger.info(f"Successfully extracted {entities_count} entities and {relationships_count} relationships from chunk")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return {"entities": [], "relationships": []}
    
    def extract_from_multiple_chunks(self, chunks: List[str], progress_callback=None) -> Dict[str, Any]:
        """Extract entities and relationships from multiple text chunks."""
        all_entities = []
        all_relationships = []
        
        total_chunks = len(chunks)
        logger.info(f"Starting extraction from {total_chunks} chunks")
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"chunk_{i}_{int(time.time())}"
                
                # Extract from this chunk
                chunk_result = self.extract_entities_and_relationships(chunk, chunk_id)
                
                # Accumulate results
                all_entities.extend(chunk_result.get("entities", []))
                all_relationships.extend(chunk_result.get("relationships", []))
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, total_chunks, f"Processed chunk {i + 1} of {total_chunks}")
                
                # Small delay to be respectful to API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
        
        logger.info(f"Extraction complete: {len(all_entities)} total entities, {len(all_relationships)} total relationships")
        
        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "chunks_processed": total_chunks
        }
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extraction service."""
        return {
            "api_key_configured": bool(self.api_key),
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "max_chunk_length": self.max_chunk_length,
            "extraction_timeout": self.extraction_timeout
        }