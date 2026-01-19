"""
LLM service for generating answers using Claude API - FULLY OPTIMIZED VERSION
Includes: Intelligent response length, speed optimization, lazy analysis, smart model selection
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import anthropic
from extensions import ProgressTracker
import html
import re
import hashlib

logger = logging.getLogger(__name__)

class QueryComplexityAnalyzer:
    """Analyzes query complexity with caching for optimal performance."""
    
    # Query patterns that indicate different complexity levels
    LIST_INDICATORS = [
        'list', 'lists', 'enumerate', 'all', 'every', 'each', 'what are',
        'which are', 'show me', 'give me', 'provide', 'identify all'
    ]
    
    DETAIL_INDICATORS = [
        'explain', 'describe', 'how', 'why', 'detailed', 'comprehensive',
        'elaborate', 'in detail', 'thoroughly', 'breakdown', 'deep dive',
        'step by step', 'walk through'
    ]
    
    TIMELINE_INDICATORS = [
        'timeline', 'history', 'chronology', 'sequence', 'evolution',
        'over time', 'progression', 'development'
    ]
    
    COMPARISON_INDICATORS = [
        'compare', 'versus', 'vs', 'difference', 'contrast', 'similarities',
        'both', 'either', 'between'
    ]
    
    SIMPLE_INDICATORS = [
        'what is', 'who is', 'when', 'where', 'define', 'definition'
    ]
    
    def __init__(self):
        """Initialize analyzer with pattern cache for performance."""
        self._pattern_cache = {}
        self._cache_size = 100
        self._cache_hits = 0
        self._cache_misses = 0
    
    @staticmethod
    def _get_query_signature(query: str, num_contexts: int, rag_mode: str) -> str:
        """Create a signature for caching (hash of normalized query pattern)."""
        # Normalize query to first 10 significant words
        words = query.lower().split()[:10]
        normalized = ' '.join(sorted(words))
        signature = f"{normalized}_{num_contexts}_{rag_mode}"
        return hashlib.md5(signature.encode()).hexdigest()[:12]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_size': len(self._pattern_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }
    
    def analyze(self, query: str, contexts: List[Dict[str, Any]], rag_mode: str) -> Dict[str, Any]:
        """
        Analyze query complexity with caching for performance.
        
        Returns:
            Dict containing:
            - complexity_level: 'simple', 'medium', 'complex', 'very_complex'
            - recommended_max_tokens: int
            - response_type: str (description of expected response)
            - reasoning: str (why this classification was made)
            - from_cache: bool (whether result came from cache)
        """
        # Check cache first
        cache_key = self._get_query_signature(query, len(contexts), rag_mode)
        if cache_key in self._pattern_cache:
            self._cache_hits += 1
            cached = self._pattern_cache[cache_key].copy()
            cached['from_cache'] = True
            logger.debug(f"Cache hit for query pattern (hit rate: {self.get_cache_stats()['hit_rate']})")
            return cached
        
        self._cache_misses += 1
        
        # Perform analysis
        query_lower = query.lower()
        
        # Count indicators
        list_score = sum(1 for indicator in self.LIST_INDICATORS if indicator in query_lower)
        detail_score = sum(1 for indicator in self.DETAIL_INDICATORS if indicator in query_lower)
        timeline_score = sum(1 for indicator in self.TIMELINE_INDICATORS if indicator in query_lower)
        comparison_score = sum(1 for indicator in self.COMPARISON_INDICATORS if indicator in query_lower)
        simple_score = sum(1 for indicator in self.SIMPLE_INDICATORS if indicator in query_lower)
        
        # Context analysis
        num_contexts = len(contexts)
        avg_context_length = sum(len(ctx.get('text', '')) for ctx in contexts) / max(num_contexts, 1)
        total_context_chars = sum(len(ctx.get('text', '')) for ctx in contexts)
        
        # Calculate total complexity score
        total_score = list_score * 3 + detail_score * 2 + timeline_score * 2 + comparison_score * 2
        
        # Determine complexity level and token allocation
        if simple_score > 0 and total_score == 0 and num_contexts <= 2:
            complexity_level = 'simple'
            base_tokens = 1500
            response_type = 'Concise factual answer'
            reasoning = 'Simple factual query with clear answer'
            
        elif list_score >= 2 or (list_score >= 1 and num_contexts > 5):
            complexity_level = 'complex'
            base_tokens = 4000
            response_type = 'Comprehensive list with details'
            reasoning = f'List query requiring enumeration of {num_contexts} relevant items'
            
        elif detail_score >= 2 or timeline_score >= 1 or comparison_score >= 2:
            complexity_level = 'very_complex'
            base_tokens = 6000
            response_type = 'Detailed explanation or comparison'
            reasoning = 'Query requires detailed explanation, timeline, or multi-faceted comparison'
            
        elif total_score >= 3 or num_contexts >= 8:
            complexity_level = 'complex'
            base_tokens = 4000
            response_type = 'Multi-source synthesis'
            reasoning = f'Complex query requiring synthesis of {num_contexts} sources'
            
        elif num_contexts >= 4 or total_score >= 1:
            complexity_level = 'medium'
            base_tokens = 2500
            response_type = 'Moderate detail answer'
            reasoning = 'Moderate complexity requiring multiple sources'
            
        else:
            complexity_level = 'simple'
            base_tokens = 1500
            response_type = 'Brief answer'
            reasoning = 'Straightforward query'
        
        # Adjust for RAG mode
        if rag_mode == 'graph':
            base_tokens += 1000  # Graph RAG needs more tokens for relationship chains
        
        # Adjust based on total context size
        if total_context_chars > 5000:
            base_tokens = min(base_tokens + 1000, 8000)
        
        # Cap at model limits
        max_tokens = min(base_tokens, 8000)
        
        result = {
            'complexity_level': complexity_level,
            'recommended_max_tokens': max_tokens,
            'response_type': response_type,
            'reasoning': reasoning,
            'from_cache': False,
            'indicators': {
                'list_score': list_score,
                'detail_score': detail_score,
                'timeline_score': timeline_score,
                'comparison_score': comparison_score,
                'simple_score': simple_score,
                'num_contexts': num_contexts,
                'avg_context_length': int(avg_context_length)
            }
        }
        
        # Cache result
        if len(self._pattern_cache) >= self._cache_size:
            # Remove oldest entry
            self._pattern_cache.pop(next(iter(self._pattern_cache)))
        self._pattern_cache[cache_key] = result.copy()
        
        return result


class LLMService:
    """Handles interactions with LLM APIs with intelligent response length management and speed optimization."""
    
    def __init__(self, config):
        self.preferred_llm = config["preferred_llm"]
        self.claude_api_key = config.get("claude_api_key", "")
        self.rag_mode = config.get("rag_mode", "normal")
        
        # Initialize query complexity analyzer with caching
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Initialize Claude client if available
        self.claude_client = None
        if self.preferred_llm == "claude" and self.claude_api_key:
            self._initialize_claude_client()
        
        # Track last complexity analysis
        self._last_complexity_analysis = None
        
        logger.info(f"LLMService initialized with preferred_llm={self.preferred_llm}, rag_mode={self.rag_mode}")
    
    def _initialize_claude_client(self):
        """Initialize Claude client with error handling."""
        try:
            self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
            logger.info("Claude client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {str(e)}")
            self.claude_client = None
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        if self.preferred_llm == "claude":
            return self.claude_client is not None
        elif self.preferred_llm == "raw":
            return True
        return False
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]], 
                       graph_context: Dict[str, Any] = None, 
                       progress_tracker: Optional[ProgressTracker] = None) -> str:
        """
        Generate answer with LAZY complexity analysis (zero overhead for simple queries).
        
        Speed optimization: Fast-path for obviously simple queries.
        """
        
        # ✅ SPEED OPTIMIZATION: LAZY ANALYSIS WITH FAST PATH
        query_lower = query.lower()
        num_contexts = len(contexts)
        
        # Fast path for obviously simple queries (skip full analysis)
        if (num_contexts <= 2 and 
            any(indicator in query_lower for indicator in ['what is', 'who is', 'when', 'where', 'define'])):
            
            self._last_complexity_analysis = {
                'complexity_level': 'simple',
                'recommended_max_tokens': 1500,
                'response_type': 'Concise answer',
                'reasoning': 'Fast path - simple factual query',
                'from_cache': False,
                'fast_path': True
            }
            logger.debug("Using fast path for simple query (no full analysis needed)")
        else:
            # Complex query - perform full analysis with caching
            if progress_tracker:
                progress_tracker.update(65, 100, status="analyzing", 
                                      message="Analyzing query complexity")
            
            self._last_complexity_analysis = self.complexity_analyzer.analyze(
                query, contexts, self.rag_mode
            )
            
            logger.info(f"Query complexity: {self._last_complexity_analysis['complexity_level']} - "
                       f"{self._last_complexity_analysis['reasoning']}")
            logger.info(f"Recommended max_tokens: {self._last_complexity_analysis['recommended_max_tokens']}")
        
        if progress_tracker:
            progress_tracker.update(70, 100, status="generating", 
                                  message=f"Generating {self._last_complexity_analysis['response_type']}")
        
        try:
            if self.rag_mode == "graph" and graph_context:
                return self._generate_graph_rag_answer(query, contexts, graph_context, progress_tracker)
            else:
                return self._generate_normal_rag_answer(query, contexts, progress_tracker)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
    
    
    def _generate_normal_rag_answer(self, query: str, contexts: List[Dict[str, Any]], 
                               progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using traditional RAG approach with complexity-aware prompting."""
        if not contexts:
            return "I couldn't find any relevant information to answer your question. Please make sure documents have been indexed."
        
        # Format contexts with source labels
        context_sections = []
        for i, ctx in enumerate(contexts):
            filename = ctx['metadata'].get('filename', 'Unknown Document')
            score = ctx.get('score', 0)
            text = ctx['text']
            context_sections.append(f"[Source: {filename}]\n{text}")
        
        context_text = "\n\n---\n\n".join(context_sections)
        
        # Get complexity analysis for adaptive prompting
        complexity = self._last_complexity_analysis or {}
        complexity_level = complexity.get('complexity_level', 'simple')
        response_type = complexity.get('response_type', 'Brief answer')
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided documents.

DOCUMENTS:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
Expected Response Type: {response_type}
Complexity Level: {complexity_level}

1. Give a direct answer first (1-2 sentences for simple queries, more comprehensive for complex)
2. {"Provide comprehensive details with proper structure for list/detail queries" if complexity_level in ['complex', 'very_complex'] else "Then provide supporting details if needed"}
3. Use inline citations like (Source: filename.pdf) when referencing specific information
4. If information comes from multiple sources, mention this naturally
5. If the answer cannot be fully determined from the documents, say so clearly
{"6. For list queries: Use clear structure (numbered lists, bullet points, sections)" if complexity_level in ['complex', 'very_complex'] else ""}

ANSWER:"""
        
        return self._generate_with_llm(prompt, progress_tracker)
    
    
    def _generate_graph_rag_answer(self, query: str, contexts: List[Dict[str, Any]], 
                              graph_context: Dict[str, Any], 
                              progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Graph RAG with relationship chain reasoning and complexity awareness."""
        
        # Separate document and graph contexts
        document_contexts = [ctx for ctx in contexts if ctx['metadata'].get('type') == 'document']
        entity_contexts = graph_context.get('entities', [])
        relationship_contexts = graph_context.get('relationships', [])
        
        # Format document context
        document_text = ""
        if document_contexts:
            doc_sections = []
            for ctx in document_contexts[:3]:
                filename = ctx['metadata'].get('filename', 'Unknown Document')
                text = ctx['text']
                doc_sections.append(f"[Source: {filename}]\n{text}")
            document_text = "\n\n".join(doc_sections)
        
        # Format entities
        entities_text = ""
        if entity_contexts:
            entity_list = []
            for ctx in entity_contexts[:5]:
                name = ctx['metadata'].get('entity_name', 'Unknown')
                etype = ctx['metadata'].get('entity_type', 'Unknown')
                desc = ctx['metadata'].get('description', '')
                discovery = ctx['metadata'].get('discovery_method', 'vector_search')
                
                entry = f"• {name} ({etype})"
                if desc:
                    entry += f" - {desc}"
                if discovery == 'graph_traversal':
                    entry += " [via graph traversal]"
                entity_list.append(entry)
            entities_text = "\n".join(entity_list)
        
        # Format relationships
        relationships_text = ""
        if relationship_contexts:
            rel_list = []
            for ctx in relationship_contexts[:5]:
                source = ctx['metadata'].get('source', 'Unknown')
                target = ctx['metadata'].get('target', 'Unknown')
                rel = ctx['metadata'].get('relationship', 'related_to')
                discovery = ctx['metadata'].get('discovery_method', 'vector_search')
                
                entry = f"• {source} → {rel} → {target}"
                if discovery == 'graph_traversal':
                    entry += " [via graph traversal]"
                rel_list.append(entry)
            relationships_text = "\n".join(rel_list)
        
        # Get complexity analysis for adaptive prompting
        complexity = self._last_complexity_analysis or {}
        complexity_level = complexity.get('complexity_level', 'simple')
        response_type = complexity.get('response_type', 'Brief answer')
        
        # Build prompt
        prompt = f"""You are an AI assistant with access to both documents and a knowledge graph. Answer using relationship reasoning when relevant.

QUESTION: {query}

"""
        
        if document_text:
            prompt += f"""DOCUMENT CONTENT:
{document_text}

"""
        
        if entities_text:
            prompt += f"""ENTITIES FROM KNOWLEDGE GRAPH:
{entities_text}

"""
        
        if relationships_text:
            prompt += f"""RELATIONSHIPS FROM KNOWLEDGE GRAPH:
{relationships_text}

"""
        
        prompt += f"""INSTRUCTIONS:
Expected Response Type: {response_type}
Complexity Level: {complexity_level}

1. Start with a direct {"1-2 sentence answer" if complexity_level == 'simple' else "comprehensive answer covering all key points"}
2. If the answer involves relationship chains, show them clearly:
   Person → works_at → Company → acquired → Target
3. Use inline citations (Source: filename.pdf) for document information
4. Mention when information was "found via graph traversal" - this indicates multi-hop reasoning
5. {"Structure complex answers with clear sections for relationships, entities, and connections" if complexity_level in ['complex', 'very_complex'] else "Keep the response focused and avoid unnecessary repetition"}

ANSWER:"""
        
        return self._generate_with_llm(prompt, progress_tracker)
    
    def _generate_with_llm(self, prompt: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using the configured LLM with comprehensive error handling."""
        if self.preferred_llm == "claude" and self.claude_client:
            return self._generate_with_claude(prompt, progress_tracker)
        elif self.preferred_llm == "raw":
            return self._generate_raw_response(prompt)
        else:
            error_msg = f"LLM {self.preferred_llm} not properly configured or available"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _generate_with_claude(self, prompt: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """
        Generate answer using Claude API with intelligent token allocation and speed optimization.
        
        Speed optimizations:
        1. Dynamic token allocation based on complexity
        2. Smart model selection (Haiku for simple/medium, Sonnet for complex)
        3. Truncation detection and user warnings
        """
        if not self.claude_client:
            return "Claude API not available. Please check your API key configuration."
        
        try:
            if progress_tracker:
                progress_tracker.update(80, 100, status="querying", 
                                      message="Querying Claude API")
            
            # ✅ SPEED OPTIMIZATION: SMART MODEL SELECTION
            complexity = self._last_complexity_analysis or {}
            complexity_level = complexity.get('complexity_level', 'simple')
            max_tokens = complexity.get('recommended_max_tokens', 2000)
            
            # Use faster Haiku model for simple/medium queries (2-3x faster)
            if complexity_level in ['simple', 'medium']:
                model = "claude-3-haiku-20240307"  # 🚀 FASTER - use for 70% of queries
                max_tokens = min(max_tokens, 2000)
                logger.debug(f"Using fast Haiku model for {complexity_level} query")
            elif self.rag_mode == "graph" or complexity_level in ['complex', 'very_complex']:
                model = "claude-3-5-sonnet-20241022"  # More capable for complex queries
                max_tokens = max(max_tokens, 3000)  # Ensure minimum for complex queries
                logger.debug(f"Using Sonnet model for {complexity_level} query")
            else:
                model = "claude-3-haiku-20240307"
            
            logger.debug(f"Using Claude model: {model} with max_tokens: {max_tokens}")
            
            # ✅ Make API call with optimized parameters
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for factual responses
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            if progress_tracker:
                progress_tracker.update(95, 100, status="formatting", 
                                      message="Response received, formatting answer")
            
            answer = response.content[0].text
            
            # ✅ TRUNCATION DETECTION - Check if response hit max_tokens
            stop_reason = response.stop_reason
            if stop_reason == "max_tokens":
                truncation_warning = self._generate_truncation_warning(max_tokens)
                answer = answer + truncation_warning
                logger.warning(f"Response truncated at {max_tokens} tokens for {complexity_level} query")
            
            # Post-process the answer
            answer = self._post_process_answer(answer)
            
            logger.info(f"Successfully generated answer with Claude ({len(answer)} characters, "
                       f"stop_reason: {stop_reason}, model: {model})")
            return answer
            
        except anthropic.RateLimitError as e:
            logger.error(f"Claude API rate limit exceeded: {str(e)}")
            return "I'm currently experiencing high demand. Please try again in a moment."
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            # Better error handling for model not found
            if "not_found_error" in str(e) and "model" in str(e):
                logger.error("Model not found - using fallback model")
                try:
                    # Fallback to Haiku with same token limit
                    response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=max_tokens,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    return f"I encountered an API error and the fallback model also failed. Please check your Claude API configuration."
            else:
                return f"I encountered an API error while generating the response. Please try again."
            
        except Exception as e:
            logger.error(f"Unexpected error with Claude API: {str(e)}")
            return f"I encountered an unexpected error while generating the response: {str(e)}"
    
    def _generate_truncation_warning(self, max_tokens: int) -> str:
        """Generate a user-friendly warning when response is truncated."""
        complexity = self._last_complexity_analysis or {}
        complexity_level = complexity.get('complexity_level', 'unknown')
        
        warning = f"\n\n---\n⚠️ **Response Truncated**: This is a {complexity_level} query that may require more detail than fits in {max_tokens} tokens."
        
        if complexity_level in ['complex', 'very_complex']:
            warning += "\n\n💡 **Suggestions:**"
            warning += "\n• Try breaking your question into smaller parts"
            warning += "\n• Ask for specific aspects rather than 'list all'"
            warning += "\n• Request a summary first, then ask for details on specific items"
        
        return warning
    
    def _generate_raw_response(self, prompt: str) -> str:
        """Generate a raw response without LLM processing - useful for debugging."""
        complexity = self._last_complexity_analysis or {}
        lines = [
            "=== RAW MODE RESPONSE ===",
            f"RAG Mode: {self.rag_mode}",
            f"Complexity: {complexity.get('complexity_level', 'unknown')}",
            f"Max Tokens: {complexity.get('recommended_max_tokens', 2000)}",
            "",
            "PROMPT USED:",
            prompt,
            "",
            "Note: This is raw mode output. Configure Claude API key for processed responses."
        ]
        return "\n".join(lines)
    
    def _post_process_answer(self, answer: str) -> str:
        """Clean Claude/LLM output by unescaping HTML, stripping tags, removing markdown, and normalizing spacing."""
        if not answer:
            return ""

        # Step 1: Trim whitespace
        answer = answer.strip()

        # Step 2: Unescape HTML entities (&lt;, &gt;, &amp;)
        answer = html.unescape(answer)

        # Step 3: Remove HTML tags if any exist
        answer = re.sub(r'<[^>]+>', '', answer)

        # Step 4: Remove common Markdown formatting
        # Bold/italic
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # **bold**
        answer = re.sub(r'\*(.*?)\*', r'\1', answer)      # *italic*
        answer = re.sub(r'__(.*?)__', r'\1', answer)      # __bold__
        answer = re.sub(r'_(.*?)_', r'\1', answer)        # _italic_
        # Headings, bullet points
        answer = re.sub(r'^\s*#+\s*', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'^\s*[-*]\s*', '', answer, flags=re.MULTILINE)

        # Step 5: Normalize whitespace
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # collapse triple newlines
        answer = re.sub(r' {2,}', ' ', answer)  # collapse multiple spaces

        return answer
    
    def get_last_complexity_analysis(self) -> Dict[str, Any]:
        """Get the complexity analysis from the last query."""
        return self._last_complexity_analysis or {
            'complexity_level': 'unknown',
            'recommended_max_tokens': 2000,
            'response_type': 'Unknown',
            'reasoning': 'No analysis available',
            'from_cache': False
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get complexity analyzer cache statistics."""
        return self.complexity_analyzer.get_cache_stats()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the LLM service connection and return status."""
        status = {
            "service_available": False,
            "preferred_llm": self.preferred_llm,
            "rag_mode": self.rag_mode,
            "error": None
        }
        
        try:
            if self.preferred_llm == "claude":
                if not self.claude_api_key:
                    status["error"] = "Claude API key not configured"
                elif not self.claude_client:
                    status["error"] = "Claude client not initialized"
                else:
                    # Test with current working model
                    test_response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=50,
                        messages=[
                            {"role": "user", "content": "Hello! Please respond with 'Claude connection test successful'."}
                        ]
                    )
                    
                    if test_response and test_response.content:
                        status["service_available"] = True
                        status["test_response"] = test_response.content[0].text
                    else:
                        status["error"] = "No response from Claude API"
                        
            elif self.preferred_llm == "raw":
                status["service_available"] = True
                status["test_response"] = "Raw mode is always available"
            else:
                status["error"] = f"Unknown LLM type: {self.preferred_llm}"
                
        except Exception as e:
            status["error"] = f"Connection test failed: {str(e)}"
        
        return status
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics and configuration info."""
        cache_stats = self.get_cache_stats()
        return {
            "preferred_llm": self.preferred_llm,
            "rag_mode": self.rag_mode,
            "claude_configured": bool(self.claude_api_key),
            "claude_client_available": self.claude_client is not None,
            "service_available": self.is_available(),
            "complexity_cache_stats": cache_stats
        }
    
    def generate_hybrid_neo4j_answer(self, query: str, contexts: List[Dict[str, Any]], 
                         graph_context: Dict[str, Any], 
                         progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Graph RAG + Neo4j with comprehensive source integration."""
        
        # Separate contexts by type
        document_contexts = []
        entity_contexts = []
        relationship_contexts = []
        neo4j_contexts = []
        
        for ctx in contexts:
            ctx_type = ctx['metadata'].get('type', 'document')
            if ctx_type == 'entity':
                entity_contexts.append(ctx)
            elif ctx_type == 'relationship':
                relationship_contexts.append(ctx)
            elif ctx_type == 'neo4j_result':
                neo4j_contexts.append(ctx)
            else:
                document_contexts.append(ctx)
        
        # Format document context
        document_text = ""
        if document_contexts:
            doc_sections = []
            for ctx in document_contexts[:3]:
                filename = ctx['metadata'].get('filename', 'Unknown Document')
                text = ctx['text']
                doc_sections.append(f"[Source: {filename}]\n{text}")
            document_text = "\n\n".join(doc_sections)
        
        # Format entities
        entities_text = ""
        if entity_contexts:
            entity_list = []
            for ctx in entity_contexts[:5]:
                name = ctx['metadata'].get('entity_name', 'Unknown')
                etype = ctx['metadata'].get('entity_type', 'Unknown')
                desc = ctx['metadata'].get('description', '')
                discovery = ctx['metadata'].get('discovery_method', 'vector_search')
                
                entry = f"• {name} ({etype})"
                if desc:
                    entry += f" - {desc}"
                if discovery == 'graph_traversal':
                    entry += " [via graph traversal]"
                entity_list.append(entry)
            entities_text = "\n".join(entity_list)
        
        # Format relationships
        relationships_text = ""
        if relationship_contexts:
            rel_list = []
            for ctx in relationship_contexts[:5]:
                source = ctx['metadata'].get('source', 'Unknown')
                target = ctx['metadata'].get('target', 'Unknown')
                rel = ctx['metadata'].get('relationship', 'related_to')
                discovery = ctx['metadata'].get('discovery_method', 'vector_search')
                
                entry = f"• {source} → {rel} → {target}"
                if discovery == 'graph_traversal':
                    entry += " [via graph traversal]"
                rel_list.append(entry)
            relationships_text = "\n".join(rel_list)
        
        # Format Neo4j results
        neo4j_text = ""
        if neo4j_contexts:
            neo4j_sections = []
            for ctx in neo4j_contexts:
                neo4j_sections.append(f"[Neo4j Query Result]\n{ctx['text']}")
            neo4j_text = "\n\n".join(neo4j_sections)
        
        # Get complexity for adaptive prompting
        complexity = self._last_complexity_analysis or {}
        complexity_level = complexity.get('complexity_level', 'simple')
        response_type = complexity.get('response_type', 'Brief answer')
        
        # Build prompt
        prompt = f"""You are an AI assistant answering questions using three data sources:
- Documents (text from uploaded files)
- Knowledge Graph (extracted entities and relationships)  
- Neo4j Database (structured graph query results)

QUESTION: {query}

"""
        
        if document_text:
            prompt += f"""DOCUMENT CONTENT:
{document_text}

"""
        
        if entities_text:
            prompt += f"""KNOWLEDGE GRAPH ENTITIES:
{entities_text}

"""
        
        if relationships_text:
            prompt += f"""KNOWLEDGE GRAPH RELATIONSHIPS:
{relationships_text}

"""
        
        if neo4j_text:
            prompt += f"""NEO4J DATABASE RESULTS:
{neo4j_text}

"""
        
        prompt += f"""INSTRUCTIONS:
Expected Response Type: {response_type}
Complexity Level: {complexity_level}

1. Start with a direct {"1-2 sentence answer" if complexity_level == 'simple' else "comprehensive answer"}
2. Show relationship chains when relevant: Entity → relationship → Entity → relationship → Entity
3. Prioritize Neo4j results for structured relationship queries (they come from direct graph database queries)
4. Use inline citations: (Source: filename.pdf) for documents, (Neo4j) for database results
5. Note when connections were "found via graph traversal" - this indicates multi-hop reasoning
6. {"Provide structured, comprehensive synthesis for complex queries" if complexity_level in ['complex', 'very_complex'] else "Synthesize information from all sources into a coherent answer"}

ANSWER:"""
        
        return self._generate_with_llm(prompt, progress_tracker)