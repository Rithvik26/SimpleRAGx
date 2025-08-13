"""
LLM service for generating answers using Claude API - FIXED VERSION
"""

import logging
from typing import List, Dict, Any, Optional
import anthropic
from extensions import ProgressTracker

logger = logging.getLogger(__name__)

class LLMService:
    """Handles interactions with LLM APIs with enhanced prompt engineering for Graph RAG."""
    
    def __init__(self, config):
        self.preferred_llm = config["preferred_llm"]
        self.claude_api_key = config.get("claude_api_key", "")
        self.rag_mode = config.get("rag_mode", "normal")
        
        # Initialize Claude client if available
        self.claude_client = None
        if self.preferred_llm == "claude" and self.claude_api_key:
            self._initialize_claude_client()
        
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
        """Generate an answer based on the query and retrieved contexts."""
        if progress_tracker:
            progress_tracker.update(70, 100, status="generating", 
                                  message="Generating answer with LLM")
        
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
        """Generate answer using traditional RAG approach with improved prompting."""
        if not contexts:
            return "I couldn't find any relevant information to answer your question. Please make sure documents have been indexed."
        
        # Format contexts for the prompt
        context_sections = []
        for i, ctx in enumerate(contexts):
            filename = ctx['metadata'].get('filename', 'Unknown Document')
            score = ctx.get('score', 0)
            text = ctx['text']
            
            context_sections.append(f"""
Document {i+1}: {filename} (Relevance: {score:.2f})
{text}
""")
        
        context_text = "\n".join(context_sections)
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context. Your goal is to provide accurate, well-sourced answers.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question using only information from the provided context
2. If the answer cannot be determined from the context, clearly state this
3. Include specific references to the source documents when relevant
4. If multiple documents contain related information, synthesize them thoughtfully
5. Be precise and avoid speculation beyond what's explicitly stated in the context

ANSWER:"""
        
        return self._generate_with_llm(prompt, progress_tracker)
    
    def _generate_graph_rag_answer(self, query: str, contexts: List[Dict[str, Any]], 
                                  graph_context: Dict[str, Any], 
                                  progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Graph RAG approach with entity and relationship awareness."""
        
        # Separate document contexts from graph contexts
        document_contexts = []
        entity_contexts = []
        relationship_contexts = []
        
        for ctx in contexts:
            ctx_type = ctx['metadata'].get('type', 'document')
            if ctx_type == 'entity':
                entity_contexts.append(ctx)
            elif ctx_type == 'relationship':
                relationship_contexts.append(ctx)
            else:
                document_contexts.append(ctx)
        
        # Format document context
        document_text = ""
        if document_contexts:
            doc_sections = []
            for i, ctx in enumerate(document_contexts):
                filename = ctx['metadata'].get('filename', 'Unknown Document')
                score = ctx.get('score', 0)
                text = ctx['text']
                doc_sections.append(f"Document {i+1}: {filename} (Score: {score:.2f})\n{text}")
            document_text = "\n\n".join(doc_sections)
        
        # Format entity context
        entities_text = ""
        if entity_contexts:
            entity_sections = []
            for ctx in entity_contexts:
                entity_name = ctx['metadata'].get('entity_name', 'Unknown Entity')
                entity_type = ctx['metadata'].get('entity_type', 'Unknown')
                description = ctx['metadata'].get('description', '')
                score = ctx.get('score', 0)
                
                entity_info = f"• {entity_name} ({entity_type})"
                if description:
                    entity_info += f": {description}"
                entity_info += f" [Relevance: {score:.2f}]"
                entity_sections.append(entity_info)
            
            entities_text = "\n".join(entity_sections)
        
        # Format relationship context
        relationships_text = ""
        if relationship_contexts:
            rel_sections = []
            for ctx in relationship_contexts:
                source = ctx['metadata'].get('source', 'Unknown')
                target = ctx['metadata'].get('target', 'Unknown')
                relationship = ctx['metadata'].get('relationship', 'unknown')
                description = ctx['metadata'].get('description', '')
                score = ctx.get('score', 0)
                
                rel_info = f"• {source} → {relationship} → {target}"
                if description:
                    rel_info += f" ({description})"
                rel_info += f" [Relevance: {score:.2f}]"
                rel_sections.append(rel_info)
            
            relationships_text = "\n".join(rel_sections)
        
        # Create comprehensive prompt for Graph RAG
        prompt = f"""You are an advanced AI assistant that answers questions using both document content and knowledge graph information (entities and relationships). You excel at understanding connections and providing comprehensive, well-reasoned answers.

USER QUESTION: {query}

AVAILABLE INFORMATION:
"""
        
        if document_text:
            prompt += f"""
DOCUMENT CONTEXT:
{document_text}
"""
        
        if entities_text:
            prompt += f"""
RELEVANT ENTITIES:
{entities_text}
"""
        
        if relationships_text:
            prompt += f"""
RELEVANT RELATIONSHIPS:
{relationships_text}
"""
        
        prompt += f"""
INSTRUCTIONS:
1. Provide a comprehensive answer that leverages both document content and knowledge graph information
2. Explain how entities and relationships are relevant to the question
3. Draw connections between different pieces of information when appropriate
4. If the knowledge graph reveals important relationships not explicitly stated in documents, highlight these insights
5. Cite specific documents, entities, or relationships that support your answer
6. If information is incomplete or conflicting, acknowledge this clearly
7. Structure your answer logically, building from basic facts to more complex relationships

Focus on providing value through the enhanced understanding that comes from combining document content with structured knowledge relationships.

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
        """Generate answer using Claude API with retry logic and error handling - FIXED MODELS."""
        if not self.claude_client:
            return "Claude API not available. Please check your API key configuration."
        
        try:
            if progress_tracker:
                progress_tracker.update(80, 100, status="querying", 
                                      message="Querying Claude API")
            
            # FIXED: Use current Claude models that actually exist
            model = "claude-3-haiku-20240307"  # This model works and is fast
            max_tokens = 2000
            
            # For Graph RAG, use Sonnet but with correct model name
            if self.rag_mode == "graph":
                model = "claude-3-5-sonnet-20241022"  # FIXED: Use current Sonnet model
                max_tokens = 3000
            
            logger.debug(f"Using Claude model: {model}")
            
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
            
            # Post-process the answer
            answer = self._post_process_answer(answer)
            
            logger.info(f"Successfully generated answer with Claude ({len(answer)} characters)")
            return answer
            
        except anthropic.RateLimitError as e:
            logger.error(f"Claude API rate limit exceeded: {str(e)}")
            return "I'm currently experiencing high demand. Please try again in a moment."
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            # FIXED: Better error handling for model not found
            if "not_found_error" in str(e) and "model" in str(e):
                logger.error("Model not found - using fallback model")
                try:
                    # Fallback to Haiku which should always work
                    response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=2000,
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
    
    def _generate_raw_response(self, prompt: str) -> str:
        """Generate a raw response without LLM processing - useful for debugging."""
        lines = [
            "=== RAW MODE RESPONSE ===",
            f"RAG Mode: {self.rag_mode}",
            "",
            "PROMPT USED:",
            prompt,
            "",
            "Note: This is raw mode output. Configure Claude API key for processed responses."
        ]
        return "\n".join(lines)
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the LLM answer for better formatting and quality."""
        # Remove excessive whitespace
        answer = answer.strip()
        
        # Ensure proper paragraph spacing
        import re
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
        
        # Fix common formatting issues
        answer = answer.replace('  ', ' ')  # Remove double spaces
        
        return answer
    
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
                        model="claude-3-haiku-20240307",  # FIXED: Use working model
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
        return {
            "preferred_llm": self.preferred_llm,
            "rag_mode": self.rag_mode,
            "claude_configured": bool(self.claude_api_key),
            "claude_client_available": self.claude_client is not None,
            "service_available": self.is_available()
        }