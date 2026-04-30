"""
LLM service for generating answers using Gemini via LiteLLM.
Gemini-only — Claude/Anthropic dependency removed.
"""

import logging
import os
import threading
import litellm
from typing import List, Dict, Any, Optional
from extensions import ProgressTracker
import html
import re
import hashlib

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini/gemini-2.5-flash"


def _clean_source_name(metadata: dict) -> str:
    """Return a human-readable source name from chunk metadata."""
    company = metadata.get('company_name', '')
    raw = metadata.get('source', metadata.get('filename', metadata.get('doc_name', '')))
    # Strip the hash prefix (32 hex chars + underscore) if present
    clean = re.sub(r'^[0-9a-f]{32}_', '', raw) if raw else ''
    if company and clean:
        return f"{company} ({clean})"
    return company or clean or 'Unknown Document'

# Silence LiteLLM's verbose success logs
litellm.success_callback = []


class QueryComplexityAnalyzer:
    """Analyzes query complexity with caching for optimal performance."""

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
        self._pattern_cache = {}
        self._cache_size = 100
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _get_query_signature(query: str, num_contexts: int, rag_mode: str) -> str:
        words = query.lower().split()[:10]
        normalized = ' '.join(sorted(words))
        signature = f"{normalized}_{num_contexts}_{rag_mode}"
        return hashlib.md5(signature.encode()).hexdigest()[:12]

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_size': len(self._pattern_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }

    def analyze(self, query: str, contexts: List[Dict[str, Any]], rag_mode: str) -> Dict[str, Any]:
        cache_key = self._get_query_signature(query, len(contexts), rag_mode)
        if cache_key in self._pattern_cache:
            self._cache_hits += 1
            cached = self._pattern_cache[cache_key].copy()
            cached['from_cache'] = True
            return cached

        self._cache_misses += 1
        query_lower = query.lower()

        list_score       = sum(1 for i in self.LIST_INDICATORS       if i in query_lower)
        detail_score     = sum(1 for i in self.DETAIL_INDICATORS     if i in query_lower)
        timeline_score   = sum(1 for i in self.TIMELINE_INDICATORS   if i in query_lower)
        comparison_score = sum(1 for i in self.COMPARISON_INDICATORS if i in query_lower)
        simple_score     = sum(1 for i in self.SIMPLE_INDICATORS     if i in query_lower)

        num_contexts = len(contexts)
        avg_context_length = sum(len(ctx.get('text', '')) for ctx in contexts) / max(num_contexts, 1)
        total_context_chars = sum(len(ctx.get('text', '')) for ctx in contexts)

        total_score = list_score * 3 + detail_score * 2 + timeline_score * 2 + comparison_score * 2

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

        if rag_mode == 'graph':
            base_tokens += 1000
        if total_context_chars > 5000:
            base_tokens = min(base_tokens + 1000, 8000)

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

        if len(self._pattern_cache) >= self._cache_size:
            self._pattern_cache.pop(next(iter(self._pattern_cache)))
        self._pattern_cache[cache_key] = result.copy()
        return result


class LLMService:
    """Handles interactions with Gemini via LiteLLM with intelligent response length management."""

    def __init__(self, config):
        self.gemini_api_key = config.get("gemini_api_key", "")
        self.rag_mode = config.get("rag_mode", "normal")
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self._thread_local = threading.local()

        if self.gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", self.gemini_api_key)

        logger.info(f"LLMService (Gemini) initialized, rag_mode={self.rag_mode}")

    def is_available(self) -> bool:
        return bool(self.gemini_api_key)

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]],
                        graph_context: Dict[str, Any] = None,
                        rag_mode: str = None,
                        progress_tracker: Optional[ProgressTracker] = None) -> str:
        effective_mode = rag_mode if rag_mode is not None else self.rag_mode
        query_lower = query.lower()
        num_contexts = len(contexts)

        if (num_contexts <= 2 and
                any(indicator in query_lower for indicator in ['what is', 'who is', 'when', 'where', 'define'])):
            self._thread_local.last_complexity_analysis = {
                'complexity_level': 'simple',
                'recommended_max_tokens': 1500,
                'response_type': 'Concise answer',
                'reasoning': 'Fast path - simple factual query',
                'from_cache': False,
                'fast_path': True
            }
        else:
            if progress_tracker:
                progress_tracker.update(65, 100, status="analyzing",
                                        message="Analyzing query complexity")
            self._thread_local.last_complexity_analysis = self.complexity_analyzer.analyze(
                query, contexts, effective_mode
            )
            ca = self._thread_local.last_complexity_analysis
            logger.info(f"Query complexity: {ca['complexity_level']} — {ca['reasoning']}")

        if progress_tracker:
            ca = self._thread_local.last_complexity_analysis
            progress_tracker.update(70, 100, status="generating",
                                    message=f"Generating {ca['response_type']}")

        try:
            if effective_mode == "graph" and graph_context:
                return self._generate_graph_rag_answer(query, contexts, graph_context, progress_tracker)
            else:
                return self._generate_normal_rag_answer(query, contexts, progress_tracker)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {e}"

    def _generate_normal_rag_answer(self, query: str, contexts: List[Dict[str, Any]],
                                    progress_tracker: Optional[ProgressTracker] = None) -> str:
        if not contexts:
            return "I couldn't find any relevant information to answer your question. Please make sure documents have been indexed."

        context_sections = []
        for ctx in contexts:
            filename = _clean_source_name(ctx['metadata'])
            text = ctx['text']
            context_sections.append(f"[Source: {filename}]\n{text}")
        context_text = "\n\n---\n\n".join(context_sections)

        complexity = getattr(self._thread_local, 'last_complexity_analysis', None) or {}
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

        return self._generate_with_gemini(prompt, progress_tracker)

    def _generate_graph_rag_answer(self, query: str, contexts: List[Dict[str, Any]],
                                   graph_context: Dict[str, Any],
                                   progress_tracker: Optional[ProgressTracker] = None) -> str:
        document_contexts = [ctx for ctx in contexts if ctx['metadata'].get('type') not in ('entity', 'relationship')]
        entity_contexts = graph_context.get('entities', [])
        relationship_contexts = graph_context.get('relationships', [])

        document_text = ""
        if document_contexts:
            doc_sections = []
            for ctx in document_contexts[:8]:
                filename = _clean_source_name(ctx['metadata'])
                doc_sections.append(f"[Source: {filename}]\n{ctx['text']}")
            document_text = "\n\n".join(doc_sections)

        entities_text = ""
        if entity_contexts:
            entity_list = []
            for ctx in entity_contexts[:10]:
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

        relationships_text = ""
        if relationship_contexts:
            rel_list = []
            for ctx in relationship_contexts[:10]:
                source = _clean_source_name(ctx['metadata'])
                target = ctx['metadata'].get('target', 'Unknown')
                rel = ctx['metadata'].get('relationship', 'related_to')
                discovery = ctx['metadata'].get('discovery_method', 'vector_search')
                entry = f"• {source} → {rel} → {target}"
                if discovery == 'graph_traversal':
                    entry += " [via graph traversal]"
                rel_list.append(entry)
            relationships_text = "\n".join(rel_list)

        complexity = getattr(self._thread_local, 'last_complexity_analysis', None) or {}
        complexity_level = complexity.get('complexity_level', 'simple')

        prompt = f"""You are an AI assistant with access to both documents and a knowledge graph. Answer using relationship reasoning when relevant.

QUESTION: {query}

"""
        if document_text:
            prompt += f"DOCUMENT CONTENT:\n{document_text}\n\n"
        if entities_text:
            prompt += f"ENTITIES FROM KNOWLEDGE GRAPH:\n{entities_text}\n\n"
        if relationships_text:
            prompt += f"RELATIONSHIPS FROM KNOWLEDGE GRAPH:\n{relationships_text}\n\n"

        # Detect question type for targeted prompt instructions
        _yn_starters = ("did ", "does ", "is ", "was ", "were ", "has ", "have ", "do ", "are ")
        _q_lower = query.strip().lower()
        _is_yn = _q_lower.startswith(_yn_starters)
        _is_comparison = _is_yn and any(
            kw in _q_lower for kw in ("agree", "align", "consistent", "same as", "match", "both claim", "also claim")
        )

        prompt += "INSTRUCTIONS:\n"
        if _is_comparison:
            prompt += (
                "1. This is an agreement/alignment question. If the named sources appear anywhere in "
                "the retrieved context above, synthesize their claims and state whether they agree. "
                "Do NOT say 'documents do not contain' if relevant context is shown above — trust the evidence.\n"
            )
        elif _is_yn:
            prompt += "1. This is a Yes/No question. Start your answer with 'Yes' or 'No' then explain briefly.\n"
        else:
            prompt += "1. State the answer directly and concisely. If context strongly implies an entity name, state it.\n"
        prompt += """2. Answer from the context above. Do not speculate beyond what is stated.
3. Once you have stated your answer, stop — do not add disclaimers about what the documents lack.
4. Only say "Insufficient information" if the context contains nothing relevant at all.
5. Use inline citations: (Source: filename)
6. If showing relationship chains, use format: Entity -> relationship -> Entity

ANSWER:"""

        return self._generate_with_gemini(prompt, progress_tracker)

    def _generate_with_gemini(self, prompt: str,
                              progress_tracker: Optional[ProgressTracker] = None) -> str:
        if not self.gemini_api_key:
            return "Gemini API key not configured. Please add gemini_api_key to your configuration."

        complexity = getattr(self._thread_local, 'last_complexity_analysis', None) or {}
        max_tokens = min(complexity.get('recommended_max_tokens', 2000), 8192)

        try:
            if progress_tracker:
                progress_tracker.update(80, 100, status="querying",
                                        message="Querying Gemini API")

            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
            resp = litellm.completion(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
            )

            if progress_tracker:
                progress_tracker.update(95, 100, status="formatting",
                                        message="Response received, formatting answer")

            answer = resp.choices[0].message.content.strip()
            answer = self._post_process_answer(answer)
            logger.info(f"Gemini answer generated ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response from Gemini: {e}"

    def _post_process_answer(self, answer: str) -> str:
        if not answer:
            return ""
        answer = answer.strip()
        answer = html.unescape(answer)
        answer = re.sub(r'<[^>]+>', '', answer)
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
        answer = re.sub(r'\*(.*?)\*', r'\1', answer)
        answer = re.sub(r'__(.*?)__', r'\1', answer)
        answer = re.sub(r'_(.*?)_', r'\1', answer)
        answer = re.sub(r'^\s*#+\s*', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'^\s*[-*]\s*', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
        answer = re.sub(r' {2,}', ' ', answer)
        return answer

    def _generate_raw_response(self, prompt: str) -> str:
        complexity = getattr(self._thread_local, 'last_complexity_analysis', None) or {}
        lines = [
            "=== RAW MODE RESPONSE ===",
            f"RAG Mode: {self.rag_mode}",
            f"Complexity: {complexity.get('complexity_level', 'unknown')}",
            "",
            "PROMPT USED:",
            prompt,
            "",
            "Note: This is raw mode output. Configure Gemini API key for processed responses."
        ]
        return "\n".join(lines)

    def get_last_complexity_analysis(self) -> Dict[str, Any]:
        return getattr(self._thread_local, 'last_complexity_analysis', None) or {
            'complexity_level': 'unknown',
            'recommended_max_tokens': 2000,
            'response_type': 'Unknown',
            'reasoning': 'No analysis available',
            'from_cache': False
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.complexity_analyzer.get_cache_stats()

    def test_connection(self) -> Dict[str, Any]:
        status = {
            "service_available": False,
            "preferred_llm": "gemini",
            "rag_mode": self.rag_mode,
            "error": None
        }
        if not self.gemini_api_key:
            status["error"] = "Gemini API key not configured"
            return status
        try:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
            resp = litellm.completion(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": "Respond with: Gemini connection test successful."}],
                temperature=0,
                max_tokens=20,
                extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
            )
            content = resp.choices[0].message.content or ""
            status["service_available"] = True
            status["test_response"] = content.strip()
        except Exception as e:
            status["error"] = f"Connection test failed: {e}"
        return status

    def get_usage_stats(self) -> Dict[str, Any]:
        cache_stats = self.get_cache_stats()
        return {
            "preferred_llm": "gemini",
            "model": GEMINI_MODEL,
            "rag_mode": self.rag_mode,
            "gemini_configured": bool(self.gemini_api_key),
            "service_available": self.is_available(),
            "complexity_cache_stats": cache_stats
        }

    def generate_hybrid_neo4j_answer(self, query: str, contexts: List[Dict[str, Any]],
                                     graph_context: Dict[str, Any],
                                     progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Graph RAG + Neo4j with comprehensive source integration."""
        document_contexts, entity_contexts, relationship_contexts, neo4j_contexts = [], [], [], []

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

        document_text = ""
        if document_contexts:
            doc_sections = []
            for ctx in document_contexts[:5]:
                filename = _clean_source_name(ctx['metadata'])
                doc_sections.append(f"[Source: {filename}]\n{ctx['text']}")
            document_text = "\n\n".join(doc_sections)

        entities_text = ""
        if entity_contexts:
            entity_list = []
            for ctx in entity_contexts[:10]:
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

        relationships_text = ""
        if relationship_contexts:
            rel_list = []
            for ctx in relationship_contexts[:10]:
                source = _clean_source_name(ctx['metadata'])
                target = ctx['metadata'].get('target', 'Unknown')
                rel = ctx['metadata'].get('relationship', 'related_to')
                discovery = ctx['metadata'].get('discovery_method', 'vector_search')
                entry = f"• {source} → {rel} → {target}"
                if discovery == 'graph_traversal':
                    entry += " [via graph traversal]"
                rel_list.append(entry)
            relationships_text = "\n".join(rel_list)

        neo4j_text = ""
        if neo4j_contexts:
            neo4j_text = "\n\n".join(f"[Neo4j Query Result]\n{ctx['text']}" for ctx in neo4j_contexts)

        complexity = getattr(self._thread_local, 'last_complexity_analysis', None) or {}
        complexity_level = complexity.get('complexity_level', 'simple')
        response_type = complexity.get('response_type', 'Brief answer')

        prompt = f"""You are an AI assistant answering questions using three data sources:
- Documents (text from uploaded files)
- Knowledge Graph (extracted entities and relationships)
- Neo4j Database (structured graph query results)

QUESTION: {query}

"""
        if document_text:
            prompt += f"DOCUMENT CONTENT:\n{document_text}\n\n"
        if entities_text:
            prompt += f"KNOWLEDGE GRAPH ENTITIES:\n{entities_text}\n\n"
        if relationships_text:
            prompt += f"KNOWLEDGE GRAPH RELATIONSHIPS:\n{relationships_text}\n\n"
        if neo4j_text:
            prompt += f"NEO4J DATABASE RESULTS:\n{neo4j_text}\n\n"

        prompt += f"""INSTRUCTIONS:
Expected Response Type: {response_type}
Complexity Level: {complexity_level}

1. Start with a direct {"1-2 sentence answer" if complexity_level == 'simple' else "comprehensive answer"}
2. Show relationship chains when relevant: Entity → relationship → Entity
3. Prioritize Neo4j results for structured relationship queries
4. Use inline citations: (Source: filename.pdf) for documents, (Neo4j) for database results
5. Note when connections were "found via graph traversal"
6. {"Provide structured, comprehensive synthesis for complex queries" if complexity_level in ['complex', 'very_complex'] else "Synthesize information from all sources into a coherent answer"}

ANSWER:"""

        return self._generate_with_gemini(prompt, progress_tracker)
