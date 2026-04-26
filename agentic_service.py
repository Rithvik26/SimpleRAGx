"""
Agentic AI Service using LangChain 1.x create_agent (function-calling, no ReAct prompt).
Tools: document search, graph search, pageindex deep research, web search, verify.
"""

import logging
import json
from typing import List, Dict, Any, Optional

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from extensions import ProgressTracker

logger = logging.getLogger(__name__)


class AgenticRAGService:
    def __init__(self, config, simple_rag_instance):
        self.config = config
        self.simple_rag = simple_rag_instance

        self.llm = None
        if config.get("gemini_api_key"):
            try:
                self.llm = ChatGoogleGenerativeAI(
                    google_api_key=config["gemini_api_key"],
                    model="gemini-2.0-flash",
                    temperature=0.1,
                )
                logger.info("LangChain Gemini LLM initialized: gemini-2.0-flash")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini LLM: {e}")

        self.tools = self._create_tools()
        self.agent = None
        if self.llm and self.tools:
            self._initialize_agent()

    # ------------------------------------------------------------------
    # Tool creation
    # ------------------------------------------------------------------

    def _create_tools(self) -> List[Tool]:
        tools = []

        tools.append(Tool(
            name="search_documents",
            description=(
                "Search document chunks using semantic similarity (Normal RAG). "
                "Use for straightforward factual questions about document content. "
                "Input: a specific question or search query."
            ),
            func=self._search_documents_tool,
        ))

        tools.append(Tool(
            name="search_knowledge_graph",
            description=(
                "Search entities and relationships in the knowledge graph (Graph RAG). "
                "Use for questions about connections between people, organisations, or concepts. "
                "Input: a question about entities or relationships."
            ),
            func=self._search_graph_tool,
        ))

        tools.append(Tool(
            name="search_pageindex",
            description=(
                "Deep multi-hop research over indexed documents using PageIndex reasoning. "
                "Use for complex questions that need reading across many pages, multi-step "
                "reasoning, or when simpler search tools are insufficient. "
                "Input: a research question."
            ),
            func=self._search_pageindex_tool,
        ))

        tools.append(Tool(
            name="web_search",
            description=(
                "Search the internet for current information not present in the knowledge base. "
                "Use for recent events, public facts, definitions, or anything beyond the "
                "uploaded documents. "
                "Input: a search query string."
            ),
            func=self._web_search_tool,
        ))

        tools.append(Tool(
            name="verify_information",
            description=(
                "Cross-check a claim by searching for supporting or contradicting evidence. "
                "Use when you need to validate a statement or find additional confirmation. "
                "Input: a statement or claim to verify."
            ),
            func=self._verify_information_tool,
        ))

        return tools

    # ------------------------------------------------------------------
    # Agent initialisation (LangChain 1.x create_agent)
    # ------------------------------------------------------------------

    def _initialize_agent(self):
        try:
            system_prompt = (
                "You are a helpful research assistant with access to a document knowledge base "
                "and internet search. Use the available tools to answer questions accurately. "
                "For factual questions about uploaded documents use search_documents. "
                "For relationship questions use search_knowledge_graph. "
                "For deep multi-document research use search_pageindex. "
                "For current events or public facts use web_search. "
                "To verify claims use verify_information. "
                "Prefer answering from documents when possible; use web_search to supplement."
            )
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt,
            )
            logger.info("LangChain 1.x create_agent initialised (%d tools)", len(self.tools))
        except Exception as e:
            logger.error(f"Failed to initialise agent: {e}")
            self.agent = None

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _search_documents_tool(self, query: str) -> str:
        try:
            if not query or len(query.strip()) < 3:
                return "Error: query too short."
            original_mode = self.simple_rag.rag_mode
            self.simple_rag.set_rag_mode("normal")
            result = self.simple_rag._query_normal_mode(query)
            self.simple_rag.set_rag_mode(original_mode)
            if len(result) > 1500:
                result = result[:1500] + "... [truncated]"
            return f"Document search results for '{query}':\n{result}"
        except Exception as e:
            logger.error(f"Error in document search tool: {e}")
            return f"Error searching documents: {str(e)}"

    def _search_graph_tool(self, query: str) -> str:
        try:
            if not query or len(query.strip()) < 3:
                return "Error: query too short."
            if not self.simple_rag.is_graph_ready():
                return "Knowledge graph not available. Please ensure Graph RAG is configured."
            original_mode = self.simple_rag.rag_mode
            self.simple_rag.set_rag_mode("graph")
            result = self.simple_rag._query_graph_mode(query)
            self.simple_rag.set_rag_mode(original_mode)
            if len(result) > 1500:
                result = result[:1500] + "... [truncated]"
            return f"Knowledge graph results for '{query}':\n{result}"
        except Exception as e:
            logger.error(f"Error in graph search tool: {e}")
            return f"Error searching knowledge graph: {str(e)}"

    def _search_pageindex_tool(self, query: str) -> str:
        try:
            if not query or len(query.strip()) < 3:
                return "Error: query too short."
            if not self.simple_rag.is_pageindex_ready():
                return "PageIndex not available. Try search_documents or search_knowledge_graph instead."
            result = self.simple_rag.pageindex_service.query(query)
            answer = result.get("answer", "No answer returned by PageIndex.")
            if len(answer) > 1500:
                answer = answer[:1500] + "... [truncated]"
            return f"PageIndex research results for '{query}':\n{answer}"
        except Exception as e:
            logger.error(f"Error in PageIndex tool: {e}")
            return f"Error using PageIndex: {str(e)}"

    def _web_search_tool(self, query: str) -> str:
        try:
            if not query or len(query.strip()) < 3:
                return "Error: query too short."
            try:
                from ddgs import DDGS
            except ImportError:
                return "Web search not available. Install with: pip install ddgs"
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=5))
            if not raw:
                return f"No web results found for '{query}'."
            lines = [f"Web search results for '{query}':\n"]
            for i, r in enumerate(raw, 1):
                title = r.get("title", "No title")
                body = r.get("body", "")[:200]
                url = r.get("href", "")
                lines.append(f"{i}. {title}")
                lines.append(f"   {body}")
                lines.append(f"   Source: {url}\n")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error in web search tool: {e}")
            return f"Web search failed: {str(e)}"

    def _verify_information_tool(self, claim: str) -> str:
        try:
            if not claim or len(claim.strip()) < 5:
                return "Error: claim too short."
            verification_query = f"evidence about {claim}"
            doc_evidence = self._search_documents_tool(verification_query)
            result = (
                f"Verification for: {claim}\n\n"
                f"Evidence found:\n"
                f"{doc_evidence[:800] if len(doc_evidence) > 800 else doc_evidence}\n\n"
                f"Verification status: based on available evidence in the knowledge base."
            )
            return result
        except Exception as e:
            logger.error(f"Error in verification tool: {e}")
            return f"Error verifying information: {str(e)}"

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    def process_agentic_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "answer": "Agentic AI service not available. Please check Gemini API configuration.",
                "reasoning_steps": [],
                "tools_used": [],
                "success": False,
            }

        if not query or len(query.strip()) < 3:
            return {
                "answer": "Please provide a more detailed question (at least 3 characters).",
                "reasoning_steps": [],
                "tools_used": [],
                "success": False,
            }

        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "agentic_query")
            progress_tracker.update(0, 100, status="starting", message="Starting agentic analysis...")

        try:
            logger.info(f"Processing agentic query: {query[:100]}...")

            if progress_tracker:
                progress_tracker.update(20, 100, status="planning", message="Agent planning approach...")

            max_iter = self.config.get("agentic_max_iterations", 5)
            # LangGraph recursion_limit ≈ 2× iterations (each step = 2 graph nodes)
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": max_iter * 2 + 2},
            )

            if progress_tracker:
                progress_tracker.update(80, 100, status="synthesizing", message="Synthesising final answer...")

            # Extract final answer from last AI message
            messages = result.get("messages", [])
            final_answer = "No answer generated."
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            # Extract reasoning steps: pair each AIMessage tool_call with its ToolMessage
            reasoning_steps = []
            tools_used = []
            tool_call_map: Dict[str, dict] = {}

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_input = tc.get("args", {})
                        tool_id = tc.get("id", "")
                        tools_used.append(tool_name)
                        input_str = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                        tool_call_map[tool_id] = {
                            "tool": tool_name,
                            "input": input_str[:200] + "..." if len(input_str) > 200 else input_str,
                            "observation": "",
                        }
                elif isinstance(msg, ToolMessage):
                    entry = tool_call_map.get(msg.tool_call_id)
                    if entry is not None:
                        obs = str(msg.content)
                        entry["observation"] = obs[:400] + "..." if len(obs) > 400 else obs
                        reasoning_steps.append(entry)

            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", message="Agentic analysis complete")

            return {
                "answer": final_answer,
                "reasoning_steps": reasoning_steps,
                "tools_used": list(dict.fromkeys(tools_used)),  # deduplicate, preserve order
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error in agentic query processing: {e}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", message=f"Error: {str(e)}")

            if "recursion" in str(e).lower():
                error_msg = "The agent reached its thinking limit. Please try a more specific question."
            elif "timeout" in str(e).lower():
                error_msg = "The query took too long. Please try a simpler question."
            else:
                error_msg = f"Error in agentic processing: {str(e)}"

            return {
                "answer": error_msg,
                "reasoning_steps": [],
                "tools_used": [],
                "success": False,
            }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return (
            self.llm is not None
            and self.agent is not None
            and self.simple_rag is not None
            and self.simple_rag.is_ready()
        )

    def get_available_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description.strip()}
            for t in self.tools
        ]

    def get_agentic_stats(self) -> Dict[str, Any]:
        try:
            tools = getattr(self, "tools", [])
            simple_rag = getattr(self, "simple_rag", None)
            web_search_available = False
            try:
                from ddgs import DDGS  # noqa: F401
                web_search_available = True
            except ImportError:
                pass
            return {
                "service_available": self.is_available(),
                "llm_configured": getattr(self, "llm", None) is not None,
                "agent_initialized": getattr(self, "agent", None) is not None,
                "tools_count": len(tools),
                "available_tools": [t.name for t in tools],
                "web_search_available": web_search_available,
                "underlying_rag_ready": simple_rag.is_ready() if simple_rag else False,
                "graph_rag_ready": simple_rag.is_graph_ready() if simple_rag else False,
                "pageindex_ready": simple_rag.is_pageindex_ready() if simple_rag else False,
                "max_iterations": self.config.get("agentic_max_iterations", 5),
            }
        except Exception as e:
            logger.warning(f"get_agentic_stats failed: {e}")
            return {"service_available": False, "error": str(e)}
