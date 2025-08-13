"""
Agentic AI Service using LangChain for autonomous tool selection and multi-step reasoning
FIXED VERSION with better error handling and iteration limits
"""

import logging
import json
from typing import List, Dict, Any, Optional, Callable
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from extensions import ProgressTracker

logger = logging.getLogger(__name__)

class AgenticRAGService:
    """
    Agentic AI service that uses LangChain agents to autonomously select tools
    and perform multi-step reasoning with your existing RAG components.
    FIXED VERSION with better error handling.
    """
    
    def __init__(self, config, simple_rag_instance):
        self.config = config
        self.simple_rag = simple_rag_instance
        
        # Initialize LangChain LLM - use same model as your working LLM service
        self.llm = None
        if config.get("claude_api_key"):
            try:
                # FIXED: Use current working Claude model
                model = "claude-3-haiku-20240307"  # Fast and reliable model that works
                
                self.llm = ChatAnthropic(
                    api_key=config["claude_api_key"],
                    model=model,
                    temperature=0.1,
                    timeout=30,  # Add timeout
                    max_retries=2  # Limit retries
                )
                logger.info(f"LangChain Claude LLM initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain LLM: {e}")
        
        # Create tools that wrap your existing RAG functionality
        self.tools = self._create_tools()
        
        # Initialize agent
        self.agent = None
        self.agent_executor = None
        if self.llm and self.tools:
            self._initialize_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools that wrap your existing RAG functionality."""
        tools = []
        
        # Tool 1: Document Search (Normal RAG)
        tools.append(Tool(
            name="search_documents",
            description="""
            Search through document chunks using semantic similarity.
            Use this for straightforward factual questions about document content.
            Input: A specific question or search query as a string.
            Returns: Relevant document passages with source information.
            """,
            func=self._search_documents_tool
        ))
        
        # Tool 2: Knowledge Graph Search (Graph RAG)
        tools.append(Tool(
            name="search_knowledge_graph",
            description="""
            Search through entities and relationships in the knowledge graph.
            Use this for questions about connections, relationships, or when you need
            to understand how different entities relate to each other.
            Input: A question about entities or relationships as a string.
            Returns: Related entities and their connections.
            """,
            func=self._search_graph_tool
        ))
        
        # Tool 3: Hybrid Analysis
        tools.append(Tool(
            name="analyze_with_both_methods",
            description="""
            Perform both document search and graph analysis for comprehensive results.
            Use this for complex questions that might benefit from multiple perspectives
            or when initial searches don't provide sufficient information.
            Input: A complex question as a string.
            Returns: Combined insights from both document and graph analysis.
            """,
            func=self._hybrid_analysis_tool
        ))
        
        # Tool 4: Verify Information
        tools.append(Tool(
            name="verify_information",
            description="""
            Cross-check information by searching for supporting or contradicting evidence.
            Use this when you need to validate claims or find additional confirmation.
            Input: A statement or claim to verify as a string.
            Returns: Supporting or contradicting evidence from the knowledge base.
            """,
            func=self._verify_information_tool
        ))
        
        return tools
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and prompt."""
        try:
            # FIXED: Simplified and more robust prompt
            prompt = PromptTemplate.from_template("""You are a helpful research assistant with access to a document knowledge base.

Available tools:
{tools}

Tool Names: {tool_names}

IMPORTANT RULES:
1. Always use the exact tool names provided: {tool_names}
2. For simple questions, start with search_documents
3. For relationship questions, use search_knowledge_graph
4. If you get sufficient information from one tool, provide your final answer
5. Don't use the same tool twice with the same input
6. Keep responses concise and helpful

Use this format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}""")
            
            # Create the ReAct agent
            self.agent = create_react_agent(self.llm, self.tools, prompt)
            
            # FIXED: Create agent executor with better limits and error handling
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,  # FIXED: Reduced from 5 to 3 to prevent loops
                max_execution_time=60,  # FIXED: Add time limit (60 seconds)
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                early_stopping_method="generate"  # FIXED: Stop early if possible
            )
            
            logger.info("LangChain ReAct agent initialized successfully with strict limits")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain agent: {e}")
            self.agent = None
            self.agent_executor = None
    
    def _search_documents_tool(self, query: str) -> str:
        """Tool wrapper for document search using existing Normal RAG."""
        try:
            logger.debug(f"Agent using document search for: {query}")
            
            # FIXED: Add input validation
            if not query or len(query.strip()) < 3:
                return "Error: Query too short or empty. Please provide a meaningful search query."
            
            # Set to normal mode temporarily
            original_mode = self.simple_rag.rag_mode
            self.simple_rag.set_rag_mode("normal")
            
            # Use existing query method
            result = self.simple_rag._query_normal_mode(query)
            
            # Restore original mode
            self.simple_rag.set_rag_mode(original_mode)
            
            # FIXED: Truncate very long results to prevent context overflow
            if len(result) > 1500:
                result = result[:1500] + "... [truncated for brevity]"
            
            return f"Document search results for '{query}':\n{result}"
            
        except Exception as e:
            logger.error(f"Error in document search tool: {e}")
            return f"Error searching documents: {str(e)}"
    
    def _search_graph_tool(self, query: str) -> str:
        """Tool wrapper for knowledge graph search using existing Graph RAG."""
        try:
            logger.debug(f"Agent using graph search for: {query}")
            
            # FIXED: Add input validation
            if not query or len(query.strip()) < 3:
                return "Error: Query too short or empty. Please provide a meaningful search query."
            
            if not self.simple_rag.is_graph_ready():
                return "Knowledge graph not available. Please ensure Graph RAG is properly configured."
            
            # Set to graph mode temporarily
            original_mode = self.simple_rag.rag_mode
            self.simple_rag.set_rag_mode("graph")
            
            # Use existing graph search
            result = self.simple_rag._query_graph_mode(query)
            
            # Restore original mode
            self.simple_rag.set_rag_mode(original_mode)
            
            # FIXED: Truncate very long results
            if len(result) > 1500:
                result = result[:1500] + "... [truncated for brevity]"
            
            return f"Knowledge graph results for '{query}':\n{result}"
            
        except Exception as e:
            logger.error(f"Error in graph search tool: {e}")
            return f"Error searching knowledge graph: {str(e)}"
    
    def _hybrid_analysis_tool(self, query: str) -> str:
        """Tool that combines both document and graph search for comprehensive analysis."""
        try:
            logger.debug(f"Agent using hybrid analysis for: {query}")
            
            # FIXED: Add input validation
            if not query or len(query.strip()) < 3:
                return "Error: Query too short or empty. Please provide a meaningful search query."
            
            # Get results from both methods (but truncated)
            doc_results = self._search_documents_tool(query)
            
            # Only do graph search if available
            if self.simple_rag.is_graph_ready():
                graph_results = self._search_graph_tool(query)
            else:
                graph_results = "Graph search not available - Graph RAG not configured."
            
            # FIXED: Create more concise combined result
            combined_result = f"""HYBRID ANALYSIS FOR: {query}

DOCUMENT FINDINGS:
{doc_results[:700] if len(doc_results) > 700 else doc_results}

GRAPH FINDINGS:
{graph_results[:700] if len(graph_results) > 700 else graph_results}

This analysis combines document content with relationship insights for a comprehensive view."""
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis tool: {e}")
            return f"Error in hybrid analysis: {str(e)}"
    
    def _verify_information_tool(self, claim: str) -> str:
        """Tool to verify information by searching for supporting evidence."""
        try:
            logger.debug(f"Agent verifying information: {claim}")
            
            # FIXED: Add input validation
            if not claim or len(claim.strip()) < 5:
                return "Error: Claim too short or empty. Please provide a specific claim to verify."
            
            # Search for supporting evidence with focused query
            verification_query = f"evidence about {claim}"
            
            # Use document search for verification
            doc_evidence = self._search_documents_tool(verification_query)
            
            # FIXED: More focused verification result
            result = f"""VERIFICATION FOR: {claim}

EVIDENCE FOUND:
{doc_evidence[:800] if len(doc_evidence) > 800 else doc_evidence}

VERIFICATION STATUS: Based on available evidence in the knowledge base."""
            
            return result
            
        except Exception as e:
            logger.error(f"Error in verification tool: {e}")
            return f"Error verifying information: {str(e)}"
    
    def process_agentic_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process a query using the agentic approach with autonomous tool selection.
        FIXED VERSION with better error handling and limits.
        """
        if not self.is_available():
            return {
                "answer": "Agentic AI service not available. Please check Claude API configuration.",
                "reasoning_steps": [],
                "tools_used": [],
                "success": False
            }
        
        # FIXED: Add input validation
        if not query or len(query.strip()) < 3:
            return {
                "answer": "Please provide a more detailed question (at least 3 characters).",
                "reasoning_steps": [],
                "tools_used": [],
                "success": False
            }
        
        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "agentic_query")
            progress_tracker.update(0, 100, status="starting", 
                                   message="Starting agentic analysis...")
        
        try:
            logger.info(f"Processing agentic query: {query[:100]}...")
            
            if progress_tracker:
                progress_tracker.update(20, 100, status="planning", 
                                       message="Agent planning approach...")
            
            # FIXED: Let the agent decide which tools to use with strict limits
            result = self.agent_executor.invoke({
                "input": query
            })
            
            if progress_tracker:
                progress_tracker.update(80, 100, status="synthesizing", 
                                       message="Synthesizing final answer...")
            
            # FIXED: Extract information about the process with better error handling
            reasoning_steps = []
            tools_used = []
            
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    try:
                        if len(step) >= 2:
                            # For ReAct agent, step format is (AgentAction, observation)
                            agent_action = step[0]
                            observation = step[1]
                            
                            # Safely extract tool name
                            tool_name = getattr(agent_action, 'tool', 'unknown_tool')
                            tools_used.append(tool_name)
                            
                            # Safely extract tool input
                            tool_input = getattr(agent_action, 'tool_input', 'No input provided')
                            
                            # Create safe reasoning step
                            reasoning_step = {
                                "tool": str(tool_name),
                                "input": str(tool_input)[:100] + "..." if len(str(tool_input)) > 100 else str(tool_input),
                                "reasoning": "Agent selected this tool for the query",
                                "observation": str(observation)[:200] + "..." if len(str(observation)) > 200 else str(observation)
                            }
                            reasoning_steps.append(reasoning_step)
                    except Exception as e:
                        logger.warning(f"Error processing reasoning step: {e}")
                        continue
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                       message="Agentic analysis complete")
            
            # FIXED: Extract the final answer safely
            final_answer = result.get("output", "No answer generated")
            
            return {
                "answer": final_answer,
                "reasoning_steps": reasoning_steps,
                "tools_used": list(set(tools_used)),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in agentic query processing: {e}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                       message=f"Error: {str(e)}")
            
            # FIXED: Provide more helpful error messages
            if "parsing" in str(e).lower():
                error_msg = "The agent had trouble understanding the response format. Please try rephrasing your question."
            elif "timeout" in str(e).lower() or "time" in str(e).lower():
                error_msg = "The query took too long to process. Please try a simpler question or break it into smaller parts."
            elif "iteration" in str(e).lower():
                error_msg = "The agent reached its thinking limit. Please try asking a more specific question."
            else:
                error_msg = f"Error in agentic processing: {str(e)}"
            
            return {
                "answer": error_msg,
                "reasoning_steps": [],
                "tools_used": [],
                "success": False
            }
    
    def is_available(self) -> bool:
        """Check if the agentic service is ready for use."""
        return (self.llm is not None and 
                self.agent_executor is not None and 
                self.simple_rag is not None and 
                self.simple_rag.is_ready())
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get information about available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description.strip()
            }
            for tool in self.tools
        ]
    
    def get_agentic_stats(self) -> Dict[str, Any]:
        """Get statistics about the agentic service."""
        return {
            "service_available": self.is_available(),
            "llm_configured": self.llm is not None,
            "agent_initialized": self.agent_executor is not None,
            "tools_count": len(self.tools),
            "available_tools": [tool.name for tool in self.tools],
            "underlying_rag_ready": self.simple_rag.is_ready() if self.simple_rag else False,
            "graph_rag_ready": self.simple_rag.is_graph_ready() if self.simple_rag else False,
            "max_iterations": 3,  # FIXED: Show the limit
            "max_execution_time": 60  # FIXED: Show the time limit
        }