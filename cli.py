"""
Command Line Interface for Enhanced SimpleRAG
"""

import sys
import argparse
import logging
from pathlib import Path

# Import the main SimpleRAG class
from simple_rag import EnhancedSimpleRAG
from config import get_config_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimpleRAG-CLI')

def setup_argument_parser():
    """Set up the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced SimpleRAG - Retrieval-Augmented Generation with Graph RAG support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Configure API keys
  python cli.py config --gemini-key YOUR_GEMINI_KEY --claude-key YOUR_CLAUDE_KEY
  
  # Set RAG mode
  python cli.py mode graph
  
  # Index a document
  python cli.py index document.pdf --mode graph
  
  # Query documents
  python cli.py query "What are the main findings?" --mode normal
  
  # Get system status
  python cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure API keys and settings")
    config_parser.add_argument("--gemini-key", help="Set Gemini API key")
    config_parser.add_argument("--claude-key", help="Set Claude API key")
    config_parser.add_argument("--qdrant-key", help="Set Qdrant API key")
    config_parser.add_argument("--qdrant-url", help="Set Qdrant URL")
    config_parser.add_argument("--preferred-llm", choices=["claude", "raw"], help="Set preferred LLM")
    config_parser.add_argument("--rag-mode", choices=["normal", "graph"], help="Set default RAG mode")
    config_parser.add_argument("--chunk-size", type=int, help="Set chunk size")
    config_parser.add_argument("--chunk-overlap", type=int, help="Set chunk overlap")
    config_parser.add_argument("--top-k", type=int, help="Set number of results to retrieve")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    # Mode command
    mode_parser = subparsers.add_parser("mode", help="Get or set RAG mode")
    mode_parser.add_argument("rag_mode", nargs="?", choices=["normal", "graph"], 
                           help="RAG mode to set (if not provided, shows current mode)")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", help="Path to document to index")
    index_parser.add_argument("--mode", choices=["normal", "graph"], 
                            help="RAG mode for this document (overrides default)")
    index_parser.add_argument("--validate-only", action="store_true", 
                            help="Only validate the file, don't index it")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--mode", choices=["normal", "graph"], 
                            help="RAG mode for this query (overrides default)")
    query_parser.add_argument("--verbose", action="store_true", 
                            help="Show detailed search information")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--detailed", action="store_true", 
                             help="Show detailed service information")
    
    # Collections command
    collections_parser = subparsers.add_parser("collections", help="Manage vector database collections")
    collections_parser.add_argument("--list", action="store_true", help="List all collections")
    collections_parser.add_argument("--create", choices=["normal", "graph"], 
                                   help="Create a collection")
    collections_parser.add_argument("--delete", help="Delete a collection by name")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test system components")
    test_parser.add_argument("--embedding", action="store_true", help="Test embedding service")
    test_parser.add_argument("--vector-db", action="store_true", help="Test vector database")
    test_parser.add_argument("--llm", action="store_true", help="Test LLM service")
    test_parser.add_argument("--all", action="store_true", help="Test all services")
    
    return parser

def handle_config_command(args):
    """Handle configuration command."""
    config_manager = get_config_manager()
    
    if args.show:
        print("Current Configuration:")
        print("=" * 50)
        config = config_manager.get_all()
        for key, value in config.items():
            if key.endswith("_api_key") and value:
                print(f"{key}: {'*' * 20}")
            else:
                print(f"{key}: {value}")
        return
    
    # Update configuration
    updates = {}
    if args.gemini_key:
        updates["gemini_api_key"] = args.gemini_key
    if args.claude_key:
        updates["claude_api_key"] = args.claude_key
    if args.qdrant_key:
        updates["qdrant_api_key"] = args.qdrant_key
    if args.qdrant_url:
        updates["qdrant_url"] = args.qdrant_url
    if args.preferred_llm:
        updates["preferred_llm"] = args.preferred_llm
    if args.rag_mode:
        updates["rag_mode"] = args.rag_mode
    if args.chunk_size:
        updates["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        updates["chunk_overlap"] = args.chunk_overlap
    if args.top_k:
        updates["top_k"] = args.top_k
    
    if updates:
        config_manager.update(updates)
        config_manager.save()
        print("Configuration updated successfully!")
        print("Updated settings:")
        for key, value in updates.items():
            if key.endswith("_api_key"):
                print(f"  {key}: {'*' * 20}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No configuration changes specified.")

def handle_mode_command(args):
    """Handle mode command."""
    try:
        simple_rag = EnhancedSimpleRAG()
        
        if args.rag_mode:
            # Set mode
            simple_rag.set_rag_mode(args.rag_mode)
            print(f"RAG mode set to: {args.rag_mode}")
            
            # Show capabilities for the mode
            if args.rag_mode == "graph":
                if simple_rag.is_graph_ready():
                    print("✓ Graph RAG mode is ready")
                else:
                    print("⚠ Graph RAG mode set but service not fully ready")
                    print("  Check that all dependencies are properly configured")
            else:
                print("✓ Normal RAG mode is ready")
        else:
            # Show current mode
            print(f"Current RAG mode: {simple_rag.rag_mode}")
            print(f"System ready: {simple_rag.is_ready()}")
            if simple_rag.rag_mode == "graph":
                print(f"Graph ready: {simple_rag.is_graph_ready()}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_index_command(args):
    """Handle index command."""
    try:
        simple_rag = EnhancedSimpleRAG()
        
        # Validate file first
        file_path = args.file_path
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        # Validate the file
        validation = simple_rag.validate_file(file_path)
        
        print(f"File Validation for: {file_path}")
        print(f"Valid: {validation['valid']}")
        print(f"File exists: {validation['file_exists']}")
        print(f"Supported format: {validation['supported_format']}")
        
        if validation.get("estimated_processing"):
            est = validation["estimated_processing"]
            print(f"Estimated processing time: {est.get('estimated_seconds', 'unknown')} seconds")
            print(f"File size: {est.get('file_size_mb', 0):.1f} MB")
        
        if validation.get("warnings"):
            print("Warnings:")
            for warning in validation["warnings"]:
                print(f"  ⚠ {warning}")
        
        if validation.get("errors"):
            print("Errors:")
            for error in validation["errors"]:
                print(f"  ✗ {error}")
            sys.exit(1)
        
        if args.validate_only:
            print("Validation complete (--validate-only specified)")
            return
        
        # Set mode if specified
        if args.mode:
            simple_rag.set_rag_mode(args.mode)
            print(f"Using {args.mode} mode for this document")
        
        print(f"\nStarting indexing in {simple_rag.rag_mode} mode...")
        
        # Index the document
        success = simple_rag.index_document(file_path)
        
        if success:
            print(f"✓ Document indexed successfully: {file_path}")
        else:
            print(f"✗ Failed to index document: {file_path}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_query_command(args):
    """Handle query command."""
    try:
        simple_rag = EnhancedSimpleRAG()
        
        if not simple_rag.is_ready():
            print("Error: SimpleRAG is not ready. Please check your configuration.")
            sys.exit(1)
        
        # Set mode if specified
        if args.mode:
            simple_rag.set_rag_mode(args.mode)
            print(f"Using {args.mode} mode for this query")
        
        question = args.question
        print(f"Question: {question}")
        print(f"Mode: {simple_rag.rag_mode}")
        print("-" * 50)
        
        # Process the query
        answer = simple_rag.query(question)
        
        print("Answer:")
        print(answer)
        
        if args.verbose:
            print("\n" + "=" * 50)
            print("System Information:")
            status = simple_rag.get_status()
            print(f"Services ready: {status['ready']}")
            if simple_rag.rag_mode == "graph":
                print(f"Graph ready: {status['graph_ready']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_status_command(args):
    """Handle status command."""
    try:
        simple_rag = EnhancedSimpleRAG()
        status = simple_rag.get_status()
        
        print("Enhanced SimpleRAG Status")
        print("=" * 50)
        print(f"Overall ready: {status['ready']}")
        print(f"Current RAG mode: {status['rag_mode']}")
        print(f"Graph RAG ready: {status['graph_ready']}")
        
        print("\nServices:")
        services = status['services']
        for service_name, available in services.items():
            status_icon = "✓" if available else "✗"
            print(f"  {status_icon} {service_name}: {'Available' if available else 'Not Available'}")
        
        if status.get('initialization_errors'):
            print("\nErrors:")
            for error in status['initialization_errors']:
                print(f"  ✗ {error}")
        
        if status.get('initialization_warnings'):
            print("\nWarnings:")
            for warning in status['initialization_warnings']:
                print(f"  ⚠ {warning}")
        
        if args.detailed:
            print("\nDetailed Information:")
            
            if 'vector_db_status' in status:
                vdb_status = status['vector_db_status']
                print(f"\nVector Database:")
                print(f"  Connected: {vdb_status.get('connected', False)}")
                print(f"  URL: {vdb_status.get('url', 'Not configured')}")
                if 'total_collections' in vdb_status:
                    print(f"  Collections: {vdb_status['total_collections']}")
            
            if 'embedding_stats' in status:
                emb_stats = status['embedding_stats']
                print(f"\nEmbedding Service:")
                print(f"  Cache enabled: {emb_stats.get('cache_enabled', False)}")
                print(f"  Dimensions: {emb_stats.get('embedding_dimension', 'Unknown')}")
                if 'cached_embeddings' in emb_stats:
                    print(f"  Cached embeddings: {emb_stats['cached_embeddings']}")
            
            if 'graph_stats' in status:
                graph_stats = status['graph_stats']
                print(f"\nKnowledge Graph:")
                print(f"  Nodes: {graph_stats.get('total_nodes', 0)}")
                print(f"  Edges: {graph_stats.get('total_edges', 0)}")
                print(f"  Connected components: {graph_stats.get('connected_components', 0)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_collections_command(args):
    """Handle collections command."""
    try:
        simple_rag = EnhancedSimpleRAG()
        
        if not simple_rag.vector_db_service:
            print("Error: Vector database service not available")
            sys.exit(1)
        
        if args.list:
            print("Vector Database Collections:")
            print("=" * 50)
            
            collections_info = simple_rag.get_collections_info()
            
            if 'error' in collections_info:
                print(f"Error: {collections_info['error']}")
                return
            
            collections = collections_info.get('collections', [])
            normal_collection = collections_info.get('normal_collection')
            graph_collection = collections_info.get('graph_collection')
            
            if not collections:
                print("No collections found.")
                return
            
            for collection in collections:
                name = collection['name']
                points = collection.get('points_count', 0)
                coll_type = collection.get('type', 'other')
                
                type_icon = "📚" if coll_type == "normal_rag" else "🕸️" if coll_type == "graph_rag" else "📄"
                print(f"{type_icon} {name}")
                print(f"    Points: {points:,}")
                print(f"    Type: {coll_type}")
                if 'config' in collection:
                    config = collection['config']
                    print(f"    Distance: {config.get('distance', 'unknown')}")
                    print(f"    Dimensions: {config.get('size', 'unknown')}")
                print()
        
        elif args.create:
            collection_type = args.create
            print(f"Creating {collection_type} collection...")
            
            try:
                if collection_type == "normal":
                    collection_name = simple_rag.config["collection_name"]
                else:
                    collection_name = simple_rag.config["graph_collection_name"]
                
                created = simple_rag.vector_db_service.ensure_collection_exists(collection_name)
                
                if created:
                    print(f"✓ Created collection: {collection_name}")
                else:
                    print(f"Collection already exists: {collection_name}")
            
            except Exception as e:
                print(f"Error creating collection: {str(e)}")
                sys.exit(1)
        
        elif args.delete:
            collection_name = args.delete
            print(f"Deleting collection: {collection_name}")
            
            # Confirm deletion
            response = input(f"Are you sure you want to delete '{collection_name}'? This cannot be undone. (yes/no): ")
            if response.lower() != 'yes':
                print("Deletion cancelled.")
                return
            
            try:
                simple_rag.vector_db_service.delete_collection(collection_name)
                print(f"✓ Deleted collection: {collection_name}")
            except Exception as e:
                print(f"Error deleting collection: {str(e)}")
                sys.exit(1)
        
        else:
            print("Please specify an action: --list, --create, or --delete")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_test_command(args):
    """Handle test command."""
    try:
        simple_rag = EnhancedSimpleRAG()
        
        tests_to_run = []
        if args.all:
            tests_to_run = ["embedding", "vector_db", "llm"]
        else:
            if args.embedding:
                tests_to_run.append("embedding")
            if args.vector_db:
                tests_to_run.append("vector_db")
            if args.llm:
                tests_to_run.append("llm")
        
        if not tests_to_run:
            print("Please specify which tests to run: --embedding, --vector-db, --llm, or --all")
            return
        
        print("Running Component Tests")
        print("=" * 50)
        
        # Test embedding service
        if "embedding" in tests_to_run:
            print("Testing Embedding Service...")
            try:
                if simple_rag.embedding_service:
                    test_text = "This is a test sentence for embedding generation."
                    embedding = simple_rag.embedding_service.get_embedding(test_text)
                    
                    if embedding and len(embedding) > 0:
                        print(f"✓ Embedding service working (dimension: {len(embedding)})")
                    else:
                        print("✗ Embedding service failed - no embedding returned")
                else:
                    print("✗ Embedding service not initialized")
            except Exception as e:
                print(f"✗ Embedding service error: {str(e)}")
            print()
        
        # Test vector database
        if "vector_db" in tests_to_run:
            print("Testing Vector Database...")
            try:
                if simple_rag.vector_db_service:
                    status = simple_rag.vector_db_service.get_status()
                    if status.get('connected'):
                        print(f"✓ Vector database connected ({status.get('total_collections', 0)} collections)")
                    else:
                        print("✗ Vector database not connected")
                        if 'last_error' in status:
                            print(f"    Error: {status['last_error']}")
                else:
                    print("✗ Vector database service not initialized")
            except Exception as e:
                print(f"✗ Vector database error: {str(e)}")
            print()
        
        # Test LLM service
        if "llm" in tests_to_run:
            print("Testing LLM Service...")
            try:
                if simple_rag.llm_service:
                    test_result = simple_rag.llm_service.test_connection()
                    if test_result.get('service_available'):
                        print(f"✓ LLM service working ({simple_rag.llm_service.preferred_llm})")
                        if 'test_response' in test_result:
                            print(f"    Response: {test_result['test_response'][:100]}...")
                    else:
                        print(f"✗ LLM service not available: {test_result.get('error', 'Unknown error')}")
                else:
                    print("✗ LLM service not initialized")
            except Exception as e:
                print(f"✗ LLM service error: {str(e)}")
            print()
        
        print("Test complete.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    """Main CLI function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "config":
            handle_config_command(args)
        elif args.command == "mode":
            handle_mode_command(args)
        elif args.command == "index":
            handle_index_command(args)
        elif args.command == "query":
            handle_query_command(args)
        elif args.command == "status":
            handle_status_command(args)
        elif args.command == "collections":
            handle_collections_command(args)
        elif args.command == "test":
            handle_test_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()