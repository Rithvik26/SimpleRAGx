"""
Updated Flask application using the modular SimpleRAG structure
"""

from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import os
import tempfile
import logging
import json
import threading
import re
import markdown
from werkzeug.utils import secure_filename

# Import the modular SimpleRAG components
from simple_rag import EnhancedSimpleRAG
from config import get_config_manager
from extensions import ProgressTracker

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'simplerag_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

@app.template_filter('nl2br')
def nl2br_filter(text):
    """Convert newlines to HTML breaks and process markdown."""
    if not text:
        return ""
    # First convert <br> tags to newlines if any
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Convert markdown to HTML
    html = markdown.markdown(text)
    return html

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimpleRAG-Web')

# Global SimpleRAG instance
simplerag_instance = None

def initialize_services():
    """Initialize SimpleRAG services based on configuration."""
    global simplerag_instance
    
    try:
        # Log environment variables (without exposing keys)
        logger.info("Checking environment variables...")
        logger.info(f"GEMINI_API_KEY: {'SET' if os.environ.get('GEMINI_API_KEY') else 'NOT SET'}")
        logger.info(f"QDRANT_API_KEY: {'SET' if os.environ.get('QDRANT_API_KEY') else 'NOT SET'}")
        logger.info(f"QDRANT_URL: {os.environ.get('QDRANT_URL', 'NOT SET')}")
        
        logger.info("Initializing Enhanced SimpleRAG services...")
        simplerag_instance = EnhancedSimpleRAG()
        
        if simplerag_instance.is_ready():
            logger.info("✓ SimpleRAG services initialized successfully")
            return True
        else:
            logger.warning("⚠ SimpleRAG initialized with some issues")
            status = simplerag_instance.get_status()
            for error in status.get('initialization_errors', []):
                logger.error(f"  Error: {error}")
            return False
        
    except Exception as e:
        logger.error(f"Error initializing SimpleRAG services: {e}")
        simplerag_instance = None
        return False

def index_document_background(file_path, session_id=None):
    """Index document in the background with comprehensive error handling."""
    global simplerag_instance
    
    try:
        if simplerag_instance and simplerag_instance.is_ready():
            logger.info(f"Starting background indexing for: {file_path}")
            success = simplerag_instance.index_document(file_path, session_id)
            
            if success:
                logger.info(f"Successfully indexed: {file_path}")
            else:
                logger.warning(f"Indexing returned False for: {file_path}")
                
        else:
            logger.error("SimpleRAG instance not ready for indexing")
            if session_id:
                tracker = ProgressTracker.get_tracker(session_id, "index_document")
                if tracker:
                    tracker.update(100, 100, status="error", 
                                 message="SimpleRAG services not ready")
                    
    except Exception as e:
        logger.error(f"Background indexing error for {file_path}: {str(e)}")
        if session_id:
            tracker = ProgressTracker.get_tracker(session_id, "index_document")
            if tracker:
                tracker.update(100, 100, status="error", 
                             message=f"Indexing error: {str(e)}")
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {e}")

def process_query_background(question, session_id=None):
    """Process query in background with progress tracking."""
    global simplerag_instance
    
    try:
        if not simplerag_instance or not simplerag_instance.is_ready():
            result = "SimpleRAG services not ready. Please check configuration."
        else:
            logger.info(f"Processing background query: {question[:50]}...")
            result = simplerag_instance.query(question, session_id)
            logger.info("Background query processing completed")
        
        # Store result in session-based storage
        if session_id:
            session_results = getattr(process_query_background, 'results', {})
            session_results[session_id] = result
            process_query_background.results = session_results
            
            # Update progress tracker
            progress_tracker = ProgressTracker.get_tracker(session_id, "query")
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message="Query processing complete")
                
    except Exception as e:
        logger.error(f"Background query error: {str(e)}")
        error_message = f"Error processing query: {str(e)}"
        
        if session_id:
            session_results = getattr(process_query_background, 'results', {})
            session_results[session_id] = error_message
            process_query_background.results = session_results
            
            progress_tracker = ProgressTracker.get_tracker(session_id, "query")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Query error: {str(e)}")

@app.route('/')
def home():
    """Render home page with current configuration status."""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    config_manager = get_config_manager()
    config = config_manager.get_all()
    is_configured = bool(config.get("gemini_api_key"))
    
    current_mode = "Unknown"
    if simplerag_instance:
        current_mode = simplerag_instance.rag_mode
    
    return render_template('index.html', 
                          is_configured=is_configured, 
                          config=config,
                          current_mode=current_mode)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    status = {
        "status": "OK",
        "simplerag_initialized": simplerag_instance is not None,
        "simplerag_ready": simplerag_instance.is_ready() if simplerag_instance else False,
        "rag_mode": simplerag_instance.rag_mode if simplerag_instance else "Unknown"
    }
    return jsonify(status), 200

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Configure API keys and RAG mode settings."""
    config_manager = get_config_manager()
    
    if request.method == 'POST':
        config_changed = False
        
        # Get form data
        form_data = {
            "gemini_api_key": request.form.get('gemini_api_key', '').strip(),
            "claude_api_key": request.form.get('claude_api_key', '').strip(),
            "qdrant_api_key": request.form.get('qdrant_api_key', '').strip(),
            "qdrant_url": request.form.get('qdrant_url', '').strip(),
            "preferred_llm": request.form.get('preferred_llm', ''),
            "rag_mode": request.form.get('rag_mode', '')
        }
        
        # Update configuration
        updates = {}
        for key, value in form_data.items():
            if value and value != config_manager.get(key, ''):
                updates[key] = value
                config_changed = True
        
        # Handle numeric settings
        numeric_fields = {
            "chunk_size": (100, 5000, 1000),
            "chunk_overlap": (0, 1000, 200),
            "top_k": (1, 20, 5),
            "max_entities_per_chunk": (5, 50, 20),
            "graph_reasoning_depth": (1, 5, 2)
        }
        
        for field, (min_val, max_val, default_val) in numeric_fields.items():
            try:
                value = int(request.form.get(field, default_val))
                value = max(min_val, min(max_val, value))
                if value != config_manager.get(field, default_val):
                    updates[field] = value
                    config_changed = True
            except (ValueError, TypeError):
                flash(f"Invalid value for {field}, using default", "warning")
        
        # Handle float settings
        try:
            threshold = float(request.form.get("entity_similarity_threshold", 0.8))
            threshold = max(0.5, min(1.0, threshold))
            if threshold != config_manager.get("entity_similarity_threshold", 0.8):
                updates["entity_similarity_threshold"] = threshold
                config_changed = True
        except (ValueError, TypeError):
            flash("Invalid entity similarity threshold, using default", "warning")
        
        # Save changes
        if config_changed:
            try:
                config_manager.update(updates)
                config_manager.save()
                
                # Reinitialize services
                initialize_services()
                
                flash("Configuration saved and services reinitialized successfully!", "success")
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                flash("Error saving configuration", "danger")
        else:
            flash("No changes detected in configuration", "info")
        
        return redirect(url_for('home'))
    
    return render_template('setup.html', config=config_manager.get_all())

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document uploads with RAG mode selection."""
    config_manager = get_config_manager()
    
    if request.method == 'POST':
        if 'document' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['document']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        # Get RAG mode
        upload_rag_mode = request.form.get('rag_mode', 'normal')
        if upload_rag_mode not in ['normal', 'graph']:
            upload_rag_mode = 'normal'
        
        # Check if SimpleRAG is ready
        if not simplerag_instance or not simplerag_instance.is_ready():
            flash('SimpleRAG not ready. Please configure API keys first.', 'danger')
            return redirect(url_for('setup'))
        
        # Update RAG mode if different
        if upload_rag_mode != simplerag_instance.rag_mode:
            try:
                simplerag_instance.set_rag_mode(upload_rag_mode)
                logger.info(f"RAG mode changed to {upload_rag_mode}")
            except Exception as e:
                logger.error(f"Error changing RAG mode: {e}")
                flash(f"Error changing RAG mode: {e}", 'warning')
        
        # Validate file
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session.get('session_id', 'unknown')}_{filename}")
            file.save(file_path)
            
            # Validate the file
            validation = simplerag_instance.validate_file(file_path)
            if not validation['valid']:
                os.remove(file_path)  # Clean up
                error_messages = '; '.join(validation['errors'])
                flash(f"File validation failed: {error_messages}", 'danger')
                return redirect(request.url)
            
            logger.info(f"File saved and validated: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            flash(f"Error saving file: {e}", 'danger')
            return redirect(request.url)
        
        # Start indexing
        session_id = session.get('session_id')
        progress_tracker = ProgressTracker(session_id, "index_document")
        progress_tracker.update(0, 100, status="starting", 
                               message=f"Starting to index {filename} in {upload_rag_mode} mode")
        
        try:
            indexing_thread = threading.Thread(
                target=index_document_background,
                args=(file_path, session_id),
                daemon=True
            )
            indexing_thread.start()
            
            flash(f"Document '{filename}' uploaded successfully. Indexing in {upload_rag_mode} mode.", 'success')
            return redirect(url_for('upload_progress'))
            
        except Exception as e:
            logger.error(f"Error starting indexing thread: {e}")
            flash(f"Error starting document processing: {e}", 'danger')
            return redirect(request.url)
    
    return render_template('upload.html', config=config_manager.get_all())

@app.route('/upload/progress')
def upload_progress():
    """Show document upload progress page."""
    return render_template('upload_progress.html')

@app.route('/api/progress/<operation_type>')
def get_progress(operation_type):
    """API endpoint to get progress information."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found"}), 404
    
    tracker = ProgressTracker.get_tracker(session_id, operation_type)
    if not tracker:
        return jsonify({"error": "No progress tracker found"}), 404
    
    return jsonify(tracker.get_info())

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Handle document querying with RAG mode selection."""
    config_manager = get_config_manager()
    answer = None
    question = None
    in_progress = False
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        query_rag_mode = request.form.get('rag_mode', None)
        
        if not question:
            flash('Please enter a question', 'warning')
            return render_template('query.html', config=config_manager.get_all())
        
        if not simplerag_instance or not simplerag_instance.is_ready():
            flash('SimpleRAG not ready. Please check configuration.', 'danger')
            return redirect(url_for('setup'))
        
        # Switch RAG mode if specified
        if query_rag_mode and query_rag_mode in ['normal', 'graph']:
            current_mode = simplerag_instance.rag_mode
            if query_rag_mode != current_mode:
                try:
                    simplerag_instance.set_rag_mode(query_rag_mode)
                    logger.info(f"RAG mode temporarily changed to {query_rag_mode} for query")
                except Exception as e:
                    logger.error(f"Error changing RAG mode: {e}")
                    flash(f"Error changing RAG mode: {e}", 'warning')
        
        session_id = session.get('session_id')
        progress_tracker = ProgressTracker(session_id, "query")
        progress_tracker.update(0, 100, status="starting", 
                               message="Processing your question")
        
        # Determine async processing
        current_mode = simplerag_instance.rag_mode
        use_async = (
            len(question) > 100 or 
            request.form.get('async', 'false') == 'true' or 
            current_mode == 'graph'
        )
        
        if use_async:
            in_progress = True
            try:
                query_thread = threading.Thread(
                    target=process_query_background,
                    args=(question, session_id),
                    daemon=True
                )
                query_thread.start()
                logger.info("Started async query processing")
            except Exception as e:
                logger.error(f"Error starting query thread: {e}")
                flash(f"Error processing query: {e}", 'danger')
                in_progress = False
        else:
            # Synchronous processing
            try:
                answer = simplerag_instance.query(question, session_id)
                logger.info("Completed sync query processing")
            except Exception as e:
                logger.error(f"Error in sync query processing: {e}")
                answer = f"Error processing your query: {str(e)}"
    
    return render_template('query.html', 
                          question=question, 
                          answer=answer, 
                          in_progress=in_progress,
                          config=config_manager.get_all())

@app.route('/api/query/result')
def get_query_result():
    """API endpoint to get the result of an asynchronous query."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found"}), 404
    
    session_results = getattr(process_query_background, 'results', {})
    result = session_results.get(session_id)
    
    if result:
        del session_results[session_id]
        return jsonify({"result": result})
    
    return jsonify({"error": "No result available"}), 404

@app.route('/advanced')
def advanced():
    """Advanced settings and system information."""
    config_manager = get_config_manager()
    config = config_manager.get_all()
    
    system_status = {
        "simplerag_initialized": simplerag_instance is not None,
        "simplerag_ready": simplerag_instance.is_ready() if simplerag_instance else False,
        "current_rag_mode": simplerag_instance.rag_mode if simplerag_instance else "Unknown",
        "graph_ready": simplerag_instance.is_graph_ready() if simplerag_instance else False
    }
    
    return render_template('advanced.html', config=config, system_status=system_status)

@app.route('/api/system/status')
def system_status():
    """Enhanced API endpoint for system status information."""
    config_manager = get_config_manager()
    config = config_manager.get_all()
    
    if simplerag_instance:
        status = simplerag_instance.get_status()
        
        response_status = {
            "simplerag_initialized": True,
            "simplerag_ready": status['ready'],
            "graph_ready": status['graph_ready'],
            "current_rag_mode": status['rag_mode'],
            "api_keys_configured": {
                "gemini": bool(config.get("gemini_api_key")),
                "claude": bool(config.get("claude_api_key")),
                "qdrant": bool(config.get("qdrant_api_key"))
            },
            "services": status['services'],
            "initialization_errors": status['initialization_errors'],
            "initialization_warnings": status['initialization_warnings'],
            "config": {
                "rag_mode": config.get("rag_mode", "normal"),
                "preferred_llm": config.get("preferred_llm", "claude"),
                "chunk_size": config.get("chunk_size", 1000),
                "top_k": config.get("top_k", 5)
            },
            "collections": {
                "normal": config.get("collection_name", "simple_rag_docs"),
                "graph": config.get("graph_collection_name", "simple_rag_graph")
            }
        }
        
        # Add service-specific status
        if 'vector_db_status' in status:
            response_status["qdrant"] = status['vector_db_status']
        
        if 'embedding_stats' in status:
            response_status["embedding_service"] = status['embedding_stats']
        
        if 'graph_stats' in status:
            response_status["graph_stats"] = status['graph_stats']
    else:
        response_status = {
            "simplerag_initialized": False,
            "simplerag_ready": False,
            "graph_ready": False,
            "current_rag_mode": "Unknown",
            "error": "SimpleRAG not initialized"
        }
    
    return jsonify(response_status)

# Admin routes
@app.route('/admin')
def admin_panel():
    """Admin panel for system management."""
    return render_template('admin.html')

@app.route('/admin/qdrant')
def qdrant_admin():
    """Qdrant collection management interface."""
    return render_template('qdrant_admin.html')

@app.route('/api/admin/qdrant/status')
def qdrant_status():
    """Get Qdrant connection status and basic info."""
    if not simplerag_instance or not simplerag_instance.vector_db_service:
        return jsonify({
            "connected": False,
            "error": "Vector database service not available"
        })
    
    try:
        status = simplerag_instance.vector_db_service.get_status()
        return jsonify({
            "connected": status.get('connected', False),
            "url": status.get('url', 'Unknown'),
            "collection_count": status.get('total_collections', 0),
            "error": status.get('last_error')
        })
    except Exception as e:
        return jsonify({
            "connected": False,
            "error": f"Status check failed: {str(e)}"
        })

@app.route('/api/admin/qdrant/collections')
def list_collections():
    """List all Qdrant collections with details."""
    if not simplerag_instance or not simplerag_instance.vector_db_service:
        return jsonify({"error": "Vector database service not available"}), 500
    
    try:
        collections_info = simplerag_instance.get_collections_info()
        return jsonify(collections_info)
    except Exception as e:
        return jsonify({"error": f"Failed to list collections: {str(e)}"}), 500

@app.route('/api/admin/qdrant/collections', methods=['POST'])
def create_collection():
    """Create a new collection."""
    if not simplerag_instance or not simplerag_instance.vector_db_service:
        return jsonify({"error": "Vector database service not available"}), 500
    
    try:
        data = request.get_json()
        collection_type = data.get('type', 'normal')
        
        config_manager = get_config_manager()
        config = config_manager.get_all()
        
        if collection_type == 'normal':
            collection_name = config["collection_name"]
        elif collection_type == 'graph':
            collection_name = config["graph_collection_name"]
        else:
            return jsonify({"error": "Invalid collection type"}), 400
        
        created = simplerag_instance.vector_db_service.ensure_collection_exists(collection_name)
        return jsonify({
            "success": True,
            "collection_name": collection_name,
            "created": created,
            "message": f"Collection '{collection_name}' {'created' if created else 'already exists'}"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create collection: {str(e)}"}), 500

@app.route('/api/admin/qdrant/collections/<collection_name>', methods=['DELETE'])
def delete_collection(collection_name):
    """Delete a collection."""
    if not simplerag_instance or not simplerag_instance.vector_db_service:
        return jsonify({"error": "Vector database service not available"}), 500
    
    try:
        simplerag_instance.vector_db_service.delete_collection(collection_name)
        return jsonify({
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to delete collection: {str(e)}"}), 500

@app.route('/api/admin/qdrant/collections/<collection_name>/inspect')
def inspect_collection(collection_name):
    """Get detailed information about a collection."""
    if not simplerag_instance or not simplerag_instance.vector_db_service:
        return jsonify({"error": "Vector database service not available"}), 500
    
    try:
        collection_info = simplerag_instance.vector_db_service.get_collection_info(collection_name)
        return jsonify(collection_info)
    except Exception as e:
        return jsonify({"error": f"Failed to inspect collection: {str(e)}"}), 500

@app.route('/api/admin/cache/clear', methods=['POST'])
def clear_cache():
    """Clear embedding cache."""
    if not simplerag_instance or not simplerag_instance.embedding_service:
        return jsonify({"error": "Embedding service not available"}), 500
    
    try:
        success = simplerag_instance.embedding_service.clear_cache()
        return jsonify({
            "success": success,
            "message": "Cache cleared successfully" if success else "Cache not available or already empty"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.route('/api/admin/collections/search/<collection_name>')
def search_collection(collection_name):
    """Search in a specific collection for debugging."""
    if not simplerag_instance or not simplerag_instance.vector_db_service:
        return jsonify({"error": "Vector database service not available"}), 500
    
    try:
        query = request.args.get('query', 'test')
        limit = min(int(request.args.get('limit', 5)), 20)
        
        query_embedding = simplerag_instance.embedding_service.get_embedding(query)
        results = simplerag_instance.vector_db_service.search_similar(
            query_embedding,
            top_k=limit,
            collection_name=collection_name
        )
        
        return jsonify({
            "collection": collection_name,
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "score": result["score"],
                    "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                    "metadata": result["metadata"]
                } for result in results
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

# Error handlers
@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('upload'))

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Initialize services on startup
try:
    initialize_services()
    logger.info("Flask app initialized with Enhanced SimpleRAG")
except Exception as e:
    logger.error(f"Error during app initialization: {e}")

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)