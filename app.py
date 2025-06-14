from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import os
import tempfile
import logging
import json
import threading
import re
import markdown
from werkzeug.utils import secure_filename

# Import enhanced SimpleRAG - Updated to use the new class name
from simplerag import EnhancedSimpleRAG
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

# Configuration path
CONFIG_PATH = os.path.join(tempfile.gettempdir(), 'simplerag_config.json')

# Enhanced default configuration with Graph RAG support
DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "claude_api_key": "",
    "qdrant_url": "https://3cbcacc0-1fe5-42a1-8be0-81515a21771b.us-west-2-0.aws.cloud.qdrant.io",
    "qdrant_api_key": "",
    "collection_name": "simple_rag_docs",
    "graph_collection_name": "simple_rag_graph",
    "embedding_dimension": 768,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "preferred_llm": "claude",
    "rag_mode": "normal",  # "normal" or "graph"
    "rate_limit": 60,
    "enable_cache": True,
    "cache_dir": None,
    # Graph RAG specific settings
    "max_entities_per_chunk": 20,
    "relationship_extraction_prompt": "extract_relationships",
    "graph_reasoning_depth": 2,
    "entity_similarity_threshold": 0.8
}

# Global SimpleRAG instance
simplerag_instance = None

def load_config():
    """Load configuration from file or use default."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            config = DEFAULT_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()
        # Create config file with defaults
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            logger.warning(f"Could not save default config: {e}")
    
    # Ensure all new config options are present
    config_updated = False
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
            config_updated = True
    
    # Save updated config if new options were added
    if config_updated:
        try:
            save_config(config)
        except Exception as e:
            logger.warning(f"Could not save updated config: {e}")
    
    return config

def save_config(config):
    """Save configuration to file."""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved successfully")
    except IOError as e:
        logger.error(f"Error saving configuration: {e}")
        raise

def initialize_services():
    """Initialize SimpleRAG services based on configuration."""
    global simplerag_instance
    
    try:
        # Initialize Enhanced SimpleRAG
        simplerag_instance = EnhancedSimpleRAG()
        
        # Check if services were properly initialized
        if simplerag_instance.vector_db_service is None:
            logger.warning("Vector DB service not initialized - check Qdrant configuration")
            return False
        
        logger.info("SimpleRAG services initialized successfully")
        
        # Set RAG mode from config
        config = load_config()
        rag_mode = config.get("rag_mode", "normal")
        simplerag_instance.set_rag_mode(rag_mode)
        logger.info(f"RAG mode set to: {rag_mode}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing SimpleRAG services: {e}")
        simplerag_instance = None
        return False
def index_document_background(file_path, session_id=None):
    """Index document in the background with comprehensive error handling."""
    global simplerag_instance
    
    try:
        if simplerag_instance:
            logger.info(f"Starting background indexing for: {file_path}")
            success = simplerag_instance.index_document(file_path, session_id)
            
            if success:
                logger.info(f"Successfully indexed: {file_path}")
            else:
                logger.warning(f"Indexing returned False for: {file_path}")
                
        else:
            logger.error("SimpleRAG instance not available for indexing")
            if session_id:
                tracker = ProgressTracker.get_tracker(session_id, "index_document")
                if tracker:
                    tracker.update(100, 100, status="error", 
                                 message="SimpleRAG services not initialized")
                    
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
        if not simplerag_instance:
            result = "SimpleRAG services not initialized. Please configure API keys."
        else:
            logger.info(f"Processing background query: {question[:50]}...")
            result = simplerag_instance.query(question, session_id)
            logger.info("Background query processing completed")
        
        # Store result in session-based storage
        if session_id:
            # Store result for retrieval
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
            # Store error result
            session_results = getattr(process_query_background, 'results', {})
            session_results[session_id] = error_message
            process_query_background.results = session_results
            
            # Update progress tracker with error
            progress_tracker = ProgressTracker.get_tracker(session_id, "query")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Query error: {str(e)}")

def process_query_sync(question, session_id=None):
    """Process query synchronously for quick responses."""
    global simplerag_instance
    
    if not simplerag_instance:
        return "SimpleRAG services not initialized. Please configure API keys."
    
    try:
        logger.info(f"Processing sync query: {question[:50]}...")
        
        # Create progress tracker for sync queries too
        if session_id:
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting", 
                                   message="Processing your question")
        
        result = simplerag_instance.query(question, session_id)
        logger.info("Sync query processing completed")
        return result
        
    except Exception as e:
        logger.error(f"Sync query error: {str(e)}")
        return f"Error processing your query: {str(e)}"

@app.route('/')
def home():
    """Render home page with current configuration status."""
    # Generate session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    config = load_config()
    is_configured = bool(config.get("gemini_api_key"))
    
    # Get current RAG mode status
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
        "rag_mode": simplerag_instance.rag_mode if simplerag_instance else "Unknown"
    }
    return jsonify(status), 200

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Configure API keys and RAG mode settings."""
    if request.method == 'POST':
        config = load_config()
        config_changed = False
        
        # Update API keys
        for key in ["gemini_api_key", "claude_api_key", "qdrant_api_key", "qdrant_url"]:
            new_value = request.form.get(key, "").strip()
            if new_value != config.get(key, ""):
                config[key] = new_value
                config_changed = True
        
        # Update LLM and RAG mode settings
        for key in ["preferred_llm", "rag_mode"]:
            new_value = request.form.get(key, "").strip()
            if new_value and new_value != config.get(key, ""):
                config[key] = new_value
                config_changed = True
        
        # Update advanced settings with validation
        try:
            numeric_settings = {
                "chunk_size": (100, 5000, 1000),
                "chunk_overlap": (0, 1000, 200),
                "top_k": (1, 20, 5),
                "rate_limit": (10, 300, 60),
                "max_entities_per_chunk": (5, 50, 20),
                "graph_reasoning_depth": (1, 5, 2)
            }
            
            for key, (min_val, max_val, default_val) in numeric_settings.items():
                try:
                    new_value = int(request.form.get(key, default_val))
                    new_value = max(min_val, min(max_val, new_value))  # Clamp to range
                    if new_value != config.get(key, default_val):
                        config[key] = new_value
                        config_changed = True
                except (ValueError, TypeError):
                    flash(f"Invalid value for {key}, using default", "warning")
            
            # Handle float settings
            try:
                threshold = float(request.form.get("entity_similarity_threshold", 0.8))
                threshold = max(0.5, min(1.0, threshold))  # Clamp to 0.5-1.0
                if threshold != config.get("entity_similarity_threshold", 0.8):
                    config["entity_similarity_threshold"] = threshold
                    config_changed = True
            except (ValueError, TypeError):
                flash("Invalid entity similarity threshold, using default", "warning")
            
            # Handle boolean settings
            enable_cache = bool(request.form.get("enable_cache"))
            if enable_cache != config.get("enable_cache", True):
                config["enable_cache"] = enable_cache
                config_changed = True
                
        except Exception as e:
            logger.error(f"Error updating advanced settings: {e}")
            flash("Error updating some advanced settings", "warning")
        
        # Save configuration if changed
        if config_changed:
            try:
                save_config(config)
                
                # Reinitialize services with new config
                initialize_services()
                
                flash("Configuration saved and services reinitialized successfully!", "success")
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                flash("Error saving configuration", "danger")
        else:
            flash("No changes detected in configuration", "info")
        
        return redirect(url_for('home'))
    
    return render_template('setup.html', config=load_config())

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document uploads with RAG mode selection."""
    if request.method == 'POST':
        # Validate file upload
        if 'document' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['document']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        # Validate file type
        allowed_extensions = {'pdf', 'txt', 'docx', 'html', 'htm'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            flash('File type not supported. Please upload PDF, TXT, DOCX, or HTML files.', 'danger')
            return redirect(request.url)
        
        # Get RAG mode for this upload
        upload_rag_mode = request.form.get('rag_mode', 'normal')
        if upload_rag_mode not in ['normal', 'graph']:
            upload_rag_mode = 'normal'
        
        # Check if SimpleRAG is initialized
        if not simplerag_instance:
            flash('SimpleRAG not initialized. Please configure API keys first.', 'danger')
            return redirect(url_for('setup'))
        
        # Update RAG mode if different from current
        current_mode = simplerag_instance.rag_mode
        if upload_rag_mode != current_mode:
            try:
                simplerag_instance.set_rag_mode(upload_rag_mode)
                config = load_config()
                config["rag_mode"] = upload_rag_mode
                save_config(config)
                logger.info(f"RAG mode changed from {current_mode} to {upload_rag_mode}")
            except Exception as e:
                logger.error(f"Error changing RAG mode: {e}")
                flash(f"Error changing RAG mode: {e}", 'warning')
        
        # Save the file
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session.get('session_id', 'unknown')}_{filename}")
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            flash(f"Error saving file: {e}", 'danger')
            return redirect(request.url)
        
        # Start indexing in background
        session_id = session.get('session_id')
        
        # Create progress tracker
        progress_tracker = ProgressTracker(session_id, "index_document")
        progress_tracker.update(0, 100, status="starting", 
                               message=f"Starting to index {filename} in {upload_rag_mode} mode")
        
        # Start background indexing thread
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
    
    config = load_config()
    return render_template('upload.html', config=config)

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
    answer = None
    question = None
    in_progress = False
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        query_rag_mode = request.form.get('rag_mode', None)
        
        if not question:
            flash('Please enter a question', 'warning')
            config = load_config()
            return render_template('query.html', config=config)
        
        if not simplerag_instance:
            flash('SimpleRAG not initialized. Please configure API keys first.', 'danger')
            return redirect(url_for('setup'))
        
        # Switch RAG mode if specified for this query
        if query_rag_mode and query_rag_mode in ['normal', 'graph']:
            current_mode = simplerag_instance.rag_mode
            if query_rag_mode != current_mode:
                try:
                    simplerag_instance.set_rag_mode(query_rag_mode)
                    logger.info(f"RAG mode temporarily changed to {query_rag_mode} for this query")
                except Exception as e:
                    logger.error(f"Error changing RAG mode: {e}")
                    flash(f"Error changing RAG mode: {e}", 'warning')
        
        session_id = session.get('session_id')
        
        # Create progress tracker
        progress_tracker = ProgressTracker(session_id, "query")
        progress_tracker.update(0, 100, status="starting", 
                               message="Processing your question")
        
        # Determine if we should use async processing
        current_mode = simplerag_instance.rag_mode if simplerag_instance else "normal"
        use_async = (
            len(question) > 100 or 
            request.form.get('async', 'false') == 'true' or 
            current_mode == 'graph'
        )
        
        if use_async:
            # Process query asynchronously
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
            # Process query synchronously for quick responses
            try:
                answer = process_query_sync(question, session_id)
                logger.info("Completed sync query processing")
            except Exception as e:
                logger.error(f"Error in sync query processing: {e}")
                answer = f"Error processing your query: {str(e)}"
    
    config = load_config()
    return render_template('query.html', 
                          question=question, 
                          answer=answer, 
                          in_progress=in_progress,
                          config=config)

@app.route('/api/query/result')
def get_query_result():
    """API endpoint to get the result of an asynchronous query."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found"}), 404
    
    # Get result from background processing
    session_results = getattr(process_query_background, 'results', {})
    result = session_results.get(session_id)
    
    if result:
        # Clear the result after retrieval
        del session_results[session_id]
        return jsonify({"result": result})
    
    return jsonify({"error": "No result available"}), 404

@app.route('/advanced')
def advanced():
    """Advanced settings and system information."""
    config = load_config()
    
    # Add system status information
    system_status = {
        "simplerag_initialized": simplerag_instance is not None,
        "current_rag_mode": simplerag_instance.rag_mode if simplerag_instance else "Unknown",
        "config_file_exists": os.path.exists(CONFIG_PATH),
        "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER'])
    }
    
    return render_template('advanced.html', config=config, system_status=system_status)

@app.route('/api/rag-mode', methods=['GET', 'POST'])
def rag_mode_api():
    """API endpoint to get or set RAG mode."""
    if request.method == 'POST':
        try:
            data = request.get_json()
            mode = data.get('mode')
            
            if mode not in ['normal', 'graph']:
                return jsonify({"error": "Invalid RAG mode. Must be 'normal' or 'graph'"}), 400
            
            if not simplerag_instance:
                return jsonify({"error": "SimpleRAG not initialized"}), 500
            
            # Update SimpleRAG instance
            old_mode = simplerag_instance.rag_mode
            simplerag_instance.set_rag_mode(mode)
            
            # Update and save config
            config = load_config()
            config["rag_mode"] = mode
            save_config(config)
            
            logger.info(f"RAG mode changed from {old_mode} to {mode} via API")
            return jsonify({"success": True, "mode": mode, "previous_mode": old_mode})
            
        except Exception as e:
            logger.error(f"Error changing RAG mode via API: {e}")
            return jsonify({"error": f"Error changing RAG mode: {str(e)}"}), 500
    else:
        # GET request - return current mode
        if simplerag_instance:
            current_mode = simplerag_instance.rag_mode
        else:
            config = load_config()
            current_mode = config.get("rag_mode", "normal")
        
        return jsonify({
            "mode": current_mode,
            "simplerag_initialized": simplerag_instance is not None
        })

# Replace the system status route in app.py with this improved version

@app.route('/api/system/status')
def system_status():
    """Enhanced API endpoint for system status information."""
    config = load_config()
    
    # Get SimpleRAG status
    simplerag_status = {
        "initialized": simplerag_instance is not None,
        "ready": False,
        "errors": []
    }
    
    if simplerag_instance:
        if hasattr(simplerag_instance, 'get_status'):
            simplerag_status.update(simplerag_instance.get_status())
        else:
            # Fallback status check
            simplerag_status["ready"] = (
                hasattr(simplerag_instance, 'vector_db_service') and 
                simplerag_instance.vector_db_service is not None
            )
    
    # Check Qdrant connection specifically
    qdrant_status = {
        "configured": bool(config.get("qdrant_url")) and bool(config.get("qdrant_api_key")),
        "connected": False,
        "error": None
    }
    
    if simplerag_instance and simplerag_instance.vector_db_service:
        try:
            collections = simplerag_instance.vector_db_service.client.get_collections()
            qdrant_status["connected"] = True
            qdrant_status["collection_count"] = len(collections.collections)
        except Exception as e:
            qdrant_status["error"] = str(e)
    
    status = {
        "simplerag_initialized": simplerag_status["initialized"],
        "simplerag_ready": simplerag_status["ready"],
        "current_rag_mode": simplerag_instance.rag_mode if simplerag_instance else "Unknown",
        "api_keys_configured": {
            "gemini": bool(config.get("gemini_api_key")),
            "claude": bool(config.get("claude_api_key")),
            "qdrant": bool(config.get("qdrant_api_key"))
        },
        "qdrant": qdrant_status,
        "services": simplerag_status.get("services", {}),
        "initialization_errors": simplerag_status.get("errors", []),
        "config": {
            "rag_mode": config.get("rag_mode", "normal"),
            "preferred_llm": config.get("preferred_llm", "claude"),
            "chunk_size": config.get("chunk_size", 1000),
            "top_k": config.get("top_k", 5),
            "qdrant_url": config.get("qdrant_url", "Not configured")
        },
        "collections": {
            "normal": config.get("collection_name", "simple_rag_docs"),
            "graph": config.get("graph_collection_name", "simple_rag_graph")
        }
    }
    
    return jsonify(status)

@app.route('/admin/qdrant')
def qdrant_admin():
    """Serve the Qdrant admin interface."""
    # You can either render the admin.html template or serve the standalone HTML
    # For now, let's serve the standalone version
    from flask import send_from_directory, make_response
    import os
    
    # Create the admin HTML content directly
    admin_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qdrant Admin - Collection Management</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-success { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .collection-card { border-left: 4px solid; margin-bottom: 1rem; }
        .collection-normal { border-left-color: #007bff; }
        .collection-graph { border-left-color: #28a745; }
        .collection-other { border-left-color: #6c757d; }
    </style>
</head>
<body>
    <!-- Insert the full HTML content from the qdrant_admin_ui artifact here -->
""" + open('path/to/qdrant_admin_ui.html', 'r').read().split('<body>')[1] if os.path.exists('path/to/qdrant_admin_ui.html') else ""
    
    # For simplicity, let's just return the template
    return render_template('qdrant_admin.html')
@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('upload'))

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

# Initialize services on startup
try:
    initialize_services()
    logger.info("Flask app initialized with SimpleRAG services")
except Exception as e:
    logger.error(f"Error during app initialization: {e}")

if __name__ == '__main__':
    # Development server settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)

# Add these routes to your app.py file
# Add these corrected routes to your app.py file, replacing the existing admin routes

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
    try:
        # Check if SimpleRAG is initialized
        if not simplerag_instance:
            return jsonify({
                "connected": False,
                "error": "SimpleRAG not initialized"
            })
        
        # Check if vector DB service exists
        if not simplerag_instance.vector_db_service:
            return jsonify({
                "connected": False,
                "error": "Vector DB service not available"
            })
        
        # Test connection by getting collections
        collections_response = simplerag_instance.vector_db_service.client.get_collections()
        
        return jsonify({
            "connected": True,
            "url": simplerag_instance.vector_db_service.qdrant_url,
            "collection_count": len(collections_response.collections)
        })
        
    except Exception as e:
        logger.error(f"Qdrant status check failed: {str(e)}")
        return jsonify({
            "connected": False,
            "error": f"Connection failed: {str(e)}"
        })

@app.route('/api/admin/qdrant/collections')
def list_collections():
    """List all Qdrant collections with details."""
    try:
        # Check if SimpleRAG is initialized
        if not simplerag_instance:
            return jsonify({"error": "SimpleRAG not initialized"}), 500
        
        # Check if vector DB service exists
        if not simplerag_instance.vector_db_service:
            return jsonify({"error": "Vector DB service not available"}), 500
        
        # Get collections with proper error handling
        try:
            collections_response = simplerag_instance.vector_db_service.client.get_collections()
        except Exception as e:
            logger.error(f"Failed to get collections: {str(e)}")
            return jsonify({"error": f"Failed to get collections: {str(e)}"}), 500
        
        collections_info = []
        config = load_config()
        
        for collection in collections_response.collections:
            try:
                # Get collection info with better error handling
                info = simplerag_instance.vector_db_service.client.get_collection(collection.name)
                
                # Extract metrics safely
                vectors_count = 0
                indexed_vectors_count = 0
                points_count = 0
                
                # Handle different Qdrant client versions
                if hasattr(info, 'vectors_count'):
                    vectors_count = info.vectors_count or 0
                if hasattr(info, 'indexed_vectors_count'):
                    indexed_vectors_count = info.indexed_vectors_count or 0
                if hasattr(info, 'points_count'):
                    points_count = info.points_count or 0
                
                # Extract config safely
                config_info = {"distance": "cosine", "size": 768}  # defaults
                try:
                    if hasattr(info, 'config') and info.config:
                        if hasattr(info.config, 'params') and info.config.params:
                            if hasattr(info.config.params, 'vectors'):
                                vectors_config = info.config.params.vectors
                                if hasattr(vectors_config, 'distance'):
                                    distance = vectors_config.distance
                                    config_info["distance"] = distance.value if hasattr(distance, 'value') else str(distance)
                                if hasattr(vectors_config, 'size'):
                                    config_info["size"] = vectors_config.size
                except Exception as config_error:
                    logger.warning(f"Error extracting config for {collection.name}: {config_error}")
                
                collections_info.append({
                    "name": collection.name,
                    "vectors_count": vectors_count,
                    "indexed_vectors_count": indexed_vectors_count,
                    "points_count": points_count,
                    "config": config_info
                })
                
            except Exception as e:
                logger.error(f"Error getting info for collection {collection.name}: {str(e)}")
                # Add collection with error info
                collections_info.append({
                    "name": collection.name,
                    "vectors_count": 0,
                    "indexed_vectors_count": 0,
                    "points_count": 0,
                    "config": {"distance": "unknown", "size": 0},
                    "error": str(e)
                })
        
        return jsonify({
            "collections": collections_info,
            "normal_collection": config.get("collection_name", "simple_rag_docs"),
            "graph_collection": config.get("graph_collection_name", "simple_rag_graph")
        })
        
    except Exception as e:
        logger.error(f"Error in list_collections: {str(e)}")
        return jsonify({"error": f"Failed to list collections: {str(e)}"}), 500

@app.route('/api/admin/qdrant/collections', methods=['POST'])
def create_collection():
    """Create a new collection."""
    try:
        if not simplerag_instance or not simplerag_instance.vector_db_service:
            return jsonify({"error": "SimpleRAG not initialized"}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        collection_type = data.get('type', 'normal')
        
        config = load_config()
        
        if collection_type == 'normal':
            collection_name = config["collection_name"]
        elif collection_type == 'graph':
            collection_name = config["graph_collection_name"]
        else:
            return jsonify({"error": "Invalid collection type. Must be 'normal' or 'graph'"}), 400
        
        # Create collection
        try:
            created = simplerag_instance.vector_db_service.ensure_collection_exists(collection_name)
            return jsonify({
                "success": True,
                "collection_name": collection_name,
                "created": created,
                "message": f"Collection '{collection_name}' {'created' if created else 'already exists'}"
            })
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            return jsonify({"error": f"Failed to create collection: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error in create_collection: {str(e)}")
        return jsonify({"error": f"Collection creation failed: {str(e)}"}), 500

@app.route('/api/admin/qdrant/collections/<collection_name>', methods=['DELETE'])
def delete_collection(collection_name):
    """Delete a collection."""
    try:
        if not simplerag_instance or not simplerag_instance.vector_db_service:
            return jsonify({"error": "SimpleRAG not initialized"}), 500
        
        # Validate collection name
        if not collection_name or not collection_name.strip():
            return jsonify({"error": "Invalid collection name"}), 400
        
        # Delete collection
        try:
            simplerag_instance.vector_db_service.client.delete_collection(collection_name)
            return jsonify({
                "success": True,
                "message": f"Collection '{collection_name}' deleted successfully"
            })
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            return jsonify({"error": f"Failed to delete collection: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error in delete_collection: {str(e)}")
        return jsonify({"error": f"Collection deletion failed: {str(e)}"}), 500

@app.route('/api/admin/qdrant/collections/<collection_name>/inspect')
def inspect_collection(collection_name):
    """Get detailed information about a collection."""
    try:
        if not simplerag_instance or not simplerag_instance.vector_db_service:
            return jsonify({"error": "SimpleRAG not initialized"}), 500
        
        # Validate collection name
        if not collection_name or not collection_name.strip():
            return jsonify({"error": "Invalid collection name"}), 400
        
        # Get collection info
        try:
            info = simplerag_instance.vector_db_service.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {str(e)}")
            return jsonify({"error": f"Collection not found: {str(e)}"}), 404
        
        # Get sample points
        sample_points = []
        try:
            scroll_result = simplerag_instance.vector_db_service.client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            if scroll_result and len(scroll_result) > 0:
                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
                sample_points = [
                    {
                        "id": str(point.id),
                        "payload": dict(point.payload) if point.payload else {}
                    } for point in points
                ]
                
        except Exception as e:
            logger.warning(f"Could not get sample points for {collection_name}: {str(e)}")
            sample_points = []
        
        # Extract collection metrics safely
        vectors_count = getattr(info, 'vectors_count', 0) or 0
        indexed_vectors_count = getattr(info, 'indexed_vectors_count', 0) or 0
        points_count = getattr(info, 'points_count', 0) or 0
        
        # Extract config safely
        config_info = {"distance": "unknown", "size": 0}
        try:
            if hasattr(info, 'config') and info.config:
                if hasattr(info.config, 'params') and info.config.params:
                    if hasattr(info.config.params, 'vectors'):
                        vectors_config = info.config.params.vectors
                        if hasattr(vectors_config, 'distance'):
                            distance = vectors_config.distance
                            config_info["distance"] = distance.value if hasattr(distance, 'value') else str(distance)
                        if hasattr(vectors_config, 'size'):
                            config_info["size"] = vectors_config.size
        except Exception as config_error:
            logger.warning(f"Error extracting config for {collection_name}: {config_error}")
        
        return jsonify({
            "name": collection_name,
            "vectors_count": vectors_count,
            "indexed_vectors_count": indexed_vectors_count,
            "points_count": points_count,
            "config": config_info,
            "sample_points": sample_points
        })
        
    except Exception as e:
        logger.error(f"Error inspecting collection {collection_name}: {str(e)}")
        return jsonify({"error": f"Inspection failed: {str(e)}"}), 500

@app.route('/api/admin/cache/clear', methods=['POST'])
def clear_cache():
    """Clear embedding cache."""
    try:
        if simplerag_instance and hasattr(simplerag_instance, 'embedding_service'):
            if hasattr(simplerag_instance.embedding_service, 'cache'):
                cache_dir = simplerag_instance.embedding_service.cache.cache_dir
                import shutil
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    return jsonify({"success": True, "message": "Cache cleared successfully"})
                else:
                    return jsonify({"success": True, "message": "Cache directory not found"})
            else:
                return jsonify({"success": False, "message": "Cache not enabled"})
        else:
            return jsonify({"success": False, "message": "SimpleRAG not initialized"})
            
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({"error": f"Cache clear failed: {str(e)}"}), 500

@app.route('/api/admin/collections/search/<collection_name>')
def search_collection(collection_name):
    """Search in a specific collection for debugging."""
    try:
        if not simplerag_instance or not simplerag_instance.vector_db_service:
            return jsonify({"error": "SimpleRAG not initialized"}), 500
        
        # Validate collection name
        if not collection_name or not collection_name.strip():
            return jsonify({"error": "Invalid collection name"}), 400
        
        query = request.args.get('query', 'test')
        limit = min(int(request.args.get('limit', 5)), 20)  # Cap at 20
        
        # Generate query embedding
        try:
            query_embedding = simplerag_instance.embedding_service.get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {str(e)}")
            return jsonify({"error": f"Failed to generate embedding: {str(e)}"}), 500
        
        # Search collection
        try:
            results = simplerag_instance.vector_db_service.search_similar(
                query_embedding,
                top_k=limit,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"Failed to search collection {collection_name}: {str(e)}")
            return jsonify({"error": f"Search failed: {str(e)}"}), 500
        
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
        logger.error(f"Error searching collection {collection_name}: {str(e)}")
        return jsonify({"error": f"Collection search failed: {str(e)}"}), 500