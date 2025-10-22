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
import uuid
# Import the modular SimpleRAG components
from simple_rag import EnhancedSimpleRAG
from config import get_config_manager
from extensions import ProgressTracker
import time
import concurrent.futures
from typing import Dict, Any
from datetime import datetime
parallel_query_results = {}

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'simplerag_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
n8n_config = {
    "monitoring": False,
    "folder_link": None,
    "folder_id": None,
    "rag_mode": "normal",
    "poll_interval": 5,
    "files_indexed": 0,
    "last_check": None,
    "recent_files": []
}
@app.template_filter('nl2br')
def nl2br_filter(text):
    """Convert newlines to HTML breaks and process markdown."""
    if not text:
        return ""
    
    # Clean any existing HTML tags first
    import re
    # Remove any stray HTML tags that shouldn't be there
    text = re.sub(r'<(?!/?(?:br|p|div|span|a|strong|em|ul|ol|li|h[1-6]|blockquote|code|pre)\b)[^>]+>', '', text)
    
    # Convert markdown to HTML
    html = markdown.markdown(text, extensions=['nl2br', 'fenced_code'])
    
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
        logger.info("Initializing Enhanced SimpleRAG services...")
        simplerag_instance = EnhancedSimpleRAG()
        
        # Remove the is_ready() check - always return True
        logger.info("✓ SimpleRAG services initialized")
        return True
        
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
        config = get_config_manager().get_all()
        if not config.get("gemini_api_key") or not config.get("qdrant_api_key"):
            flash('Please configure your API keys first.', 'danger')
            return redirect(url_for('setup'))
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

def process_parallel_queries(question: str, session_id: str) -> Dict[str, Any]:
    """Process queries in all three modes simultaneously with progress tracking."""
    global simplerag_instance
    import threading
    
    progress_tracker = ProgressTracker.get_tracker(session_id, "query")
    
    results = {
        'normal': {'answer': None, 'time': 0, 'error': None},
        'graph': {'answer': None, 'time': 0, 'error': None},
        'hybrid_neo4j': {'answer': None, 'time': 0, 'error': None}
    }
    
    if progress_tracker:
        progress_tracker.update(20, 100, status="processing", 
                               message="Starting parallel execution in 3 modes")
    
    # Create a lock for thread-safe mode switching
    mode_lock = threading.Lock()
    completed_modes = {'count': 0}
    
    def run_query_mode(mode: str) -> tuple:
        """Run query in a specific mode and measure time."""
        start_time = time.time()
        try:
            with mode_lock:
                # Save original mode
                original_mode = simplerag_instance.rag_mode
                
                # Check availability for special modes BEFORE setting mode
                if mode == 'hybrid_neo4j':
                    if not simplerag_instance.is_graph_ready():
                        elapsed_time = time.time() - start_time
                        return (mode, None, elapsed_time, "Graph RAG not available for hybrid mode")
                    if not simplerag_instance.is_neo4j_ready():
                        elapsed_time = time.time() - start_time
                        return (mode, None, elapsed_time, "Neo4j not available for hybrid mode")
                elif mode == 'graph':
                    if not simplerag_instance.is_graph_ready():
                        elapsed_time = time.time() - start_time
                        return (mode, None, elapsed_time, "Graph RAG not initialized")
                
                # Set mode temporarily
                simplerag_instance.set_rag_mode(mode)
            
            # Run query (outside lock to allow parallel execution)
            # FIXED: Just use the standard query method for all modes
            answer = simplerag_instance.query(question, f"{session_id}_{mode}")
            
            with mode_lock:
                # Restore original mode
                simplerag_instance.set_rag_mode(original_mode)
                
                # Update progress
                completed_modes['count'] += 1
                if progress_tracker:
                    progress = 20 + (completed_modes['count'] * 20)  # 20-80%
                    progress_tracker.update(progress, 100, status="processing", 
                                        message=f"Completed {mode} mode ({completed_modes['count']}/3)")
        
            elapsed_time = time.time() - start_time
            return (mode, answer, elapsed_time, None)
            
        except Exception as e:
            with mode_lock:
                # Restore original mode on error
                try:
                    simplerag_instance.set_rag_mode(original_mode)
                except:
                    pass
                    
                completed_modes['count'] += 1
                if progress_tracker:
                    progress = 20 + (completed_modes['count'] * 20)
                    progress_tracker.update(progress, 100, status="processing", 
                                        message=f"Error in {mode} mode ({completed_modes['count']}/3)")
                    
            elapsed_time = time.time() - start_time
            logger.error(f"Error in {mode} mode: {e}")
            return (mode, None, elapsed_time, str(e))
    
    # DELETE the run_hybrid_query function entirely - it's not needed
    
    # Run all modes in parallel
    threads = []
    mode_results = {}
    
    def thread_worker(mode):
        """Worker function for thread that properly stores results."""
        result = run_query_mode(mode)
        # Store the result in the shared dictionary
        mode_results[mode] = result
        return result
    
    # Create and start threads with the fixed worker function
    for mode in ['normal', 'graph', 'hybrid_neo4j']:
        thread = threading.Thread(target=thread_worker, args=(mode,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Process results
    for mode in ['normal', 'graph', 'hybrid_neo4j']:
        if mode in mode_results:
            mode_name, answer, elapsed_time, error = mode_results[mode]
            results[mode]['answer'] = answer
            results[mode]['time'] = round(elapsed_time, 2)
            results[mode]['error'] = error
        else:
            results[mode]['error'] = "Thread execution failed"
    
    # Calculate analytics
    successful_modes = [mode for mode in results if not results[mode]['error']]
    fastest_mode = min(successful_modes, key=lambda m: results[m]['time']) if successful_modes else None
    
    analytics = {
        'total_modes': 3,
        'successful_modes': len(successful_modes),
        'failed_modes': 3 - len(successful_modes),
        'fastest_mode': fastest_mode,
        'total_time': max(results[mode]['time'] for mode in results),
        'average_time': sum(results[mode]['time'] for mode in results if not results[mode]['error']) / len(successful_modes) if successful_modes else 0
    }
    
    return {
        'question': question,
        'results': results,
        'analytics': analytics,
        'timestamp': time.time()
    }
    
@app.route('/')
def home():
    """Render home page with current configuration status."""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    config_manager = get_config_manager()
    config = config_manager.get_all()
    # Change this logic
    is_configured = bool(config.get("gemini_api_key") and config.get("qdrant_api_key") and config.get("qdrant_url"))
    
    current_mode = config.get("rag_mode", "normal")  # Get from config instead
    
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
        if 'neo4j_enabled' in request.form:
            updates["neo4j_enabled"] = True
        else:
            updates["neo4j_enabled"] = False

        neo4j_fields = ["neo4j_uri", "neo4j_username", "neo4j_password"]
        for field in neo4j_fields:
            value = request.form.get(field, '').strip()
            if value != config_manager.get(field, ''):
                updates[field] = value
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

@app.route('/agentic')
def agentic_interface():
    """Agentic AI interface for autonomous query processing."""
    config_manager = get_config_manager()
    
    # Check if agentic AI is available
    agentic_available = False
    available_tools = []
    agentic_stats = {}
    
    if simplerag_instance and simplerag_instance.is_agentic_ready():
        agentic_available = True
        available_tools = simplerag_instance.agentic_service.get_available_tools()
        agentic_stats = simplerag_instance.agentic_service.get_agentic_stats()
    
    return render_template('agentic.html', 
                          config=config_manager.get_all(),
                          agentic_available=agentic_available,
                          available_tools=available_tools,
                          agentic_stats=agentic_stats)

@app.route('/agentic/query', methods=['POST'])
def agentic_query():
    """Handle agentic AI queries with autonomous tool selection."""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    if not simplerag_instance or not simplerag_instance.is_agentic_ready():
        return jsonify({
            "error": "Agentic AI not available. Please check Claude API configuration."
        }), 503
    
    try:
        session_id = session.get('session_id')
        result = simplerag_instance.query_agentic(question, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in agentic query: {e}")
        return jsonify({"error": f"Agentic query failed: {str(e)}"}), 500

@app.route('/api/agentic/tools')
def get_agentic_tools():
    """Get information about available agentic tools."""
    if not simplerag_instance or not simplerag_instance.is_agentic_ready():
        return jsonify({"error": "Agentic AI not available"}), 503
    
    try:
        tools = simplerag_instance.agentic_service.get_available_tools()
        stats = simplerag_instance.agentic_service.get_agentic_stats()
        
        return jsonify({
            "tools": tools,
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to get tools: {str(e)}"}), 500

@app.route('/api/webhook/n8n-binary-upload', methods=['POST'])
def n8n_binary_upload():
    """Webhook endpoint for n8n with binary file upload"""
    try:
        # Get filename and rag_mode from form data
        filename = request.form.get('filename', 'document.pdf')
        rag_mode = request.form.get('rag_mode', 'normal')
        
        # Get binary file data
        if 'data' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file data received'
            }), 400
        
        file_data = request.files['data']
        
        logger.info(f"n8n binary upload triggered for file: {filename}")
        
        # Create session and save file
        session_id = str(uuid.uuid4())
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{safe_filename}")
        
        # Save the file
        file_data.save(file_path)
        
        logger.info(f"File saved to: {file_path}")
        
        # Validate file
        validation = simplerag_instance.validate_file(file_path)
        if not validation['valid']:
            os.remove(file_path)
            return jsonify({
                'success': False,
                'error': 'File validation failed',
                'details': validation['errors']
            }), 400
        
        # Set RAG mode
        if rag_mode in ['normal', 'graph', 'neo4j']:
            simplerag_instance.set_rag_mode(rag_mode)
        
        # Index document
        result = simplerag_instance.index_document(file_path, safe_filename)
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'rag_mode': rag_mode,
            'message': f'Document indexed successfully in {rag_mode} mode',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"n8n binary upload error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@app.route('/api/webhook/n8n-upload', methods=['POST'])
def n8n_webhook_upload():
    """Webhook endpoint for n8n automation - auto-indexes documents from external sources"""
    try:
        data = request.json
        file_id = data.get('file_id')  # Changed from file_url
        filename = data.get('filename')
        rag_mode = data.get('rag_mode', 'normal')
        
        logger.info(f"n8n webhook triggered for file: {filename} (ID: {file_id})")
        
        # For Google Drive, we need to use the n8n downloaded file
        # This endpoint now expects n8n to have already downloaded the file
        # and sent it as base64 or we use Google Drive API
        
        # TEMPORARY FIX: Use a direct download URL format
        # Google Drive direct download format
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logger.info(f"Attempting to download from: {download_url}")
        
        # Download file from URL
        import requests
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Create session and save file
        session_id = str(uuid.uuid4())
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{safe_filename}")
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"File downloaded to: {file_path}")
        
        # Validate file
        validation = simplerag_instance.validate_file(file_path)
        if not validation['valid']:
            os.remove(file_path)
            return jsonify({
                'success': False,
                'error': 'File validation failed',
                'details': validation['errors']
            }), 400
        
        # Set RAG mode
        if rag_mode in ['normal', 'graph', 'neo4j']:
            simplerag_instance.set_rag_mode(rag_mode)
        
        # Index document
        result = simplerag_instance.index_document(file_path, safe_filename)
        
        return jsonify({
            'success': True,  # Changed to True (was 'result')
            'filename': safe_filename,
            'rag_mode': rag_mode,
            'message': f'Document indexed successfully in {rag_mode} mode',
            'session_id': session_id
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file from URL: {e}")
        return jsonify({'success': False, 'error': f'Failed to download file: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"n8n webhook error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    
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
        if upload_rag_mode not in ['normal', 'graph', 'neo4j']:
            upload_rag_mode = 'normal'
            
        if upload_rag_mode == 'neo4j' and not simplerag_instance.is_neo4j_ready():
            flash('Neo4j service not available. Please configure Neo4j settings first.', 'danger')
            return redirect(url_for('setup'))
        
        # Check if SimpleRAG is ready
        config = get_config_manager().get_all()
        if not config.get("gemini_api_key") or not config.get("qdrant_api_key"):
            flash('Please configure your API keys first.', 'danger')
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
    """Handle document querying with RAG mode selection including parallel mode."""
    config_manager = get_config_manager()
    answer = None
    question = None
    in_progress = False
    parallel_results = None
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        query_rag_mode = request.form.get('rag_mode', None)
        
        if not question:
            flash('Please enter a question', 'warning')
            return render_template('query.html', config=config_manager.get_all())
        
        config = get_config_manager().get_all()
        if not config.get("gemini_api_key") or not config.get("qdrant_api_key"):
            flash('Please configure your API keys first.', 'danger')
            return redirect(url_for('setup'))
        
        session_id = session.get('session_id')
        
        
        # Handle parallel mode
        if query_rag_mode == 'parallel':
            try:
                logger.info("Starting parallel query processing")
                
                # Initialize progress tracker for parallel mode
                progress_tracker = ProgressTracker(session_id, "query")
                progress_tracker.update(0, 100, status="starting", 
                                    message="Initializing parallel processing for all RAG modes")
                
                # Start async parallel processing
                parallel_thread = threading.Thread(
                    target=process_parallel_queries_async,
                    args=(question, session_id),
                    daemon=True
                )
                parallel_thread.start()
                
                # Return template with progress tracking enabled
                return render_template('query.html', 
                                    question=question, 
                                    in_progress=True,  # Enable progress tracking
                                    parallel_mode=True,  # Flag for parallel mode
                                    config=config_manager.get_all())
                                    
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                flash(f"Error processing parallel query: {e}", 'danger')
                return render_template('query.html', 
                                    question=question,
                                    config=config_manager.get_all())
                
        # Original single-mode processing logic
        if query_rag_mode and query_rag_mode in ['normal', 'graph', 'hybrid_neo4j']:
            current_mode = simplerag_instance.rag_mode
            
            # Special handling for hybrid_neo4j mode
            if query_rag_mode == 'hybrid_neo4j':
                if not simplerag_instance.is_graph_ready():
                    flash('Graph RAG not available for hybrid mode', 'warning')
                    query_rag_mode = 'normal'
                elif not simplerag_instance.is_neo4j_ready():
                    flash('Neo4j not available, using Graph RAG only', 'warning')
                    query_rag_mode = 'graph'
                else:
                    # Temporarily set to hybrid mode for this query
                    simplerag_instance.rag_mode = 'hybrid_neo4j'
            
            if query_rag_mode != current_mode and query_rag_mode != 'hybrid_neo4j':
                try:
                    simplerag_instance.set_rag_mode(query_rag_mode)
                    logger.info(f"RAG mode temporarily changed to {query_rag_mode} for query")
                except Exception as e:
                    logger.error(f"Error changing RAG mode: {e}")
                    flash(f"Error changing RAG mode: {e}", 'warning')
        
        progress_tracker = ProgressTracker(session_id, "query")
        progress_tracker.update(0, 100, status="starting", 
                               message="Processing your question")
        
        # Determine async processing
        current_mode = simplerag_instance.rag_mode
        use_async = True
        
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
                          parallel_results=parallel_results,
                          config=config_manager.get_all())

def process_parallel_queries_async(question: str, session_id: str):
    """Async wrapper for parallel queries with progress tracking."""
    global parallel_query_results
    try:
        progress_tracker = ProgressTracker.get_tracker(session_id, "query")
        if progress_tracker:
            progress_tracker.update(10, 100, status="processing", 
                                   message="Running queries in parallel across all modes")
        
        # Run the actual parallel processing
        parallel_results = process_parallel_queries(question, session_id)
        
        if progress_tracker:
            progress_tracker.update(90, 100, status="finalizing", 
                                   message="Compiling results from all modes")
        
        # Store results in global dictionary
        parallel_query_results[session_id] = parallel_results
        
        if progress_tracker:
            progress_tracker.update(100, 100, status="complete", 
                                   message="Parallel processing complete")
        
        logger.info(f"Parallel query completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in async parallel processing: {e}")
        progress_tracker = ProgressTracker.get_tracker(session_id, "query")
        if progress_tracker:
            progress_tracker.update(100, 100, status="error", 
                                   message=f"Parallel processing error: {str(e)}")
        # Store error in results
        parallel_query_results[session_id] = {
            'error': str(e),
            'results': {
                'normal': {'answer': None, 'time': 0, 'error': str(e)},
                'graph': {'answer': None, 'time': 0, 'error': str(e)},
                'hybrid_neo4j': {'answer': None, 'time': 0, 'error': str(e)}
            }
        }

@app.route('/api/query/parallel-result')
def get_parallel_query_result():
    """API endpoint to get the result of parallel query processing."""
    global parallel_query_results
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found"}), 404
    
    result = parallel_query_results.get(session_id)
    
    if result:
        # Remove from storage after retrieval
        del parallel_query_results[session_id]
        return jsonify({"result": result})
    
    return jsonify({"error": "No result available"}), 404

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



#n8n endpoints

@app.route('/n8n-setup')
def n8n_setup():
    """n8n automation configuration page"""
    return render_template('n8n_config.html')

@app.route('/api/n8n/configure', methods=['POST'])
def configure_n8n():
    """Configure n8n Google Drive monitoring"""
    try:
        data = request.json
        folder_link = data.get('folder_link')
        rag_mode = data.get('rag_mode', 'normal')
        poll_interval = int(data.get('poll_interval', 5))
        
        # Extract folder ID from Google Drive link
        # Format: https://drive.google.com/drive/folders/FOLDER_ID
        if '/folders/' in folder_link:
            folder_id = folder_link.split('/folders/')[-1].split('?')[0]
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid Google Drive folder link. Please use the shareable link format.'
            }), 400
        
        # Update configuration
        n8n_config.update({
            "monitoring": True,
            "folder_link": folder_link,
            "folder_id": folder_id,
            "rag_mode": rag_mode,
            "poll_interval": poll_interval
        })
        
        logger.info(f"n8n configured: Folder {folder_id}, Mode {rag_mode}, Interval {poll_interval}min")
        
        return jsonify({
            'success': True,
            'folder_id': folder_id,
            'folder_name': f'Drive folder ({folder_id[:8]}...)',
            'message': 'n8n monitoring configured successfully'
        })
        
    except Exception as e:
        logger.error(f"Error configuring n8n: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/n8n/status')
def n8n_status():
    """Get current n8n monitoring status"""
    return jsonify(n8n_config)

@app.route('/api/n8n/stop', methods=['POST'])
def stop_n8n():
    """Stop n8n monitoring"""
    n8n_config["monitoring"] = False
    logger.info("n8n monitoring stopped")
    return jsonify({'success': True, 'message': 'Monitoring stopped'})

@app.route('/api/n8n/file-indexed', methods=['POST'])
def n8n_file_indexed():
    """Called by n8n when a file is successfully indexed"""
    try:
        data = request.json
        filename = data.get('filename')
        
        # Update stats
        n8n_config["files_indexed"] += 1
        n8n_config["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n8n_config["recent_files"].insert(0, {
            'name': filename,
            'time': datetime.now().strftime("%H:%M:%S")
        })
        
        # Keep only last 10 files
        n8n_config["recent_files"] = n8n_config["recent_files"][:10]
        
        logger.info(f"n8n indexed file: {filename}")
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error updating n8n stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500




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
        # Try to reconnect if not connected
        if not simplerag_instance.vector_db_service.is_connected:
            simplerag_instance.vector_db_service.retry_connection()
        
        status = simplerag_instance.vector_db_service.get_status()
        return jsonify({
            "connected": status.get('connected', False),
            "url": status.get('url', 'Unknown'),
            "collection_count": status.get('total_collections', 0),
            "error": status.get('last_error'),
            "retry_available": True
        })
    except Exception as e:
        return jsonify({
            "connected": False,
            "error": f"Status check failed: {str(e)}",
            "retry_available": True
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
@app.route('/n8n-test')
def n8n_test():
    """Test page for n8n integration"""
    return render_template('n8n_test.html')
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


# Add these routes to app.py before the error handlers section
