# Enhanced SimpleRAG

A Dual-Mode Retrieval-Augmented Generation System with Normal and Graph RAG Capabilities

![Enhanced SimpleRAG](https://via.placeholder.com/800x200?text=Enhanced%20SimpleRAG)

## Overview

Enhanced SimpleRAG is an advanced document question-answering system that offers two powerful RAG modes: traditional semantic search and cutting-edge knowledge graph reasoning. Upload PDFs, DOCX, TXT, or HTML files and ask questions in natural language to get accurate, contextually-aware answers extracted directly from your content.

## ✨ What's New in Enhanced SimpleRAG

### 🕸️ Graph RAG Mode
- **Entity Extraction**: Automatically identifies people, organizations, concepts, locations, and events
- **Relationship Mapping**: Discovers and maps connections between entities
- **Knowledge Graph**: Builds a NetworkX graph for advanced reasoning
- **Hybrid Search**: Combines semantic search with graph traversal for richer context

### 📚 Normal RAG Mode
- **Fast Processing**: Traditional chunking and embedding for quick results
- **Semantic Search**: Vector similarity search for relevant document sections
- **Efficient Storage**: Optimized for speed and straightforward Q&A

## Features

### Core Capabilities
- 📄 **Multi-Format Support**: Process PDF, TXT, DOCX, and HTML files
- 🔄 **Dual RAG Modes**: Switch between Normal and Graph RAG based on your needs
- 🤖 **LLM Integration**: High-quality answers using Claude LLM
- 🗄️ **Advanced Storage**: Dual Qdrant collections for documents and graph elements
- ⚡ **Performance Optimizations**: Embedding cache, rate limiting, and progress tracking
- 🔧 **Admin Interface**: Comprehensive collection management and debugging tools

### Enhanced Features
- 🧠 **Knowledge Graph Reasoning**: Understand relationships and connections
- 🔍 **Hybrid Search**: Combine document content with entity relationships
- 📊 **Real-time Progress**: Detailed progress tracking for complex operations
- 🛠️ **Advanced Configuration**: Fine-tune entity extraction and graph parameters
- 📈 **System Monitoring**: Health checks and status monitoring

## Architecture

Enhanced SimpleRAG employs a sophisticated dual-mode architecture:

### Normal RAG Flow
1. **Document Processing**: Parse and extract text from uploaded files
2. **Text Chunking**: Split documents into overlapping chunks
3. **Vector Embeddings**: Generate embeddings using Gemini API
4. **Vector Storage**: Store in primary Qdrant collection
5. **Semantic Search**: Find similar chunks using vector similarity
6. **Answer Generation**: Use Claude LLM with retrieved context

### Graph RAG Flow
1. **Document Processing**: Parse and extract text from uploaded files
2. **Context-Rich Chunking**: Create larger chunks for better entity context
3. **Entity Extraction**: Use Gemini to identify entities and relationships
4. **Knowledge Graph**: Build NetworkX graph with entities as nodes, relationships as edges
5. **Graph Embeddings**: Generate embeddings for entities and relationships
6. **Dual Storage**: Store documents in primary collection, graph elements in graph collection
7. **Hybrid Search**: Search both document chunks and graph elements
8. **Enhanced Answer Generation**: Use Claude with both document and graph context

## Installation

### Prerequisites

- Python 3.8+
- API keys for:
  - **Gemini API** (required for embeddings and entity extraction)
  - **Claude API** (optional, for LLM-generated answers)
  - **Qdrant Cloud** (required for vector database)

### Quick Install (macOS/Linux)

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-simplerag.git
cd enhanced-simplerag

# Run installation script
./install.sh
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your API keys
python simplerag.py config --gemini-key YOUR_GEMINI_API_KEY --claude-key YOUR_CLAUDE_API_KEY --qdrant-key YOUR_QDRANT_KEY --qdrant-url YOUR_QDRANT_URL
```

## Usage

### Web Interface

Enhanced SimpleRAG provides an intuitive web interface with admin capabilities:

```bash
# Start the web server
python app.py
```

Navigate to http://localhost:5001 and access:
- **Home**: Overview and mode selection
- **Upload**: Document indexing with RAG mode selection
- **Query**: Question answering with mode switching
- **Settings**: Configuration and RAG mode preferences
- **Advanced**: System information and performance details
- **Admin**: Collection management and debugging tools

### Command Line

```bash
# Configure API keys and RAG mode
simplerag config --gemini-key YOUR_GEMINI_API_KEY --rag-mode graph

# Index a document in Graph RAG mode
simplerag index /path/to/document.pdf --mode graph

# Query using Graph RAG
simplerag query "How are Company A and Company B related?" --mode graph

# Switch to Normal RAG for faster queries
simplerag query "What is the revenue?" --mode normal
```

## Configuration

### Basic Configuration

Configure through the web interface Settings tab or via command line:

```bash
# Set RAG mode and basic parameters
simplerag config --rag-mode graph --chunk-size 1000 --top-k 5

# Configure Graph RAG specific settings
simplerag config --max-entities-per-chunk 20 --graph-reasoning-depth 2
```

### Key Parameters

#### Normal RAG Settings
- `chunk_size`: Size of text chunks in characters (500-5000)
- `chunk_overlap`: Overlap between chunks (50-500)
- `top_k`: Number of results to retrieve (1-20)

#### Graph RAG Settings
- `max_entities_per_chunk`: Maximum entities to extract per chunk (5-50)
- `graph_reasoning_depth`: Relationship hops to consider (1-5)
- `entity_similarity_threshold`: Threshold for merging entities (0.5-1.0)

#### Performance Settings
- `rate_limit`: API calls per minute (10-300)
- `enable_cache`: Enable embedding caching (true/false)
- `preferred_llm`: Choose between `claude` or `raw`

## RAG Mode Comparison

| Feature | Normal RAG | Graph RAG |
|---------|------------|-----------|
| **Speed** | Fast ⚡ | Slower but thorough 🧠 |
| **Processing** | Simple chunking | Entity extraction + graph building |
| **Storage** | Single collection | Dual collections (docs + graph) |
| **Best For** | Direct facts, simple Q&A | Relationships, complex reasoning |
| **Use Cases** | "What is the revenue?" | "How are X and Y connected?" |

## Advanced Features

### Knowledge Graph Construction

Graph RAG automatically:
- Extracts entities (people, organizations, concepts, locations, events)
- Identifies relationships between entities
- Builds a NetworkX graph for traversal and analysis
- Stores graph elements with semantic embeddings

### Hybrid Search Strategy

When using Graph RAG, the system:
1. Searches document chunks for direct content matches
2. Searches graph elements for entity and relationship matches
3. Combines results for comprehensive context
4. Generates answers using both document and graph information

### Admin Interface

Access `/admin` for:
- **System Status**: Real-time service health monitoring
- **Collection Management**: Create, inspect, and delete Qdrant collections
- **Debug Tools**: Test queries and inspect search results
- **Performance Metrics**: Monitor API usage and cache effectiveness

### Rate Limiting & Caching

- **Smart Rate Limiting**: Prevents API quota exhaustion with automatic throttling
- **Embedding Cache**: Reduces costs by caching frequently used embeddings
- **Progress Tracking**: Real-time updates for long-running operations

## API Endpoints

Enhanced SimpleRAG provides RESTful API endpoints:

```bash
# System status
GET /api/system/status

# RAG mode management
GET /api/rag-mode
POST /api/rag-mode {"mode": "graph"}

# Collection management
GET /api/admin/qdrant/collections
POST /api/admin/qdrant/collections {"type": "graph"}
DELETE /api/admin/qdrant/collections/{name}

# Progress tracking
GET /api/progress/{operation_type}
```

## Use Cases

### Normal RAG - Perfect For:
- ✅ Quick factual queries
- ✅ Direct information extraction
- ✅ Simple document search
- ✅ Performance-critical applications

### Graph RAG - Ideal For:
- 🧠 Understanding relationships between entities
- 🔗 "How are X and Y connected?" questions
- 📊 Complex multi-entity queries
- 🎯 Contextual reasoning and inference

## Example Queries

```bash
# Normal RAG queries
"What is TechCorp's revenue?"
"Who is the CEO?"
"What products does the company offer?"

# Graph RAG queries  
"How are TechCorp and Microsoft related?"
"What partnerships exist between the mentioned companies?"
"Who are the key people involved in the acquisitions?"
"What is the relationship between Sarah Chen and the board members?"
```

## Troubleshooting

### Common Issues

**No results in Graph RAG mode:**
- Check that both document and graph collections exist in admin panel
- Verify entity extraction is working in the debug console
- Try simpler queries first to test connectivity

**Slow Graph RAG performance:**
- Reduce `max_entities_per_chunk` for faster processing
- Enable embedding cache to reduce API calls
- Consider using Normal RAG for simple queries

**Collection errors:**
- Use the admin interface to recreate collections
- Check Qdrant connection status
- Verify API keys are correctly configured

### Debug Tools

Access the admin panel (`/admin`) for:
- Collection inspection and management
- Query debugging with result scoring
- System status and error monitoring
- Cache management and clearing

## Performance Tips

1. **Choose the Right Mode**: Use Normal RAG for simple queries, Graph RAG for relationship questions
2. **Enable Caching**: Significantly reduces API costs and improves speed
3. **Optimize Chunk Size**: Larger chunks for Graph RAG, smaller for Normal RAG
4. **Monitor Progress**: Use the real-time progress indicators for long operations
5. **Use Admin Tools**: Regular collection maintenance improves performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/claude) for advanced language modeling
- [Google Gemini](https://ai.google.dev/) for embedding generation and entity extraction
- [Qdrant](https://qdrant.tech/) for high-performance vector storage
- [NetworkX](https://networkx.org/) for knowledge graph construction

## Contributing

Contributions are welcome! Areas of particular interest:
- Additional entity types and relationship patterns
- Performance optimizations for large knowledge graphs
- Integration with other vector databases
- Enhanced visualization of knowledge graphs

Please feel free to submit a Pull Request or open an issue for discussion.
- [Google Gemini](https://ai.google.dev/) for embedding generation
- [Qdrant](https://qdrant.tech/) for vector storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
