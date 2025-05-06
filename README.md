# SimpleRAG

A Retrieval-Augmented Generation System for Document Q&A

![SimpleRAG](https://via.placeholder.com/800x200?text=SimpleRAG)

## Overview

SimpleRAG is a powerful document question-answering system that combines vector embeddings with large language models to provide accurate answers based on your documents. Upload PDFs, DOCX, TXT, or HTML files and ask questions in natural language to get relevant information extracted directly from your content.

## Features

- 📚 **Document Indexing**: Upload and process PDF, TXT, DOCX, and HTML files
- 🔍 **Semantic Search**: Find relevant document sections based on meaning, not just keywords
- 🤖 **LLM Integration**: Get high-quality answers using Claude LLM or raw document extracts
- ⚡ **Performance Optimizations**: Embedding cache and rate limiting for faster responses
- 📈 **Progress Tracking**: Real-time progress indicators for long-running operations
- 🔧 **Configurable**: Adjust chunk sizes, overlap, and other parameters to fit your needs

## Architecture

SimpleRAG employs a classic RAG (Retrieval-Augmented Generation) architecture:

1. **Document Processing**: Documents are parsed, extracted, and split into chunks with configurable overlap
2. **Vector Embeddings**: Each chunk is converted into a vector embedding using the Gemini API
3. **Vector Storage**: Embeddings are stored in Qdrant vector database for efficient similarity search
4. **Query Processing**: When you ask a question, it's converted to an embedding and used to find the most similar chunks
5. **LLM Integration**: The most relevant chunks are sent to Claude LLM along with your question to generate a comprehensive answer

## Installation

### Prerequisites

- Python 3.8+
- API keys for:
  - Gemini API (required for embeddings)
  - Claude API (optional, for LLM-generated answers)
  - Qdrant Cloud (optional, for vector database)

### Quick Install (macOS/Linux)

```bash
# Clone the repository
git clone https://github.com/yourusername/simplerag.git
cd simplerag

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
python simplerag.py config --gemini-key YOUR_GEMINI_API_KEY --claude-key YOUR_CLAUDE_API_KEY
```

## Usage

### Web Interface

SimpleRAG provides a web interface for easy document uploading and querying:

```bash
# Start the web server
python app.py
```

Then navigate to http://localhost:5001 in your browser.

### Command Line

```bash
# Configure API keys
simplerag config --gemini-key YOUR_GEMINI_API_KEY --claude-key YOUR_CLAUDE_API_KEY

# Index a document
simplerag index /path/to/document.pdf

# Ask a question
simplerag query "What is the main theme of the document?"
```

## Configuration

Configure SimpleRAG through the web interface in the Settings tab or via command line:

```bash
simplerag config --chunk-size 1000 --chunk-overlap 200 --top-k 5 --preferred-llm claude
```

### Key Parameters

- `chunk_size`: Size of text chunks in characters (500-5000 recommended)
- `chunk_overlap`: Overlap between adjacent chunks (50-500 recommended)
- `top_k`: Number of results to retrieve for each query (1-20)
- `preferred_llm`: Choose between `claude` or `raw` (document extracts only)

## Advanced Features

### Rate Limiting

SimpleRAG implements rate limiting for API calls to prevent exceeding API provider quotas and avoid service interruptions. The system automatically throttles requests if they exceed the configured limit.

### Embedding Cache

To improve performance and reduce API costs, SimpleRAG can cache embeddings locally. When the same text needs to be embedded multiple times, the system will use the cached version instead of making additional API calls.

### Progress Tracking

For long-running operations like indexing large documents, SimpleRAG provides real-time progress indicators that show the current status of the operation.

## Troubleshooting

- **No results returned**: Check that your document was properly indexed and that your questions are related to the document content
- **Slow performance**: Consider enabling the embedding cache and adjusting the chunk size
- **Indexing errors**: For large documents, try decreasing the chunk size or increasing the timeout value

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/claude) for LLM capabilities
- [Google Gemini](https://ai.google.dev/) for embedding generation
- [Qdrant](https://qdrant.tech/) for vector storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
