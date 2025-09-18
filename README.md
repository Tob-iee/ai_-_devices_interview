# AI Document Assistant

This application acts as a personal assistant when you are working with any document. You provide the system with your document, and then you can query the system to summarize, explain, or teach you things based on the information (context) in your document.

## Architecture Diagram

![AI Document Assistant Architecture](images/AI_Document_Assistant_Architecture.png)

## Features

- **Document Processing**: Load and process PDF documents using LlamaIndex/LlamaParse
- **Semantic Search**: Query documents using natural language with HuggingFace embeddings
- **Local LLM**: Uses Meta's Llama-3.2-3B-Instruct model for responses
- **Chunking Strategy**: Intelligent document splitting with overlap for better context
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware answers

## Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- CUDA-compatible GPU (recommended for better performance)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_-_devices_interview
   ```

2. **Install dependencies using Poetry**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

## Usage

1. **Prepare your documents**
   - Place your PDF files in the `data/` directory
   - Update the file path in `src/rag.py` if needed:
     ```python
     data_file = "./data/your-document.pdf"
     ```

2. **Run the RAG script**
   ```bash
   python src/rag.py
   ```

3. **Customize queries**
   - Modify the query at the bottom of `src/rag.py`:
     ```python
     response = query_engine.query("Your question here")
     ```

## Configuration

### Model Settings
- **LLM**: `meta-llama/Llama-3.2-3B-Instruct`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Context Window**: 4096 tokens
- **Max New Tokens**: 1000

### Chunking Parameters
- **Chunk Size**: 1024 characters
- **Chunk Overlap**: 128 characters
- **Similarity Top-K**: 4 (number of relevant chunks retrieved)


