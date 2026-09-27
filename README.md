# AI Chatbot with RAG

A chatbot that answers questions using Retrieval-Augmented Generation (RAG) — instead of relying only on the language model's training data, it retrieves relevant information from your own documents and uses that as context to generate accurate, grounded answers.

## Features

- Retrieves relevant context from a document knowledge base before answering
- Reduces hallucination by grounding responses in real source material
- Simple chat interface to ask questions

## Tech Stack## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS (vector store)
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- PyPDF (for loading PDF documents)

## How It Works

1. User asks a question
2. The query is converted into an embedding and matched against stored document chunks
3. The most relevant chunks are retrieved and added as context
4. The language model generates an answer using that context

## Installation

\`\`\`bash
git clone https://github.com/aaryampatil456-sys/ai-chatbot-rag.git
cd ai-chatbot-rag
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
[app.py run command — e.g. streamlit run app.py]
\`\`\`

## Project Structure

\`\`\`
├── app.py           # Main application file
├── rag_utils.py      # RAG pipeline helper functions
└── requirements.txt  # Python dependencies
\`\`\`
