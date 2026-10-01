# DocuMind - Basic Document Q&A Bot

A basic Retrieval-Augmented Generation (RAG) application that allows users
to ask natural-language questions against a local collection of documents.

The system extracts document text, creates overlapping chunks, generates
embeddings in batches, stores them in a persistent FAISS vector database,
retrieves the most relevant chunks for a question, and generates a grounded
answer using an LLM. Each response also displays the source filename and
page/section information.

---

## 1. Objective

The objective of this project is to implement a complete RAG pipeline:

1. Document ingestion
2. Text extraction
3. Text chunking
4. Batched embeddings
5. Persistent vector storage
6. Similarity-based retrieval
7. Grounded LLM answer generation
8. Source citation
9. Interactive command-line interface

---

## 2. Tech Stack

| Technology | Version | Purpose |
|---|---:|---|
| Python | 3.11+ | Main programming language |
| OpenAI | 3.20.0 | LLM answer generation |
| Sentence Transformers | 6.1.0 | Local text embeddings |
| FAISS CPU | 1.15.1 | Vector database |
| PyMuPDF | 1.28.2 | PDF text extraction |
| python-docx | 1.2.0 | DOCX text extraction |
| python-dotenv | 1.1.1 | Environment variables |
| NumPy | 2.3.3 | Numerical operations |

---

## 3. Architecture

```text
Documents
   |
   v
Document Ingestion
   |
   v
Text Extraction
   |
   v
Chunking
(900 characters / 150 overlap)
   |
   v
Batch Embeddings
(Sentence Transformers)
   |
   v
Persistent FAISS Vector Database
   |
   v
User Question
   |
   v
Query Embedding
   |
   v
Top-K Similarity Search
   |
   v
Retrieved Context
   |
   v
OpenAI LLM
   |
   v
Grounded Answer
   |
   v
Source Filename + Page/Section