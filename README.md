![Screenshot 2024-06-28 113015](https://github.com/Devanshu3214/Web-Based-QA-System/assets/97178720/96c0004a-0dc7-4682-9d2b-020ee864767a)

# Web-Based QA System

## Overview

This project is a web-based Question-Answering (QA) system that uses advanced language models and various libraries to provide accurate answers based on a set of predefined documents. The application leverages Flask for the web interface, LangChain for text processing and question answering, and Google's Generative AI for embeddings and language modeling.

## Features

- **Document Loading**: Loads and processes documents from specified URLs.
- **Text Splitting**: Splits large documents into manageable chunks.
- **Embeddings and Vector Index**: Creates embeddings and a vector index for efficient document retrieval.
- **QA Chain**: Uses a question-answering chain to provide answers based on the context of the documents.
- **Web Interface**: User-friendly web interface for interacting with the QA system.

## Development

- **Load and Process Documents**: The `load_and_process_documents` function loads documents from specified URLs and processes them into a single context.
- **Split Text**: The `split_text` function splits the context into smaller chunks for better processing.
- **Create Vector Index**: The `create_vector_index` function creates embeddings and a vector index for efficient retrieval of relevant documents.
- **Create QA Chain**: The `create_qa_chain` function sets up the question-answering chain using Google's Generative AI.
