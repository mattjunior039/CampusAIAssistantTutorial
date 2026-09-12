# Campus Assistant Pipeline

This project is a three-phase NLP and retrieval system designed around a university campus information assistant. It progresses from classic lexical search, to dense semantic embeddings and intent classification, and finally to retrieval-augmented generation (RAG) with local model inference.

The repository is built as a teaching-oriented lab and demo pipeline for campus-style FAQ and policy retrieval tasks. It uses synthetic campus data and reproducible notebook-based workflows to demonstrate how a simple question-answering system evolves from brittle keyword matching into a grounded generative assistant.

## Project Overview

The overall system is organized into four main learning stages:

1. Phase 1: Lexical Search and Rule-Based Matching
   - TF-IDF vectorization
   - token sanitization and stop-word removal
   - cosine similarity matching
   - ranked retrieval and diagnostics

2. Phase 2: Dense Semantic Embeddings and Intent Classification
   - sentence embeddings with SentenceTransformers
   - intent labeling for campus requests
   - logistic regression and SVM classification
   - performance metrics and confusion matrices

3. Phase 3: Retrieval-Augmented Generation
   - PDF ingestion and parsing
   - semantic chunking and overlap
   - FAISS vector indexing
   - retrieval + grounded generation workflow

4. Appendix A: Live RAG Inference with Local Ollama
   - connection to a local Ollama server
   - live LLM generation using local open-weight models
   - strict retrieval grounding and low-temperature factual generation

## Repository Structure

- `Phase_1_Lexical_Search_and_Rule_Based_Matching.ipynb` — classic keyword-based retrieval workflow
- `Phase_2_Dense_Semantic_Embeddings_and_Intent_Classification.ipynb` — embedding-based classification pipeline
- `Phase_3_Retrieval_Augmented_Generation.ipynb` — FAISS + chunked retrieval + RAG mock generator
- `Appendix_A_Live_RAG_Inference_with_Local_Ollama.ipynb` — local Ollama integration for live generation
- `generate_notebook.py` — script used to generate Phase 1 notebook content
- `generate_phase2_notebook.py` — script used to generate Phase 2 notebook content
- `generate_phase3_notebook.py` — script used to generate Phase 3 notebook content
- `generate_appendix_a_notebook.py` — script used to generate the Appendix A notebook
- `generate_synthetic_data.py` — synthetic data creation utilities
- `data/phase1_campus_faqs.csv` — FAQ dataset used in the lexical retrieval phase
- `data/phase2_intent_queries.csv` — synthetic query dataset used in the intent classification phase
- `Campus_Student_Handbook_and_Policy_Manual.pdf` — generated campus policy PDF used by the RAG chapters

## Data and Educational Purpose

The project uses synthetic campus data to replicate realistic university operational queries. Examples include:

- parking and permit questions
- tuition payments and fee deadlines
- dorm maintenance and housing issues
- library hours and study access
- academic advising and course policies
- drone and campus policy restrictions

This makes the project useful for teaching information retrieval, NLP fundamentals, vector search, and grounded generation without needing a proprietary production dataset.

## Phase 1: Lexical Search and Rule-Based Matching

This phase demonstrates why naive keyword matching fails in realistic human-language tasks.

Key elements:

- text normalization and cleaning
- lowercase conversion
- punctuation stripping
- tokenization with NLTK
- stop-word filtering
- TF-IDF feature extraction with scikit-learn
- cosine similarity scoring
- ranking of the most relevant FAQ entries

The notebook intentionally includes semantically equivalent but lexically different phrasing patterns, such as:

- “parking permit” vs. “automobile storage permit authorization”
- “pay tuition” vs. “bursar fee deposit and wire remittance”
- “dorm room” vs. “residential living quad dwelling allocation”

This reveals the vocabulary mismatch problem directly.

## Phase 2: Dense Semantic Embeddings and Intent Classification

This phase moves from sparse lexical features to dense embedding vectors using SentenceTransformers.

Core ideas:

- transform user queries into dense embeddings
- map queries into structured intent categories
- train logistic regression or SVM classifiers
- examine semantic similarity in hidden vector space
- evaluate with accuracy, precision, recall, and F1-score
- visualize clustering and embeddings via PCA and t-SNE

The synthetic dataset includes balanced campus query classes such as:

- `tuition_payment`
- `parking_permit`
- `dorm_maintenance`
- `library_hours`
- `academic_advising`

This phase shows how dense representations reduce the synonym problem and improve generalization over lexical overlap.

## Phase 3: Retrieval-Augmented Generation

This phase builds a full campus policy RAG pipeline.

Components:

- policy document ingestion via PyPDF2
- chunking with `RecursiveCharacterTextSplitter`
- embeddings from SentenceTransformers
- vector indexing with FAISS
- top-k retrieval based on semantic similarity
- prompt construction for grounded answer generation
- rule-based generation or a mocked answer pipeline for reproducible educational exercises

This is the first point where the project demonstrates the difference between:

- retrieval-only matching
- semantic classification
- grounded answer synthesis using evidence from the campus policy corpus

## Appendix A: Local Ollama RAG Inference

The appendix connects the retrieval pipeline to a local LLM runtime using Ollama.

This stage includes:

- local Ollama model health checks
- model selection such as `llama3.1`, `llama3.2`, and `qwen2.5:7b`
- prompt templates with strict grounding instructions
- low-temperature generation for factual stability
- refusal logic when the retrieved evidence is weak or missing
- robust exception handling for local inference failures

The key principle is that the model should answer from retrieved context rather than rely on memory alone.

## Local Setup

### Requirements

- Python 3.9+
- pip
- a working Ollama installation for the live inference appendix
- optional GPU acceleration for faster embeddings and inference

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: install Ollama

For the appendix that runs the local LLM:

1. Install Ollama: https://ollama.com/
2. Start the service:

```bash
ollama serve
```

3. Pull a model:

```bash
ollama pull llama3.1
```

Then the live RAG code can query the local runtime using the Python Ollama SDK.

## Running the Notebooks

Open the notebooks in Jupyter or VS Code and run cells in order.

Typical flow:

```bash
jupyter notebook
```

Then open:

- `Phase_1_Lexical_Search_and_Rule_Based_Matching.ipynb`
- `Phase_2_Dense_Semantic_Embeddings_and_Intent_Classification.ipynb`
- `Phase_3_Retrieval_Augmented_Generation.ipynb`
- `Appendix_A_Live_RAG_Inference_with_Local_Ollama.ipynb`

## Regenerating the Notebooks

The notebooks can be regenerated from the Python generator scripts if needed:

```bash
python generate_notebook.py
python generate_phase2_notebook.py
python generate_phase3_notebook.py
python generate_appendix_a_notebook.py
```

This can be useful if you want to update the content or re-export the lab materials.

## Notes for Reproducibility

- Some packages may need to be installed before running the notebooks for the first time.
- NLTK downloads stopwords and tokenization resources on first use.
- Sentence transformers may download model weights the first time they are used.
- If you are running on a constrained machine, the smaller embedding models will be easier to use.
- If Ollama is not running, the Appendix A notebook will still show the setup logic and diagnostics, but live inference cannot complete until the daemon is active.

## Typical Use Cases

This repository is useful for:

- teaching classical and modern NLP pipelines
- illustrating the transition from lexical matching to dense retrieval and generative AI
- demonstrating RAG and hallucination control in a local environment
- prototyping a campus support assistant with a small domain-specific knowledge base
- exploring semantic search, classification, and grounded rule-based answer generation

## Dependencies Summary

The project relies on:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk`
- `sentence-transformers`
- `torch`
- `faiss-cpu`
- `PyPDF2`
- `reportlab`
- `langchain`
- `langchain-community`
- `langchain-text-splitters`
- `ollama`

## License

This project is provided for educational and research purposes. Please review any third-party package licenses before production deployment or redistribution.

## Summary

This repository is a compact end-to-end demonstration of how a campus assistant can evolve from simple keyword retrieval into a robust grounded AI system. It is especially useful for learning the conceptual transition from lexical matching to semantic search and retrieval-augmented generation, while keeping the environment reproducible and local.
