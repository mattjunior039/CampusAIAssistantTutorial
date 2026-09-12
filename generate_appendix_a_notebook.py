import json
import os

def create_appendix_a_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # =========================================================================
    # CELL 1: TITLE & METADATA
    # =========================================================================
    add_md("""# Appendix A: Live RAG Inference with Local Ollama
## Connecting FAISS Dense Vector Search to Local Llama 3.1

**Curriculum Module:** Supplemental Hands-On Production Appendix to Phase 3  
**Prerequisites:** Phase 3 (Retrieval-Augmented Generation / FAISS), Local Ollama CLI  
**Target Hardware:** Apple Silicon (M1/M2/M3/M4), NVIDIA RTX GPUs, or modern Multi-Core CPUs  
**Estimated Time:** 30–45 minutes  

---

### Learning Objectives
By completing this appendix, you will:
1. **Transition from Mock to Live Autoregressive Synthesis:** Replace the deterministic rule-based generator from Phase 3 with an actual local Large Language Model (`Llama 3.1`).
2. **Execute Zero-Cost, Private Local Inference:** Use **Ollama** to serve open-weights models locally without cloud API subscription keys, data leaks, or token billing.
3. **Master Temperature & Hyperparameter Control for RAG:** Configure low-entropy sampling (`temperature=0.1`) to enforce strict factual fidelity and citation adherence.
4. **Implement Production Error Handling:** Construct robust error-handling wrappers catching `ollama.ResponseError` and connection exceptions.
5. **Evaluate Local Model Architecture Trade-Offs:** Benchmark latency vs. reasoning quality across different model parameter scales (`llama3.1:8b`, `llama3.2:3b`, `qwen2.5:7b`).""")

    # =========================================================================
    # CELL 2: SECTION 1: ARCHITECTURE OVERVIEW
    # =========================================================================
    add_md("""## 1. Architectural Overview: Decoupling Retrieval and Local Generation

In **Phase 3**, we developed the complete information retrieval infrastructure:
- Parsing campus policy PDF handbooks with `PyPDF2`
- Segmenting text using `RecursiveCharacterTextSplitter` with 20% semantic overlap
- Populating a high-performance `FAISS` vector database using `all-MiniLM-L6-v2` dense embeddings

To ensure the primary curriculum was 100% executable in air-gapped or non-GPU environments, Phase 3 utilized a deterministic mock client. 

In this Appendix, we complete the production loop by connecting our **FAISS vector retriever** to a **live local Ollama LLM runtime**:

```
+-----------------------------------------------------------------------------------------+
|                         LIVE LOCAL OLLAMA RAG ARCHITECTURE                              |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [Student Query: "How much is a parking permit?"]                                       |
|               |                                                                         |
|               v                                                                         |
|  +-------------------------------------------------------------+                        |
|  | SentenceTransformer: all-MiniLM-L6-v2 (Embedding to R^384) |                        |
|  +-------------------------------------------------------------+                        |
|               |                                                                         |
|               v                                                                         |
|  +-------------------------------------------------------------+                        |
|  | FAISS Vector Store Search (Top-k Chunks Retrieval)          |                        |
|  +-------------------------------------------------------------+                        |
|               |                                                                         |
|               v                                                                         |
|  [Retrieved Context Chunk: CHUNK-09 ($185/semester permit)]                             |
|               |                                                                         |
|               v                                                                         |
|  +-------------------------------------------------------------+                        |
|  | Strict Factual Grounding Prompt Formatter                   |                        |
|  +-------------------------------------------------------------+                        |
|               |                                                                         |
|               v                                                                         |
|  +-------------------------------------------------------------+                        |
|  | ollama.chat(model='llama3.1', options={'temperature': 0.1}) |                        |
|  | Local Model Weights executing on Metal / CUDA / CPU         |                        |
|  +-------------------------------------------------------------+                        |
|               |                                                                         |
|               v                                                                         |
|  [Live Generated Answer with Verifiable Citation: [CHUNK-09]]                           |
+-----------------------------------------------------------------------------------------+
```

### Why Local Ollama for Campus AI?
- **Zero Cloud Costs:** Unlimited student queries without per-token API charges.
- **Complete Privacy & FERPA Compliance:** Student inquiries and internal university documents never leave local campus infrastructure.
- **Low Latency:** Direct memory access on unified-memory hardware (e.g., Apple Silicon M-series or local servers).""")

    # =========================================================================
    # CELL 3: STEP 2: ENVIRONMENT SETUP
    # =========================================================================
    add_code("""# Step 2: Environment Setup & Library Verification
import sys
import subprocess
import os
import time
import re
from typing import List, Dict, Any, Tuple, Optional

# Verify and install required libraries
required_packages = ["ollama", "faiss-cpu", "sentence-transformers", "pandas", "numpy"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"[Setup] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import faiss
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer

print(f"[OK] Python version: {sys.version.split()[0]}")
print(f"[OK] FAISS version: {faiss.__version__}")
print(f"[OK] Ollama Python SDK installed and ready.")""")

    # =========================================================================
    # CELL 4: STEP 2.2: OLLAMA DAEMON HEALTH CHECK
    # =========================================================================
    add_code("""# Step 2.2: Verify Local Ollama Daemon & Discover Installed Models

def check_local_ollama() -> Tuple[bool, List[str]]:
    \"\"\"
    Checks if the local Ollama daemon is active and lists all downloaded models.
    \"\"\"
    try:
        response = ollama.list()
        # Handle both object attributes and dict structures across SDK versions
        models_list = []
        if hasattr(response, "models"):
            for m in response.models:
                name = getattr(m, "model", None) or getattr(m, "name", str(m))
                models_list.append(name)
        elif isinstance(response, dict) and "models" in response:
            for m in response["models"]:
                name = m.get("name") or m.get("model", str(m))
                models_list.append(name)
                
        return True, models_list
    except Exception as e:
        print(f"[Ollama Warning] Could not connect to local daemon: {e}")
        return False, []

is_active, installed_models = check_local_ollama()

print("=" * 70)
print(f"Ollama Daemon Active : {is_active}")
print(f"Installed Models     : {installed_models if installed_models else 'None detected'}")
print("=" * 70)

if not is_active:
    print(\"\"\"
[ACTION REQUIRED]:
Local Ollama service is not running. 
To start it:
  1. Open a new Terminal window.
  2. Run: 'ollama serve'
  3. Pull the target model: 'ollama pull llama3.1'
\"\"\")
else:
    print("✓ Local Ollama daemon is running and reachable.")""")

    # =========================================================================
    # CELL 5: SECTION 3: THE LIVE OLLAMA RAG CLIENT MARKDOWN
    # =========================================================================
    add_md(r"""## 3. The Live RAG Client Implementation

Let us construct the `LiveOllamaRAGClient` class.

### Key Engineering Details:
1. **Low Temperature Sampling ($\tau = 0.1$):**
   In standard creative writing, temperature $\tau \approx 0.7\text{--}1.0$ promotes lexical diversity. In RAG systems, high entropy causes the model to drift from injected facts. Setting $\tau = 0.1$ flattens the output distribution toward the highest-probability factual tokens conditioned on context $\mathcal{C}$:
   $$P(y_t = v \mid \mathbf{h}_t) = \frac{\exp(\mathbf{w}_v^T \mathbf{h}_t / \tau)}{\sum_{j} \exp(\mathbf{w}_j^T \mathbf{h}_t / \tau)}$$

2. **Negative Constraint & Abstention Trigger:**
   If retrieved chunks lack semantic relevance, the system prompt instructs the model to reply with the exact refusal sentinel:
   `"I CANNOT FIND THIS IN THE CAMPUS DOCUMENTS"`.

3. **Exception Handling:**
   Catches `ollama.ResponseError` or network connection drops, returning clean diagnostic messages rather than crashing the pipeline.""")

    # =========================================================================
    # CELL 6: STEP 3.1: LIVE CLIENT CODE
    # =========================================================================
    add_code("""# Step 3.1: Define the LiveOllamaRAGClient Class

class LiveOllamaRAGClient:
    \"\"\"
    Connects a dense retrieval pipeline to a locally hosted LLM served via Ollama.
    
    Attributes:
        model_name (str): Identifier of the Ollama model (e.g., 'llama3.1', 'llama3.2', 'qwen2.5:7b').
        temperature (float): Sampling temperature (0.1 for high-fidelity RAG).
        confidence_threshold (float): Minimum cosine similarity to attempt generative synthesis.
    \"\"\"
    
    def __init__(
        self,
        model_name: str = "llama3.1",
        temperature: float = 0.1,
        confidence_threshold: float = 0.25
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
    def build_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        \"\"\"
        Formats the strict context-grounded prompt template.
        \"\"\"
        context_entries = []
        for chunk in retrieved_chunks:
            entry = f"--- SOURCE [{chunk['chunk_id']}] (Cosine Score: {chunk.get('similarity_score', 1.0):.4f}) ---\\n{chunk['text']}"
            context_entries.append(entry)
            
        context_block = "\\n\\n".join(context_entries)
        
        prompt = f\"\"\"You are the official AI Campus Policy Assistant.
Your task is to answer the student's question accurately, concisely, and STRICTLY using the retrieved context chunks below.

### RETRIEVED CONTEXT CHUNKS:
{context_block}

### OPERATIONAL RULES:
1. Base your answer ONLY on the retrieved context above. Do NOT use outside general knowledge or assumptions.
2. If the retrieved context does not contain enough information to answer the question truthfully, reply EXACTLY with:
   "I CANNOT FIND THIS IN THE CAMPUS DOCUMENTS. Please contact the Dean of Students Office for guidance."
3. Cite the source chunk IDs (e.g., [CHUNK-02]) for each policy fact you mention.

Student Question: {query}
Helpful & Cited Answer:\"\"\"
        return prompt

    def generate(
        self,
        prompt: str,
        retrieved_chunks: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        \"\"\"
        Sends the formatted prompt to the local Ollama LLM and returns the synthesized response.
        
        Args:
            prompt (str): The structured system and context prompt.
            retrieved_chunks (List[Dict[str, Any]]): Retrieved document chunks with similarity scores.
            query (str): The raw student question.
            
        Returns:
            Dict[str, Any]: Payload containing the synthesized answer, latency, and status.
        \"\"\"
        # Check similarity threshold guardrail
        max_score = max([c.get("similarity_score", 1.0) for c in retrieved_chunks]) if retrieved_chunks else 0.0
        if max_score < self.confidence_threshold:
            return {
                "answer": "I CANNOT FIND THIS IN THE CAMPUS DOCUMENTS. Please contact the Dean of Students Office for guidance.",
                "status": "GUARDRAIL_REFUSAL",
                "latency_ms": 0.0,
                "model_used": self.model_name
            }
            
        t0 = time.perf_counter()
        try:
            # Call Ollama Python API
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.9
                }
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            # Extract generated text
            answer_text = response["message"]["content"].strip()
            
            return {
                "answer": answer_text,
                "status": "SUCCESS",
                "latency_ms": round(elapsed_ms, 2),
                "model_used": self.model_name
            }
            
        except ollama.ResponseError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "answer": f"[Ollama API Error]: {e.error} (Status code: {e.status_code})",
                "status": "API_ERROR",
                "latency_ms": round(elapsed_ms, 2),
                "model_used": self.model_name
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "answer": f"[Connection Error]: Could not reach Ollama daemon. Ensure 'ollama serve' is running. Details: {e}",
                "status": "CONNECTION_ERROR",
                "latency_ms": round(elapsed_ms, 2),
                "model_used": self.model_name
            }""")

    # =========================================================================
    # CELL 7: STEP 4: EXECUTION BLOCK (MOCK SIMULATION)
    # =========================================================================
    add_md("""## 4. Execution Block: Live RAG Synthesis & Verification

We now demonstrate the live generation client:
1. **Mock Retrieval Test Case:** We simulate retrieving official campus parking policy chunks (`CHUNK-09`).
2. **Prompt Formulation:** The query and chunks are formatted into our factual template.
3. **Live Ollama Inference:** Sent to the local `Llama 3.1` model to observe live generation and citation fidelity.""")

    # =========================================================================
    # CELL 8: STEP 4.1: LIVE SYNTHESIS CODE
    # =========================================================================
    add_code("""# Step 4.1: Live Synthesis on Mock Retrieved Context

# Determine active model tag from installed list (fallback to 'llama3.1' or first available)
target_model = "llama3.1"
if installed_models:
    for m in installed_models:
        if "llama3.1" in m or "llama3" in m:
            target_model = m
            break
    else:
        gen_m = [m for m in installed_models if "embed" not in m]
        if gen_m:
            target_model = gen_m[0]

print(f"[Client Config] Initializing LiveOllamaRAGClient with model='{target_model}'...")
live_rag_client = LiveOllamaRAGClient(model_name=target_model, temperature=0.1)

# Simulated Retrieved Document Chunks (from Campus Policy Manual Section 5)
mock_retrieved_chunks = [
    {
        "chunk_id": "CHUNK-09",
        "rank": 1,
        "similarity_score": 0.7850,
        "text": (
            "SECTION 5: PARKING PERMITS & AUTOMOBILE STORAGE\\n"
            "5.1 Commuter Permit Authorization: Commuter decals ($185/semester) authorize parking in Lots A, B, and C.\\n"
            "5.2 Resident Student Storage: Residential students residing on campus must purchase a Red Zone storage permit.\\n"
            "Vehicles parked in visitor stalls without validation between 2:00 AM and 6:00 AM are subject to immediate towing."
        )
    },
    {
        "chunk_id": "CHUNK-10",
        "rank": 2,
        "similarity_score": 0.4210,
        "text": (
            "5.3 Citations & Fines: Unregistered vehicles parked in faculty spaces will receive a $45 fine per violation.\\n"
            "Appeals must be submitted to the Department of Motor Transportation within 14 instructional days."
        )
    }
]

# User inquiry
sample_query = "How much does a commuter student parking pass cost, and where can I park?"

# Build prompt
live_prompt = live_rag_client.build_prompt(sample_query, mock_retrieved_chunks)

# Execute Live Generation
result = live_rag_client.generate(live_prompt, mock_retrieved_chunks, sample_query)

print("=" * 80)
print(f"QUERY: '{sample_query}'")
print("=" * 80)
print(f"STATUS     : {result['status']}")
print(f"MODEL USED : {result['model_used']}")
print(f"LATENCY    : {result['latency_ms']:.2f} ms")
print(f"\\nLIVE GENERATED ANSWER:\\n{result['answer']}")
print("=" * 80)""")

    # =========================================================================
    # CELL 9: STEP 4.2: END-TO-END FAISS + OLLAMA PIPELINE CODE
    # =========================================================================
    add_code("""# Step 4.2: End-to-End Vector Retrieval + Live Generation Demonstration

print("[End-to-End] Embedding sample campus chunks with all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Synthesize a compact 4-chunk campus policy knowledge base
campus_corpus = [
    {
        "chunk_id": "CHUNK-01",
        "text": "Academic AI Policy: The utilization of Large Language Models is strictly prohibited on coursework unless explicitly authorized in the course syllabus with formal disclosure."
    },
    {
        "chunk_id": "CHUNK-02",
        "text": "Drone Operations: Operating recreational drones is strictly prohibited across all campus grounds. Academic research exemptions require a 72-hour registration notice."
    },
    {
        "chunk_id": "CHUNK-03",
        "text": "Residential Quiet Hours: Quiet hours are strictly enforced from 10:00 PM to 8:00 AM Sunday through Thursday, and 12:00 Midnight to 9:00 AM Friday and Saturday."
    },
    {
        "chunk_id": "CHUNK-04",
        "text": "Bursar Tuition Refund: 100% refund prior to Day 5; 75% refund Day 6 to 10; 50% refund Day 11 to 15. No refunds are issued after Day 15 of instruction."
    }
]

# Build FAISS Index
corpus_texts = [c["text"] for c in campus_corpus]
corpus_embeddings = embedding_model.encode(corpus_texts, normalize_embeddings=True).astype(np.float32)

faiss_idx = faiss.IndexFlatIP(corpus_embeddings.shape[1])
faiss_idx.add(corpus_embeddings)

def run_live_rag_search(query: str, top_k: int = 2) -> None:
    \"\"\"Performs dense FAISS retrieval followed by live Ollama synthesis.\"\"\"
    q_vec = embedding_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_idx.search(q_vec, k=top_k)
    
    retrieved = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        item = campus_corpus[idx].copy()
        item["rank"] = rank
        item["similarity_score"] = float(score)
        retrieved.append(item)
        
    prompt = live_rag_client.build_prompt(query, retrieved)
    response_payload = live_rag_client.generate(prompt, retrieved, query)
    
    print(f"\\n--------------------------------------------------------------------------------")
    print(f"QUERY: '{query}'")
    print(f"TOP RETRIEVED CHUNK: [{retrieved[0]['chunk_id']}] (Cosine Score: {retrieved[0]['similarity_score']:.4f})")
    print(f"GENERATED RESPONSE ({response_payload['latency_ms']:.1f} ms):")
    print(response_payload["answer"])
    print(f"--------------------------------------------------------------------------------")

# Test 1: In-distribution inquiry
run_live_rag_search("What are the quiet hours in the dorms on weekend nights?")

# Test 2: Out-of-distribution inquiry (testing refusal guardrail)
run_live_rag_search("How do I bake a chocolate cake?")""")

    # =========================================================================
    # CELL 10: SECTION 5: SELF-CHECK UNIT TESTS
    # =========================================================================
    add_code("""# =============================================================================
# SECTION 5: SELF-CHECK UNIT TESTS (ASSERTIONS)
# =============================================================================

def run_appendix_unit_tests():
    print("[Testing Suite] Running Appendix A verification checks...")
    
    # Check 1: Prompt Construction Integrity
    dummy_chunks = [{"chunk_id": "CHUNK-99", "text": "Sample policy test text.", "similarity_score": 0.85}]
    formatted_p = live_rag_client.build_prompt("Test Question?", dummy_chunks)
    assert "CHUNK-99" in formatted_p, "Prompt must inject chunk identifiers."
    assert "Test Question?" in formatted_p, "Prompt must include user question."
    assert "I CANNOT FIND THIS IN THE CAMPUS DOCUMENTS" in formatted_p, "Prompt must include refusal rule."
    print("  ✓ Check 1: Prompt formatting and negative constraints verified.")
    
    # Check 2: Low-Confidence Abstention Guardrail
    low_conf_chunks = [{"chunk_id": "CHUNK-01", "text": "Irrelevant text", "similarity_score": 0.12}]
    low_p = live_rag_client.build_prompt("OOD Query", low_conf_chunks)
    res_ood = live_rag_client.generate(low_p, low_conf_chunks, "OOD Query")
    assert res_ood["status"] == "GUARDRAIL_REFUSAL", "Low similarity must trigger guardrail refusal."
    assert "I CANNOT FIND THIS IN THE CAMPUS DOCUMENTS" in res_ood["answer"], "Must return exact refusal sentinel."
    print("  ✓ Check 2: Low-confidence abstention guardrail verified.")
    
    # Check 3: Client Data Contract
    assert hasattr(live_rag_client, "generate"), "Client must implement generate()."
    assert hasattr(live_rag_client, "model_name"), "Client must expose model_name."
    print("  ✓ Check 3: Client schema and method contract verified.")
    
    print("\\n" + "=" * 70)
    print("🎉 ALL APPENDIX A UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

run_appendix_unit_tests()""")

    # =========================================================================
    # CELL 11: SECTION 6: STUDENT CHALLENGE MARKDOWN
    # =========================================================================
    add_md("""## 6. Student Lab Challenge: Model Architecture & Latency Trade-Offs

Now that you have connected the RAG pipeline to a live local model, explore how different open-weight model architectures perform on your local hardware.

---

### Student Task:
1. Pull an ultra-compact lightweight model in your terminal:
   ```bash
   ollama pull llama3.2:3b
   # or
   ollama pull qwen2.5:7b
   # or
   ollama pull phi3:mini
   ```
2. Re-initialize `LiveOllamaRAGClient` with the new model name:
   ```python
   compact_rag_client = LiveOllamaRAGClient(model_name="llama3.2:3b", temperature=0.1)
   ```
3. Run identical campus queries across both models and record:
   - **Generation Latency (ms)**
   - **Time to First Token (TTFT)**
   - **Citation Accuracy & Adherence to Negative Constraints**

---

### Model Trade-Off Comparison Matrix

| Model Identifier | Parameter Count | VRAM / RAM Footprint | Typical Generation Latency | Reasoning & Citation Fidelity |
| :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:3b`** | 3.2 Billion | $\approx 2.0\text{ GB}$ | **Ultra-Fast (< 800 ms)** | Good; occasional minor prompt formatting leaks. |
| **`phi3:mini`** | 3.8 Billion | $\approx 2.3\text{ GB}$ | **Fast (< 1000 ms)** | Strong logical reasoning; strict adherence. |
| **`qwen2.5:7b`** | 7.6 Billion | $\approx 4.7\text{ GB}$ | **Balanced ($\approx 1500\text{ ms}$)** | Excellent multilingual & technical extraction. |
| **`llama3.1:8b`** | 8.0 Billion | $\approx 4.9\text{ GB}$ | **Standard ($\approx 2000\text{ ms}$)** | **State-of-the-Art local grounding & zero hallucination.** |

---
*End of Appendix A: Live RAG Inference with Local Ollama.*""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "Appendix_A_Live_RAG_Inference_with_Local_Ollama.ipynb")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"[SUCCESS] Wrote Appendix A notebook with {len(cells)} cells to {path}")

if __name__ == "__main__":
    create_appendix_a_notebook()
