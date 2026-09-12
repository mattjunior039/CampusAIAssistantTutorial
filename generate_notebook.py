import json
import os

def create_notebook():
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

    # CELL 1: TITLE & METADATA
    add_md("""# Project 1: AI Campus Assistant Pipeline
## Phase 1: Lexical Search & Rule-Based Matching

**Curriculum Level:** Undergraduate / Graduate Natural Language Processing & Information Retrieval  
**Prerequisites:** Python 3.9+, Linear Algebra (Dot Products & Vector Norms), Basic Probability  
**Estimated Time:** 90–120 minutes  

---

### Learning Objectives
By completing this hands-on laboratory, you will:
1. **Deconstruct the NLP Evolution:** Contrast discrete lexical retrieval (TF-IDF), dense semantic representation (bi-encoders), and generative orchestration (RAG).
2. **Master the Mathematical Foundations of Vector Space Models:** Formulate Term Frequency (TF), smoothed Inverse Document Frequency (IDF), $L_2$ document normalization, and Cosine Similarity.
3. **Build an End-to-End Production Lexical Search Engine:** Implement text sanitation, tokenization, stop-word elimination, TF-IDF vectorization, inverted indexing, and ranked retrieval.
4. **Empirically Diagnose Lexical Retrieval Failures:** Observe and analyze *Synonym Blindness* (vocabulary mismatch), *Polysemy / Context Blindness* (ambiguous token collisions), and *Sparsity Explosion*.
5. **Implement Robust Fallbacks & Evaluate Performance:** Build automated thresholding, investigate n-gram dimensionality scaling, and execute unit tests.""")

    # CELL 2: SECTION 1: ROADMAP & THE CORE PREMISE
    add_md("""## 1. Course Introduction & The NLP Evolution Roadmap

Modern conversational AI systems—such as university campus assistants, customer support bots, and enterprise search platforms—did not emerge overnight. They represent a three-decade evolution from discrete symbolic methods to continuous dense vector spaces, and ultimately to generative foundation models.

```
+---------------------------------------------------------------------------------------------------+
|                                 THE NLP & IR EVOLUTIONARY SPECTRUM                                |
+------------------------------------+----------------------------------+---------------------------+
| Phase 1: Lexical & Rule-Based      | Phase 2: Dense Semantic Matching | Phase 3: RAG & GenAI      |
| (1990s - 2010s)                    | (2018 - 2022)                    | (2023 - Present)          |
+------------------------------------+----------------------------------+---------------------------+
| - Bag-of-Words & TF-IDF            | - Word2Vec, GloVe, FastText      | - LLM Generation (GPT-4o) |
| - Inverted Indexes & BM25          | - Dense Bi-Encoders (SBERT)      | - Vector Databases (HNSW) |
| - Exact token & character overlap  | - Continuous embedding spaces    | - Hybrid Retrieval + Rerank|
| - Fast, interpretable, rigid       | - Handles synonyms & paraphrasing| - Context-grounded synthesis|
| * THIS LAB (Phase 1) *             | * NEXT LAB (Phase 2) *           | * FINAL PROJECT (Phase 3) *|
+------------------------------------+----------------------------------+---------------------------+
```

### The Core Premise: Why Lexical Search Fails Human Intent
Lexical search operates under a fundamental assumption: **relevance is a function of shared surface-form tokens**. If a user's query and a reference document share words, they are assumed to be semantically related.

However, natural human language violates this assumption in three critical ways:
1. **Synonymy (The Vocabulary Mismatch Problem):** Different words express identical concepts (e.g., *"car"* vs. *"automobile"*, *"tuition deposit"* vs. *"bursar payment"*). A pure lexical engine produces a similarity score of zero when vocabulary does not overlap.
2. **Polysemy & Ambiguity:** A single token carries distinct meanings depending on context (e.g., *"river bank"* vs. *"commercial bank"*, *"Python course"* vs. *"ball python"*). Lexical systems blindly match the token regardless of meaning.
3. **Word Order & Negation Blindness:** Bag-of-words architectures treat text as unordered multisets. Consequently, *"The exam is not hard, it is easy"* and *"The exam is not easy, it is hard"* produce identical vector representations.""")

    # CELL 3: SECTION 2: MATHEMATICAL FOUNDATIONS
    add_md(r"""## 2. Mathematical Foundations

Let our campus knowledge base consist of a corpus $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ containing $N$ documents, over a vocabulary $\mathcal{V} = \{t_1, t_2, \dots, t_{|V|}\}$ of $|V|$ unique terms.

---

### 2.1 Bag-of-Words (BoW) & Loss of Syntax
In a Bag-of-Words representation, a document $d$ is mapped to a vector $\mathbf{x} \in \mathbb{R}^{|V|}$, where $x_i$ represents the frequency of term $t_i$ in document $d$:

$$\mathbf{x} = [f(t_1, d), f(t_2, d), \dots, f(t_{|V|}, d)]^T$$

**Limitation:** The permutation matrix $P$ that reorders tokens is discarded: $\text{BoW}(\text{sequence}) = \text{BoW}(\pi(\text{sequence}))$.

---

### 2.2 Term Frequency-Inverse Document Frequency (TF-IDF)
Raw frequency counts disproportionately favor ubiquitous words (e.g., *"the"*, *"university"*, *"campus"*) that appear in nearly every document and carry minimal discriminative power. TF-IDF resolves this by balancing local frequency against global rarity.

#### 1. Term Frequency (Relative Frequency):
Measures how frequently term $t$ occurs in document $d$, normalized by document length:

$$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

where $f_{t, d}$ is the raw count of term $t$ in document $d$.

#### 2. Smooth Inverse Document Frequency (scikit-learn Formulation):
Measures the informational specificity of term $t$ across the entire corpus of $N$ documents:

$$IDF(t) = \ln\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$

where $DF(t) = |\{d \in \mathcal{D} : t \in d\}|$ is the document frequency of term $t$ (count of documents containing $t$).
- The $+1$ inside the logarithm prevents division-by-zero (smoothing).
- The $+1$ outside ensures terms appearing in all documents retain non-zero positive weights rather than being completely zeroed out.

#### 3. Combined TF-IDF Weight:
$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

#### 4. Euclidean ($L_2$) Normalization:
To prevent longer documents from dominating similarity scores simply due to larger raw vector lengths, each document vector $\mathbf{v} \in \mathbb{R}^{|V|}$ is projected onto the unit hypersphere:

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{|V|} v_i^2}}$$

---

### 2.3 Cosine Similarity in High-Dimensional Vector Space
Given a vectorized query $\mathbf{q} \in \mathbb{R}^{|V|}$ and document vector $\mathbf{d} \in \mathbb{R}^{|V|}$, Cosine Similarity computes the cosine of the angle $\theta$ between them:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^{|V|} q_i d_i}{\sqrt{\sum_{i=1}^{|V|} q_i^2} \sqrt{\sum_{i=1}^{|V|} d_i^2}}$$

When vectors are already $L_2$-normalized ($\|\mathbf{q}\|_2 = 1$ and $\|\mathbf{d}\|_2 = 1$), this reduces to the standard inner product:

$$\text{Cosine Similarity}(\mathbf{q}_{\text{norm}}, \mathbf{d}_{\text{norm}}) = \mathbf{q}_{\text{norm}} \cdot \mathbf{d}_{\text{norm}} = \sum_{i=1}^{|V|} q_i d_i$$

#### Why Cosine Similarity Isolates Angular Orientation:
Unlike Euclidean distance $\|\mathbf{q} - \mathbf{d}\|_2 = \sqrt{\sum (q_i - d_i)^2}$, which scales with magnitude (document length), cosine similarity measures strictly the **directional alignment** of term distributions. Two documents with identical relative keyword proportions will have a cosine similarity of $1.0$, regardless of whether one document is 10 words and the other is 10,000 words.""")

    # CELL 4: SECTION 3: STEP-BY-STEP IMPLEMENTATION PIPELINE
    add_md("""## 3. Step-by-Step Implementation Pipeline

Let us construct our lexical search engine from scratch through a modular, typed architecture.

```
+-------------------------------------------------------------------------------------+
|                              LEXICAL RETRIEVAL PIPELINE                             |
+-------------------------------------------------------------------------------------+
|                                                                                     |
|  [Raw User Query]                          [Campus Knowledge Base: 20 FAQs]        |
|          |                                                |                         |
|          v                                                v                         |
|  +------------------------+                     +------------------------+          |
|  | Regex Sanitization     |                     | Regex Sanitization     |          |
|  | Lowercasing & Punct.   |                     | Lowercasing & Punct.   |          |
|  | NLTK Tokenization      |                     | NLTK Tokenization      |          |
|  | Stop-Word Filtering    |                     | Stop-Word Filtering    |          |
|  +------------------------+                     +------------------------+          |
|          |                                                |                         |
|          v                                                v                         |
|  [Sanitized Tokens]                             [Sanitized Corpus Matrix]           |
|          |                                                |                         |
|          v                                                v                         |
|  +--------------------------------------------------------------------+             |
|  |                  scikit-learn TfidfVectorizer                      |             |
|  |            Vocabulary V, Document-Term Matrix (N x |V|)            |             |
|  +--------------------------------------------------------------------+             |
|                                |                                                    |
|                                v                                                    |
|            +---------------------------------------+                                |
|            | Cosine Similarity (Query dot Docs)    |                                |
|            | Matched Feature Contribution Extraction|                                |
|            | Top-K Ranking & Threshold Filtering   |                                |
|            +---------------------------------------+                                |
|                                |                                                    |
|                                v                                                    |
|                  [Ranked Results + Diagnostics]                                     |
+-------------------------------------------------------------------------------------+
```""")

    # CELL 5: STEP 3.1: ENVIRONMENT SETUP
    add_code("""# Step 3.1: Environment Setup & Verification
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional

# Install required packages if missing in the environment
required_packages = ["scikit-learn", "nltk", "pandas", "numpy", "matplotlib", "seaborn"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"[Setup] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK tokenizers and stop-word corpora
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print(f"[OK] Python version: {sys.version.split()[0]}")
print(f"[OK] NumPy version: {np.__version__}")
print(f"[OK] Pandas version: {pd.__version__}")
print(f"[OK] NLTK & Scikit-Learn successfully configured.")""")

    # CELL 6: STEP 3.2: CORPUS SYNTHESIS MARKDOWN
    add_md("""### 3.2 Campus FAQ Knowledge Base Synthesis

To make our pedagogical failure analysis transparent, we synthesize an in-memory dataset of 20 realistic university FAQ entries.

**Crucial Pedagogical Design Note:**
Several entries are intentionally authored using formal administrative, bureaucratic, or antiquated terminology:
- *Parking* is indexed as **"automobile storage permit authorization"**.
- *Paying tuition* is indexed as **"bursar fee deposit and wire remittance"**.
- *Dorm rooms* are indexed as **"residential living quad dwelling allocations"**.
- *Tutoring* is indexed as **"pedagogical consultation and academic remediation"**.

This deliberate design will allow us to demonstrate the exact failure mechanisms of lexical matching later in the lab.""")

    # CELL 7: STEP 3.2: CORPUS CODE
    add_code("""# Step 3.2: Define the 20 Campus FAQ Knowledge Base Entries

FAQ_DATA: List[Dict[str, Any]] = [
    {
        "faq_id": "FAQ-01",
        "category": "Parking & Transportation",
        "question": "How do students register for an on-campus automobile storage permit?",
        "answer": "Automobile storage permits must be requested via the Department of Motor Transportation portal with vehicle registration documentation."
    },
    {
        "faq_id": "FAQ-02",
        "category": "Financial Services",
        "question": "What is the mandatory procedure for bursar fee deposit and wire remittance?",
        "answer": "All bursar fee deposit and tuition wire remittance procedures must clear through the central treasury portal prior to the third academic calendar week."
    },
    {
        "faq_id": "FAQ-03",
        "category": "Residential Life",
        "question": "What regulations govern residential living quad dwelling room assignments?",
        "answer": "Undergraduate residential living quad dwelling allocations are finalized every July by the Office of Student Housing Administration."
    },
    {
        "faq_id": "FAQ-04",
        "category": "Academic Support",
        "question": "Where can undergraduates receive pedagogical consultation and academic remediation?",
        "answer": "Pedagogical consultation and peer remediation workshops take place Mondays through Thursdays at the Academic Success Pavilion."
    },
    {
        "faq_id": "FAQ-05",
        "category": "Library & Research",
        "question": "How many books and research manuscripts can a student borrow from the central library?",
        "answer": "Undergraduate scholars may borrow up to twenty-five print monographs and research manuscripts for a four-week renewable loan period."
    },
    {
        "faq_id": "FAQ-06",
        "category": "Health & Wellness",
        "question": "Where is the student health and wellness clinical infirmary located?",
        "answer": "The student health center and clinical infirmary is located on North Campus adjacent to the recreation facility."
    },
    {
        "faq_id": "FAQ-07",
        "category": "Dining Services",
        "question": "How do meal plans and dining hall culinary swipe credits work?",
        "answer": "Campus culinary swipe credits reload automatically on Sunday midnight and can be utilized across all three residential dining commons."
    },
    {
        "faq_id": "FAQ-08",
        "category": "Information Technology",
        "question": "How do I reset my campus network credentials and institutional password?",
        "answer": "Visit the campus identity management portal and complete multi-factor authentication to initiate an institutional password reset."
    },
    {
        "faq_id": "FAQ-09",
        "category": "Recreation & Athletics",
        "question": "What are the operating hours for the campus aquatic center and gymnasium?",
        "answer": "The varsity gymnasium and aquatic swimming complex are open daily from 6:00 AM until 11:00 PM with valid student ID access."
    },
    {
        "faq_id": "FAQ-10",
        "category": "Career Development",
        "question": "How do I schedule an appointment with a career development counselor?",
        "answer": "Log into the university career network portal to book mock interviews, resume critiques, and internship advising sessions."
    },
    {
        "faq_id": "FAQ-11",
        "category": "Registrar & Enrollment",
        "question": "What is the official deadline to drop a course without academic transcript penalty?",
        "answer": "The deadline to drop an academic course without receiving a withdrawal mark on your transcript is the end of the tenth instructional day."
    },
    {
        "faq_id": "FAQ-12",
        "category": "Financial Services",
        "question": "Which commercial bank institution handles university wire transfers and student deposits?",
        "answer": "The university partners with First State Bank for institutional escrow, international tuition wire transfers, and student deposit accounts."
    },
    {
        "faq_id": "FAQ-13",
        "category": "Campus Safety",
        "question": "How do students summon university police emergency escort services after hours?",
        "answer": "Dial 555-SAFE from any blue-light callbox station on campus to request an immediate university police escort to your residence."
    },
    {
        "faq_id": "FAQ-14",
        "category": "Accessibility Services",
        "question": "How do I register for disability classroom accommodations and exam proctoring?",
        "answer": "Submit medical documentation to the Accessibility Resources Office at least four weeks prior to midsemester examination periods."
    },
    {
        "faq_id": "FAQ-15",
        "category": "Student Activities",
        "question": "How can students charter a new registered campus student organization or club?",
        "answer": "New student organizations require at least ten enrolled active members, a faculty advisor, and approval from the Student Union Senate."
    },
    {
        "faq_id": "FAQ-16",
        "category": "International Student Services",
        "question": "How do international scholars maintain F-1 visa compliance and employment authorization?",
        "answer": "F-1 visa holders must maintain full-time academic enrollment (minimum twelve credits) and secure designated school official authorization before off-campus employment."
    },
    {
        "faq_id": "FAQ-17",
        "category": "Facilities & Maintenance",
        "question": "How do I submit an urgent maintenance repair work order for my dorm room?",
        "answer": "Submit a facilities work order request through the campus housing portal for plumbing, electrical, heating, or key replacement repairs."
    },
    {
        "faq_id": "FAQ-18",
        "category": "Graduation & Commencement",
        "question": "What are the graduation requirements and commencement regalia ordering procedures?",
        "answer": "Students must complete all departmental capstone requirements and order their graduation cap and gown regalia via the university bookstore by April 1."
    },
    {
        "faq_id": "FAQ-19",
        "category": "Sustainability & Waste",
        "question": "Where are zero-waste compost receptacles and electronic recycling bins located?",
        "answer": "Compost receptacles and electronic waste recycling drop-off stations are stationed in the lobby of every academic building and dining commons."
    },
    {
        "faq_id": "FAQ-20",
        "category": "Outdoor Recreation",
        "question": "Are students permitted to fish or kayak along the scenic campus river bank?",
        "answer": "Recreational non-motorized kayaking is allowed on the river, but fishing along the campus river bank requires a state conservation permit."
    }
]

# Convert into a structured pandas DataFrame
faq_df = pd.DataFrame(FAQ_DATA)

# Combine Question + Answer to form rich, retrievable document text
faq_df["full_text"] = faq_df["question"] + " " + faq_df["answer"]

print(f"[Corpus] Successfully initialized {len(faq_df)} FAQ knowledge base documents.")
print(f"[Corpus] Dataset Columns: {list(faq_df.columns)}")
display(faq_df[["faq_id", "category", "question"]].head(5))""")

    # CELL 8: STEP 3.3: PREPROCESSING PIPELINE MARKDOWN
    add_md("""### 3.3 Text Normalization & Preprocessing Pipeline

Before numerical vectorization, raw natural text must be transformed into clean, canonical tokens through a reproducible pipeline:

1. **Lowercasing:** Normalizes `"Tuition"` and `"tuition"` to a single vocabulary entry.
2. **Regex Punctuation Removal:** Strips symbols (e.g., `?`, `!`, `,`, `$`, `-`) using the regular expression `[^a-zA-Z0-9\s]`.
3. **NLTK Word Tokenization:** Segments character sequences into linguistic word tokens.
4. **Single-Character & Stop-Word Handling:**
   - Single alphanumeric tokens (such as `'f'` and `'1'` in `"F-1 visa"`, course section `'a'`, or building `'b'`) are explicitly preserved (`len(tok) >= 1`), as they carry critical domain meaning.
   - Grammatical stopwords (e.g., *"is"*, *"at"*, *"which"*, *"the"*, *"and"*) are eliminated using NLTK's English stop-word dictionary.

Let us implement this pipeline with full type signatures and verify it with a side-by-side inspection.""")

    # CELL 9: STEP 3.3: PREPROCESSOR CODE
    add_code("""# Step 3.3: Preprocessing Pipeline Implementation

ENGLISH_STOPWORDS: set = set(stopwords.words("english"))

def tokenize_and_clean(text: str, remove_stopwords: bool = True) -> List[str]:
    \"\"\"
    Sanitizes, lowercases, tokenizes, and filters stop words from an input string.
    
    Design Note on Single-Character Tokens:
    We preserve single alphanumeric tokens (e.g., '1' and 'f' in 'F-1 visa', building 'A', grade 'A')
    by checking len(tok) >= 1, while stripping empty tokens and filtering out stop words.

    Args:
        text (str): The raw input string to be preprocessed.
        remove_stopwords (bool): Whether to filter out standard English stop words.

    Returns:
        List[str]: A list of clean, normalized word tokens.
    \"\"\"
    if not isinstance(text, str):
        return []
    
    # 1. Lowercase
    text_lower = text.lower()
    
    # 2. Strip punctuation and special characters (retain alphanumeric and spaces)
    text_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text_lower)
    
    # 3. NLTK word tokenization
    tokens = word_tokenize(text_clean)
    
    # 4. Optional Stop-Word Filtering & Single-Character Handling (len >= 1)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in ENGLISH_STOPWORDS and len(tok) >= 1]
    else:
        tokens = [tok for tok in tokens if len(tok) >= 1]
        
    return tokens

# Side-by-Side Comparison Demonstration
sample_raw_sentences = [
    "How do international scholars maintain F-1 visa compliance?",
    "Where is the student health & wellness clinical infirmary located?",
    "Are students permitted to fish along the scenic campus river bank?"
]

print(f"{'RAW TEXT':<70} | {'SANITIZED TOKENS'}")
print("-" * 115)
for s in sample_raw_sentences:
    cleaned = tokenize_and_clean(s)
    print(f"{s:<70} | {cleaned}")""")

    # CELL 10: STEP 3.3: ASSERTION TESTS
    add_code("""# Self-Check Unit Test: Preprocessing Pipeline
def test_preprocessing():
    test_str = "What is the fee for 2026 bursar wire deposits for F-1 visa students??!"
    tokens = tokenize_and_clean(test_str, remove_stopwords=True)
    
    assert isinstance(tokens, list), "Output must be a list of tokens."
    assert "what" not in tokens, "Stop words like 'what' must be filtered."
    assert "is" not in tokens, "Stop words like 'is' must be filtered."
    assert "fee" in tokens, "'fee' should be preserved as an informative token."
    assert "2026" in tokens, "Alphanumeric tokens should be preserved."
    assert "deposits" in tokens, "'deposits' should be preserved."
    assert "f" in tokens and "1" in tokens, "Single alphanumeric characters like 'f' and '1' from F-1 visa must be preserved."
    assert all(not re.search(r"[^\w\s]", tok) for tok in tokens), "No punctuation allowed in tokens."
    print("[PASS] Preprocessing Pipeline Unit Tests Passed Successfully!")

test_preprocessing()""")

    # CELL 11: STEP 3.4: VECTOR SPACE MODELING MARKDOWN
    add_md(r"""### 3.4 Vector Space Modeling: Fitting `TfidfVectorizer`

We now fit scikit-learn's `TfidfVectorizer` to our 20 FAQ documents.

**Scikit-Learn Tokenizer Harmonization:**
When passing a custom callable `tokenizer=tokenize_and_clean` to `TfidfVectorizer`, scikit-learn requires setting `token_pattern=None`. Otherwise, scikit-learn will raise a `UserWarning` indicating that `token_pattern` is overridden by `tokenizer`, or could attempt double-tokenization.

Key Configuration Parameters:
- `tokenizer=tokenize_and_clean`: Applies our custom normalization, tokenization, single-character preservation, and stop-word removal.
- `token_pattern=None`: Harmonizes the vectorizer interface and disables the default regex tokenizer.
- `ngram_range=(1, 1)`: Extracts unigram tokens.
- `norm='l2'`: Applies Euclidean unit-length normalization to each document row.
- `smooth_idf=True`: Implements $IDF(t) = \ln\left(\frac{1+N}{1+DF(t)}\right) + 1$.

Let us inspect the resulting **Document-Term Matrix (DTM)**, compute its **sparsity**, and visualize a feature heatmap.""")

    # CELL 12: STEP 3.4: VECTORIZER FIT CODE
    add_code("""# Step 3.4: Fitting the Harmonized Vector Space Model

# Instantiate the vectorizer with custom tokenizer and token_pattern=None
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_and_clean,
    token_pattern=None,
    ngram_range=(1, 1),
    norm="l2",
    smooth_idf=True
)

# Fit on the combined corpus text (N documents)
doc_matrix = vectorizer.fit_transform(faq_df["full_text"])

# Extract vocabulary and dimensions
vocab = vectorizer.vocabulary_
feature_names = np.array(vectorizer.get_feature_names_out())
num_docs, vocab_size = doc_matrix.shape

# Compute Document-Term Matrix Sparsity
non_zero_elements = doc_matrix.nnz
total_elements = num_docs * vocab_size
sparsity = (1.0 - (non_zero_elements / total_elements)) * 100.0

print(f"================ TF-IDF VECTOR SPACE SUMMARY ================")
print(f"Total Documents in Corpus (N)      : {num_docs}")
print(f"Vocabulary Dimension (|V|)          : {vocab_size} unique terms")
print(f"Sparse Matrix Non-Zero Entries     : {non_zero_elements} / {total_elements}")
print(f"Matrix Sparsity Percentage         : {sparsity:.2f}% sparse")
print(f"=============================================================")

# Display a sample of the learned vocabulary indices
sample_vocab_items = list(vocab.items())[:10]
print(f"Sample Vocabulary Index Mapping: {sample_vocab_items}")""")

    # CELL 13: STEP 3.4: HEATMAP VISUALIZATION
    add_code("""# Step 3.4 Visualization: Document-Term TF-IDF Heatmap
# To maintain readability, we select 8 sample documents and 15 representative vocabulary features.

sample_doc_indices = [0, 1, 2, 3, 4, 11, 12, 19]
sample_doc_labels = [f"{faq_df.loc[i, 'faq_id']} ({faq_df.loc[i, 'category'][:15]})" for i in sample_doc_indices]

# Select 15 terms across different domains
selected_terms = [
    "automobile", "storage", "permit", "bursar", "deposit", 
    "wire", "bank", "river", "dwelling", "pedagogical", 
    "library", "escort", "police", "dining", "kayak"
]

# Find column indices for selected terms
selected_term_indices = [vectorizer.vocabulary_[t] for t in selected_terms if t in vectorizer.vocabulary_]
selected_term_names = [feature_names[idx] for idx in selected_term_indices]

# Extract dense submatrix
dense_submatrix = doc_matrix[sample_doc_indices, :][:, selected_term_indices].toarray()

# Plot Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    dense_submatrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=selected_term_names,
    yticklabels=sample_doc_labels,
    cbar_kws={"label": "Normalized TF-IDF Weight"}
)
plt.title("Document-Term TF-IDF Weight Distribution (Selected Subspace)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Vocabulary Terms ($t \in \mathcal{V}$)", fontsize=11)
plt.ylabel("Corpus Documents ($d \in \mathcal{D}$)", fontsize=11)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()""")

    # CELL 14: STEP 3.4: ASSERTION TESTS
    add_code("""# Self-Check Unit Test: Document-Term Matrix Properties
def test_vector_space_properties():
    # 1. Verify dimensions
    assert doc_matrix.shape[0] == len(faq_df), "Document count must match FAQ rows (20)."
    assert doc_matrix.shape[1] == len(vectorizer.vocabulary_), "Matrix columns must equal vocabulary size."
    
    # 2. Verify L2 unit-norm property for non-empty documents
    dense_matrix = doc_matrix.toarray()
    row_norms = np.linalg.norm(dense_matrix, axis=1)
    np.testing.assert_allclose(
        row_norms, 
        np.ones(len(faq_df)), 
        rtol=1e-5, 
        err_msg="Every document vector must have Euclidean L2 norm of 1.0."
    )
    
    # 3. Verify non-negativity
    assert (dense_matrix >= 0.0).all(), "TF-IDF weights must be non-negative."
    print("[PASS] Vector Space Model Properties Validated (Shapes, L2 Unit Norms, Non-negativity)!")

test_vector_space_properties()""")

    # CELL 15: STEP 3.5: INFERENCE ENGINE MARKDOWN
    add_md(r"""### 3.5 Inference & Search Engine Implementation

When a user submits a query $\mathbf{q}_{\text{raw}}$:
1. It is preprocessed and projected into the exact same $|V|$-dimensional TF-IDF vector space:
   $$\mathbf{q} = \text{TfidfVectorizer.transform}([\mathbf{q}_{\text{raw}}])$$
2. Cosine similarities are computed against all $N$ corpus document vectors in a single matrix-vector dot product:
   $$\mathbf{s} = \mathbf{q} \mathbf{D}^T \in \mathbb{R}^{1 \times N}$$
   where $\mathbf{D} \in \mathbb{R}^{N \times |V|}$ is the normalized Document-Term matrix.
3. Documents are sorted in descending order of similarity score $s_j$.
4. Matched terms are extracted by taking the Hadamard (element-wise) product $\mathbf{q} \odot \mathbf{d}_j$ to identify which vocabulary features drove the score.
5. Scores falling below a predefined `threshold` are flagged or filtered.""")

    # CELL 16: STEP 3.5: SEARCH FUNCTION CODE
    add_code("""# Step 3.5: Implement the Lexical Search Engine with Feature Attribution

def search_campus_faq(
    query: str,
    vectorizer: TfidfVectorizer,
    doc_matrix: Any,
    faq_df: pd.DataFrame,
    top_k: int = 3,
    threshold: float = 0.10
) -> List[Dict[str, Any]]:
    \"\"\"
    Searches the campus FAQ corpus using TF-IDF vectorization and Cosine Similarity.

    Args:
        query (str): The natural language query from the student.
        vectorizer (TfidfVectorizer): The fitted scikit-learn TF-IDF vectorizer.
        doc_matrix (scipy.sparse.csr_matrix): The N x |V| document-term matrix.
        faq_df (pd.DataFrame): The reference FAQ DataFrame containing metadata.
        top_k (int): Maximum number of ranked results to return.
        threshold (float): Minimum cosine similarity required to consider a match valid.

    Returns:
        List[Dict[str, Any]]: A ranked list of matching result dictionaries.
    \"\"\"
    # 1. Transform query into the learned vector space
    query_vec = vectorizer.transform([query])
    
    # 2. Compute Cosine Similarities across all corpus documents
    # Since both query_vec and doc_matrix are L2-normalized, linear dot product is exact cosine similarity
    similarities = cosine_similarity(query_vec, doc_matrix).flatten()
    
    # 3. Sort document indices by descending cosine similarity
    ranked_indices = np.argsort(similarities)[::-1]
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    query_dense = query_vec.toarray().flatten()
    
    results = []
    for rank, doc_idx in enumerate(ranked_indices[:top_k], start=1):
        score = float(similarities[doc_idx])
        doc_dense = doc_matrix[doc_idx].toarray().flatten()
        
        # 4. Identify matched keywords via element-wise product: q_i * d_i > 0
        overlap_weights = query_dense * doc_dense
        matched_feature_indices = np.where(overlap_weights > 0)[0]
        
        matched_keywords = []
        for feat_idx in matched_feature_indices:
            matched_keywords.append({
                "term": feature_names[feat_idx],
                "query_weight": float(query_dense[feat_idx]),
                "doc_weight": float(doc_dense[feat_idx]),
                "contribution": float(overlap_weights[feat_idx])
            })
            
        # Sort matched terms by contribution descending
        matched_keywords.sort(key=lambda x: x["contribution"], reverse=True)
        
        matched_terms_list = [mk["term"] for mk in matched_keywords]
        
        row = faq_df.iloc[doc_idx]
        results.append({
            "rank": rank,
            "faq_id": row["faq_id"],
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "cosine_score": round(score, 4),
            "is_above_threshold": score >= threshold,
            "matched_keywords": matched_terms_list,
            "detailed_contributions": matched_keywords
        })
        
    return results

def print_search_results(query: str, results: List[Dict[str, Any]]) -> None:
    \"\"\"CLI formatted display helper for search results.\"\"\"
    print(f"\\n================================================================================")
    print(f"SEARCH QUERY: \\\"{query}\\\"")
    print(f"================================================================================")
    for r in results:
        status = "[VALID MATCH]" if r["is_above_threshold"] else "[BELOW THRESHOLD]"
        print(f"Rank {r['rank']} | Score: {r['cosine_score']:.4f} {status}")
        print(f"FAQ ID: {r['faq_id']} | Category: {r['category']}")
        print(f"Question: {r['question']}")
        print(f"Answer:   {r['answer']}")
        print(f"Matched Tokens: {r['matched_keywords']}")
        print("-" * 80)""")

    # CELL 17: STEP 3.5: ASSERTION TESTS
    add_code("""# Self-Check Unit Test: Search Engine Retrieval Output
def test_search_engine():
    res = search_campus_faq(
        query="bursar fee deposit wire",
        vectorizer=vectorizer,
        doc_matrix=doc_matrix,
        faq_df=faq_df,
        top_k=3,
        threshold=0.1
    )
    
    assert len(res) == 3, "Must return exactly top_k (3) results."
    assert res[0]["cosine_score"] >= res[1]["cosine_score"] >= res[2]["cosine_score"], "Results must be sorted descending."
    assert 0.0 <= res[0]["cosine_score"] <= 1.0, "Cosine score must be bounded within [0, 1]."
    assert "faq_id" in res[0] and "matched_keywords" in res[0], "Expected output keys missing."
    print("[PASS] Search Engine Basic Invariants Validated Successfully!")

test_search_engine()""")

    # CELL 18: SECTION 4: DELIBERATE FAILURE ANALYSIS MARKDOWN
    add_md("""## 4. Student Lab Exercises & Deliberate Failure Analysis

Now comes the core pedagogical inquiry of Phase 1: **Where and why does Lexical Search break down?**

We will evaluate three controlled test cases:
1. **The Happy Path (Exact Token Overlap):** When the user speaks the exact vocabulary of the knowledge base.
2. **Synonym Blindness (Lexical Fragility):** When the user asks a completely natural question, but uses synonyms absent from the index.
3. **Polysemy & Context Blindness (False-Positive Retrieval):** When identical tokens represent completely distinct concepts in different domains.""")

    # CELL 19: TEST CASE 1: HAPPY PATH
    add_code("""# =============================================================================
# TEST CASE 1: THE HAPPY PATH (EXACT VOCABULARY MATCH)
# =============================================================================
# Query matches the exact vocabulary tokens in FAQ-02: 'bursar', 'fee', 'deposit', 'wire', 'remittance'

query_1 = "bursar fee deposit and wire remittance"
results_1 = search_campus_faq(query_1, vectorizer, doc_matrix, faq_df, top_k=3, threshold=0.15)
print_search_results(query_1, results_1)

# Diagnostic Analysis:
top_1 = results_1[0]
query_1_tokens = tokenize_and_clean(query_1)
doc_2_tokens = set(tokenize_and_clean(faq_df.loc[1, "full_text"]))
matched_overlap = [tok for tok in query_1_tokens if tok in doc_2_tokens]
token_recall = len(matched_overlap) / len(query_1_tokens)

print(f"\\n[Analysis Case 1]: Perfect lexical overlap achieved on tokens {top_1['matched_keywords']}.")
print(f"Query Token Recall against FAQ-02: {token_recall * 100:.1f}% ({matched_overlap})")
print(f"Cosine Similarity is strong ({top_1['cosine_score']:.4f}) because both TF and IDF align precisely.")
print(\"\"\"
[Pedagogical Note on Morphological Inflection]:
Notice that in un-stemmed lexical search, words must match their exact surface forms.
If the knowledge base contained 'remittances' and 'deposits' while the user searched 'remittance' and 'deposit',
exact string matching would fail without stemming (e.g., PorterStemmer) or lemmatization.
\"\"\")""")

    # CELL 20: TEST CASE 2: SYNONYM BLINDNESS
    add_code("""# =============================================================================
# TEST CASE 2: SYNONYM BLINDNESS / LEXICAL FRAGILITY (VOCABULARY MISMATCH)
# =============================================================================
# A natural student query using everyday words: "Where can I park my car?"
# The knowledge base entry FAQ-01 uses: "automobile storage permit authorization"

query_2a = "Where can I park my car?"
results_2a = search_campus_faq(query_2a, vectorizer, doc_matrix, faq_df, top_k=3, threshold=0.10)
print_search_results(query_2a, results_2a)

# Pedagogical Breakdown:
print("\\n" + "=" * 80)
print("[FAILURE DIAGNOSIS — TEST CASE 2: SYNONYM BLINDNESS]")
print("=" * 80)
print(f"User Query Cleaned Tokens : {tokenize_and_clean(query_2a)}")
print(f"FAQ-01 Ground Truth Tokens: {tokenize_and_clean(faq_df.loc[0, 'full_text'])}")
print(f"Top Similarity Score      : {results_2a[0]['cosine_score']:.4f}")
print(\"\"\"
EXPLANATION:
Even though a human understands that 'park my car' is 100% semantically identical to
'automobile storage', the TF-IDF dot product is EXACTLY 0.0000 because the intersection of 
query tokens {'park', 'car'} and document tokens {'automobile', 'storage', 'permit'} is EMPTY.

This is the classic 'Vocabulary Mismatch Problem' that motivates Dense Semantic Embeddings (Phase 2)!
\"\"\")""")

    # CELL 21: TEST CASE 3: POLYSEMY / CONTEXT BLINDNESS
    add_code("""# =============================================================================
# TEST CASE 3: POLYSEMY & CONTEXT BLINDNESS (FALSE-POSITIVE COLLISION)
# =============================================================================
# We submit two queries that share the single token "bank", but with totally different meanings:
# 1. A river bank (geographical / outdoor recreation)
# 2. A financial bank (wire transfers / banking institution)

query_river = "Can I fish or kayak on the campus river bank?"
query_finance = "Which commercial bank handles international tuition wires?"

print(">>> RUNNING QUERY A (Recreation - River Bank):")
res_river = search_campus_faq(query_river, vectorizer, doc_matrix, faq_df, top_k=2)
print_search_results(query_river, res_river)

print("\\n>>> RUNNING QUERY B (Finance - Banking Institution):")
res_finance = search_campus_faq(query_finance, vectorizer, doc_matrix, faq_df, top_k=2)
print_search_results(query_finance, res_finance)

print("\\n" + "=" * 80)
print("[FAILURE DIAGNOSIS — TEST CASE 3: POLYSEMY / CONTEXT COLLISION]")
print("=" * 80)
print(\"\"\"
Notice how the token 'bank' is shared between:
  - FAQ-12 (Commercial Bank / Tuition Wire Remittance)
  - FAQ-20 (Scenic River Bank / Fishing & Kayaking)

Because Bag-of-Words treats each token independently:
If a student asks: 'Can I fish on the bank?', the word 'bank' provides a positive TF-IDF
weight that partially activates FAQ-12 (Commercial Banking) as a false-positive candidate!
Lexical vectors have NO mechanism to realize that 'river bank' and 'investment bank' are
orthogonal semantic domains.
\"\"\")""")

    # CELL 22: SECTION 5: STUDENT LAB TASKS MARKDOWN
    add_md("""## 5. Student Lab Tasks (Hands-On Implementation)

To solidify your understanding of vector space mechanics and production engineering, complete the following two structured tasks:

---

### Task A: N-Gram Range Exploration & Dimensional Explosion
Currently, our vectorizer uses unigrams `ngram_range=(1, 1)`. When we expand the n-gram range to include bigrams `(1, 2)` or trigrams `(1, 3)`:
- We preserve local word order (e.g., `"river bank"` becomes a distinct feature from `"commercial bank"`).
- **The Tradeoff:** The vocabulary size $|V|$ explodes, and matrix sparsity increases dramatically.

**Your Objective:** Write a function `compare_ngram_dimensions(corpus_texts, ngram_configs)` that iterates over multiple n-gram settings, calculates vocabulary size, non-zero entries, and matrix sparsity, and displays the comparative table.

---

### Task B: Production Fallback Mechanism with Confidence Thresholding
In a live campus assistant, returning an irrelevant FAQ with a low similarity score (e.g., 0.04) causes user frustration and hallucinations.

**Your Objective:** Implement an enhanced retrieval function `search_with_fallback(...)` that:
1. Evaluates if the top retrieved result meets a confidence threshold $\tau$ (e.g., $\tau = 0.20$).
2. If `top_score < threshold`, returns a structured fallback payload directing the student to campus human support, rather than delivering incorrect information.""")

    # CELL 23: TASK A CODE
    add_code("""# =============================================================================
# STUDENT TASK A: N-GRAM RANGE & DIMENSIONAL EXPLOSION
# =============================================================================

def compare_ngram_dimensions(
    corpus_texts: List[str], 
    ngram_configs: List[Tuple[int, int]]
) -> pd.DataFrame:
    \"\"\"
    Analyzes how different n-gram ranges affect vocabulary size and matrix sparsity.

    Args:
        corpus_texts (List[str]): List of document strings.
        ngram_configs (List[Tuple[int, int]]): List of (min_n, max_n) tuples to evaluate.

    Returns:
        pd.DataFrame: Comparative metrics across n-gram configurations.
    \"\"\"
    records = []
    
    for (min_n, max_n) in ngram_configs:
        v = TfidfVectorizer(
            tokenizer=tokenize_and_clean,
            token_pattern=None,
            ngram_range=(min_n, max_n),
            norm="l2"
        )
        dtm = v.fit_transform(corpus_texts)
        
        num_docs, vocab_dim = dtm.shape
        nnz = dtm.nnz
        total_slots = num_docs * vocab_dim
        sparsity_pct = (1.0 - (nnz / total_slots)) * 100.0
        
        # Sample some bigram/trigram features if present
        features = v.get_feature_names_out()
        multi_word_features = [f for f in features if " " in f]
        sample_feature_str = ", ".join(multi_word_features[:3]) if multi_word_features else "None (Unigrams only)"
        
        records.append({
            "N-Gram Range": f"({min_n}, {max_n})",
            "Vocabulary Size (|V|)": vocab_dim,
            "Non-Zero Elements": nnz,
            "Matrix Sparsity (%)": round(sparsity_pct, 2),
            "Sample Higher-Order N-Grams": sample_feature_str
        })
        
    df_comparison = pd.DataFrame(records)
    return df_comparison

# Run Task A Comparison
configs = [(1, 1), (1, 2), (1, 3), (2, 2)]
ngram_results_df = compare_ngram_dimensions(faq_df["full_text"].tolist(), configs)
display(ngram_results_df)

print(\"\"\"
[Task A Key Takeaway]:
Notice that expanding from Unigrams (1, 1) to Unigrams+Bigrams (1, 2) more than doubles 
the vocabulary dimension |V|! On massive real-world corpora (millions of documents), 
n-gram expansion causes severe memory overhead and requires sparse matrix indexing optimizations.
\"\"\")""")

    # CELL 24: TASK B CODE
    add_code("""# =============================================================================
# STUDENT TASK B: ROBUST FALLBACK MECHANISM WITH THRESHOLDING
# =============================================================================

def search_with_fallback(
    query: str,
    vectorizer: TfidfVectorizer,
    doc_matrix: Any,
    faq_df: pd.DataFrame,
    top_k: int = 3,
    confidence_threshold: float = 0.20
) -> Dict[str, Any]:
    \"\"\"
    Performs FAQ retrieval with a robust confidence threshold and automated fallback.

    Args:
        query (str): The raw user query.
        vectorizer (TfidfVectorizer): The fitted vectorizer.
        doc_matrix (Any): The document-term matrix.
        faq_df (pd.DataFrame): The reference knowledge base.
        top_k (int): Number of top results to return if confident.
        confidence_threshold (float): Minimum cosine score for the top-1 result.

    Returns:
        Dict[str, Any]: Either high-confidence ranked results or fallback guidance.
    \"\"\"
    raw_results = search_campus_faq(
        query=query,
        vectorizer=vectorizer,
        doc_matrix=doc_matrix,
        faq_df=faq_df,
        top_k=top_k,
        threshold=confidence_threshold
    )
    
    top_result = raw_results[0] if raw_results else None
    top_score = top_result["cosine_score"] if top_result else 0.0
    
    # Check if top candidate satisfies the minimum confidence threshold
    if top_score < confidence_threshold:
        return {
            "status": "FALLBACK_TRIGGERED",
            "query": query,
            "max_confidence_score": top_score,
            "fallback_message": (
                "I'm sorry, I could not find a verified campus policy matching your exact wording. "
                "Please contact the Student Services Central Desk at help@campus.edu or (555) 019-9000."
            ),
            "suggested_actions": [
                "Try rephrasing your question using official administrative terms.",
                "Check the campus directory at directory.campus.edu.",
                "Visit North Hall Room 101 for in-person support."
            ],
            "low_confidence_candidates": raw_results
        }
    else:
        return {
            "status": "SUCCESS",
            "query": query,
            "max_confidence_score": top_score,
            "results": raw_results
        }

# Demonstrate Task B with an Out-of-Vocabulary / Unmatched Query
oov_query = "Where can I buy vegan gluten-free pizza near the football stadium?"
fallback_response = search_with_fallback(oov_query, vectorizer, doc_matrix, faq_df, confidence_threshold=0.20)

print(f"Query: '{oov_query}'")
print(f"Status: {fallback_response['status']}")
print(f"Max Cosine Score: {fallback_response['max_confidence_score']:.4f}")
print(f"System Message: {fallback_response.get('fallback_message')}")""")

    # CELL 25: SECTION 6: COMPREHENSIVE UNIT TEST SUITE
    add_code("""# =============================================================================
# SECTION 6: COMPREHENSIVE STUDENT SELF-CHECK TEST SUITE
# =============================================================================

def run_comprehensive_self_check():
    print("[Testing Suite] Initiating comprehensive verification checks...")
    
    # Test 1: Preprocessor idempotence and stop words
    s1 = "The university is closed for winter break!"
    toks1 = tokenize_and_clean(s1)
    assert "university" in toks1, "Content token 'university' must be retained."
    assert "the" not in toks1 and "is" not in toks1 and "for" not in toks1, "Stopwords must be stripped."
    print("  ✓ Check 1: Text sanitization and stop-word filtering verified.")
    
    # Test 2: Vocabulary bounds
    assert len(vectorizer.vocabulary_) > 100, "Vocabulary must contain over 100 domain tokens."
    print("  ✓ Check 2: Vocabulary dimension verified.")
    
    # Test 3: Cosine Similarity mathematical boundary conditions
    # Identical text must yield cosine similarity == 1.0 (within numerical float tolerance)
    exact_q = faq_df.loc[0, "full_text"]
    q_vec = vectorizer.transform([exact_q])
    sim_exact = cosine_similarity(q_vec, doc_matrix[0:1])[0][0]
    np.testing.assert_allclose(sim_exact, 1.0, rtol=1e-4, err_msg="Self-similarity of identical text must equal 1.0")
    print("  ✓ Check 3: Identity vector cosine projection (sim == 1.0) verified.")
    
    # Test 4: Orthogonal / Zero-overlap queries must yield similarity == 0.0
    gibberish_q = "xyzqwk123 nonexistingtoken999"
    res_gibberish = search_campus_faq(gibberish_q, vectorizer, doc_matrix, faq_df)
    assert res_gibberish[0]["cosine_score"] == 0.0, "Disjoint vocabulary query must yield exact 0.0 cosine similarity."
    print("  ✓ Check 4: Orthogonal disjoint vocabulary handling (sim == 0.0) verified.")
    
    # Test 5: Fallback trigger thresholding
    fb_res = search_with_fallback(gibberish_q, vectorizer, doc_matrix, faq_df, confidence_threshold=0.25)
    assert fb_res["status"] == "FALLBACK_TRIGGERED", "Fallback must trigger on zero-score queries."
    print("  ✓ Check 5: Automated fallback threshold logic verified.")
    
    print("\\n" + "=" * 80)
    print("🎉 ALL SELF-CHECK UNIT TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80)

run_comprehensive_self_check()""")

    # CELL 26: CONCLUSION & TRANSITION TO PHASE 2
    add_md("""## 6. Summary, Critical Reflection & Transition to Phase 2

### Summary of Phase 1 Key Findings

| Metric / Dimension | Lexical Search (TF-IDF + Cosine Similarity) |
| :--- | :--- |
| **Computational Complexity** | Highly efficient: sparse dot-product inference in $\\mathcal{O}(|V_{\\text{active}}|)$ time. |
| **Training Requirements** | Unsupervised; no GPU or neural gradient updates required. |
| **Explainability** | $100\\%$ interpretable via individual term TF-IDF dot-product attribution. |
| **Synonym Handling** | **Fails completely ($0.0$ similarity)** without manual synonym thesauri. |
| **Polysemy Handling** | **Blind to context** (matches ambiguous words like *"bank"* across unrelated domains). |
| **Word Order / Negation** | Ignored by Bag-of-Words assumption. |

---

### Student Reflection Questions (Pre-Lab for Phase 2)
1. If we added stemming (e.g., Porter Stemmer) or lemmatization (WordNet Lemmatizer) to our pipeline, which failure modes would be mitigated, and which would remain unsolved?
2. Why can't we simply build an exhaustive dictionary of all synonyms for every word on a university campus? (Consider domain drift, slang, polysemy, and maintenance overhead).
3. How do continuous dense embedding spaces (e.g., Sentence Transformers / SBERT) represent the sentence *"Where can I park my car?"* such that its cosine similarity to *"automobile storage permits"* is $>0.85$, despite sharing **zero** vocabulary tokens?

---

**Next Up — Phase 2:** *Dense Semantic Representations, Sentence-BERT Bi-Encoders & Approximate Nearest Neighbors (ANN).*""")

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
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(base_dir, "Phase_1_Lexical_Search_and_Rule_Based_Matching.ipynb")
    path2 = os.path.join(base_dir, ".ipynb")

    with open(path1, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    with open(path2, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"[SUCCESS] Wrote notebook with {len(cells)} cells to {path1} and {path2}")

if __name__ == "__main__":
    create_notebook()
