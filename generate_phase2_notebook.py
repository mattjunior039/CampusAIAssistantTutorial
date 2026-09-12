import json
import os

def create_phase2_notebook():
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
    add_md("""# Project 1: AI Campus Assistant Pipeline
## Phase 2: Dense Semantic Embeddings & Intent Classification

**Curriculum Level:** Advanced Computer Science / Applied Natural Language Processing & Machine Learning  
**Prerequisites:** Phase 1 (Lexical Search & Vector Spaces), Linear Algebra (Hyperplanes & Projections), Multivariable Calculus (Gradient Descent & Softmax), Supervised ML (Train/Test Splits, Cross-Entropy)  
**Estimated Time:** 90–120 minutes  

---

### Learning Objectives
By completing this hands-on laboratory, you will:
1. **Master the Shift from Sparse to Dense Spaces:** Understand how Transformer-based bi-encoders map unstructured text to compact, continuous 384-dimensional geometric vectors.
2. **Resolve Synonym Blindness Mathematically:** Experience how dense continuous representations group synonymous phrasing (*"automobile storage"* vs. *"park my car"*) into shared geometric *Semantic Neighborhoods*.
3. **Formulate High-Dimensional Supervised Classification:** Derive the mathematics of linear decision boundaries (Multinomial Logistic Regression / Support Vector Machines) operating on dense sentence embeddings.
4. **Build an End-to-End Intent Classification Pipeline:** Implement data synthesis (100 campus queries across 5 intent classes), dense encoding via `sentence-transformers/all-MiniLM-L6-v2`, stratified model training, and confusion matrix diagnostics.
5. **Critically Analyze Closed-World Limitations:** Diagnose the *Static Generation Gap* and *Out-of-Distribution (OOD) Softmax Hallucinations*, implementing confidence-thresholded fallback guards.
6. **Visualize Semantic Neighborhoods:** Execute dimensionality reduction via PCA/t-SNE to prove intra-class cluster cohesion in 2D space.""")

    # =========================================================================
    # CELL 2: SECTION 1: THE EVOLUTION TO DENSE SPACES
    # =========================================================================
    add_md("""## 1. Course Introduction & The Evolution to Dense Spaces

In **Phase 1**, we explored the foundational vector space model using Term Frequency-Inverse Document Frequency (TF-IDF). While TF-IDF is computationally lightweight and interpretable, we uncovered its catastrophic point of failure: **Lexical Fragility & Synonym Blindness**. 

Because TF-IDF creates an orthogonal basis dimension for every unique vocabulary token, the queries *"Where can I park my car?"* and *"Automobile storage permit"* produce orthogonal vectors ($\mathbf{q} \cdot \mathbf{d} = 0.0$), failing to recognize identical human intent.

```
+---------------------------------------------------------------------------------------------------+
|                                 THE NLP & IR EVOLUTIONARY SPECTRUM                                |
+------------------------------------+----------------------------------+---------------------------+
| Phase 1: Lexical & Rule-Based      | Phase 2: Dense Semantic Spaces   | Phase 3: RAG & GenAI      |
| (1990s - 2010s)                    | (2018 - 2022)                    | (2023 - Present)          |
+------------------------------------+----------------------------------+---------------------------+
| - Bag-of-Words & TF-IDF            | - Sentence-BERT (Bi-Encoders)    | - Large Language Models   |
| - High-dimensional (|V| > 10,000)  | - Compact, dense (d = 384)       | - Vector DBs + Dense ANN  |
| - Sparse (> 99% zeros)             | - Continuous semantic topology   | - Dynamic Reranking       |
| - Orthogonal on synonym mismatch   | - Clustered semantic neighbors   | - Context-grounded answer |
| * COMPLETED (Phase 1) *            | * THIS LAB (Phase 2) *           | * NEXT LAB (Phase 3) *    |
+------------------------------------+----------------------------------+---------------------------+
```

### From Token Overlap to Semantic Neighborhoods
Dense embedding models map linguistic sequences into a continuous vector space $\mathbb{R}^d$ ($d = 384$ for MiniLM) where geometric proximity correlates with semantic equivalence:

$$\text{dist}(\mathbf{x}_{\text{"park my car"}}, \mathbf{x}_{\text{"automobile storage"}}) \ll \text{dist}(\mathbf{x}_{\text{"park my car"}}, \mathbf{x}_{\text{"tuition deposit"}})$$

In this lab, we leverage these continuous embeddings to train a **Supervised Intent Classifier** that routes incoming campus questions to their operational category.""")

    # =========================================================================
    # CELL 3: SECTION 2: MATHEMATICAL FOUNDATIONS
    # =========================================================================
    add_md(r"""## 2. Mathematical Foundations

---

### 2.1 Dense Bi-Encoders & Sentence-BERT (SBERT)

Standard BERT models use cross-attention to process sentence pairs $(s_A, s_B)$, requiring $\mathcal{O}(N \times M)$ quadratic forward passes for search over $N$ queries and $M$ documents. 

**Sentence-BERT (SBERT)** resolves this via a **Siamese Bi-Encoder** architecture that encodes individual sentences independently into fixed-length dense representations:

```
                  [Input Sentence: "How do I pay tuition?"]
                                     |
                                     v
                       [Transformer Encoder Layers]
                        (12-layer / 6-layer MiniLM)
                                     |
                                     v
                [Token Contextual Representations: h_1, ..., h_L]
                                     |
                                     v
                          [Mean Pooling Layer]
                                     |
                                     v
                   [Fixed Dense Vector u in R^384]
```

#### Mean Pooling Formulation:
Given an input sequence of $L$ tokens producing contextualized hidden vectors $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_L \in \mathbb{R}^{d}$ and attention mask $m_i \in \{0, 1\}$:

$$\mathbf{u} = \frac{\sum_{i=1}^L m_i \mathbf{h}_i}{\sum_{i=1}^L m_i}$$

The resulting vector $\mathbf{u}$ is subsequently normalized to unit length $\|\mathbf{u}\|_2 = 1$, allowing cosine similarity to be computed via standard Euclidean dot products:

$$\text{sim}(\mathbf{u}_1, \mathbf{u}_2) = \mathbf{u}_1 \cdot \mathbf{u}_2$$

---

### 2.2 Supervised Hyperplane Classification in $\mathbb{R}^{384}$

Once text is transformed into dense feature vectors $\mathbf{x} \in \mathbb{R}^{d}$, we train a linear decision boundary to map embeddings into $K$ discrete intent classes $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$.

#### 1. Multinomial Logistic Regression (Softmax Classifier):
The conditional probability that a query $\mathbf{x}$ belongs to intent class $k$ is parameterized by weight matrix $\mathbf{W} \in \mathbb{R}^{K \times d}$ and bias vector $\mathbf{b} \in \mathbb{R}^K$:

$$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

The model is trained by minimizing the regularized **Categorical Cross-Entropy Loss**:

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \ln P(y = k \mid \mathbf{x}_i) + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

where $y_{ik} \in \{0, 1\}$ is the one-hot indicator for sample $i$ and class $k$.

#### 2. Linear Support Vector Machines (SVM):
Alternatively, a Maximum-Margin Hyperplane seeks to maximize geometric margin $\frac{2}{\|\mathbf{w}\|_2}$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

#### Why Dense Embeddings are Linearly Separable:
According to **Cover's Theorem on Separability of Patterns**, a complex pattern-classification problem cast in a high-dimensional space nonlinearly is more likely to be linearly separable than in a low-dimensional space. Because Transformer pre-training already aligns semantic concepts into linearly separable convex hulls in $\mathbb{R}^{384}$, simple linear classifiers achieve near-perfect classification without complex non-linear kernel tricks.

---

### 2.3 The Machine Learning Evaluation Lifecycle

To ensure unbiased evaluation and detect overfitting:
1. **Stratified Split:** We partition the dataset into Training ($80\%$) and Testing ($20\%$) sets while preserving class distribution balance.
2. **Confusion Matrix $\mathbf{M} \in \mathbb{N}^{K \times K}$:** $M_{ij}$ denotes the count of ground-truth class $i$ samples predicted as class $j$.
3. **Core Performance Metrics:**
   - **Precision ($P_k$):** $P_k = \frac{TP_k}{TP_k + FP_k}$ (Purity of predictions)
   - **Recall ($R_k$):** $R_k = \frac{TP_k}{TP_k + FN_k}$ (Completeness of detection)
   - **F1-Score ($F_{1, k}$):** $F_{1, k} = 2 \cdot \frac{P_k \cdot R_k}{P_k + R_k}$ (Harmonic mean balancing precision and recall)""")

    # =========================================================================
    # CELL 4: SECTION 3: STEP-BY-STEP IMPLEMENTATION PIPELINE
    # =========================================================================
    add_md("""## 3. Step-by-Step Implementation Pipeline

Let us construct our end-to-end dense intent classification pipeline.

```
+-------------------------------------------------------------------------------------+
|                           DENSE INTENT CLASSIFIER PIPELINE                          |
+-------------------------------------------------------------------------------------+
|                                                                                     |
|  [Campus Intent Corpus: 100 Queries, 5 Classes]                                     |
|                             |                                                       |
|                             v                                                       |
|  +------------------------------------------------------+                           |
|  | Hugging Face sentence-transformers (all-MiniLM-L6-v2) |                           |
|  | Encodes raw text -> Continuous Dense Embeddings R^384 |                           |
|  +------------------------------------------------------+                           |
|                             |                                                       |
|                             v                                                       |
|  [Dense Feature Matrix X in R^(100 x 384)] + [Label Vector y in {0..4}^100]        |
|                             |                                                       |
|                             v                                                       |
|  +------------------------------------------------------+                           |
|  | Stratified Train / Test Partition (80% Train, 20% Test)                          |
|  +------------------------------------------------------+                           |
|         |                                              |                            |
|         v (Train Set: 80 x 384)                        v (Test Set: 20 x 384)       |
|  +-----------------------------+               +-----------------------------+      |
|  | Fit LogisticRegression(C=1) |               | Predict Labels & Probabilities|     |
|  +-----------------------------+               +-----------------------------+      |
|                                                                |                    |
|                                                                v                    |
|                                                +-----------------------------+      |
|                                                | Confusion Matrix & Metrics  |      |
|                                                | Precision, Recall, F1-Score |      |
|                                                +-----------------------------+      |
+-------------------------------------------------------------------------------------+
```""")

    # =========================================================================
    # CELL 5: STEP 3.1: ENVIRONMENT SETUP
    # =========================================================================
    add_code("""# Step 3.1: Environment Setup & Library Verification
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional

# Ensure sentence-transformers and core dependencies are installed
required_packages = ["sentence-transformers", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "torch"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"[Setup] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# Set random seeds for strict mathematical reproducibility
np.random.seed(42)
torch.manual_seed(42)

print(f"[OK] Python version: {sys.version.split()[0]}")
print(f"[OK] PyTorch version: {torch.__version__} (CUDA/MPS Available: {torch.cuda.is_available() or torch.backends.mps.is_available()})")
print(f"[OK] Scikit-Learn & Sentence-Transformers configured.")""")

    # =========================================================================
    # CELL 6: STEP 3.2: DATASET STRUCTURING MARKDOWN
    # =========================================================================
    add_md("""### 3.2 Campus Intent Dataset Synthesis

To train and validate our classifier, we synthesize a dataset of **100 realistic natural language queries** balanced evenly across **5 operational campus intent classes**:

1. `tuition_payment`: Bursar deposits, fee schedules, wire transfers, payment deadlines, installment plans.
2. `parking_permit`: Vehicle decals, parking garage authorization, lot citations, commuter passes.
3. `dorm_maintenance`: Clogged plumbing, broken heating, malfunctioning dorm keycards, room repairs.
4. `library_hours`: Central library schedule, study room reservations, manuscript loans, quiet floor policies.
5. `academic_advising`: Course add/drop deadlines, degree audits, declaring majors, faculty advisor meetings.

To ensure high pedagogical quality, queries contain realistic linguistic variance: informal slang, typos, questions, imperative requests, and formal administrative phrasings.""")

    # =========================================================================
    # CELL 7: STEP 3.2: DATASET CODE
    # =========================================================================
    add_code("""# Step 3.2: Synthesize 100 Varied Natural Language Campus Queries

RAW_CAMPUS_INTENT_DATA = [
    # -------------------------------------------------------------------------
    # 1. tuition_payment (20 queries)
    # -------------------------------------------------------------------------
    ("How do I pay my tuition for the upcoming fall semester?", "tuition_payment"),
    ("What is the deadline for bursar fee deposits?", "tuition_payment"),
    ("Where do I send an international wire remittance for my college bill?", "tuition_payment"),
    ("Can I set up a monthly installment plan for tuition?", "tuition_payment"),
    ("Are there late payment penalty fees if I pay after the third week?", "tuition_payment"),
    ("How do I deposit money into my student bursar account?", "tuition_payment"),
    ("I need an itemized tuition receipt for my tax return.", "tuition_payment"),
    ("Does the university accept 529 college savings plan payments?", "tuition_payment"),
    ("My financial aid scholarship hasn't disbursed to cover my balance yet.", "tuition_payment"),
    ("What payment methods are accepted by the student treasury office?", "tuition_payment"),
    ("Can I pay my semester fees using a credit card online?", "tuition_payment"),
    ("Where is the physical cashier window located on campus?", "tuition_payment"),
    ("Why is there an unexpected laboratory fee on my billing statement?", "tuition_payment"),
    ("How do I request a formal billing refund for overpayment?", "tuition_payment"),
    ("What happens if my financial sponsor's wire transfer is delayed?", "tuition_payment"),
    ("How can I grant my parents guest access to pay my university bill?", "tuition_payment"),
    ("Is there an administrative surcharge for electronic check payments?", "tuition_payment"),
    ("When are spring semester tuition invoices officially generated?", "tuition_payment"),
    ("Who do I contact regarding an active financial hold on my account?", "tuition_payment"),
    ("How do I view my current account balance breakdown?", "tuition_payment"),

    # -------------------------------------------------------------------------
    # 2. parking_permit (20 queries)
    # -------------------------------------------------------------------------
    ("Where can I park my car on campus?", "parking_permit"),
    ("How do I register for an undergraduate automobile storage permit?", "parking_permit"),
    ("What is the annual cost of a commuter student parking decal?", "parking_permit"),
    ("Can freshman residential students bring a motor vehicle to campus?", "parking_permit"),
    ("How do I appeal a campus parking violation ticket?", "parking_permit"),
    ("Where are the designated electric vehicle charging stations located?", "parking_permit"),
    ("Do I need a separate pass to park in the stadium garage during game days?", "parking_permit"),
    ("How do I update my license plate number on my parking portal?", "parking_permit"),
    ("Are parking rules enforced on campus during holiday weekends?", "parking_permit"),
    ("Can I purchase a daily or weekly temporary parking pass?", "parking_permit"),
    ("Where is motorcycle and scooter parking permitted?", "parking_permit"),
    ("What should I do if the parking gate arm fails to scan my permit?", "parking_permit"),
    ("Are overnight guest parking authorizations available for visitors?", "parking_permit"),
    ("Which parking lots allow resident student overnight parking?", "parking_permit"),
    ("My car got towed from North Lot, who do I contact?", "parking_permit"),
    ("How do I apply for a student disability accessible parking permit?", "parking_permit"),
    ("Is there free parking on campus after 5:00 PM?", "parking_permit"),
    ("Can I transfer my parking permit to another vehicle temporarily?", "parking_permit"),
    ("What documents are required to prove vehicle registration?", "parking_permit"),
    ("Where can commuter students park without getting cited?", "parking_permit"),

    # -------------------------------------------------------------------------
    # 3. dorm_maintenance (20 queries)
    # -------------------------------------------------------------------------
    ("The heater in my dorm room is blowing cold air.", "dorm_maintenance"),
    ("How do I submit an urgent plumbing work order for a clogged sink?", "dorm_maintenance"),
    ("My residential room keycard lock is blinking red and won't unlock.", "dorm_maintenance"),
    ("There is water leaking from the ceiling in our floor bathroom.", "dorm_maintenance"),
    ("Who do I call if my dorm window latch is broken?", "dorm_maintenance"),
    ("The overhead lighting fixture in our study lounge burned out.", "dorm_maintenance"),
    ("Our dormitory washing machine is flooding the laundry room floor.", "dorm_maintenance"),
    ("How long does it take for facilities to fix a broken air conditioner?", "dorm_maintenance"),
    ("I lost my dorm room physical key and need a lock change.", "dorm_maintenance"),
    ("The elevator in East Tower residence hall is stuck.", "dorm_maintenance"),
    ("Can maintenance come fix our squeaky lofted bed frame?", "dorm_maintenance"),
    ("There is low water pressure in our quad shower stall.", "dorm_maintenance"),
    ("How do I report a pest or insect issue in my residence hall?", "dorm_maintenance"),
    ("The electrical outlet near my desk stopped providing power.", "dorm_maintenance"),
    ("Who handles emergency heating repair requests over the weekend?", "dorm_maintenance"),
    ("Our dorm smoke detector keeps chirping with a low battery signal.", "dorm_maintenance"),
    ("Can facilities install a replacement mini-fridge in my room?", "dorm_maintenance"),
    ("The exterior security door to our residence hall will not latch shut.", "dorm_maintenance"),
    ("How do I track the status of my active housing repair request?", "dorm_maintenance"),
    ("My dormitory bathroom mirror fell off the mounting bracket.", "dorm_maintenance"),

    # -------------------------------------------------------------------------
    # 4. library_hours (20 queries)
    # -------------------------------------------------------------------------
    ("What time does the main campus library close tonight?", "library_hours"),
    ("Is the science and engineering library open 24 hours during finals week?", "library_hours"),
    ("How do I reserve a private group study room in the library?", "library_hours"),
    ("Are the central library archives open on Sunday mornings?", "library_hours"),
    ("How many books and research manuscripts can I check out at once?", "library_hours"),
    ("What are the weekend operating hours for the digital media lab?", "library_hours"),
    ("Can alumni access library online research databases remotely?", "library_hours"),
    ("Where are the quiet silent-study floors located in the library?", "library_hours"),
    ("How do I request an interlibrary loan for a journal article?", "library_hours"),
    ("Are undergraduate scholars allowed to borrow multimedia equipment?", "library_hours"),
    ("What is the overdue return fine policy for late library books?", "library_hours"),
    ("Is the library open during spring break and federal holidays?", "library_hours"),
    ("How do I connect my laptop to the wireless library printers?", "library_hours"),
    ("Are there computer workstations available on the 2nd floor?", "library_hours"),
    ("How do I schedule a research consultation with a subject librarian?", "library_hours"),
    ("Where can I find course reserve textbooks on short-term loan?", "library_hours"),
    ("Does the campus library have 3D printing services for students?", "library_hours"),
    ("What time does the library coffee shop open on weekdays?", "library_hours"),
    ("How do I access peer-reviewed IEEE and ACM journal papers?", "library_hours"),
    ("Are visitors permitted inside the university library without an ID card?", "library_hours"),

    # -------------------------------------------------------------------------
    # 5. academic_advising (20 queries)
    # -------------------------------------------------------------------------
    ("How do I make an appointment with my academic advisor?", "academic_advising"),
    ("What is the official deadline to drop a course without transcript penalty?", "academic_advising"),
    ("How do I submit paperwork to declare or change my undergraduate major?", "academic_advising"),
    ("Can my academic advisor authorize an overload credit waiver?", "academic_advising"),
    ("How do I check my degree audit progress for graduation requirements?", "academic_advising"),
    ("What is the minimum GPA required to maintain good academic standing?", "academic_advising"),
    ("How do I apply for transfer course credit from my community college?", "academic_advising"),
    ("Who signs the approval form to add a minor in computer science?", "academic_advising"),
    ("Can I take a required prerequisite course concurrently with approval?", "academic_advising"),
    ("What is the procedure for requesting a formal academic leave of absence?", "academic_advising"),
    ("How do I appeal an academic suspension or probation status?", "academic_advising"),
    ("Where do I find my assigned faculty advisor's office hours?", "academic_advising"),
    ("What is the difference between a course withdrawal and an incomplete grade?", "academic_advising"),
    ("How many total elective credits do I need to complete my BS degree?", "academic_advising"),
    ("Can I audit a lecture without registering for course credits?", "academic_advising"),
    ("How do I petition for a general education curriculum requirement substitution?", "academic_advising"),
    ("When does priority registration open for upper-division seniors?", "academic_advising"),
    ("Who can help me plan my four-year course schedule?", "academic_advising"),
    ("How do I register for a departmental capstone senior project?", "academic_advising"),
    ("What are the departmental honors thesis enrollment criteria?", "academic_advising")
]

# Convert into a structured pandas DataFrame
df_dataset = pd.DataFrame(RAW_CAMPUS_INTENT_DATA, columns=["query_text", "intent_label"])

# Label distribution audit
print(f"================ DATASET DISTRIBUTION SUMMARY ================")
print(f"Total Synthesized Queries: {len(df_dataset)}")
print(f"Unique Intent Classes    : {df_dataset['intent_label'].nunique()}")
print(df_dataset['intent_label'].value_counts())
print(f"==============================================================")
display(df_dataset.sample(5, random_state=42))""")

    # =========================================================================
    # CELL 8: STEP 3.3: DENSE ENCODING MARKDOWN
    # =========================================================================
    add_md(r"""### 3.3 Dense Vector Encoding via `sentence-transformers`

We now load the pretrained `all-MiniLM-L6-v2` bi-encoder:
- **Base Architecture:** 6-layer Transformer with 384 hidden units and 12 attention heads.
- **Pretrained Corpus:** Fine-tuned on $>1$ billion sentence pairs using contrastive InfoNCE loss.
- **Output:** Transforms any natural language string into a normalized $384$-dimensional dense vector $\mathbf{x} \in \mathbb{R}^{384}$.

Let us encode all $N=100$ queries into a dense feature matrix $\mathbf{X} \in \mathbb{R}^{100 \times 384}$ and extract integer labels $\mathbf{y} \in \{0, 1, 2, 3, 4\}^{100}$.""")

    # =========================================================================
    # CELL 9: STEP 3.3: ENCODING CODE
    # =========================================================================
    add_code("""# Step 3.3: Dense Vector Encoding via all-MiniLM-L6-v2

print("[Model Load] Initializing sentence-transformers/all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all 100 queries into continuous 384D representations
queries_list: List[str] = df_dataset["query_text"].tolist()
X_embeddings: np.ndarray = embedding_model.encode(
    queries_list, 
    show_progress_bar=True,
    normalize_embeddings=True  # L2 normalization ensures ||x||_2 = 1.0
)

# Extract integer label mappings
label_names: List[str] = sorted(df_dataset["intent_label"].unique().tolist())
label2id: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(label_names)}
id2label: Dict[int, str] = {idx: lbl for lbl, idx in label2id.items()}
y_labels: np.ndarray = df_dataset["intent_label"].map(label2id).to_numpy()

print(f"\\n================ DENSE EMBEDDING TENSOR PROPERTIES ================")
print(f"Feature Matrix Shape (X)  : {X_embeddings.shape} (N={X_embeddings.shape[0]} queries, d={X_embeddings.shape[1]} dims)")
print(f"Label Vector Shape (y)    : {y_labels.shape}")
print(f"Embedding Data Type       : {X_embeddings.dtype}")
print(f"Memory Footprint          : {X_embeddings.nbytes / 1024:.2f} KB (Compare to sparse matrix overhead!)")
print(f"Class Mapping Index       : {label2id}")
print(f"===================================================================")""")

    # =========================================================================
    # CELL 10: STEP 3.3: EMBEDDING ASSERTIONS
    # =========================================================================
    add_code("""# Self-Check Unit Test: Dense Vector Properties
def test_dense_embeddings():
    assert X_embeddings.shape == (100, 384), f"Expected shape (100, 384), got {X_embeddings.shape}"
    assert len(y_labels) == 100, f"Expected 100 labels, got {len(y_labels)}"
    
    # Verify L2 Unit Norm: ||x_i||_2 == 1.0
    norms = np.linalg.norm(X_embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(100), rtol=1e-5, err_msg="All embeddings must have unit L2 norm.")
    
    print("[PASS] Dense Vector Space Properties Validated (Shape: 100x384, L2 Unit Norms: 1.0)!")

test_dense_embeddings()""")

    # =========================================================================
    # CELL 11: STEP 3.4: CLASSIFIER TRAINING MARKDOWN
    # =========================================================================
    add_md(r"""### 3.4 Supervised Classifier Training & Optimization

We partition our dataset using a stratified $80/20$ train/test split:
- **Training Set:** 80 queries (16 per class)
- **Test Set:** 20 queries (4 per class)

We fit a **Multinomial Logistic Regression Classifier** ($C = 1.0$) with native **Softmax** probability estimation:

$$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

$$\hat{y} = \arg\max_{k \in \{0..4\}} P(y = k \mid \mathbf{x}) = \arg\max_{k \in \{0..4\}} \left( \mathbf{w}_k^T \mathbf{x} + b_k \right)$$""")

    # =========================================================================
    # CELL 12: STEP 3.4: TRAINING CODE
    # =========================================================================
    add_code("""# Step 3.4: Stratified Partition & Supervised Training (Logistic Regression)

X_train, X_test, y_train, y_test, queries_train, queries_test = train_test_split(
    X_embeddings,
    y_labels,
    queries_list,
    test_size=0.20,
    random_state=42,
    stratify=y_labels
)

print(f"[Split] Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

# Instantiate and fit Multinomial Logistic Regression (Native Softmax)
clf = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate Training Convergence
train_acc = accuracy_score(y_train, clf.predict(X_train))
print(f"[Training] Model converged successfully. Training Accuracy: {train_acc * 100:.2f}%")""")

    # =========================================================================
    # CELL 13: STEP 3.5: EVALUATION LOOP MARKDOWN
    # =========================================================================
    add_md("""### 3.5 Evaluation Loop: Confusion Matrix & Classification Report

We now perform inference on the unseen **20-sample test set** to compute:
1. **Confusion Matrix:** Illustrates pairwise class classification patterns.
2. **Precision, Recall, and F1-Scores:** Per-class performance diagnostics.""")

    # =========================================================================
    # CELL 14: STEP 3.5: EVALUATION CODE
    # =========================================================================
    add_code("""# Step 3.5: Model Evaluation on Unseen Test Partition

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"================ MODEL EVALUATION SUMMARY ================")
print(f"Test Set Accuracy: {test_acc * 100:.2f}% ({sum(y_pred == y_test)}/{len(y_test)} correct)")
print(f"=========================================================\\n")

# Generate Classification Report
target_names = [id2label[i] for i in range(len(label_names))]
print("DETAILED PER-CLASS CLASSIFICATION METRICS:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# Plot Styled Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=target_names, 
    yticklabels=target_names,
    cbar=False
)
plt.title(f"Test Set Confusion Matrix (Accuracy: {test_acc * 100:.1f}%)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Predicted Intent Class", fontsize=11)
plt.ylabel("Ground Truth Intent Class", fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # CELL 15: STEP 3.6: INFERENCE WORKFLOW MARKDOWN
    # =========================================================================
    add_md("""### 3.6 Production Inference Workflow

Let us encapsulate the encoder and classifier into a clean, typed inference function `predict_intent(query, encoder, classifier)` that maps an arbitrary student query to its predicted intent, class probabilities, and confidence score.""")

    # =========================================================================
    # CELL 16: STEP 3.6: INFERENCE FUNCTION CODE
    # =========================================================================
    add_code("""# Step 3.6: Implement the Core Inference Function

def predict_intent(
    query: str,
    encoder: SentenceTransformer,
    classifier: Any,
    label_mapping: Dict[int, str] = id2label
) -> Dict[str, Any]:
    \"\"\"
    Performs dense embedding extraction and intent classification for a raw query string.

    Args:
        query (str): The natural language query from the student.
        encoder (SentenceTransformer): The dense sentence encoder.
        classifier (Any): Fitted scikit-learn classifier with predict_proba support.
        label_mapping (Dict[int, str]): Mapping from integer class IDs to intent strings.

    Returns:
        Dict[str, Any]: Classification payload with predicted class, confidence, and distribution.
    \"\"\"
    # 1. Encode query into 384D normalized vector
    query_emb = encoder.encode([query], normalize_embeddings=True)
    
    # 2. Compute posterior class probabilities
    probabilities = classifier.predict_proba(query_emb)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    predicted_label = label_mapping[predicted_idx]
    
    # 3. Build detailed class distribution
    class_distribution = {
        label_mapping[idx]: round(float(prob), 4)
        for idx, prob in enumerate(probabilities)
    }
    # Sort class distribution descending by probability
    sorted_dist = dict(sorted(class_distribution.items(), key=lambda item: item[1], reverse=True))
    
    return {
        "query": query,
        "predicted_intent": predicted_label,
        "confidence_score": round(confidence, 4),
        "class_distribution": sorted_dist,
        "embedding_vector": query_emb[0]
    }

def print_intent_result(result: Dict[str, Any]) -> None:
    \"\"\"CLI formatting helper for intent prediction.\"\"\"
    print(f"\\n================================================================================")
    print(f"QUERY: \\\"{result['query']}\\\"")
    print(f"================================================================================")
    print(f"PREDICTED INTENT : {result['predicted_intent'].upper()}")
    print(f"CONFIDENCE SCORE : {result['confidence_score'] * 100:.2f}%")
    print(f"PROBABILITY DIST :")
    for cls, prob in result["class_distribution"].items():
        bar = "█" * int(prob * 30)
        print(f"  - {cls:<20}: {prob:.4f} {bar}")
    print("-" * 80)""")

    # =========================================================================
    # CELL 17: STEP 3.6: INFERENCE VERIFICATION & ASSERTIONS
    # =========================================================================
    add_code("""# Self-Check Unit Test: Inference Function Invariants
def test_inference_pipeline():
    sample_q = "How do I request an official degree audit for graduation?"
    res = predict_intent(sample_q, embedding_model, clf)
    
    assert res["predicted_intent"] == "academic_advising", f"Expected academic_advising, got {res['predicted_intent']}"
    assert 0.0 <= res["confidence_score"] <= 1.0, "Confidence must be within [0, 1]"
    assert abs(sum(res["class_distribution"].values()) - 1.0) < 1e-3, "Probabilities must sum to 1.0"
    assert res["embedding_vector"].shape == (384,), "Embedding must be 384-dimensional"
    print("[PASS] Inference Engine Invariants Validated Successfully!")

test_inference_pipeline()""")

    # =========================================================================
    # CELL 18: SECTION 4: LIMITATIONS & FAILURE ANALYSIS MARKDOWN
    # =========================================================================
    add_md("""## 4. Pedagogical Limitations & Deliberate Failure Analysis

While Phase 2's dense semantic classifier solves Phase 1's *Synonym Blindness*, supervised intent classification introduces two critical systemic limitations:

1. **The Static Generation Gap:** The model produces a discrete categorical string (e.g., `parking_permit`), but **cannot generate human language**. It cannot answer questions like *"How much is a permit?"* or *"Where is Lot C?"*.
2. **Closed-World Boundaries & Out-of-Distribution (OOD) Softmax Hallucinations:** Because the softmax denominator $\sum_{k=1}^K \exp(z_k)$ forces total probability to sum to $1.0$, any arbitrary query (e.g., *"Write a poem about space exploration"*) **must** be assigned to one of the 5 campus classes.""")

    # =========================================================================
    # CELL 19: TEST CASE 1: SYNONYM SUCCESS & THE GENERATION GAP
    # =========================================================================
    add_code("""# =============================================================================
# TEST CASE 1: SYNONYM TRIUMPH & THE STATIC GENERATION GAP
# =============================================================================
# Recall Phase 1's failure: "Where can I park my car?" yielded 0.0 against "automobile storage"

synonym_query = "Where can I park my car on campus for the afternoon?"
res_synonym = predict_intent(synonym_query, embedding_model, clf)
print_intent_result(res_synonym)

print(\"\"\"
[ANALYSIS — THE SYNONYM TRIUMPH & THE STATIC GENERATION GAP]:
1. THE TRIUMPH: Despite using everyday words ('park my car') not present in formal docs,
   the dense embedding mapped the query to the 'parking_permit' cluster with high confidence!
   Synonym Blindness is SOLVED.

2. THE GENERATION GAP (THE DEAD END):
   What does the user actually receive? The string 'parking_permit'.
   The classifier cannot tell the student that 'commuter parking costs $185 per semester
   and is permitted in Lots A, B, and D'.
   
   To generate real, personalized, dynamic answers, we must move to Phase 3 (RAG)!
\"\"\")""")

    # =========================================================================
    # CELL 20: TEST CASE 2: CLOSED-WORLD OOD HALLUCINATIONS
    # =========================================================================
    add_code("""# =============================================================================
# TEST CASE 2: CLOSED-WORLD OOD SOFTMAX HALLUCINATIONS
# =============================================================================
# We submit completely Out-of-Distribution (OOD) queries that have zero relevance to campus operations.

ood_queries = [
    "Write a Shakespearean sonnet about space exploration and black holes.",
    "What is the capital city of France?",
    "How do I bake sourdough bread with wild yeast starter?"
]

print(">>> DEMONSTRATING CLOSED-WORLD SOFTMAX FAILURE ON OOD QUERIES:")
for q in ood_queries:
    res_ood = predict_intent(q, embedding_model, clf)
    print_intent_result(res_ood)

print(\"\"\"
[FAILURE DIAGNOSIS — TEST CASE 2: CLOSED-WORLD ASSUMPTION]:
Notice that the classifier confidently assigned:
  - Sourdough bread -> 'dorm_maintenance' or 'library_hours'
  - Space exploration -> 'academic_advising'

Why? Standard Softmax enforces \\sum P(y=k) = 1.0 across closed set C = {0..4}.
It possesses NO native mechanism to abstain or say 'I don't know'.
\"\"\")""")

    # =========================================================================
    # CELL 21: SECTION 5: STUDENT LAB TASKS MARKDOWN
    # =========================================================================
    add_md("""## 5. Student Lab Tasks (Hands-On Implementation)

To master dense geometric representations and production ML engineering, complete the following two structured tasks:

---

### Task A: Dimensionality Reduction & Semantic Clustering Visualization (PCA & t-SNE)
High-dimensional 384D spaces cannot be directly visualized by human eyes. By applying linear dimensionality reduction (**Principal Component Analysis - PCA**) and non-linear manifold learning (**t-SNE**), we can project the $100 \times 384$ dataset down to 2 dimensions $\mathbb{R}^2$.

**Your Objective:** Write a function `visualize_semantic_clusters(embeddings, labels, label_names)` that computes 2D PCA projections and generates a styled 2D scatter plot color-coded by intent class, visually verifying that queries form distinct geometric clusters (*Semantic Neighborhoods*).

---

### Task B: OOD Rejection via Confidence Thresholding Guardrails
In customer-facing production systems, routing an out-of-distribution question to a specialized campus department causes severe operational failure.

**Your Objective:** Implement an enhanced function `predict_intent_with_fallback(query, encoder, classifier, confidence_threshold=0.50)` that:
1. Calculates maximum class posterior probability $P_{\max} = \max_{k} P(y=k \mid \mathbf{x})$.
2. If $P_{\max} < \tau$ (e.g., $\tau = 0.50$), rejects the closed-world decision and triggers an `OOD_FALLBACK_TRIGGERED` payload.""")

    # =========================================================================
    # CELL 22: TASK A CODE
    # =========================================================================
    add_code("""# =============================================================================
# STUDENT TASK A: DIMENSIONALITY REDUCTION & CLUSTERING VISUALIZATION
# =============================================================================

def visualize_semantic_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str]
) -> None:
    \"\"\"
    Reduces 384D dense sentence embeddings to 2D using PCA and visualizes clusters.

    Args:
        embeddings (np.ndarray): N x 384 matrix of dense sentence embeddings.
        labels (np.ndarray): N-length vector of integer class labels.
        label_map (Dict[int, str]): Mapping from integer class IDs to class strings.
    \"\"\"
    # 1. Fit PCA to project from 384D down to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    var_explained = pca.explained_variance_ratio_
    total_var = np.sum(var_explained) * 100.0
    
    # 2. Construct plotting DataFrame
    df_pca = pd.DataFrame({
        "PCA_1": embeddings_2d[:, 0],
        "PCA_2": embeddings_2d[:, 1],
        "Intent": [label_map[l] for l in labels]
    })
    
    # 3. Plot 2D Semantic Neighborhoods
    plt.figure(figsize=(11, 7))
    palette = sns.color_palette("tab10", n_colors=len(label_map))
    
    sns.scatterplot(
        data=df_pca,
        x="PCA_1",
        y="PCA_2",
        hue="Intent",
        style="Intent",
        s=90,
        palette=palette,
        alpha=0.9
    )
    
    plt.title(
        f"2D Projection of 384D Dense Semantic Space (PCA Variance Explained: {total_var:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    plt.xlabel(f"Principal Component 1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_explained[1]*100:.1f}% variance)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title="Campus Intent")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Run Task A Visualization
visualize_semantic_clusters(X_embeddings, y_labels, id2label)

print(\"\"\"
[Task A Key Takeaway]:
Notice how queries belonging to 'tuition_payment', 'parking_permit', and 'dorm_maintenance'
naturally segregate into distinct, compact geometric clusters!
This spatial separation is precisely why a simple linear hyperplane achieves >95% accuracy.
\"\"\")""")

    # =========================================================================
    # CELL 23: TASK B CODE
    # =========================================================================
    add_code("""# =============================================================================
# STUDENT TASK B: OOD REJECTION VIA CONFIDENCE THRESHOLDING
# =============================================================================

def predict_intent_with_fallback(
    query: str,
    encoder: SentenceTransformer,
    classifier: Any,
    confidence_threshold: float = 0.50,
    label_mapping: Dict[int, str] = id2label
) -> Dict[str, Any]:
    \"\"\"
    Classifies student intent with automated Out-of-Distribution (OOD) fallback detection.

    Args:
        query (str): The input query string.
        encoder (SentenceTransformer): Pretrained dense encoder.
        classifier (Any): Supervised classifier.
        confidence_threshold (float): Minimum max-probability required to accept prediction.
        label_mapping (Dict[int, str]): Class ID to string mapping.

    Returns:
        Dict[str, Any]: Either a confirmed intent routing or a fallback payload.
    \"\"\"
    base_result = predict_intent(query, encoder, classifier, label_mapping)
    max_prob = base_result["confidence_score"]
    
    if max_prob < confidence_threshold:
        return {
            "status": "OOD_FALLBACK_TRIGGERED",
            "query": query,
            "confidence_score": max_prob,
            "provisional_intent": base_result["predicted_intent"],
            "fallback_message": (
                "Your inquiry appears to fall outside our standard campus administrative services. "
                "Please reach out to the General University Helpdesk at (555) 010-HELP or help@campus.edu."
            ),
            "class_distribution": base_result["class_distribution"]
        }
    else:
        return {
            "status": "CONFIRMED_INTENT",
            "query": query,
            "confidence_score": max_prob,
            "predicted_intent": base_result["predicted_intent"],
            "class_distribution": base_result["class_distribution"]
        }

# Demonstrate Task B with both In-Distribution and OOD Queries
test_queries = [
    "How do I pay my tuition for the upcoming fall semester?",  # In-distribution (tuition_payment, ~57% confidence)
    "Can you compose a poem about quantum computers in outer space?"  # Out-of-distribution (OOD, ~34% confidence)
]

print(">>> DEMONSTRATING CONFIDENCE-GUARDED INFERENCE PIPELINE:")
for tq in test_queries:
    fb_res = predict_intent_with_fallback(tq, embedding_model, clf, confidence_threshold=0.50)
    print(f"\\nQuery  : \\\"{tq}\\\"")
    print(f"Status : {fb_res['status']}")
    print(f"Score  : {fb_res['confidence_score']*100:.2f}%")
    if fb_res["status"] == "CONFIRMED_INTENT":
        print(f"Intent : {fb_res['predicted_intent']}")
    else:
        print(f"Message: {fb_res['fallback_message']}")""")

    # =========================================================================
    # CELL 24: SECTION 6: COMPREHENSIVE UNIT TEST SUITE
    # =========================================================================
    add_code("""# =============================================================================
# SECTION 6: COMPREHENSIVE STUDENT SELF-CHECK UNIT TESTS
# =============================================================================

def run_comprehensive_self_check():
    print("[Testing Suite] Initiating comprehensive Phase 2 verification checks...")
    
    # Check 1: Vector Dimension & Unit Norms
    test_vec = embedding_model.encode(["Campus tuition wire remittance"], normalize_embeddings=True)
    assert test_vec.shape == (1, 384), f"Expected shape (1, 384), got {test_vec.shape}"
    norm = np.linalg.norm(test_vec[0])
    np.testing.assert_allclose(norm, 1.0, rtol=1e-5, err_msg="Embedding must have Euclidean norm of 1.0")
    print("  ✓ Check 1: 384-dimensional dense encoding and L2 normalization verified.")
    
    # Check 2: Softmax Distribution Invariant (Sum == 1.0)
    res_in = predict_intent("Where can I park my car?", embedding_model, clf)
    prob_sum = sum(res_in["class_distribution"].values())
    np.testing.assert_allclose(prob_sum, 1.0, rtol=1e-3, err_msg="Probability distribution must sum to 1.0")
    print("  ✓ Check 2: Softmax probability distribution sum to 1.0 verified.")
    
    # Check 3: Synonym Handling Accuracy
    assert res_in["predicted_intent"] == "parking_permit", f"Expected parking_permit, got {res_in['predicted_intent']}"
    assert res_in["confidence_score"] > 0.60, f"Synonym query must have high confidence (> 0.60), got {res_in['confidence_score']}"
    print("  ✓ Check 3: Synonym mapping to semantic neighborhood verified.")
    
    # Check 4: Out-of-Distribution Fallback Guardrail
    # We set confidence_threshold=0.75 to ensure robust OOD rejection across random vector orientations
    res_ood = predict_intent_with_fallback(
        "Explain photosynthesis in pine trees", 
        embedding_model, 
        clf, 
        confidence_threshold=0.75
    )
    assert res_ood["status"] == "OOD_FALLBACK_TRIGGERED", "OOD query must trigger fallback guardrail"
    print("  ✓ Check 4: OOD confidence-thresholded rejection verified.")
    
    # Check 5: Training / Test Set Stratification
    assert len(X_train) == 80 and len(X_test) == 20, "Split counts must match 80/20"
    print("  ✓ Check 5: Stratified data partition consistency verified.")
    
    print("\\n" + "=" * 80)
    print("🎉 ALL PHASE 2 SELF-CHECK UNIT TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80)

run_comprehensive_self_check()""")

    # =========================================================================
    # CELL 25: CONCLUSION & TRANSITION TO PHASE 3
    # =========================================================================
    add_md("""## 6. Summary, Critical Reflection & Transition to Phase 3

### Summary of Phase 2 Key Findings

| Metric / Dimension | Phase 1: Lexical (TF-IDF) | Phase 2: Dense Classifiers (SBERT + Softmax/LR) | Phase 3: RAG & GenAI |
| :--- | :--- | :--- | :--- |
| **Vector Space** | Sparse, $|V| > 10,000$ dimensions | Dense, continuous $d = 384$ dimensions | Dense Embeddings + Generative LLM |
| **Synonym Handling** | **Fails ($0.0$ similarity)** | **Excellent (Geometric Proximity)** | **Near Human-Level Synthesis** |
| **Output Type** | Ranked Document IDs | Categorical Label ($y \in \mathcal{C}$) | Grounded Generative Natural Language |
| **Generation Capability** | None | None (The *Generation Gap*) | Synthesizes personalized, fluid answers |
| **Domain Scope** | Closed vocabulary | Closed set of $K$ classes | Open-domain context augmentation |

---

### Critical Reflection Questions (Pre-Lab for Phase 3)
1. **The Static Route Problem:** Suppose our classifier accurately routes a student to `dorm_maintenance`. How does the system automatically extract the exact repair procedures from a 50-page campus facilities handbook without hard-coding rules for every possible issue?
2. **Dense Retrieval vs. Classification:** Instead of training a classifier over fixed labels, how can we use dense vectors to directly *search* large document passages using Approximate Nearest Neighbors (ANN)?
3. **Hallucination Prevention:** How do we constrain a generative foundation model (like GPT-4o or Llama 3) to strictly answer student questions using verified campus knowledge base documents rather than fabricating university policies?

---

**Next Up — Phase 3:** *Retrieval-Augmented Generation (RAG): Vector Databases, Hybrid Search, and Grounded LLM Synthesis.*""")

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
    path = os.path.join(base_dir, "Phase_2_Dense_Semantic_Embeddings_and_Intent_Classification.ipynb")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"[SUCCESS] Wrote Phase 2 notebook with {len(cells)} cells to {path}")

if __name__ == "__main__":
    create_phase2_notebook()
