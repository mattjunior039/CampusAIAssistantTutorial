#!/usr/bin/env python3
"""
Synthetic Dataset Generator for AI Campus Assistant NLP Pipeline.

Generates two datasets using local Llama 3.1 via the `ollama` Python library:
1. Phase 1: Campus FAQs (data/phase1_campus_faqs.csv) - 100 prompt-response pairs
2. Phase 2: Intent Classification Queries (data/phase2_intent_queries.csv) - 500 queries across 5 intents
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import ollama

MODEL_NAME = "llama3.1"
DATA_DIR = "data"
MAX_RETRIES = 4

# ==============================================================================
# PHASE 1 CONFIGURATION: CAMPUS FAQS
# ==============================================================================
PHASE1_OUTPUT_FILE = os.path.join(DATA_DIR, "phase1_campus_faqs.csv")
PHASE1_CATEGORIES = [
    "Parking",
    "Financial Aid",
    "Housing",
    "Registration",
    "IT Support",
    "Health Services",
    "Library",
    "Student Life",
    "Dining",
    "Career Services",
]
FAQS_PER_CATEGORY = 10

# ==============================================================================
# PHASE 2 CONFIGURATION: INTENT CLASSIFICATION QUERIES
# ==============================================================================
PHASE2_OUTPUT_FILE = os.path.join(DATA_DIR, "phase2_intent_queries.csv")
PHASE2_INTENTS = [
    "tuition_payment",
    "parking_permit",
    "dorm_maintenance",
    "library_hours",
    "academic_advising",
]
QUERIES_PER_INTENT = 100
PHASE2_BATCH_SIZE = 50  # Generated in 2 batches of 50 per intent for high diversity & output stability


def extract_json_payload(raw_content: str) -> Optional[Any]:
    """Parse JSON safely from raw string content, handling top-level lists or wrapped dicts."""
    try:
        parsed = json.loads(raw_content)
        return parsed
    except json.JSONDecodeError:
        # Fallback: attempt to find JSON array/object substring
        raw_content = raw_content.strip()
        start_arr = raw_content.find("[")
        end_arr = raw_content.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                return json.loads(raw_content[start_arr : end_arr + 1])
            except json.JSONDecodeError:
                pass

        start_obj = raw_content.find("{")
        end_obj = raw_content.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                return json.loads(raw_content[start_obj : end_obj + 1])
            except json.JSONDecodeError:
                pass
    return None


def generate_phase1_faqs() -> pd.DataFrame:
    """Generate 100 campus FAQ pairs across 10 categories with formal administrative answers."""
    print("\n" + "=" * 80)
    print("STARTING PHASE 1 DATASET GENERATION: Campus FAQs")
    print("=" * 80)

    all_faqs: List[Dict[str, Any]] = []

    for cat_idx, category in enumerate(PHASE1_CATEGORIES, 1):
        print(f"\n[{cat_idx}/{len(PHASE1_CATEGORIES)}] Generating {FAQS_PER_CATEGORY} FAQs for Category: '{category}'...")

        prompt = f"""You are an administrative data generator for a university campus assistant system.
Generate exactly {FAQS_PER_CATEGORY} unique, realistic campus Frequently Asked Questions (FAQs) and answers for the category: "{category}".

REQUIREMENTS:
1. Question: Realistic, common questions students ask regarding {category}.
2. Answer: Comprehensive, authoritative, and written in formal administrative campus language.
3. Output MUST be a strict JSON array containing exactly {FAQS_PER_CATEGORY} objects.

JSON schema:
[
  {{
    "question": "What is the deadline for ...?",
    "answer": "According to university policy, students must submit ..."
  }}
]"""

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start_time = time.time()
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant that always responds with valid JSON arrays.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options={"temperature": 0.7},
                )
                elapsed = time.time() - start_time
                content = response["message"]["content"]
                eval_tokens = response.get("eval_count", "N/A")
                prompt_tokens = response.get("prompt_eval_count", "N/A")

                parsed = extract_json_payload(content)

                # Handle if model returned a dictionary wrapping the list
                faq_items = None
                if isinstance(parsed, list):
                    faq_items = parsed
                elif isinstance(parsed, dict):
                    for val in parsed.values():
                        if isinstance(val, list):
                            faq_items = val
                            break

                if not faq_items or not isinstance(faq_items, list):
                    raise ValueError(f"Response did not contain a valid JSON list. Parsed type: {type(parsed)}")

                # Validate elements
                valid_items = []
                for item in faq_items:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        valid_items.append(
                            {
                                "category": category,
                                "question": str(item["question"]).strip(),
                                "answer": str(item["answer"]).strip(),
                            }
                        )

                if len(valid_items) < FAQS_PER_CATEGORY:
                    print(
                        f"   ⚠️  Attempt {attempt}: Received {len(valid_items)}/{FAQS_PER_CATEGORY} valid pairs. Re-prompting..."
                    )
                    continue

                # Take top required count
                category_faqs = valid_items[:FAQS_PER_CATEGORY]
                all_faqs.extend(category_faqs)
                success = True

                print(
                    f"   ✅ Successfully generated {len(category_faqs)} FAQs in {elapsed:.2f}s "
                    f"(Prompt tokens: {prompt_tokens}, Eval tokens: {eval_tokens})"
                )
                break

            except Exception as e:
                print(f"   ❌ Attempt {attempt} failed with error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(1)

        if not success:
            raise RuntimeError(f"Failed to generate valid FAQs for category '{category}' after {MAX_RETRIES} attempts.")

    # Build DataFrame
    df_faqs = pd.DataFrame(all_faqs)
    # Assign sequential faq_id
    df_faqs.insert(0, "faq_id", [f"faq_{i+1:03d}" for i in range(len(df_faqs))])

    # Ensure exact column schema
    df_faqs = df_faqs[["faq_id", "category", "question", "answer"]]

    df_faqs.to_csv(PHASE1_OUTPUT_FILE, index=False)
    print(f"\n🎉 Saved Phase 1 dataset ({len(df_faqs)} rows) to: {PHASE1_OUTPUT_FILE}")
    print(df_faqs.head(3))
    return df_faqs


def generate_phase2_intent_queries() -> pd.DataFrame:
    """Generate 500 natural language student queries across 5 intents with high linguistic variance."""
    print("\n" + "=" * 80)
    print("STARTING PHASE 2 DATASET GENERATION: Intent Classification Queries")
    print("=" * 80)

    all_queries: List[Dict[str, str]] = []

    for intent_idx, intent in enumerate(PHASE2_INTENTS, 1):
        print(f"\n[{intent_idx}/{len(PHASE2_INTENTS)}] Generating {QUERIES_PER_INTENT} queries for Intent: '{intent}'...")

        intent_queries: List[str] = []
        batches_needed = (QUERIES_PER_INTENT + PHASE2_BATCH_SIZE - 1) // PHASE2_BATCH_SIZE

        for batch_num in range(1, batches_needed + 1):
            count_to_request = min(PHASE2_BATCH_SIZE, QUERIES_PER_INTENT - len(intent_queries))
            print(f"   -> Batch {batch_num}/{batches_needed}: Requesting {count_to_request} queries...")

            prompt = f"""You are simulating a diverse population of university college students.
Generate a JSON list of exactly {count_to_request} unique, realistic user queries/messages sent to a campus chatbot for the intent: "{intent}".

CRITICAL REQUIREMENTS FOR LINGUISTIC VARIANCE:
1. Force high diversity across the dataset:
   - Slang and colloquialisms (e.g. "yo where do i pay my bill", "fr need help with classes")
   - Minor typos and informal punctuation (e.g. "dorm sink is leking plz help", "how 2 get parkin pass??")
   - Frantic / urgent tones (e.g. "URGENT my registration hold is still active classes start tomorrow!!")
   - Formal / polite queries (e.g. "Good morning, could you please provide guidance on scheduling an advising appointment?")
   - Short sentence fragments / search keywords (e.g. "library closing time tonight", "FAFSA status")
   - Complex multi-part sentences (e.g. "I transferred credits from community college last fall and need to know which major advisor I should contact to approve my upper-division electives.")
2. Ensure every single query expresses the intent: "{intent}".
3. Output MUST be a strict JSON array of strings containing {count_to_request} distinct queries.

JSON schema:
[
  "query 1...",
  "query 2..."
]"""

            batch_success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    start_time = time.time()
                    response = ollama.chat(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a specialized synthetic data generator that outputs strict JSON string arrays.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        format="json",
                        options={"temperature": 0.85},
                    )
                    elapsed = time.time() - start_time
                    content = response["message"]["content"]
                    eval_tokens = response.get("eval_count", "N/A")
                    prompt_tokens = response.get("prompt_eval_count", "N/A")

                    parsed = extract_json_payload(content)

                    extracted_strings = []
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, str) and item.strip():
                                extracted_strings.append(item.strip())
                            elif isinstance(item, dict):
                                # If item is object {"query": "..."}
                                for val in item.values():
                                    if isinstance(val, str) and val.strip():
                                        extracted_strings.append(val.strip())
                                        break
                    elif isinstance(parsed, dict):
                        for val in parsed.values():
                            if isinstance(val, list):
                                for sub in val:
                                    if isinstance(sub, str) and sub.strip():
                                        extracted_strings.append(sub.strip())
                                    elif isinstance(sub, dict):
                                        for v in sub.values():
                                            if isinstance(v, str) and v.strip():
                                                extracted_strings.append(v.strip())
                                                break

                    if not extracted_strings:
                        raise ValueError(f"No valid query strings found in response. Parsed: {type(parsed)}")

                    # Deduplicate while preserving order
                    unique_new = [q for q in extracted_strings if q not in intent_queries]
                    intent_queries.extend(unique_new)

                    print(
                        f"   ✅ Batch {batch_num} extracted {len(unique_new)} new queries in {elapsed:.2f}s "
                        f"(Progress: {len(intent_queries)}/{QUERIES_PER_INTENT}, Eval tokens: {eval_tokens})"
                    )
                    batch_success = True
                    break

                except Exception as e:
                    print(f"   ⚠️  Batch {batch_num} Attempt {attempt} failed: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(1)

            if not batch_success:
                raise RuntimeError(
                    f"Failed to generate batch {batch_num} for intent '{intent}' after {MAX_RETRIES} attempts."
                )

        # Ensure we have at least QUERIES_PER_INTENT
        final_intent_queries = intent_queries[:QUERIES_PER_INTENT]
        for q in final_intent_queries:
            all_queries.append({"query_text": q, "intent_label": intent})

        print(f"   🎯 Completed {len(final_intent_queries)} queries for '{intent}'")

    # Build DataFrame
    df_queries = pd.DataFrame(all_queries)

    # Shuffle the dataset using frac=1, random_state=42
    df_shuffled = df_queries.sample(frac=1, random_state=42).reset_index(drop=True)

    # Export to CSV
    df_shuffled.to_csv(PHASE2_OUTPUT_FILE, index=False)
    print(f"\n🎉 Saved Phase 2 dataset ({len(df_shuffled)} rows) to: {PHASE2_OUTPUT_FILE}")
    print("\nClass distribution:")
    print(df_shuffled["intent_label"].value_counts())
    print("\nSample records:")
    print(df_shuffled.head(5))
    return df_shuffled


def main():
    # Ensure destination data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Verified directory: '{DATA_DIR}/'")

    # Generate Phase 1 Dataset
    df_faqs = generate_phase1_faqs()

    # Generate Phase 2 Dataset
    df_queries = generate_phase2_intent_queries()

    print("\n" + "=" * 80)
    print("ALL SYNTHETIC DATASETS SUCCESSFULLY GENERATED!")
    print(f"1. {PHASE1_OUTPUT_FILE} ({len(df_faqs)} pairs)")
    print(f"2. {PHASE2_OUTPUT_FILE} ({len(df_queries)} queries)")
    print("=" * 80)


if __name__ == "__main__":
    main()
