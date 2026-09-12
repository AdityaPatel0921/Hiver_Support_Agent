"""
src/data_prep.py
Memory-safe ingestion, cleaning, and partitioned data preparation pipeline for Kaggle Customer Support on Twitter (@AppleSupport).
Enforces strict physical separation between Knowledge Base (FAISS retrieval) and Held-Out Evaluation Candidates.
"""

import os
import re
import random
import logging
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/twcs.csv")
MASTER_PROCESSED_PATH = Path("data/processed/apple_support_pairs.csv")
KNOWLEDGE_BASE_PATH = Path("data/processed/knowledge_base_pairs.csv")
UNANNOTATED_CANDIDATES_PATH = Path("data/processed/unannotated_eval_candidates.csv")
ANNOTATION_TEMPLATE_PATH = Path("golden_set/annotation_template.csv")
GOLDEN_SET_PATH = Path("golden_set/golden_eval_set.csv")
LEGACY_GOLDEN_BACKUP_PATH = Path("golden_set/golden_eval_set_legacy_synthetic.csv")


def clean_text(text: str) -> str:
    """
    Sanitize text:
    - Normalize whitespace
    - Strip raw URLs (t.co, http/https, etc.)
    - Remove Twitter handles (@AppleSupport, @115858, etc.)
    - Clean surrounding punctuation
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+|t\.co/\S+", "", text)
    # Remove mentions/handles
    text = re.sub(r"@\w+", "", text)
    # Replace newlines and extra spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_query_text(text: str) -> str:
    """Normalize text for strict duplicate detection across splits."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_synthetic_twcs(output_path: Path, num_pairs: int = 6000) -> None:
    """
    Fallback generator if the genuine Kaggle raw file is completely missing.
    """
    logger.info(f"Generating {num_pairs} synthetic @AppleSupport pairs at {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device_models = ["iPhone 15 Pro", "iPhone 14", "iPad Pro", "MacBook Pro", "Apple Watch"]
    os_versions = ["iOS 17.4", "iOS 17.2", "macOS Sonoma 14.3", "watchOS 10.2"]
    intents = ["OS_SOFTWARE_UPDATE", "BATTERY_POWER", "ACCOUNT_ICLOUD", "HARDWARE_DAMAGE", "GENERAL_INQUIRY"]
    
    rows = []
    tweet_id_counter = 100000
    rng = random.Random(42)
    
    for i in range(num_pairs):
        intent = intents[i % len(intents)]
        device = device_models[i % len(device_models)]
        os_ver = os_versions[i % len(os_versions)]
        
        cust_text = f"Help with {device} running {os_ver} for {intent}. @AppleSupport"
        apple_reply = f"Thanks for reaching out about your {device}. Please check our support guides."
        
        cust_tweet_id = tweet_id_counter
        apple_tweet_id = tweet_id_counter + 1
        tweet_id_counter += 2
        
        rows.append({
            "tweet_id": cust_tweet_id,
            "author_id": f"User_{rng.randint(1000, 99999)}",
            "inbound": True,
            "created_at": "Fri Sep 01 10:00:00 +0000 2023",
            "text": cust_text,
            "response_tweet_id": str(apple_tweet_id),
            "in_response_to_tweet_id": ""
        })
        rows.append({
            "tweet_id": apple_tweet_id,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Fri Sep 01 10:05:00 +0000 2023",
            "text": f"@User {apple_reply}",
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(cust_tweet_id)
        })
        
    df_synthetic = pd.DataFrame(rows)
    df_synthetic.to_csv(output_path, index=False)
    logger.info(f"Generated synthetic dataset with {len(df_synthetic)} rows at {output_path}.")


def classify_heuristic_intent_suggestion(text: str) -> Tuple[str, str, int, Optional[str]]:
    """
    Non-ground-truth heuristic assistant provided solely to speed up human review.
    Must never be confused with human ground truth.
    """
    text_lower = text.lower()
    
    if any(k in text_lower for k in ["shatter", "cracked", "broken screen", "swelling", "swollen", "explosion", "hazard", "burnt plastic", "smells like", "crackling", "sunken into", "dropped", "water damage", "dropped in water"]):
        if any(k in text_lower for k in ["swelling", "swollen", "explosion", "burnt plastic", "smells like"]):
            return "HARDWARE_DAMAGE", "ESCALATE", 5, "Critical battery safety/thermal hazard"
        return "HARDWARE_DAMAGE", "ESCALATE", 4, "Physical hardware damage requires authorized repair"

    if any(k in text_lower for k in ["apple id", "icloud", "password", "locked", "hacked", "stolen", "unauthorized", "verification prompt", "family sharing", "account recovery"]):
        if any(k in text_lower for k in ["hacked", "stolen", "unauthorized", "another region"]):
            return "ACCOUNT_ICLOUD", "ESCALATE", 4, "Potential security compromise / unauthorized access"
        return "ACCOUNT_ICLOUD", "AUTO_HANDLE", 3, None

    if any(k in text_lower for k in ["battery", "draining", "drain", "charge", "charging", "dies", "warm while charging", "service recommended"]):
        return "BATTERY_POWER", "AUTO_HANDLE", 2, None

    if any(k in text_lower for k in ["ios", "update", "ipados", "watchos", "macos", "lag", "glitch", "verifying update", "installation paused", "system data", "wifi disconnect", "software"]):
        return "OS_SOFTWARE_UPDATE", "AUTO_HANDLE", 2, None

    return "GENERAL_INQUIRY", "AUTO_HANDLE", 1, None


def split_knowledge_and_evaluation_sets(
    df_clean: pd.DataFrame,
    total_target: int = 5000,
    eval_size: int = 150,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministically partition clean dialogue pairs into:
    1. Knowledge Base Split (4,850 rows) -> strictly for FAISS index construction.
    2. Held-Out Evaluation Candidate Pool (150 rows) -> strictly for manual human annotation & evaluation.
    Preserves global unique pair IDs to guarantee absolute disjointness.
    """
    logger.info(f"Partitioning {len(df_clean)} pairs into Knowledge Base ({total_target - eval_size}) and Eval Pool ({eval_size}) [Seed={random_seed}]...")
    
    df_clean = df_clean.copy().reset_index(drop=True)
    # Assign global unique pair_id (1 to N)
    df_clean["pair_id"] = range(1, len(df_clean) + 1)
    df_clean["_temp_intent"] = [classify_heuristic_intent_suggestion(t)[0] for t in df_clean["customer_text"]]
    
    intents = df_clean["_temp_intent"].unique()
    samples_per_intent = eval_size // len(intents)
    
    eval_indices = []
    rng = random.Random(random_seed)
    
    for intent in intents:
        intent_indices = df_clean[df_clean["_temp_intent"] == intent].index.tolist()
        sampled = rng.sample(intent_indices, min(len(intent_indices), samples_per_intent))
        eval_indices.extend(sampled)
        
    if len(eval_indices) < eval_size:
        remaining_needed = eval_size - len(eval_indices)
        remaining_pool = list(set(df_clean.index) - set(eval_indices))
        eval_indices.extend(rng.sample(remaining_pool, remaining_needed))
        
    eval_indices = sorted(eval_indices[:eval_size])
    kb_indices = sorted(list(set(df_clean.index) - set(eval_indices)))
    
    df_eval = df_clean.loc[eval_indices].copy().drop(columns=["_temp_intent"]).reset_index(drop=True)
    df_kb = df_clean.loc[kb_indices].copy().drop(columns=["_temp_intent"]).reset_index(drop=True)
    
    logger.info(f"Split complete: Knowledge Base = {len(df_kb)} pairs, Evaluation Candidates = {len(df_eval)} pairs.")
    return df_kb, df_eval


def create_annotation_template(df_eval: pd.DataFrame, output_path: Path = ANNOTATION_TEMPLATE_PATH) -> pd.DataFrame:
    """
    Generate an unannotated template CSV for human labeling.
    Leaves human ground-truth fields blank and segregates heuristic predictions as non-binding suggestions.
    """
    template_rows = []
    for _, row in df_eval.iterrows():
        sugg_intent, sugg_action, sugg_urgency, sugg_reason = classify_heuristic_intent_suggestion(row["customer_text"])
        template_rows.append({
            "pair_id": row["pair_id"],
            "customer_tweet_id": row.get("customer_tweet_id", ""),
            "customer_text": row["customer_text"],
            "historical_apple_reply": row["apple_reply"],
            # Blank human ground-truth fields to be populated manually
            "ground_truth_intent": "",
            "ground_truth_action": "",
            "ground_truth_urgency": "",
            "ground_truth_reason": "",
            "annotator_notes": "",
            # Separate heuristic suggestions for reviewer reference
            "suggested_intent": sugg_intent,
            "suggested_action": sugg_action,
            "suggested_urgency": sugg_urgency,
            "suggested_reason": sugg_reason or ""
        })
        
    df_template = pd.DataFrame(template_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_template.to_csv(output_path, index=False)
    logger.info(f"Annotation template generated at {output_path} ({len(df_template)} unannotated candidates).")
    
    # Also write to golden_set/golden_eval_set.csv with blank fields awaiting human annotation
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_template.to_csv(GOLDEN_SET_PATH, index=False)
    logger.info(f"Prepared unannotated golden set template at {GOLDEN_SET_PATH} (150 unannotated candidates).")
    
    return df_template


def process_twcs_dataset(
    raw_path: Path = RAW_DATA_PATH,
    master_processed_path: Path = MASTER_PROCESSED_PATH,
    knowledge_base_path: Path = KNOWLEDGE_BASE_PATH,
    unannotated_candidates_path: Path = UNANNOTATED_CANDIDATES_PATH,
    target_clean_pairs: int = 5000,
    eval_candidates_size: int = 150,
    chunksize: int = 200000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline execution:
    1. Ingest real Kaggle twcs.csv dataset in memory-safe chunks.
    2. Extract AppleSupport replies and matching inbound customer queries.
    3. Clean and filter dialogues with length >= 15 chars and unique query texts.
    4. Export target clean pairs (5,000).
    5. Deterministically partition into knowledge_base_pairs.csv (4,850) and unannotated_eval_candidates.csv (150).
    6. Generate annotation_template.csv and unannotated golden_eval_set.csv with blank ground-truth fields.
    """
    if not raw_path.exists():
        logger.warning(f"Raw dataset not found at {raw_path}. Generating synthetic fallback dataset...")
        generate_synthetic_twcs(raw_path, num_pairs=target_clean_pairs + 1000)

    logger.info(f"Starting chunked ingestion from {raw_path} (chunksize={chunksize})...")
    
    # Pass 1: Collect AppleSupport replies
    apple_replies: Dict[str, Dict[str, Any]] = {}
    target_inbound_ids: Set[str] = set()
    total_scanned = 0
    
    for chunk_idx, chunk in enumerate(pd.read_csv(
        raw_path,
        usecols=["tweet_id", "author_id", "text", "in_response_to_tweet_id", "created_at"],
        chunksize=chunksize,
        low_memory=False
    )):
        total_scanned += len(chunk)
        apple_mask = chunk["author_id"] == "AppleSupport"
        apple_chunk = chunk[apple_mask]
        
        for _, row in apple_chunk.iterrows():
            resp_to = str(row["in_response_to_tweet_id"]).split(".")[0].strip()
            if resp_to and resp_to != "nan":
                target_inbound_ids.add(resp_to)
                apple_replies[resp_to] = {
                    "apple_tweet_id": row["tweet_id"],
                    "apple_text": str(row["text"]),
                    "created_at": str(row.get("created_at", ""))
                }
                
        if (chunk_idx + 1) % 5 == 0:
            logger.info(f"Pass 1: Processed {total_scanned:,} rows. Apple replies indexed: {len(apple_replies):,}")
            
    logger.info(f"Pass 1 complete. Total rows: {total_scanned:,}. Apple replies awaiting inbounds: {len(apple_replies):,}")
    
    # Pass 2: Collect and match inbound tweets
    paired_data: List[Dict[str, Any]] = []
    seen_normalized_texts: Set[str] = set()
    pair_id = 1
    
    for chunk_idx, chunk in enumerate(pd.read_csv(
        raw_path,
        usecols=["tweet_id", "inbound", "text"],
        chunksize=chunksize,
        low_memory=False
    )):
        inbound_mask = chunk["inbound"].astype(str).str.lower().isin(["true", "1"])
        inbound_chunk = chunk[inbound_mask]
        
        matching = inbound_chunk[inbound_chunk["tweet_id"].astype(str).isin(target_inbound_ids)]
        
        for _, row in matching.iterrows():
            cust_id = str(row["tweet_id"]).split(".")[0].strip()
            if cust_id in apple_replies:
                apple_info = apple_replies[cust_id]
                clean_cust = clean_text(str(row["text"]))
                clean_apple = clean_text(apple_info["apple_text"])
                norm_cust = normalize_query_text(clean_cust)
                
                if len(clean_cust) >= 15 and len(clean_apple) >= 15 and norm_cust and norm_cust not in seen_normalized_texts:
                    seen_normalized_texts.add(norm_cust)
                    paired_data.append({
                        "pair_id": pair_id,
                        "customer_tweet_id": cust_id,
                        "apple_tweet_id": apple_info["apple_tweet_id"],
                        "customer_text": clean_cust,
                        "apple_reply": clean_apple,
                        "created_at": apple_info["created_at"]
                    })
                    pair_id += 1
                    
                    if len(paired_data) >= target_clean_pairs:
                        break
                        
        if len(paired_data) >= target_clean_pairs:
            logger.info(f"Reached target clean dialogue pairs ({len(paired_data):,}). Stopping chunk stream.")
            break

    df_clean = pd.DataFrame(paired_data)
    master_processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(master_processed_path, index=False)
    logger.info(f"Exported master clean dataset ({len(df_clean):,} pairs) to {master_processed_path}.")

    # Deterministic Split
    df_kb, df_eval = split_knowledge_and_evaluation_sets(
        df_clean,
        total_target=target_clean_pairs,
        eval_size=eval_candidates_size,
        random_seed=random_seed
    )
    
    # Save Knowledge Base Split
    df_kb.to_csv(knowledge_base_path, index=False)
    logger.info(f"Exported Knowledge Base ({len(df_kb):,} pairs) to {knowledge_base_path}.")
    
    # Save Unannotated Candidate Pool
    df_eval.to_csv(unannotated_candidates_path, index=False)
    logger.info(f"Exported Evaluation Candidates ({len(df_eval):,} pairs) to {unannotated_candidates_path}.")

    # Generate Blank Annotation Template & Blank Golden Set
    create_annotation_template(df_eval, ANNOTATION_TEMPLATE_PATH)

    return df_clean, df_kb, df_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TWCS dataset for AppleSupport RAG pipeline.")
    parser.add_argument("--reprocess-all", action="store_true", help="Force reprocessing from raw CSV")
    parser.add_argument("--raw-path", type=str, default=str(RAW_DATA_PATH), help="Path to raw twcs.csv")
    parser.add_argument("--target-pairs", type=int, default=5000, help="Number of clean pairs to produce")
    parser.add_argument("--eval-size", type=int, default=150, help="Number of held-out evaluation pairs")
    args = parser.parse_args()
    
    process_twcs_dataset(
        raw_path=Path(args.raw_path),
        target_clean_pairs=args.target_pairs,
        eval_candidates_size=args.eval_size
    )
