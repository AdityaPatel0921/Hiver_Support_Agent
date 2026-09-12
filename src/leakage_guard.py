"""
src/leakage_guard.py
Programmatic data leakage guard to verify mathematical and text-level isolation
between the retrieval knowledge base and the golden evaluation dataset.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Set
import pandas as pd

logger = logging.getLogger(__name__)


class DataContaminationError(Exception):
    """Raised when evaluation queries or IDs are detected within the retrieval knowledge base."""
    pass


def normalize_query_text(text: str) -> str:
    """Normalize text for strict duplicate detection across splits."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def verify_no_leakage(
    eval_source: Union[pd.DataFrame, Path, str],
    retriever_metadata: Union[List[Dict[str, Any]], pd.DataFrame, Path, str]
) -> bool:
    """
    Verify that the evaluation dataset is strictly disjoint from the retrieval knowledge base.
    Checks:
    1. Pair ID collisions
    2. Normalized customer query text collisions
    
    Raises:
        DataContaminationError: If any ID or normalized text overlap is detected.
    """
    # 1. Load evaluation DataFrame
    if isinstance(eval_source, (str, Path)):
        eval_path = Path(eval_source)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}")
        eval_df = pd.read_csv(eval_path)
    elif isinstance(eval_source, pd.DataFrame):
        eval_df = eval_source
    else:
        raise ValueError("eval_source must be a DataFrame, Path, or string path.")

    # 2. Extract Evaluation IDs and Texts
    eval_ids: Set[int] = set()
    if "pair_id" in eval_df.columns:
        eval_ids = set(eval_df["pair_id"].dropna().astype(int))

    eval_texts = [normalize_query_text(t) for t in eval_df.get("customer_text", []) if normalize_query_text(t)]
    eval_text_set: Set[str] = set(eval_texts)

    # 3. Extract Knowledge Base IDs and Texts
    kb_ids: Set[int] = set()
    kb_texts: Set[str] = set()

    if isinstance(retriever_metadata, (str, Path)):
        kb_path = Path(retriever_metadata)
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base data not found at {kb_path}")
        kb_df = pd.read_csv(kb_path)
        if "pair_id" in kb_df.columns:
            kb_ids = set(kb_df["pair_id"].dropna().astype(int))
        kb_texts = set(normalize_query_text(t) for t in kb_df.get("customer_text", []) if normalize_query_text(t))
    elif isinstance(retriever_metadata, pd.DataFrame):
        if "pair_id" in retriever_metadata.columns:
            kb_ids = set(retriever_metadata["pair_id"].dropna().astype(int))
        kb_texts = set(normalize_query_text(t) for t in retriever_metadata.get("customer_text", []) if normalize_query_text(t))
    elif isinstance(retriever_metadata, list):
        # Format from FAISS metadata store: List[Dict[str, Any]]
        for item in retriever_metadata:
            cid = item.get("citation_id") or item.get("pair_id")
            if cid is not None:
                kb_ids.add(int(cid))
            c_text = item.get("customer_text", "")
            if c_text:
                kb_texts.add(normalize_query_text(c_text))
    else:
        raise ValueError("retriever_metadata must be a list of dicts, DataFrame, or file path.")

    # 4. Check for ID collisions
    id_overlap = eval_ids.intersection(kb_ids)
    if id_overlap:
        sample_ids = list(id_overlap)[:5]
        raise DataContaminationError(
            f"CRITICAL DATA LEAKAGE: Detected {len(id_overlap)} overlapping Pair IDs between evaluation set "
            f"and retrieval knowledge base! (Sample IDs: {sample_ids}). "
            f"Evaluation queries must be strictly excluded from FAISS indexing."
        )

    # 5. Check for Exact Normalized Text Collisions
    text_overlap = eval_text_set.intersection(kb_texts)
    if text_overlap:
        sample_texts = list(text_overlap)[:3]
        raise DataContaminationError(
            f"CRITICAL DATA LEAKAGE: Detected {len(text_overlap)} overlapping customer query texts between "
            f"evaluation set and retrieval knowledge base! (Sample texts: {sample_texts}). "
            f"Evaluation queries must not exist in FAISS knowledge base."
        )

    logger.info(
        f"Leakage Guard Passed: Evaluation set ({len(eval_df)} rows) is strictly disjoint from "
        f"Knowledge Base ({len(kb_ids) if kb_ids else len(kb_texts)} records). Zero ID and text collisions detected."
    )
    return True
