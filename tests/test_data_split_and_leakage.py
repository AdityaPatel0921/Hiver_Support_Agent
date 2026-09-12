"""
tests/test_data_split_and_leakage.py
Unit and integration test suite verifying:
1. Exact knowledge base vs evaluation pool record counts.
2. Complete ID and text-level disjointness (zero leakage).
3. FAISS index metadata purity.
4. Leakage guard failure detection on contaminated mocks.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_guard import verify_no_leakage, DataContaminationError, normalize_query_text
from src.retriever import AppleSupportRetriever

KNOWLEDGE_BASE_PATH = Path("data/processed/knowledge_base_pairs.csv")
UNANNOTATED_CANDIDATES_PATH = Path("data/processed/unannotated_eval_candidates.csv")
FAISS_METADATA_PATH = Path("data/processed/faiss_metadata.pkl")


class TestDataSplitAndLeakage(unittest.TestCase):
    """Test suite verifying data split integrity and leakage prevention."""

    @classmethod
    def setUpClass(cls):
        """Ensure split files exist before tests."""
        if not KNOWLEDGE_BASE_PATH.exists() or not UNANNOTATED_CANDIDATES_PATH.exists():
            from src.data_prep import process_twcs_dataset
            process_twcs_dataset()

    def test_01_split_record_counts(self):
        """Verify exact record counts: 4,850 knowledge base pairs and 150 evaluation candidates."""
        df_kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
        df_eval = pd.read_csv(UNANNOTATED_CANDIDATES_PATH)

        self.assertEqual(len(df_kb), 4850, f"Expected 4,850 KB records, found {len(df_kb)}")
        self.assertEqual(len(df_eval), 150, f"Expected 150 eval candidates, found {len(df_eval)}")
        self.assertEqual(len(df_kb) + len(df_eval), 5000, "Total sum must equal 5,000 master pairs")

    def test_02_id_disjointness(self):
        """Verify that Knowledge Base and Evaluation Candidate tweet IDs have zero intersection."""
        df_kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
        df_eval = pd.read_csv(UNANNOTATED_CANDIDATES_PATH)

        kb_tweet_ids = set(df_kb["customer_tweet_id"].astype(str))
        eval_tweet_ids = set(df_eval["customer_tweet_id"].astype(str))

        overlap = kb_tweet_ids.intersection(eval_tweet_ids)
        self.assertEqual(len(overlap), 0, f"Detected tweet ID overlap between KB and Eval: {overlap}")

    def test_03_text_overlap_disjointness(self):
        """Verify that normalized evaluation queries do not exist verbatim in Knowledge Base."""
        df_kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
        df_eval = pd.read_csv(UNANNOTATED_CANDIDATES_PATH)

        kb_texts = set(normalize_query_text(t) for t in df_kb["customer_text"])
        eval_texts = set(normalize_query_text(t) for t in df_eval["customer_text"])

        # Check unique candidate queries against KB
        overlap = eval_texts.intersection(kb_texts)
        self.assertEqual(len(overlap), 0, f"Detected exact normalized text overlap: {overlap}")

    def test_04_leakage_guard_clean_pass(self):
        """Verify that verify_no_leakage returns True on the clean knowledge base split."""
        df_kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
        df_eval = pd.read_csv(UNANNOTATED_CANDIDATES_PATH)

        is_clean = verify_no_leakage(eval_source=df_eval, retriever_metadata=df_kb)
        self.assertTrue(is_clean, "Clean split should pass leakage guard without errors")

    def test_05_leakage_guard_detects_contamination(self):
        """Verify that verify_no_leakage fails loudly when an evaluation query is artificially injected."""
        df_kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
        df_eval = pd.read_csv(UNANNOTATED_CANDIDATES_PATH)

        # Artificially inject an evaluation row into knowledge base
        contaminated_kb = pd.concat([df_kb, df_eval.head(1)]).reset_index(drop=True)

        with self.assertRaises(DataContaminationError) as ctx:
            verify_no_leakage(eval_source=df_eval, retriever_metadata=contaminated_kb)
        
        self.assertIn("CRITICAL DATA LEAKAGE", str(ctx.exception))

    def test_06_faiss_metadata_purity(self):
        """Verify that loaded FAISS retriever metadata matches Knowledge Base size and contains no eval queries."""
        retriever = AppleSupportRetriever()
        retriever.load_index()

        self.assertEqual(len(retriever.metadata), 4850, f"FAISS metadata must contain 4,850 records, found {len(retriever.metadata)}")
        
        df_eval = pd.read_csv(UNANNOTATED_CANDIDATES_PATH)
        is_clean = verify_no_leakage(eval_source=df_eval, retriever_metadata=retriever.metadata)
        self.assertTrue(is_clean, "FAISS metadata must be 100% disjoint from evaluation candidates")


if __name__ == "__main__":
    unittest.main()
