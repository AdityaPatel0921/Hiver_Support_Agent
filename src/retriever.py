"""
src/retriever.py
Vector similarity retrieval using sentence-transformers and FAISS IndexFlatL2.
Indexes ONLY historical verified @AppleSupport resolution pairs from the isolated Knowledge Base split.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Lazy imports for fast module startup
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

from src.leakage_guard import verify_no_leakage, DataContaminationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("data/processed/faiss_index.bin")
DEFAULT_METADATA_PATH = Path("data/processed/faiss_metadata.pkl")
DEFAULT_KNOWLEDGE_BASE_PATH = Path("data/processed/knowledge_base_pairs.csv")
DEFAULT_EVAL_CANDIDATES_PATH = Path("data/processed/unannotated_eval_candidates.csv")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class AppleSupportRetriever:
    """
    Production retriever for Apple Support query grounding.
    Encodes customer questions and performs dense nearest-neighbor search via FAISS IndexFlatL2.
    Strictly isolated from the held-out golden evaluation set.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH
    ):
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed. Please run `pip install -r requirements.txt`.")
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_index(
        self,
        data_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
        eval_check_path: Optional[Path] = DEFAULT_EVAL_CANDIDATES_PATH,
        batch_size: int = 256
    ) -> None:
        """
        Build FAISS IndexFlatL2 over clean knowledge_base_pairs and persist index & metadata.
        Verifies that no held-out evaluation candidates are indexed.
        """
        if faiss is None:
            raise ImportError("faiss-cpu is not installed. Please run `pip install -r requirements.txt`.")

        model = self._load_model()
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Knowledge Base dataset not found at {data_path}. Run data_prep.py first.")

        # Leakage guard check before indexing
        if eval_check_path and Path(eval_check_path).exists():
            logger.info("Executing Pre-Indexing Leakage Guard check...")
            verify_no_leakage(eval_source=eval_check_path, retriever_metadata=data_path)

        logger.info(f"Reading Knowledge Base pairs from {data_path}...")
        df = pd.read_csv(data_path)
        
        texts = df["customer_text"].tolist()
        logger.info(f"Generating dense embeddings for {len(texts):,} Knowledge Base inquiries...")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        
        # Ensure float32 dtype for FAISS
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        
        logger.info(f"Creating FAISS IndexFlatL2 (dimension={dim}, total_vectors={len(embeddings)})...")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        # Store metadata for retrieval lookups
        self.metadata = []
        for idx, row in df.iterrows():
            self.metadata.append({
                "citation_id": int(row.get("pair_id", idx + 1)),
                "customer_tweet_id": row.get("customer_tweet_id", ""),
                "customer_text": str(row["customer_text"]),
                "apple_reply": str(row["apple_reply"])
            })

        # Persist index and metadata to disk
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
        logger.info(f"FAISS index saved to {self.index_path} and metadata ({len(self.metadata)} records) saved to {self.metadata_path}.")

    def load_index(self) -> None:
        """Load the pre-built FAISS index and metadata store."""
        if faiss is None:
            raise ImportError("faiss-cpu is not installed. Please run `pip install -r requirements.txt`.")

        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info("Index or metadata file not found. Building index from Knowledge Base split...")
            self.build_index()
            return

        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.metadata)} metadata records.")

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar historical customer inquiries and verified Apple responses.
        Returns a list of dicts with citation_id, customer_text, apple_reply, distance, and similarity_score.
        """
        if self.index is None or not self.metadata:
            self.load_index()

        model = self._load_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["distance"] = float(dist)
            # Convert L2 distance to an intuitive similarity score (0 to 1)
            item["similarity_score"] = float(1.0 / (1.0 + dist))
            results.append(item)

        return results


if __name__ == "__main__":
    retriever = AppleSupportRetriever()
    retriever.build_index()
    
    test_query = "My battery is dying in 2 hours on my iPhone 14"
    docs = retriever.retrieve_similar(test_query, top_k=2)
    print(f"\nQuery: {test_query}")
    for doc in docs:
        print(f"- Citation [{doc['citation_id']}] (Score: {doc['similarity_score']:.3f}):")
        print(f"  Cust: {doc['customer_text']}")
        print(f"  Apple: {doc['apple_reply']}")
