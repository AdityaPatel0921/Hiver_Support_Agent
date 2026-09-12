"""
eval/llm_judge.py
LLM-as-a-Judge evaluation framework, rubric scoring, and Human-vs-LLM Judge agreement computation.
Evaluates agent draft responses on a 1-5 rubric across Groundedness, Brand Voice, and Actionability.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class JudgeScore(BaseModel):
    """Structured rubric evaluation score from LLM Judge."""
    groundedness: int = Field(..., ge=1, le=5, description="1-5 rating on zero-hallucination and link/policy compliance.")
    brand_voice: int = Field(..., ge=1, le=5, description="1-5 rating on Apple tone, empathy, conciseness, and professionalism.")
    actionability: int = Field(..., ge=1, le=5, description="1-5 rating on safe, clear, and actionable troubleshooting steps.")
    rationale: str = Field(..., description="Short explanation justifying the scores.")


JUDGE_PROMPT_TEMPLATE = """
You are an expert LLMOps Evaluator auditing an AI Customer Support Agent for @AppleSupport.
Evaluate the following customer inquiry and agent response based on the strict 1-5 scoring rubric.

### 1-5 SCORING RUBRIC:
1. GROUNDEDNESS (Zero-Hallucination & Policy Compliance):
   - Score 5: Strictly factual, zero invented URLs or phone numbers, uses `[Apple Support Link]` placeholder if linking, adheres to official support procedures.
   - Score 3: Mostly factual but uses vague claims or improper formatting.
   - Score 1: Hallucinates URLs, fake phone numbers, fake warranty policies, or hazardous advice.

2. BRAND VOICE (Tone & Empathy):
   - Score 5: Distinctive Apple voice—warm, empathetic, professional, highly concise (ideal for social/Twitter), respectful.
   - Score 3: Robotic, overly verbose, or mildly detached.
   - Score 1: Rude, robotic, confusing, or inappropriate.

3. ACTIONABILITY (Clear Next Steps):
   - Score 5: Provides immediate, crystal-clear diagnostic steps (e.g. Settings > Battery) or exact escalation guidance.
   - Score 3: Suggestions are generic (e.g. "restart your phone") without specific navigation.
   - Score 1: Leaves user stranded with no guidance or dangerous instructions.

### INPUT DATA:
- Customer Query: "{customer_query}"
- Agent Triage Action: "{triage_action}"
- Agent Draft Reply: "{draft_reply}"

### REQUIRED OUTPUT:
Output valid JSON strictly conforming to:
{{
  "groundedness": int (1-5),
  "brand_voice": int (1-5),
  "actionability": int (1-5),
  "rationale": "detailed concise justification"
}}
"""


class LLMJudge:
    """Evaluates customer support responses using Google GenAI or deterministic heuristic rubric fallback."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if self.api_key and genai is not None:
            self.client = genai.Client(api_key=self.api_key)

    def evaluate_response(
        self,
        customer_query: str,
        triage_action: str,
        draft_reply: str
    ) -> JudgeScore:
        """Score a draft reply across Groundedness, Brand Voice, and Actionability."""
        if self.client is None:
            return self._heuristic_evaluation(customer_query, triage_action, draft_reply)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_query=customer_query,
            triage_action=triage_action,
            draft_reply=draft_reply
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            data = json.loads(raw_text)
            return JudgeScore.model_validate(data)
        except Exception as e:
            logger.warning(f"LLM Judge call failed: {e}. Falling back to heuristic rubric scoring.")
            return self._heuristic_evaluation(customer_query, triage_action, draft_reply)

    def _heuristic_evaluation(self, customer_query: str, triage_action: str, draft_reply: str) -> JudgeScore:
        """Deterministic heuristic evaluation when LLM client is offline."""
        groundedness = 5
        brand_voice = 4
        actionability = 4
        deductions = []

        # Groundedness checks
        if re.search(r"https?://(?!apple\.com)\S+", draft_reply) or "www." in draft_reply:
            groundedness -= 2
            deductions.append("Hallucinated raw external URL instead of placeholder.")
        if re.search(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", draft_reply):
            groundedness -= 2
            deductions.append("Included unverified direct phone number.")

        # Brand voice checks
        if len(draft_reply) > 280:
            brand_voice -= 1
            deductions.append("Exceeds standard 280 character Twitter brevity.")
        if any(w in draft_reply.lower() for w in ["apologize", "help", "glad", "understand", "support", "reach"]):
            brand_voice = min(5, brand_voice + 1)

        # Actionability checks
        if any(w in draft_reply for w in ["Settings >", "Genius Bar", "Apple Store", "restore", "restart", "[Apple Support Link]", "escalat", "DM"]):
            actionability = min(5, actionability + 1)
        else:
            actionability = max(2, actionability - 1)

        groundedness = max(1, min(5, groundedness))
        brand_voice = max(1, min(5, brand_voice))
        actionability = max(1, min(5, actionability))

        rationale = "Heuristic evaluation: " + ("; ".join(deductions) if deductions else "Complies with Apple Support guidelines.")
        return JudgeScore(
            groundedness=groundedness,
            brand_voice=brand_voice,
            actionability=actionability,
            rationale=rationale
        )


def calculate_cohens_kappa(rater1_labels: List[Any], rater2_labels: List[Any]) -> float:
    """
    Calculate Cohen's Kappa coefficient (κ) for unweighted categorical agreement.
    κ = (P_o - P_e) / (1 - P_e)
    """
    if len(rater1_labels) != len(rater2_labels) or len(rater1_labels) == 0:
        return 0.0

    n = len(rater1_labels)
    r1 = [str(x) for x in rater1_labels]
    r2 = [str(x) for x in rater2_labels]
    
    categories = sorted(list(set(r1 + r2)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    cm = np.zeros((num_cats, num_cats), dtype=np.float64)
    for y1, y2 in zip(r1, r2):
        cm[cat_to_idx[y1], cat_to_idx[y2]] += 1

    p_o = np.trace(cm) / n
    row_sums = np.sum(cm, axis=1) / n
    col_sums = np.sum(cm, axis=0) / n
    p_e = np.sum(row_sums * col_sums)

    if np.isclose(1.0 - p_e, 0.0):
        return 1.0 if np.isclose(p_o, 1.0) else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


def calculate_weighted_kappa(
    rater1_scores: List[int],
    rater2_scores: List[int],
    min_rating: int = 1,
    max_rating: int = 5
) -> float:
    """
    Calculate Quadratic Weighted Cohen's Kappa (κ_w) for ordinal 1-5 rubric scores.
    Penalizes larger rating discrepancies quadratically: w_ij = 1 - (i - j)^2 / (max - min)^2.
    """
    if len(rater1_scores) != len(rater2_scores) or len(rater1_scores) == 0:
        return 0.0

    n = len(rater1_scores)
    categories = list(range(min_rating, max_rating + 1))
    num_cats = len(categories)
    cat_map = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix O
    O = np.zeros((num_cats, num_cats), dtype=np.float64)
    for s1, s2 in zip(rater1_scores, rater2_scores):
        c1 = max(min_rating, min(max_rating, int(s1)))
        c2 = max(min_rating, min(max_rating, int(s2)))
        O[cat_map[c1], cat_map[c2]] += 1

    # Build Weight Matrix W (quadratic weights)
    W = np.zeros((num_cats, num_cats), dtype=np.float64)
    for i in range(num_cats):
        for j in range(num_cats):
            W[i, j] = 1.0 - float((i - j) ** 2) / float((num_cats - 1) ** 2)

    # Expected Matrix E
    row_sums = np.sum(O, axis=1, keepdims=True)
    col_sums = np.sum(O, axis=0, keepdims=True)
    E = (row_sums @ col_sums) / n

    # Weighted agreement
    po_w = np.sum(W * O) / n
    pe_w = np.sum(W * E) / n

    if np.isclose(1.0 - pe_w, 0.0):
        return 1.0 if np.isclose(po_w, 1.0) else 0.0

    kappa_w = (po_w - pe_w) / (1.0 - pe_w)
    return float(kappa_w)


def compute_rubric_agreement(human_scores: List[int], llm_scores: List[int]) -> Dict[str, Any]:
    """
    Comprehensive multi-metric agreement suite for human vs LLM judge scores:
    - Exact Agreement Rate
    - Adjacent Agreement Rate (+/- 1 score difference)
    - Mean Absolute Error (MAE)
    - Quadratic Weighted Cohen's Kappa
    """
    if len(human_scores) != len(llm_scores) or len(human_scores) == 0:
        return {"exact_agreement": 0.0, "adjacent_agreement": 0.0, "mae": 0.0, "weighted_kappa": 0.0}

    h = np.array(human_scores, dtype=np.int32)
    m = np.array(llm_scores, dtype=np.int32)
    diff = np.abs(h - m)

    exact = float(np.mean(diff == 0))
    adjacent = float(np.mean(diff <= 1))
    mae = float(np.mean(diff))
    weighted_kappa = calculate_weighted_kappa(human_scores, llm_scores)

    return {
        "exact_agreement_rate": exact,
        "adjacent_agreement_rate": adjacent,
        "mean_absolute_error_mae": mae,
        "quadratic_weighted_kappa": weighted_kappa
    }


if __name__ == "__main__":
    judge = LLMJudge()
    score = judge.evaluate_response(
        customer_query="My iPhone battery drains in 3 hours.",
        triage_action="AUTO_HANDLE",
        draft_reply="We understand your battery concern. Check Settings > Battery > Battery Health. [Apple Support Link]"
    )
    print("Judge Score:", score)

    # Test Rubric Agreement calculation
    h_scores = [5, 4, 5, 4, 3, 5, 4, 5, 4, 5]
    l_scores = [5, 4, 4, 4, 3, 5, 5, 5, 4, 5]
    res = compute_rubric_agreement(h_scores, l_scores)
    print("Sample Rubric Agreement:", res)
