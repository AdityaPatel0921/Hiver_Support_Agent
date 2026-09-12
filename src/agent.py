"""
src/agent.py
Grounded, zero-hallucination AI Customer Support Agent for @AppleSupport using Google GenAI SDK.
Enforces strict triage policies, closed-world retrieval grounding, and safe fallback guardrails.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import IntentType, TriageAction, TriageDecision, AgentResponse
from src.retriever import AppleSupportRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


AGENT_SYSTEM_INSTRUCTION = """
You are the official Production AI Customer Support Agent for @AppleSupport on social channels.
Your mission is to deliver accurate, empathetic, concise, and 100% grounded customer triage and responses.

### STRICT GUARDRAILS & ZERO-HALLUCINATION POLICY:
1. CLOSED-WORLD GROUNDING: Only provide guidance, steps, and troubleshooting verified by Apple official practices or present in the retrieved historical support context.
2. NEVER INVENT URLS, PHONE NUMBERS, OR FAKE WARRANTIES: You must NEVER generate unverified external URLs, fabricated apple.com links, fake phone numbers, or estimated repair costs. Always use the standard placeholder token `[Apple Support Link]` for any documentation or resource links.
3. BRAND VOICE: Concise (under 280 characters if possible), helpful, empathetic, professional, clear, and proactive.

### TRIAGE & ESCALATION RULES:
1. MANDATORY ESCALATION (`action`: "ESCALATE"):
   - Physical damage: Cracked screens, shattered glass, liquid ingress, physical frame damage.
   - Critical Battery & Thermal Safety: Battery swelling, extreme heat, bulging case, smoke, spark risk.
   - Account Compromise & Security: Unauthorized charges, hacked Apple ID, account lockouts with failed 2FA.
   - Legal/Regulatory threats: Mention of lawsuits, consumer protection complaints, or regulatory filings.
   - Severe distress / abusive escalations.
   - For ANY escalated case, you MUST specify a clear, concise `escalation_reason` and an `urgency_score` from 3 to 5.

2. AUTO-HANDLE (`action`: "AUTO_HANDLE"):
   - Routine software updates (iOS, iPadOS, macOS, watchOS), basic glitches, app crashes.
   - Standard battery health inquiries (usage optimization, Low Power Mode, battery settings).
   - Standard iCloud settings (backups, storage management, photo sync).
   - General inquiries (device specs, trade-in links, feature compatibility).
   - `escalation_reason` should be null. `urgency_score` between 1 and 3.

### RETRIEVAL CITATIONS:
- You will be given retrieved verified historical Apple Support pairs.
- If you use information from a retrieved pair, include its integer citation ID in `retrieval_citations`.

### OUTPUT FORMAT:
You MUST output a valid JSON object strictly matching the following schema:
{
  "decision": {
    "intent": "OS_SOFTWARE_UPDATE" | "BATTERY_POWER" | "ACCOUNT_ICLOUD" | "HARDWARE_DAMAGE" | "GENERAL_INQUIRY",
    "confidence": float (0.0 to 1.0),
    "action": "AUTO_HANDLE" | "ESCALATE",
    "escalation_reason": string or null,
    "urgency_score": int (1 to 5)
  },
  "draft_reply": string,
  "retrieval_citations": [int, ...]
}
"""


class AppleSupportAgent:
    """
    Production AI Customer Support Agent for @AppleSupport.
    Orchestrates FAISS dense retrieval, grounded reasoning, structured Pydantic validation, and guardrail fallbacks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        retriever: Optional[AppleSupportRetriever] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.retriever = retriever or AppleSupportRetriever()
        self.client = None
        
        if self.api_key and genai is not None:
            self.client = genai.Client(api_key=self.api_key)
        else:
            logger.warning("Google GenAI client initialized in offline/fallback mode (GEMINI_API_KEY missing or SDK not found).")

    def _build_grounded_prompt(self, customer_query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Construct the prompt injecting verified retrieved historical context."""
        context_blocks = []
        for doc in retrieved_docs:
            cid = doc.get("citation_id", 0)
            c_text = doc.get("customer_text", "")
            a_text = doc.get("apple_reply", "")
            context_blocks.append(f"[Citation ID {cid}]\nCustomer: {c_text}\nVerified Apple Reply: {a_text}")
            
        retrieved_context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant historical pairs retrieved."
        
        prompt = f"""
### RETRIEVED HISTORICAL VERIFIED APPLE SUPPORT CONTEXT:
{retrieved_context_str}

### INCOMING CUSTOMER INQUIRY:
"{customer_query}"

Analyze the customer inquiry, determine the intent and triage decision according to the triage rules, and craft a grounded Apple Support response.
Output only valid JSON.
"""
        return prompt

    def generate_fallback_response(self, customer_query: str, reason: str = "Safety/Parsing Guardrail Fallback") -> AgentResponse:
        """
        Deterministic, safety-critical fallback response when LLM generation or JSON validation fails.
        Always defaults to ESCALATE to ensure human-in-the-loop review.
        """
        logger.warning(f"Triggering Guardrail Fallback for query '{customer_query[:50]}...'. Reason: {reason}")
        return AgentResponse(
            decision=TriageDecision(
                intent=IntentType.GENERAL_INQUIRY,
                confidence=0.0,
                action=TriageAction.ESCALATE,
                escalation_reason=reason,
                urgency_score=4
            ),
            draft_reply="We want to help resolve this for you. We are escalating your inquiry directly to an Apple Support specialist. In the meantime, you can view resources here: [Apple Support Link]",
            retrieval_citations=[]
        )

    def process_query(self, customer_query: str, top_k: int = 3) -> AgentResponse:
        """
        End-to-end processing pipeline:
        1. Dense retrieval of historical verified support pairs.
        2. Prompt construction with closed-world constraints.
        3. LLM generation with structured JSON schema enforcement.
        4. Pydantic v2 validation.
        5. Guardrail fallback on any exception.
        """
        if not customer_query or not customer_query.strip():
            return self.generate_fallback_response("", reason="Empty customer query input.")

        # 1. Retrieve historical context
        retrieved_docs = []
        try:
            retrieved_docs = self.retriever.retrieve_similar(customer_query, top_k=top_k)
        except Exception as e:
            logger.error(f"Retriever error: {e}. Proceeding without retrieval context.")

        # 2. Check if client is available
        if self.client is None:
            return self.generate_fallback_response(customer_query, reason="LLM Client unavailable or GEMINI_API_KEY unset.")

        # 3. Generate response via GenAI SDK
        prompt = self._build_grounded_prompt(customer_query, retrieved_docs)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.0  # Zero temperature for deterministic adherence
                )
            )
            
            raw_text = response.text.strip()
            # Clean markdown JSON wrapping if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            # 4. Strict Pydantic v2 validation
            parsed_data = json.loads(raw_text)
            agent_response = AgentResponse.model_validate(parsed_data)
            return agent_response

        except Exception as e:
            logger.error(f"Agent generation / parsing failed: {e}")
            return self.generate_fallback_response(customer_query, reason=f"Safety/Parsing Guardrail Fallback ({type(e).__name__})")


if __name__ == "__main__":
    # Test agent with sample inquiries
    agent = AppleSupportAgent()
    
    test_queries = [
        "My iPhone 14 battery is swelling and the screen is popping off!",
        "How do I update my iPhone to the latest iOS version?",
        "Someone hacked my iCloud account and stole my passwords!"
    ]
    
    for q in test_queries:
        print(f"\n==========================================")
        print(f"Customer: {q}")
        res = agent.process_query(q)
        print(f"Decision: Intent={res.decision.intent}, Action={res.decision.action}, Urgency={res.decision.urgency_score}")
        if res.decision.escalation_reason:
            print(f"Escalation Reason: {res.decision.escalation_reason}")
        print(f"Draft Reply: {res.draft_reply}")
        print(f"Citations: {res.retrieval_citations}")
