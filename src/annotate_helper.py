"""
src/annotate_helper.py
Interactive CLI assistant for manual human annotation of the 150 held-out evaluation candidates.
Provides step-by-step review, auto-saves progress, and validates human ground-truth entries.
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import IntentType, TriageAction

TEMPLATE_PATH = Path("golden_set/annotation_template.csv")
GOLDEN_OUTPUT_PATH = Path("golden_set/golden_eval_set.csv")


def run_annotation_cli():
    """Interactive loop for rapid human review and annotation."""
    if not TEMPLATE_PATH.exists():
        print(f"Error: Annotation template not found at {TEMPLATE_PATH}. Run `python src/data_prep.py` first.")
        return

    # Load existing progress or fresh template
    if GOLDEN_OUTPUT_PATH.exists():
        df = pd.read_csv(GOLDEN_OUTPUT_PATH)
        print(f"Loaded existing progress from {GOLDEN_OUTPUT_PATH} ({len(df)} total rows).")
    else:
        df = pd.read_csv(TEMPLATE_PATH)
        print(f"Loaded fresh template from {TEMPLATE_PATH} ({len(df)} candidate rows).")

    # Ensure columns exist
    required_cols = [
        "pair_id", "customer_text", "historical_apple_reply",
        "ground_truth_intent", "ground_truth_action", "ground_truth_urgency",
        "ground_truth_reason", "annotator_notes"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    unannotated = df[df["ground_truth_intent"].isna() | (df["ground_truth_intent"].astype(str).str.strip() == "")]
    print(f"\nStatus: {len(df) - len(unannotated)} / {len(df)} items annotated. ({len(unannotated)} remaining).")
    
    if len(unannotated) == 0:
        print("\nAll 150 items have been annotated! You can run `python eval/run_eval.py` to evaluate.")
        return

    print("\nStarting Interactive Annotation Session (Type 'q' to save & quit, 's' to skip to next):\n")
    valid_intents = [e.value for e in IntentType]
    valid_actions = [e.value for e in TriageAction]

    for idx, row in df.iterrows():
        # Check if already annotated
        curr_intent = str(row.get("ground_truth_intent", "")).strip()
        if curr_intent and curr_intent != "nan":
            continue

        print("=" * 80)
        print(f"Candidate #{row['pair_id']} of {len(df)}")
        print(f"Customer Query: \"{row['customer_text']}\"")
        print(f"Historical Reply: \"{row.get('historical_apple_reply', '')}\"")
        print("-" * 80)
        
        # Show optional suggestion
        sugg_intent = row.get("suggested_intent", "GENERAL_INQUIRY")
        sugg_action = row.get("suggested_action", "AUTO_HANDLE")
        sugg_urgency = row.get("suggested_urgency", 2)
        sugg_reason = row.get("suggested_reason", "")
        print(f"Heuristic Suggestion: [{sugg_intent}] | Action: [{sugg_action}] | Urgency: [{sugg_urgency}]")
        if sugg_reason:
            print(f"Suggested Reason: {sugg_reason}")

        # Choice: Accept suggestion or manual input
        user_choice = input("\nAccept suggestion? [y=yes / m=manual edit / s=skip / q=quit]: ").strip().lower()
        
        if user_choice == "q":
            df.to_csv(GOLDEN_OUTPUT_PATH, index=False)
            print(f"\nProgress saved to {GOLDEN_OUTPUT_PATH}. Exiting.")
            break
        elif user_choice == "s":
            continue
        elif user_choice == "y":
            df.at[idx, "ground_truth_intent"] = sugg_intent
            df.at[idx, "ground_truth_action"] = sugg_action
            df.at[idx, "ground_truth_urgency"] = int(sugg_urgency)
            df.at[idx, "ground_truth_reason"] = sugg_reason if sugg_action == "ESCALATE" else ""
            df.at[idx, "annotator_notes"] = "Human accepted and verified suggestion"
        else:
            # Manual Intent
            print(f"Select Intent: 1: OS_SOFTWARE_UPDATE, 2: BATTERY_POWER, 3: ACCOUNT_ICLOUD, 4: HARDWARE_DAMAGE, 5: GENERAL_INQUIRY")
            intent_choice = input("Enter Intent (1-5 or name): ").strip()
            intent_map = {"1": "OS_SOFTWARE_UPDATE", "2": "BATTERY_POWER", "3": "ACCOUNT_ICLOUD", "4": "HARDWARE_DAMAGE", "5": "GENERAL_INQUIRY"}
            chosen_intent = intent_map.get(intent_choice, intent_choice.upper())
            if chosen_intent not in valid_intents:
                print("Invalid intent, defaulting to GENERAL_INQUIRY.")
                chosen_intent = "GENERAL_INQUIRY"
            df.at[idx, "ground_truth_intent"] = chosen_intent

            # Manual Action
            action_choice = input("Action [1: AUTO_HANDLE, 2: ESCALATE]: ").strip()
            chosen_action = "ESCALATE" if action_choice in ["2", "escalate", "ESCALATE"] else "AUTO_HANDLE"
            df.at[idx, "ground_truth_action"] = chosen_action

            # Manual Urgency
            urgency_choice = input("Urgency Score (1-5): ").strip()
            try:
                chosen_urgency = max(1, min(5, int(urgency_choice)))
            except ValueError:
                chosen_urgency = 2
            df.at[idx, "ground_truth_urgency"] = chosen_urgency

            # Manual Reason
            if chosen_action == "ESCALATE":
                reason_input = input("Escalation Reason (Mandatory): ").strip()
                while not reason_input:
                    reason_input = input("Reason cannot be empty for ESCALATE. Enter reason: ").strip()
                df.at[idx, "ground_truth_reason"] = reason_input
            else:
                df.at[idx, "ground_truth_reason"] = ""

            notes_input = input("Annotator Notes (Optional): ").strip()
            df.at[idx, "annotator_notes"] = notes_input

        # Save after every entry
        df.to_csv(GOLDEN_OUTPUT_PATH, index=False)
        print("-> Saved entry.")

    print(f"\nSession finished. Current progress saved to {GOLDEN_OUTPUT_PATH}.")


if __name__ == "__main__":
    run_annotation_cli()
