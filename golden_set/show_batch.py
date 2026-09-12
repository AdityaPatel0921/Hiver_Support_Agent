import sys
import pandas as pd

def show_batch(start_idx=0, count=15):
    df = pd.read_csv('golden_set/golden_eval_set.csv')
    sub = df.iloc[start_idx:start_idx+count]
    for i, (_, r) in enumerate(sub.iterrows(), start=start_idx+1):
        c = str(r['customer_text']).encode('ascii', 'replace').decode('ascii')
        a = str(r['historical_apple_reply']).encode('ascii', 'replace').decode('ascii')
        si = r['suggested_intent']
        sa = r['suggested_action']
        su = r['suggested_urgency']
        sr = r['suggested_reason'] if pd.notna(r['suggested_reason']) else ''
        print(f"[{i}] Pair #{r['pair_id']}")
        print(f"    Customer: \"{c}\"")
        print(f"    Apple Reply: \"{a}\"")
        print(f"    Suggested: {si} | {sa} | Urgency={su} | Reason: {sr}")
        print()

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    show_batch(start, cnt)
