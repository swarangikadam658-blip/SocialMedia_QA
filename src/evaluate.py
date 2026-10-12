import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval as R  # noqa: E402

EVAL_PATH = "data/eval/eval_set.csv"
OUT_PATH = "data/eval/results.csv"
METHODS = ["tfidf", "bm25", "embedding", "hybrid"]
KS = (1, 3, 5)


def make_template(n=60):
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    unique = R.df.drop_duplicates("title_clean")
    sample = unique.sample(min(n, len(unique)), random_state=42)
    pd.DataFrame({
        "query": sample["title_clean"],
        "expected_title": sample["title_clean"],
    }).to_csv(EVAL_PATH, index=False)
    print(f"Created template at {EVAL_PATH}.")
    print("Rewrite the 'query' column as paraphrases, then run this again.")


def evaluate():
    if not os.path.exists(EVAL_PATH):
        make_template()
        return

    ev = pd.read_csv(EVAL_PATH).dropna()
    titles = R.df["title_clean"].tolist()
    rows = []

    for method in METHODS:
        hits = {k: 0 for k in KS}
        rr_total, used = 0.0, 0

        for _, item in ev.iterrows():
            ranked = R.rank_by_method(item["query"], method, k=max(KS) * 4)
            if ranked is None:          # method not available (library missing)
                break
            used += 1
            rank = None
            seen = []
            for idx in ranked:          # collapse duplicate titles
                t = titles[idx]
                if t not in seen:
                    seen.append(t)
            if item["expected_title"] in seen:
                rank = seen.index(item["expected_title"]) + 1
            if rank:
                rr_total += 1.0 / rank
                for k in KS:
                    if rank <= k:
                        hits[k] += 1

        if used == 0:
            print(f"Skipping {method} (not available)")
            continue

        rows.append({
            "method": method,
            **{f"Recall@{k}": round(hits[k] / used, 3) for k in KS},
            "MRR": round(rr_total / used, 3),
            "queries": used,
        })

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    evaluate()
