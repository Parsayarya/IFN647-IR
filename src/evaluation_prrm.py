import os
import numpy as np
import pandas as pd

# === Define Metrics ===
def average_precision(y_true, y_score):
    sorted_indices = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[sorted_indices]
    precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k] == 1]
    return np.mean(precisions) if precisions else 0.0

def precision_at_k(y_true, y_score, k=12):
    sorted_indices = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[sorted_indices]
    return np.mean(y_true[:k])

def dcg_at_k(y_true, y_score, k=12):
    sorted_indices = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[sorted_indices]
    return sum([(2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(y_true[:k])])

# === Folders ===
ranking_dir = "RankingOutputs_PRRM"
benchmark_dir = "../data/EvaluationBenchmark"

results = []

# === Evaluate All Queries R101–R150 ===
for i in range(101, 151):
    query_id = f"R{i}"
    ranking_file = os.path.join(ranking_dir, f"PRRM_{query_id}Ranking.dat")
    benchmark_file = os.path.join(benchmark_dir, f"Dataset{i}.txt")

    if not os.path.exists(ranking_file) or not os.path.exists(benchmark_file):
        print(os.path.exists(ranking_file))
        print(os.path.exists(benchmark_file))
        print(f" Skipping {query_id}: missing file.")
        continue

    # Load gold labels
    relevance_dict = {}
    with open(benchmark_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                _, doc_id, label = parts
                relevance_dict[doc_id] = int(label)

    # Load predictions
    ranked_docs = []
    with open(ranking_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                doc_id, score = parts
                try:
                    ranked_docs.append((doc_id, float(score)))
                except:
                    continue

    if not ranked_docs:
        print(f" Skipping {query_id}: no valid scores.")
        continue

    # Sort scores
    ranked_docs.sort(key=lambda x: -x[1])
    y_true = [relevance_dict.get(doc_id, 0) for doc_id, _ in ranked_docs]
    y_score = [score for _, score in ranked_docs]

    # Compute metrics
    ap = average_precision(y_true, y_score)
    p12 = precision_at_k(y_true, y_score)
    dcg12 = dcg_at_k(y_true, y_score)

    results.append((query_id, ap, p12, dcg12))

# === Create Summary Table ===
df = pd.DataFrame(results, columns=["Query", "MAP", "P@12", "DCG@12"])
df.loc[len(df)] = ["Average", df["MAP"].mean(), df["P@12"].mean(), df["DCG@12"].mean()]

# === Save or Display ===
print("\n PRRM Evaluation Results")
print(df.to_string(index=False))

# Optional: Save to CSV
df.to_csv("PRRM_Evaluation_Results.csv", index=False)
