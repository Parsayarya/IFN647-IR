import os
import numpy as np
from scipy.stats import ttest_rel
import pandas as pd

def get_paths():
    """Get correct paths for the new folder structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    parent_dir = os.path.dirname(current_dir)  # project root
    data_dir = os.path.join(parent_dir, "data")
    
    return {
        'benchmark_dir': os.path.join(data_dir, "EvaluationBenchmark"),
        'prrm_dir': os.path.join(current_dir, "RankingOutputs_PRRM"),
        'bm25_dir': os.path.join(data_dir, "RankingOutputs_BM25"),
        'lmrm_dir': os.path.join(current_dir, "RankingOutputs_LMRM")
    }

def load_scores(metric_index, method_dir, method_prefix, benchmark_dir):
    """Load scores for a specific metric and method."""
    scores = []
    processed_queries = []
    
    for i in range(101, 151):
        query_id = f"R{i}"
        ranking_file = os.path.join(method_dir, f"{method_prefix}_{query_id}Ranking.dat")
        benchmark_file = os.path.join(benchmark_dir, f"Dataset{i}.txt")

        if not os.path.exists(ranking_file) or not os.path.exists(benchmark_file):
            continue

        # Load relevance judgments
        relevance_dict = {}
        try:
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        _, doc_id, label = parts
                        relevance_dict[doc_id] = int(label)
        except Exception as e:
            print(f"Error loading benchmark file {benchmark_file}: {e}")
            continue

        # Load ranked documents
        ranked_docs = []
        try:
            with open(ranking_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle different formats
                    if line.startswith('[') and line.endswith(']'):
                        # BM25 format: ['docid', 'score']
                        line = line.strip("[]")
                        parts = [item.strip().strip("'\"") for item in line.split(",")]
                        if len(parts) == 2:
                            try:
                                ranked_docs.append((parts[0], float(parts[1])))
                            except ValueError:
                                continue
                    else:
                        # LMRM/PRRM format: docid score
                        parts = line.split()
                        if len(parts) == 2:
                            try:
                                ranked_docs.append((parts[0], float(parts[1])))
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error loading ranking file {ranking_file}: {e}")
            continue

        if not ranked_docs:
            continue

        # Sort by score (descending)
        ranked_docs.sort(key=lambda x: -x[1])
        y_true = [relevance_dict.get(doc_id, 0) for doc_id, _ in ranked_docs]
        y_score = [score for _, score in ranked_docs]

        # Calculate metrics
        ap = average_precision(y_true, y_score)
        p12 = precision_at_k(y_true, y_score)
        dcg12 = dcg_at_k(y_true, y_score)

        metric_values = [ap, p12, dcg12]
        scores.append(metric_values[metric_index])
        processed_queries.append(query_id)
    
    print(f"Loaded {len(scores)} scores for {method_prefix} from queries: {processed_queries[:5]}...{processed_queries[-5:] if len(processed_queries) > 5 else ''}")
    return scores

# Metric calculation functions (copied from evaluation_prrm.py)
def average_precision(y_true, y_score):
    if not y_true or not y_score:
        return 0.0
    
    sorted_indices = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[sorted_indices]
    
    precisions = []
    for k in range(len(y_true)):
        if y_true[k] == 1:
            precision_at_k = np.mean(y_true[:k+1])
            precisions.append(precision_at_k)
    
    return np.mean(precisions) if precisions else 0.0

def precision_at_k(y_true, y_score, k=12):
    if not y_true or not y_score:
        return 0.0
    
    sorted_indices = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[sorted_indices]
    return np.mean(y_true[:k])

def dcg_at_k(y_true, y_score, k=12):
    if not y_true or not y_score:
        return 0.0
    
    sorted_indices = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[sorted_indices]
    dcg = 0.0
    for idx, rel in enumerate(y_true[:k]):
        if idx == 0:
            dcg += (2 ** rel - 1)
        else:
            dcg += (2 ** rel - 1) / np.log2(idx + 2)
    return dcg

def perform_statistical_tests():
    """Perform t-tests comparing all models: PRRM vs BM25, PRRM vs LMRM, and BM25 vs LMRM."""
    paths = get_paths()
    
    # Check if all required directories exist
    required_dirs = [paths['benchmark_dir'], paths['prrm_dir'], paths['bm25_dir'], paths['lmrm_dir']]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print("Error: Missing required directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        return

    methods = {
        "PRRM": (paths['prrm_dir'], "PRRM"),
        "BM25": (paths['bm25_dir'], "BM25IR"),
        "LMRM": (paths['lmrm_dir'], "LMRM")
    }

    metric_names = ["MAP", "P@12", "DCG@12"]
    
    # Define all pairwise comparisons
    comparisons = [
        ("PRRM", "BM25"),
        ("PRRM", "LMRM"), 
        ("BM25", "LMRM")
    ]
    
    print("="*70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS - T-TEST RESULTS")
    print("="*70)

    for metric_idx, metric_name in enumerate(metric_names):
        print(f"\n{'-'*50}")
        print(f"T-Tests for {metric_name}")
        print(f"{'-'*50}")
        
        # Load all scores for this metric
        all_scores = {}
        for method_name, method_info in methods.items():
            scores = load_scores(metric_idx, *method_info, paths['benchmark_dir'])
            all_scores[method_name] = scores
            if scores:
                print(f"  {method_name}: {len(scores)} queries loaded")
        
        # Perform all pairwise comparisons
        for model1, model2 in comparisons:
            print(f"\n{model1} vs {model2}:")
            print("-" * 25)
            
            scores1 = all_scores.get(model1, [])
            scores2 = all_scores.get(model2, [])
            
            if not scores1 or not scores2:
                print(f"  No scores found for {model1 if not scores1 else model2}")
                continue

            # Align lengths (use minimum available)
            min_len = min(len(scores1), len(scores2))
            if min_len < 5:
                print(f"  Not enough data points ({min_len}) for meaningful comparison")
                continue
                
            scores1_aligned = scores1[:min_len]
            scores2_aligned = scores2[:min_len]
            
            # Calculate basic statistics
            mean1 = np.mean(scores1_aligned)
            mean2 = np.mean(scores2_aligned)
            std1 = np.std(scores1_aligned)
            std2 = np.std(scores2_aligned)
            improvement = ((mean1 - mean2) / mean2) * 100 if mean2 != 0 else 0
            
            print(f"  {model1} mean: {mean1:.4f} (±{std1:.4f})")
            print(f"  {model2} mean: {mean2:.4f} (±{std2:.4f})")
            print(f"  Improvement: {improvement:+.2f}%")
            print(f"  Data points: {min_len}")
            
            # Perform paired t-test
            try:
                t_stat, p_value = ttest_rel(scores1_aligned, scores2_aligned)
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                
                # Determine significance level
                if p_value < 0.001:
                    significance = "highly significant (p < 0.001)"
                elif p_value < 0.01:
                    significance = "very significant (p < 0.01)"
                elif p_value < 0.05:
                    significance = "significant (p < 0.05)"
                else:
                    significance = "not significant (p >= 0.05)"
                
                # Determine direction
                if t_stat > 0:
                    direction = f"{model1} performs better than {model2}"
                else:
                    direction = f"{model2} performs better than {model1}"
                
                print(f"  Result: {direction} ({significance})")                
            except Exception as e:
                print(f"  Error performing t-test: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL COMPARISONS")
    print(f"{'='*70}")
    
    print("\nTotal Tests Performed: 9 (3 metrics × 3 pairwise comparisons)")
    print("\nComparisons:")
    print("  1. PRRM vs BM25 (MAP, P@12, DCG@12)")
    print("  2. PRRM vs LMRM (MAP, P@12, DCG@12)")
    print("  3. BM25 vs LMRM (MAP, P@12, DCG@12)")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    perform_statistical_tests()