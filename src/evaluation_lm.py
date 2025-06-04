import math
import os

def load_relevance_judgments(eval_benchmark_dir, query_num_int_str):
    """
    Loads relevance judgments for a specific query.
    query_num_int_str: String like "101", "102".
    Returns: Dictionary {doc_id: relevance (0 or 1)}
    """
    relevance_map = {}
    # Benchmark files are named e.g., Dataset101.txt for query R101
    filepath = os.path.join(eval_benchmark_dir, f"Dataset{query_num_int_str}.txt")
    
    expected_query_id_in_file = f"R{query_num_int_str}"

    if not os.path.exists(filepath):
        print(f"Warning: Relevance judgment file not found: {filepath}")
        return relevance_map

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                file_query_id, doc_id, rel_judgment_str = parts
                if file_query_id == expected_query_id_in_file:
                    try:
                        relevance_map[doc_id] = int(rel_judgment_str)
                    except ValueError:
                        print(f"Warning: Could not parse relevance judgment in {filepath}: {line}")
    return relevance_map

def precision_at_k(ranked_doc_ids: list, relevant_doc_ids_map: dict, k: int) -> float:
    if not ranked_doc_ids or k == 0:
        return 0.0
    
    retrieved_k = ranked_doc_ids[:k]
    relevant_and_retrieved_k_count = 0
    for doc_id in retrieved_k:
        if relevant_doc_ids_map.get(doc_id, 0) == 1: # 1 means relevant
            relevant_and_retrieved_k_count += 1
            
    return relevant_and_retrieved_k_count / k

def average_precision(ranked_doc_ids: list, relevant_doc_ids_map: dict) -> float:
    if not ranked_doc_ids or not relevant_doc_ids_map:
        return 0.0

    ap_sum = 0.0
    relevant_docs_retrieved_count = 0
    
    total_relevant_in_collection = sum(1 for rel_val in relevant_doc_ids_map.values() if rel_val == 1)
    
    if total_relevant_in_collection == 0: # No relevant documents for this query in benchmark
        return 0.0 # Or 1.0 if no relevant docs were expected and none retrieved. Standard is 0.0.

    for i, doc_id in enumerate(ranked_doc_ids):
        if relevant_doc_ids_map.get(doc_id, 0) == 1:
            relevant_docs_retrieved_count += 1
            precision_at_i = relevant_docs_retrieved_count / (i + 1)
            ap_sum += precision_at_i
            
    return ap_sum / total_relevant_in_collection if total_relevant_in_collection > 0 else 0.0

def dcg_at_k(ranked_doc_ids: list, relevant_doc_ids_map: dict, k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rank = i + 1
        relevance = relevant_doc_ids_map.get(doc_id, 0) # 0 or 1 for rel_i
        
        if rank == 1:
            dcg += float(relevance)
        else:
            dcg += float(relevance) / math.log2(rank) # log base 2 of i (rank)
            
    return dcg

def print_evaluation_summary(all_query_results: list, model_name: str = "LMRM"):
    """
    Prints evaluation tables for AP, P@12, and DCG@12.
    all_query_results: list of dicts, each {'query_id': str, 'P@12': float, 'AP': float, 'DCG@12': float}
    """
    if not all_query_results:
        print(f"No results to summarize for {model_name}.")
        return

    num_queries = len(all_query_results)

    print(f"\n--- Table 1: Performance of {model_name} on Average Precision ---")
    print(f"{'Topic':<6} | {model_name:<10}")
    print("-" * 20)
    total_ap = 0
    for res in all_query_results:
        print(f"{res['query_id']:<6} | {res['AP']:.4f}")
        total_ap += res['AP']
    map_val = total_ap / num_queries if num_queries > 0 else 0
    print("-" * 20)
    print(f"{'MAP':<6} | {map_val:.4f}")

    print(f"\n--- Table 2: Performance of {model_name} on Precision@12 ---")
    print(f"{'Topic':<6} | {model_name:<10}")
    print("-" * 20)
    total_p_at_12 = 0
    for res in all_query_results:
        print(f"{res['query_id']:<6} | {res['P@12']:.4f}")
        total_p_at_12 += res['P@12']
    avg_p_at_12 = total_p_at_12 / num_queries if num_queries > 0 else 0
    print("-" * 20)
    print(f"{'AvgP@12':<6} | {avg_p_at_12:.4f}")


    print(f"\n--- Table 3: Performance of {model_name} on DCG@12 ---")
    print(f"{'Topic':<6} | {model_name:<10}")
    print("-" * 20)
    total_dcg_at_12 = 0
    for res in all_query_results:
        print(f"{res['query_id']:<6} | {res['DCG@12']:.4f}")
        total_dcg_at_12 += res['DCG@12']
    avg_dcg_at_12 = total_dcg_at_12 / num_queries if num_queries > 0 else 0
    print("-" * 20)
    print(f"{'AvgDCG@12':<6} | {avg_dcg_at_12:.4f}")