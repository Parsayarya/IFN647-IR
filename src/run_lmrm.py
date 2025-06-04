import os
import zipfile
import shutil
import csv

from data_processing_lm import (load_stopwords, parse_dataset_xml, 
                                parse_queries, calculate_collection_stats)
from LMRM import rank_documents_lmrm
from evaluation_lm import (load_relevance_judgments, precision_at_k, 
                           average_precision, dcg_at_k, print_evaluation_summary)

# Constants
LAMBDA_VAL = 0.4
K_FOR_EVAL = 12

def get_paths():
    """Get correct paths for the new folder structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    parent_dir = os.path.dirname(current_dir)  # project root
    data_dir = os.path.join(parent_dir, "data")
    
    return {
        'data_dir': data_dir,
        'dataset_base_dir': os.path.join(data_dir, "DataSets"),
        'eval_benchmark_base_dir': os.path.join(data_dir, "EvaluationBenchmark"),
        'queries_file_path': os.path.join(data_dir, "Queries-1.txt"),
        'stopwords_file_path': os.path.join(data_dir, "common-english-words.txt"),
        'ranking_output_dir': os.path.join(current_dir, "RankingOutputs_LMRM")  # in src folder
    }

def save_evaluation_to_csv(all_query_eval_results):
    """Save LMRM evaluation results to CSV file in outputs/LMRM/ folder."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    parent_dir = os.path.dirname(current_dir)  # project root
    
    # Create the LMRM output directory in outputs folder
    lmrm_output_dir = os.path.join(parent_dir, "outputs", "LMRM")
    os.makedirs(lmrm_output_dir, exist_ok=True)
    
    csv_file_path = os.path.join(lmrm_output_dir, "LMRM_Evaluation_Results.csv")
    
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Query', 'MAP', 'P@12', 'DCG@12']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write individual query results
            for result in all_query_eval_results:
                writer.writerow({
                    'Query': result['query_id'],
                    'MAP': f"{result['AP']:.4f}",
                    'P@12': f"{result['P@12']:.4f}",
                    'DCG@12': f"{result['DCG@12']:.4f}"
                })
            
            # Calculate and write average scores
            if all_query_eval_results:
                num_queries = len(all_query_eval_results)
                avg_map = sum(result['AP'] for result in all_query_eval_results) / num_queries
                avg_p12 = sum(result['P@12'] for result in all_query_eval_results) / num_queries
                avg_dcg12 = sum(result['DCG@12'] for result in all_query_eval_results) / num_queries
                
                writer.writerow({
                    'Query': 'Average',
                    'MAP': f"{avg_map:.4f}",
                    'P@12': f"{avg_p12:.4f}",
                    'DCG@12': f"{avg_dcg12:.4f}"
                })
        
        print(f"\n LMRM evaluation results saved to: {csv_file_path}")
        
    except Exception as e:
        print(f"Error saving LMRM evaluation results to CSV: {e}")

def main():
    """Main function with corrected paths and CSV output."""
    paths = get_paths()
    
    # Check if required files exist
    required_paths = [
        paths['dataset_base_dir'],
        paths['eval_benchmark_base_dir'],
        paths['queries_file_path'],
        paths['stopwords_file_path']
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Required path not found: {path}")
            return
    
    print("All required paths found")
    
    # Load stopwords
    load_stopwords(paths['stopwords_file_path'])

    # Parse queries
    queries_map = parse_queries(paths['queries_file_path'])
    if not queries_map:
        print("No queries parsed. Exiting.")
        return

    print(f"Loaded {len(queries_map)} queries")

    # Create output directory for rankings
    if not os.path.exists(paths['ranking_output_dir']):
        os.makedirs(paths['ranking_output_dir'])
    else:
        # Clean out old ranking files if directory exists
        for f_name in os.listdir(paths['ranking_output_dir']):
            if f_name.startswith("LMRM_R") and f_name.endswith("Ranking.dat"):
                os.remove(os.path.join(paths['ranking_output_dir'], f_name))

    all_query_eval_results = []
    query_numbers_to_process = list(range(101, 151))
    
    # Main Processing Loop
    for query_num_int in query_numbers_to_process:
        query_id_str_numeric = str(query_num_int)
        query_id_full = f"R{query_id_str_numeric}"
        
        dataset_folder_name = f"Dataset{query_id_str_numeric}"
        current_dataset_path = os.path.join(paths['dataset_base_dir'], dataset_folder_name)

        print(f"\nProcessing {query_id_full} for {dataset_folder_name}...")

        if not os.path.exists(current_dataset_path):
            print(f"  Dataset path {current_dataset_path} not found. Skipping.")
            continue
        
        current_query_processed_terms = queries_map.get(query_id_full)
        if not current_query_processed_terms:
            print(f"  Query {query_id_full} not found in parsed queries or has no terms. Skipping.")
            continue

        # 1. Data Processing for the current dataset
        print(f"  Parsing and preprocessing documents in {current_dataset_path}...")
        dataset_coll = parse_dataset_xml(current_dataset_path)
        
        if not dataset_coll or not dataset_coll.docs:
            print(f"  No documents found or parsed for {dataset_folder_name}. Skipping.")
            all_query_eval_results.append({'query_id': query_id_full, 'P@12': 0.0, 'AP': 0.0, 'DCG@12': 0.0})
            continue
            
        collection_term_freqs, total_collection_words = calculate_collection_stats(dataset_coll)
        if total_collection_words == 0:
            print(f"  Warning: Dataset {dataset_folder_name} has zero total processable words. Scores might be minimal.")
        
        # 2. LMRM Model & Ranking Output
        print(f"  Ranking documents for {query_id_full} using LMRM...")
        ranked_docs_with_scores = rank_documents_lmrm(dataset_coll, current_query_processed_terms,
                                                      collection_term_freqs, total_collection_words,
                                                      LAMBDA_VAL)

        ranking_file_name = f"LMRM_{query_id_full}Ranking.dat"
        ranking_file_full_path = os.path.join(paths['ranking_output_dir'], ranking_file_name)
        with open(ranking_file_full_path, 'w', encoding='utf-8') as f_rank_out:
            for doc_id, score in ranked_docs_with_scores:
                f_rank_out.write(f"{doc_id} {score}\n")
        print(f"  Generated ranking file: {ranking_file_full_path}")

        # 3. Individual Evaluation for LMRM
        print(f"  Evaluating LMRM for {query_id_full}...")
        relevance_judgments = load_relevance_judgments(paths['eval_benchmark_base_dir'], query_id_str_numeric)
        
        if not relevance_judgments:
            print(f"  Warning: No relevance judgments found for {query_id_full}. Evaluation metrics will be 0.")
            p_at_12, ap, dcg_at_12 = 0.0, 0.0, 0.0
        else:
            ranked_doc_ids_only_list = [doc_id for doc_id, score in ranked_docs_with_scores]
            p_at_12 = precision_at_k(ranked_doc_ids_only_list, relevance_judgments, K_FOR_EVAL)
            ap = average_precision(ranked_doc_ids_only_list, relevance_judgments)
            dcg_at_12 = dcg_at_k(ranked_doc_ids_only_list, relevance_judgments, K_FOR_EVAL)
        
        print(f"  {query_id_full} - P@12: {p_at_12:.4f}, AP: {ap:.4f}, DCG@12: {dcg_at_12:.4f}")
        
        all_query_eval_results.append({
            'query_id': query_id_full,
            'P@12': p_at_12,
            'AP': ap,
            'DCG@12': dcg_at_12
        })

    # Final Evaluation Summary
    if all_query_eval_results:
        print_evaluation_summary(all_query_eval_results, model_name="LMRM")
        
        # Save evaluation results to CSV file
        save_evaluation_to_csv(all_query_eval_results)
    else:
        print("No queries were processed or evaluated.")

if __name__ == "__main__":
    main()