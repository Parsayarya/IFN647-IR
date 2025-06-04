import evaluation_bm25 as evaluation
import data_processing_bm25 as data_processing
import os


if __name__ == '__main__':
    import sys
   
    # Fixed paths for the new structure
    root_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    parent_dir = os.path.dirname(root_dir)  # project root
    
    data_folder = os.path.join(parent_dir, "data")  # data folder from project root
    rank_output_folder = os.path.join(parent_dir, "data", "RankingOutputs_BM25")  # BM25 ranking outputs
    benchmark_folder = os.path.join(parent_dir, "data", "EvaluationBenchmark")  # benchmarks
    document_folder = os.path.join(parent_dir, "data", "DataSets")  # datasets
    eval_output_folder = os.path.join(parent_dir, "data", "EvaluationOutputs")  # BM25 eval outputs
    
    # Create directories if they don't exist
    os.makedirs(rank_output_folder, exist_ok=True)
    os.makedirs(eval_output_folder, exist_ok=True)
    
    # Full paths to be passed into functions as required
    queries_path = os.path.join(data_folder, "Queries-1.txt")  # the full filepath for queries
    stop_word_path = os.path.join(data_folder, "common-english-words.txt")  # the full filepath for stop words
    
    print(f"Looking for queries at: {queries_path}")
    print(f"Looking for stopwords at: {stop_word_path}")
    print(f"Looking for documents at: {document_folder}")
    
    # Check if required files exist
    if not os.path.exists(queries_path):
        print(f"Error: Queries file not found at {queries_path}")
        sys.exit(1)
    if not os.path.exists(stop_word_path):
        print(f"Error: Stop words file not found at {stop_word_path}")
        sys.exit(1)
    if not os.path.exists(document_folder):
        print(f"Error: Documents folder not found at {document_folder}")
        sys.exit(1)
    
    # Get the queries
    query_dict = data_processing.load_queries(queries_path)        
    print(f"Loaded {len(query_dict)} queries")

    # Process and rank
    data_processing.process_and_rank_datasets(document_folder, rank_output_folder, query_dict, stop_word_path)

    # Check if there are already BM25 files in the eval location because the next step appends scores per query.
    # If there are files at this step, then we don't want to repeat already output scores.
    # To output new scores, delete the BM25 files in the eval_path
    bm25_eval_files_exist = False
    if os.path.exists(eval_output_folder):
        bm25_eval_files_exist = any(
            os.path.isfile(os.path.join(eval_output_folder, f)) and "BM25" in f
            for f in os.listdir(eval_output_folder)
        )
    
    ap_list = []
    pk_list = []
    dcg12_list = []
    
    # Evaluate
    for key, value in query_dict.items():
        # We only need the code
        dataset_code = key[1:]        
        ap, pk, dcg12 = evaluation.eval(benchmark_folder, rank_output_folder, dataset_code, eval_output_folder, bm25_eval_files_exist)
        ap_list.append(ap)
        pk_list.append(pk)
        dcg12_list.append(dcg12)
        
    
    map_score = sum(ap_list) / len(ap_list)
    pk_avg = sum(pk_list) / len(pk_list)
    dcg_avg = sum(dcg12_list) / len(dcg12_list)
    print(f"\nBM25 Results:")
    print(f"MAP: {map_score:.4f}")
    print(f"P@12 avg: {pk_avg:.4f}")
    print(f"DCG@12 avg: {dcg_avg:.4f}")    