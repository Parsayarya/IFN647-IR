import os
import re
import subprocess
from PRRM import PRRMModel
from data_processing_prrm import parse_docs, parse_query, load_stop_words
from feature_extraction_prrm import extract_features
from evaluation_prrm import average_precision, precision_at_k, dcg_at_k

def get_paths():
    """Get correct paths for the new folder structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    parent_dir = os.path.dirname(current_dir)  # project root
    data_dir = os.path.join(parent_dir, "data")
    
    return {
        'data_dir': data_dir,
        'dataset_base_dir': os.path.join(data_dir, "DataSets"),
        'queries_file_path': os.path.join(data_dir, "Queries-1.txt"),
        'stopwords_file_path': os.path.join(data_dir, "common-english-words.txt"),
        'lmrm_rankings_dir': os.path.join(current_dir, "RankingOutputs_LMRM"),
        'bm25_rankings_dir': os.path.join(data_dir, "RankingOutputs_BM25"),
        'prrm_output_dir': os.path.join(current_dir, "RankingOutputs_PRRM")
    }

# Extracts queries from the Queries-1.txt file 
def extract_queries(filepath):
    queries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split("<Query>")
    for block in blocks:
        if not block.strip():
            continue

        num_match = re.search(r"<num>\s*Number:\s*R(\d+)", block)
        title_match = re.search(r"<title>(.*?)\n", block)
        desc_match = re.search(r"<desc>\s*Description:\s*(.*)", block, re.DOTALL)

        if num_match and title_match:
            query_id = num_match.group(1).strip()
            title = title_match.group(1).strip()
            description = desc_match.group(1).strip().replace("\n", " ") if desc_match else ""
            full_query = f"{title} {description}"
            queries[query_id] = full_query

    return queries

# Loads ranking scores from a .dat file into a dictionary
def load_ranking_scores(filepath):
    scores = {}
    if not os.path.exists(filepath):
        print(f"Warning: Ranking file not found: {filepath}")
        return scores
    
    with open(filepath, 'r', encoding='utf-8') as f:
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
                        scores[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
            else:
                # LMRM format: docid score
                parts = line.split()
                if len(parts) == 2:
                    try:
                        scores[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
    return scores

# Runs PRRM for a single query and dataset
def run_prrm_for_query(query_id, query_text, dataset_path, stop_words, paths):
    print(f"\nRunning PRRM for R{query_id}")
    # Parse documents
    documents = parse_docs(dataset_path, stop_words)
    if not documents:
        print(f" No documents found for R{query_id}")
        return

    query_terms = parse_query(query_text, stop_words)
    print(f" Parsed {len(documents)} documents and query terms")

    # Load scores from both models
    lmrm_file = os.path.join(paths['lmrm_rankings_dir'], f"LMRM_R{query_id}Ranking.dat")
    bm25_file = os.path.join(paths['bm25_rankings_dir'], f"BM25IR_R{query_id}Ranking.dat")

    lmrm_scores = load_ranking_scores(lmrm_file)
    bm25_scores = load_ranking_scores(bm25_file)

    print(f" Loaded {len(lmrm_scores)} LMRM scores and {len(bm25_scores)} BM25 scores")

    if not lmrm_scores or not bm25_scores:
        print(f" Skipping R{query_id}: Missing ranking files")
        return

    # Rank by score
    lmrm_ranked = sorted(lmrm_scores.items(), key=lambda x: -x[1])
    bm25_ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])

    n = min(len(lmrm_ranked), len(bm25_ranked)) // 3
    if n < 1:
        print(f" Skipping R{query_id}: Not enough documents for pseudo-labeling")
        return

    top_lmrm = set([doc_id for doc_id, _ in lmrm_ranked[:n]])
    bottom_lmrm = set([doc_id for doc_id, _ in lmrm_ranked[-n:]])
    top_bm25 = set([doc_id for doc_id, _ in bm25_ranked[:n]])
    bottom_bm25 = set([doc_id for doc_id, _ in bm25_ranked[-n:]])

    top_docs = top_lmrm & top_bm25
    bottom_docs = bottom_lmrm & bottom_bm25

    training_docs = []
    labels = []

    for doc_id in top_docs:
        if doc_id in documents:
            training_docs.append(documents[doc_id])
            labels.append(1)

    for doc_id in bottom_docs:
        if doc_id in documents:
            training_docs.append(documents[doc_id])
            labels.append(0)

    # if len(training_docs) < 5:
    #     print(f" Skipping R{query_id}: Not enough training data after intersection")
    #     return

    print(f" Training docs: {len(training_docs)} | Pos: {labels.count(1)} | Neg: {labels.count(0)}")

    # Train and rank
    try:
        X_train = extract_features(query_terms, {doc.doc_id: doc for doc in training_docs}, 
                                 bm25_scores=bm25_scores, lmrm_scores=lmrm_scores)
        model = PRRMModel()
        model.train(X_train, labels)

        X_all = extract_features(query_terms, documents, 
                               bm25_scores=bm25_scores, lmrm_scores=lmrm_scores)
        scores = model.predict(X_all)
        all_doc_ids = list(documents.keys())
        scored_docs = sorted(zip(all_doc_ids, scores), key=lambda x: -x[1])

        output_path = os.path.join(paths['prrm_output_dir'], f"PRRM_R{query_id}Ranking.dat")
        with open(output_path, 'w') as f:
            for doc_id, score in scored_docs:
                f.write(f"{doc_id} {score}\n")

        print(f" Finished R{query_id}, Output: {output_path}")
    except Exception as e:
        print(f" Error processing R{query_id}: {e}")

# Entry point
if __name__ == "__main__":
    print("Starting PRRM processing...")
    
    paths = get_paths()
    
    # Check required paths exist
    required_paths = [
        paths['queries_file_path'],
        paths['stopwords_file_path'],
        paths['dataset_base_dir'],
        paths['lmrm_rankings_dir'],
        paths['bm25_rankings_dir']
    ]
    
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("Error: Missing required paths:")
        for path in missing_paths:
            print(f"  - {path}")
        print("Make sure BM25 and LMRM have been run first.")
        exit(1)
    
    # Create output directory
    os.makedirs(paths['prrm_output_dir'], exist_ok=True)
    
    # Load queries and stop words
    queries = extract_queries(paths['queries_file_path'])
    stop_words = load_stop_words(paths['stopwords_file_path'])
    
    print(f"Loaded {len(queries)} queries and {len(stop_words)} stop words")
    
    # Process each query
    for query_id, query_text in queries.items():
        dataset_path = os.path.join(paths['dataset_base_dir'], f"Dataset{query_id}")
        if os.path.exists(dataset_path):
            run_prrm_for_query(query_id, query_text, dataset_path, stop_words, paths)
        else:
            print(f"Warning: Dataset path not found: {dataset_path}")
    
    # Run evaluation after all ranking files are generated
    print("\nAll PRRM ranking files created. Starting evaluation...")
    try:
        subprocess.run(["python", "evaluation_prrm.py"], check=True)
        print("PRRM evaluation completed")
    except subprocess.CalledProcessError as e:
        print(f"PRRM evaluation failed: {e}")
    except FileNotFoundError:
        print("Could not run evaluation_prrm.py")