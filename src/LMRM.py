import math
from data_processing_lm import BowDoc, BowColl 

# Constants
LAMBDA_VAL = 0.4
LOG_OF_ZERO_PROB = -100.0 

def calculate_lmrm_score(doc_obj: BowDoc,
                         query_terms: list[str],
                         collection_term_freqs: dict[str, int],
                         total_collection_words: int,
                         lambda_val: float = LAMBDA_VAL):
    score = 0.0
    doc_len = doc_obj.doc_len

    if not query_terms: # Empty query
        return 0.0 
        
    if doc_len == 0:
         return LOG_OF_ZERO_PROB * len(query_terms) 

    for term in query_terms: 
        f_qi_D = doc_obj.terms.get(term, 0)
        c_qi = collection_term_freqs.get(term, 0)

        term_doc_prob = f_qi_D / doc_len 
        
        if total_collection_words == 0:
            if c_qi > 0: 
                 term_coll_prob = 1.0 
            else: 
                 term_coll_prob = 0.0
        else:
            term_coll_prob = c_qi / total_collection_words
            
        smoothed_prob = (1.0 - lambda_val) * term_doc_prob + lambda_val * term_coll_prob
        
        if smoothed_prob > 1e-9: 
            score += math.log2(smoothed_prob)
        else:
            score += LOG_OF_ZERO_PROB 
            
    return score

def rank_documents_lmrm(dataset_coll: BowColl,
                        query_terms: list[str],
                        collection_term_freqs: dict[str, int],
                        total_collection_words: int,
                        lambda_val: float = LAMBDA_VAL):
    doc_scores = {}
    if dataset_coll and dataset_coll.docs:
        for doc_id, doc_obj in dataset_coll.docs.items():
            doc_scores[doc_id] = calculate_lmrm_score(doc_obj, query_terms,
                                                      collection_term_freqs,
                                                      total_collection_words,
                                                      lambda_val)
    sorted_doc_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_doc_scores