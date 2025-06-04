import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def doc_to_text(doc_obj):
    return " ".join([term for term, freq in doc_obj.terms.items() for _ in range(freq)])

def extract_features(query_terms, documents, bm25_scores=None, lmrm_scores=None):
    vectorizer = TfidfVectorizer()
    doc_texts = [doc_to_text(doc) for doc in documents.values()]
    X = vectorizer.fit_transform(doc_texts)

    query_text = " ".join([term for term, freq in query_terms.items()])
    q_vec = vectorizer.transform([query_text])
    cosine_similarities = X @ q_vec.T
    cosine_features = cosine_similarities.toarray().flatten().reshape(-1, 1)

    # Append BM25 and LMRM scores as extra features (default to 0 if missing)
    additional_features = []
    doc_ids = list(documents.keys())
    for doc_id in doc_ids:
        bm25 = bm25_scores.get(doc_id, 0) if bm25_scores else 0
        lmrm = lmrm_scores.get(doc_id, 0) if lmrm_scores else 0
        additional_features.append([bm25, lmrm])
    additional_features = np.array(additional_features)

    # Combine cosine similarity with BM25 and LMRM scores
    return np.hstack([cosine_features, additional_features])
