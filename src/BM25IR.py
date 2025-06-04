import math


def df(coll):
    """
    count document frequency of term
    
    Args:
        coll (Rcv1Coll): collection of documents

    Returns:
        dict {term:df, ...}

    """     
    df = {}
   
    #for each document
    for id, doc in coll.coll.items():
        #for each term
        for term in doc.terms.keys():
            try:
                df[term] +=1 #add the term to a dic and increment the count of that term
            except KeyError:
                df[term] = 1
                
    #return a {term:df, ...} dictionary
    return df 

def avg_length(coll):
    """
    average document length (used by bm25)
    
    Args:
        coll (Rcv1Coll): collection of documents

    Returns:
        average document length

    """      
    return coll.totalDocLength/coll.num_docs

def bm25(coll, q, df):
    """
    compute bm25 score for the given collection against a query 
    
    Args:
        coll (Rcv1Coll): collection of documents
        q (dict): the tokenised query
        df (dict): document frequency
    
    Returns:
        dict {docid:bm25_score} 

    """      
    bm25s = {}
    avg_dl = avg_length(coll)
    no_docs = coll.num_docs
    for id, doc in coll.coll.items():

        k = 1.2 * ((1 - 0.75) + 0.75 * doc.doc_size / float(avg_dl))
        bm25_ = 0.0;
        for qt in q.keys():
            n = 0
            if qt in df.keys():
                n = df[qt]     
                #f = doc.terms[qt]            
                qf = q[qt]

                try:
                    #get the tf for doc
                    f = doc.terms[qt];
                except KeyError:
                    f = 0            

                bm = math.log(1.0 / ((n + 0.5) / (3*no_docs - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((500 + 1) * qf) / float(500 + qf))
                # bm values may be negative if no_docs < 2n+1, so we may use 3*no_docs to solve this problem.
                bm25_ += bm
        bm25s[doc.doc_id] = bm25_
    
    #return dict {docid:bm25_score}    
    return bm25s
    
