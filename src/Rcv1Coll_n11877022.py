class Rcv1Coll:
  
    def __init__(self):
        self.coll = {} #empty dictionary for collection
        self.totalDocLength = 0 #total Length of all documents
        self.num_docs = 0
        
    def add_doc(self, doc):     
        try:
            self.coll[doc.doc_id] = doc
            #add this documents length to the total doc length for the collection
            self.totalDocLength += doc.get_doc_size()
            #increment the number of docs
            self.num_docs += 1
        except KeyError:
            print("skipping duplicate document: "+ doc.doc_id)
            
