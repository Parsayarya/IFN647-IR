class DocV3:
  
    def __init__(self):
        self.doc_id = None #the itemid in the news item
        self.terms = {} #empty dictionary of terms (key-value pair string term: int freq)
        self.doc_size = 0 #the length of the document (WORDS). initialise to zero
        self.number_of_terms = 0
        
    #get docID (newsID)
    def get_docid(self):
        return self.docid
        
    def set_docid(self, docid):
        self.doc_id = docid
        
    def get_doc_size(self):
        return self.doc_size
        
    def set_doc_size(self, length):
        self.doc_size = length
        
    #return sorted list of all terms and their frequency as tuples, desc by frequency
    # NOTE: I see in slack that this is optional, but also specified to sort alphabetically.. I don't understand why we would sort this alphabetically when we are asked to print 
    def get_termlist_freq(self):
        x = {k : v for k, v in sorted(self.terms.items(), key = lambda item : item[1], reverse=True)}
        return x.items()
        
    #to add a term to the dictionary terms with value 1 if it doesn't exist, if it does exist then increment value (count) by 1 for that key
    #if the term is in stop words then ignore it
    def add_term(self, term):
        try:
            self.terms[term] += 1
        except KeyError:
            self.terms[term] = 1

        self.number_of_terms += 1
        
    