import glob, os
import string
import DocV3_n11877022 as doc
import Rcv1Coll_n11877022 as collection
from stemming import stem
import BM25IR as bm25

def parse_docs(stop_words, inputfolder): 
    """
    tokenise documents and create collection 
    
    Args:
        stop_words (list): list of stop words
        input_folder (str): the dataset folder
    
    Returns:
        Rcv1Coll object (collection of DocV3 objects to represent the collection) 

    """        
    os.chdir(inputfolder) #set the directory
    list_of_xml = glob.glob('*.xml') #get list of xml files in dir
    
    doc_collection = collection.Rcv1Coll()
    
    #for every xml file in the directory
    for d in list_of_xml:
        start_end = False # set flag to indicate whether we are at the start or the end of the text element.
        myfile=open(d)
        file_=myfile.readlines() #read in file contents into a list of strings (lines)       
        word_count = 0 #initialise word count to 0     
        curr_doc = doc.DocV3() #initialise empty docv3 object
        docid = None
        
        #************************************************************************************************************************************************************************************
        #WORDS AND TERMS DEFINITION:                                                                                                                                                        *
        #Words: are fundamental constructs in many natural languages. It excludes numbers here                                                                                              *
        #Terms: are specific concepts which can contain a single word or many words (such as ngrams). We are not applying ngrams here, so our unique terms are simply                       *
        #the set which remains after removal of punctuation, numbers, and application of stemming to our document text blocks which also meet the condition of being 3 characters or more   *
        #************************************************************************************************************************************************************************************
        for line in file_: #for every line in the file
            line = line.strip() #strip leading and trailing whitespace
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1] #get the docid from newsitem@itemid attribute
                            break  
                if line.startswith("<text>"):  #at beginning of text element
                    start_end = True 
            elif line.startswith("</text>"):  #at end of text element
                break
            else:
                #process lines inside text element
                line = line.replace("<p>", "").replace("</p>", "") #remove paragraph elements
                line = line.replace("&quot;", "") # remove quotation html elements (these aren't caught as punctuation it seems)
                line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) #remove numbers and punctuation
                line = line.replace("\\s+", " ") #removing isolated s after removing punctuation (we're only taking terms > 2 in length but need to remove these so not counted as a word)
                
                #process each word, extract terms with stemming applied (ignore stop words and words < 3 char)
                for word in line.split():
                    word_count += 1
                    word = stem(word.lower()) #stem
                    if len(word) > 2 and word not in stop_words:
                        curr_doc.add_term(word)

        #populate the DocV3 attributes for this document and close the file     
        curr_doc.set_docid(docid)
        curr_doc.set_doc_size(word_count)
        myfile.close()
        
        #add the DocV3 object to the collection (this will also update the total doc length for the collection)
        doc_collection.add_doc(curr_doc)
    
    #return the collection of DocV3 objects
    return(doc_collection)
 
def parse_q(query, stop_words):   
    """
    tokenise querys 
    
    Args:
        query (dict {RXXX : Title}): this is output from load_queries()
        stop_words (list): list of stop_words
    
    Returns:
        dict: {term : freq}

    """    
    q = {}   
    query = query.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    query = query.replace("\\s+", " ") 
          
    for word in query.split():
        word = stem(word.lower())
        if len(word) > 2 and word not in stop_words:
            try:
                q[word] += 1
            except KeyError:
                q[word] = 1
      
    return q
        
def load_queries(queries_path):
    """
    Extract RXXX reference and query (title) from Queries-1.txt. Expected to be located in same directory. 

    Returns:
        dict: RXXX as key, query text as value

    """
    queries = {}
    
    #source_dir = os.path.dirname(os.path.abspath(__file__))    
    #filepath = os.path.join(queries_dir, "Queries-1.txt")   
    filepath = queries_path
    
    with open(filepath, "r", encoding="utf-8") as file:
        current_ref = None
        
        #go through each line
        for line in file:
            line = line.strip() #strip leading and trailing whitespace
            if line.startswith("<num>"):
                # Extract Rxxx reference
                parts = line.split("Number:")
                if len(parts) > 1:
                    current_ref = parts[1].strip()
                    #print("query extract key: " + current_ref)
            elif line.startswith("<title>") and current_ref:
                # Extract title (query)
                title = line.replace("<title>", "").strip()
                queries[current_ref] = title
                current_ref = None  # Reset for next query

    #return as dict with with RXXX : Title
    return queries

def load_stopwords(stop_word_path):
    """
    loads 'common-english-words.txt' file. Expected to be located in same directory as .py files. 
    
    Returns:
        stop words list
    """
    #source_dir = os.path.dirname(os.path.abspath(__file__))
    #filepath = os.path.join(source_dir, "common-english-words.txt")
    filepath = stop_word_path
    
    #load stop words
    stopwords_f = open(filepath, 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    
    return stop_words
    

def process_and_rank_datasets(inputfolder,outputfolder,queries,stop_word_path):
    """
    Iterates through each subdirectory in the input folder and parses the docs then gets df and bm25 score through call to bm25.py functions.
    prints bm25 ranking .dat files to output folder 
    
    Args:
        inputfolder (str): Path to the dataset directory 
        outputfolder (str): Path to the output directory where ranking .dat files should be saved

    """
    
    datasets = []
    
    #stop words file assumed to be in
    stop_words = load_stopwords(stop_word_path)
    
    #queries = load_queries()
    
    #inputfolder = os.path.abspath(inputfolder)
    #outputfolder = os.path.abspath(outputfolder)
    
    #for each folder (dataset) in the directory
    for folder_name in os.listdir(inputfolder):
        folder_path = os.path.join(inputfolder, folder_name)

        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            
            #get the code from this folder path so we can process against the respective query (last 3 characters)
            folder_ref = folder_name[-3:]            
            
            #find the related query and parse it
            pq = parse_q(queries["R"+folder_ref], stop_words)
            print(pq)
            
            #create the document collection for this folder
            temp_coll = parse_docs(stop_words, folder_path)
            
            #get the df for the collection
            df = bm25.df(temp_coll)
            
            #dict {docid:bm25_score} 
            bm_scores = bm25.bm25(temp_coll, pq, df)

            outputpath = outputfolder+"\BM25IR_R"+ folder_ref + "Ranking.dat"
            if not os.path.exists(outputpath): #don't append to existing files, if we want a new output we assume they've been deleted
                wFile = open(outputpath, 'a')
                #wFile.write('[')
                count = 0
                for (k, v) in sorted(bm_scores.items(), key=lambda x: x[1], reverse=True):
                    wFile.write(f"['{k}', '{v}']\n")

                wFile.close()     

