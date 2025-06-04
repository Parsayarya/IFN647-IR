import glob, os
import math

#dataset_code is only the number
def eval_input(bench_input, ranked_input, dataset_code):
    """
    gets the inputs necessary for evaluation
    
    Args:
        bench_input (str): location of benchmarks
        ranked_input (str): location of model ranked results
        dataset_code (str): the XXX code for the dataset

    """    
    coll = {}    
    
    rel_doc = {}
    C = {}
    ranked_doc = {}
    

    os.chdir(bench_input) #set directory to the benchmark inputs
    list_of_files_in_dir= glob.glob('*.txt') 
    #find the relevance file for this query
    
    print("in eval, dataset_code: " + dataset_code)
    matching_file = next((f for f in list_of_files_in_dir if dataset_code in f), None)
    for line in open(matching_file):
        line = line.strip()
        line1 = line.split()
        rel_doc[line1[1]] = int(float(line1[2]))
    
    #get the ranked output file
    os.chdir(ranked_input)
    #list_of_files_in_dir= glob.glob('*.dat') 
    list_of_files_in_dir= glob.glob('*BM25*.dat') #only get the rankings from BM25 in case other models are in the same directory
    #find the relevance file for this query
    matching_file = next((f for f in list_of_files_in_dir if dataset_code in f), None)  
    
    for line in open(matching_file):
        line = line.strip().strip("[]") #remove square brackets
        parts = [item.strip().strip("'") for item in line.split(",")] #split by comma and strip quotes
        C[parts[0]] = float(parts[1])
    # get the documents in terms of {rankingNO: documentID, ...}
    i=1
    for (k,v) in sorted(C.items(), key=lambda x:x[1], reverse=True):    
        if v >= 0:
            ranked_doc[i] = k
            i = i+1

        
    return (rel_doc, ranked_doc)

def eval(bench_folder, ranked_folder, dataset_code, eval_output_path, bm25_eval_files_exist):
    """
    evaluates the IR model and writes .dat and .csv files for each measure to the the eval_output_path directory.
    be careful not to have existing files in the location when this is run or it will probably just append to those existing files. 
    
    Args:
        bench_folder (str): location of benchmarks
        ranked_folder (str): location of model ranked results
        dataset_code (str): the XXX code for the dataset

    """    
    
    #rel_doc is our relevance judgements, ranked_doc is bm25 scores
    rel_doc, ranked_doc = eval_input(bench_folder, ranked_folder, dataset_code)

    #Average Precision
    relevant_doc_count = 0
    ap = 0.0 #to sum the precision
    for (n,id) in sorted(ranked_doc.items(), key=lambda x: int(x[0])): #sort by the integer value of the key to be safe
        if (rel_doc[id]==1): #if the doc at this rank is relevant according to ground truth
            relevant_doc_count =relevant_doc_count+1 #increment number of relevant docs 
            pi = float(relevant_doc_count)/float(int(n)) #calcuate precision for this rank
            ap = ap + pi #add to sum
    ap = ap/float(relevant_doc_count)
       
    os.chdir(eval_output_path)   
    
    if not bm25_eval_files_exist:
        #append the score the AP files   
        wFile = open("BM25_AP.csv", 'a')
        wFile.write(f"R{dataset_code}, {ap}\n")
        wFile.close()
        
        wFile = open("BM25_AP.dat", 'a')
        wFile.write((f"['R{dataset_code}', '{str(ap)}']\n"))
        wFile.close()    
   
    #Precision@12
    pk_relevant_doc_count = 0
    count = 0
    for (n,id) in sorted(ranked_doc.items(), key=lambda x: int(x[0])):
        count +=1
        if count > 11:
            break
        if (rel_doc[id]==1): #if the doc at this rank is relevant according to ground truth
            pk_relevant_doc_count =pk_relevant_doc_count+1 #increment number of relevant docs 
    pk = float(pk_relevant_doc_count)/12

    if not bm25_eval_files_exist:    
        #write the P@12 files    
        wFile = open("BM25_P12.csv", 'a')
        wFile.write(f"R{dataset_code}, {pk}\n")
        wFile.close()
        
        wFile = open("BM25_P12.dat", 'a')
        wFile.write((f"['R{dataset_code}', '{str(pk)}']\n"))
        wFile.close()
        
    #DCG@12
    dcg12 = 0
    count = 0
    for (n,id) in sorted(ranked_doc.items(), key=lambda x: int(x[0])):
        count +=1
        if count > 11:
            break            
        rel = rel_doc[id]       
        if n == 1:
            dcg12 += rel
        else:
            dcg12 += rel/math.log2(n)
    
    if not bm25_eval_files_exist:
        #write the DCG12 files
        wFile = open("BM25_DCG12.csv", 'a')
        wFile.write(f"R{dataset_code}, {dcg12}\n")
        wFile.close()
        
        wFile = open("BM25_DCG12.dat", 'a')
        wFile.write((f"['R{dataset_code}', '{str(dcg12)}']\n"))
        wFile.close()
        
    
    return ap, pk, dcg12
    