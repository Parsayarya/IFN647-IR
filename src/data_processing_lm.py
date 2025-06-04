import glob
import os
import string
from stemming import stem


stop_words_list = []

def load_stopwords(filepath="common-english-words.txt"):
    global stop_words_list
    with open(filepath, 'r', encoding='utf-8') as f:
        stop_words_list = f.read().split(',')
    stop_words_list = [word.strip().lower() for word in stop_words_list if word.strip()] 


def preprocess_text(text_content):
    if text_content is None:
        return []
    text = text_content.lower()
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    tokens = []
    for term_token in text.split():
        stemmed_term = stem(term_token)
        if len(stemmed_term) > 2 and stemmed_term not in stop_words_list:
            tokens.append(stemmed_term)
    return tokens

class BowDoc:
    def __init__(self, docid):
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_processed_terms(self, processed_terms_list):
        for term in processed_terms_list:
            self.terms[term] = self.terms.get(term, 0) + 1
            self.doc_len += 1

class BowColl:
    def __init__(self):
        self.docs = {}

    def add_doc(self, doc_obj):
        self.docs[doc_obj.docid] = doc_obj

def parse_dataset_xml(dataset_folder_path):
    dataset_coll = BowColl()
    xml_files = glob.glob(os.path.join(dataset_folder_path, "*.xml"))
    if not xml_files:
        print(f"Warning: No XML files found in {dataset_folder_path}")
        return dataset_coll
    for xml_file_path in xml_files:
        try:
            doc_id = None
            text_content_lines = []
            start_end_text = False
            with open(xml_file_path, 'r', encoding='iso-8859-1') as f_xml:
                for line in f_xml:
                    line = line.strip()
                    if not doc_id and line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                doc_id = part.split("=")[1].split("\"")[1]
                                break
                    if line.startswith("<text>"):
                        start_end_text = True
                        continue
                    elif line.startswith("</text>"):
                        start_end_text = False
                        break
                    if start_end_text:
                        line_content = line.replace("<p>", "").replace("</p>", "").strip()
                        if line_content:
                            text_content_lines.append(line_content)
            if doc_id and text_content_lines:
                full_text = " ".join(text_content_lines)
                processed_terms = preprocess_text(full_text)
                doc_obj = BowDoc(doc_id)
                doc_obj.add_processed_terms(processed_terms)
                dataset_coll.add_doc(doc_obj)
            elif not doc_id:
                 print(f"Warning: Could not find itemid in {xml_file_path}")
        except Exception as e:
            print(f"Error parsing XML file {xml_file_path}: {e}")
    return dataset_coll


def parse_queries(queries_filepath="Queries-1.txt"):
    queries = {}  # {query_id: [processed_terms]}
    current_query_id = None
    current_title = ""
    current_desc = ""
    current_narr = ""
    
    parsing_section = None # Can be "title", "desc"

    with open(queries_filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line: # Skip empty lines
                continue

            if line.startswith("<Query>"):
                # Reset for a new query
                current_query_id = None
                current_title = ""
                current_desc = ""
                current_narr = ""
                parsing_section = None
                continue

            if line.startswith("<num>"):
                try:
                    # Example: <num> Number: R101
                    parts = line.split("R")
                    if len(parts) > 1:
                        current_query_id_num_str = parts[-1].strip()
                        if current_query_id_num_str.isdigit():
                             current_query_id = "R" + current_query_id_num_str
                        else:
                            print(f"Warning (line {line_num}): Could not parse query ID num from '{line}'")
                            current_query_id = None
                    else:
                        print(f"Warning (line {line_num}): 'R' not found in num line '{line}'")
                        current_query_id = None
                    parsing_section = None # Reset section
                except Exception as e:
                    print(f"Error parsing query num line '{line}' (line {line_num}): {e}")
                    current_query_id = None
                continue

            if line.startswith("<title>"):
                parsing_section = "title"
                current_title += line.replace("<title>", "").strip() + " "
                if "</title>" in line:
                    current_title = current_title.replace("</title>", "").strip()
                    parsing_section = None
                continue
            elif line.startswith("</title>"):
                parsing_section = None
                continue
            
            if line.startswith("<desc>"):
                parsing_section = "desc"
                current_desc += line.replace("<desc>", "").replace("Description:", "").strip() + " "
                if "</desc>" in line:
                    current_desc = current_desc.replace("</desc>", "").strip()
                    parsing_section = None
                continue
            elif line.startswith("</desc>"):
                parsing_section = None
                continue

            if line.startswith("<narr>"):
                parsing_section = "narr"
                current_narr += line.replace("<narr>", "").replace("Narrative:", "").strip() + " "
                if "</narr>" in line:
                    current_narr = current_narr.replace("</narr>", "").strip()
                    parsing_section = None
                continue
            elif line.startswith("</narr>"):
                parsing_section = None
                continue

            # If we are inside a section, append the line content
            if parsing_section == "title":
                current_title += line.strip() + " "
                if "</title>" in line: # Handle case where content and closing tag are on same line
                    current_title = current_title.replace("</title>", "").strip()
                    parsing_section = None
            elif parsing_section == "desc":
                current_desc += line.strip() + " "
                if "</desc>" in line:
                    current_desc = current_desc.replace("</desc>", "").strip()
                    parsing_section = None
            elif parsing_section == "narr":
                current_narr += line.strip() + " "
                if "</narr>" in line:
                    current_narr = current_narr.replace("</narr>", "").strip()
                    parsing_section = None
            
            if line.startswith("</Query>"):
                if current_query_id:
                    # Clean up any trailing tags that might have been missed by same-line checks
                    final_title = current_title.replace("</title>", "").strip()
                    final_desc = current_desc.replace("</desc>", "").strip()
                    # final_narr = current_narr.replace("</narr>", "").strip()

                    full_query_text = (final_title + " " + 
                                       final_desc).strip()
                    
                    if full_query_text: # Only add if there's some text
                        queries[current_query_id] = preprocess_text(full_query_text)
                    else:
                        print(f"Warning: Query {current_query_id} resulted in empty text after combining fields.")
                
                # Reset for next query
                current_query_id = None
                current_title = ""
                current_desc = ""
                current_narr = ""
                parsing_section = None
                
    return queries

def calculate_collection_stats(dataset_coll: BowColl):
    collection_term_freqs = {}
    total_collection_words = 0
    if dataset_coll and dataset_coll.docs:
        for doc_id, doc_obj in dataset_coll.docs.items():
            total_collection_words += doc_obj.doc_len
            for term, freq in doc_obj.terms.items():
                collection_term_freqs[term] = collection_term_freqs.get(term, 0) + freq
    return collection_term_freqs, total_collection_words

if __name__ == '__main__':
    print("Testing data_processing_lm.py with revised extended query parsing...")
    load_stopwords() 
    print(f"Loaded {len(stop_words_list)} stopwords.")

    if os.path.exists("Queries-1.txt"):
        queries = parse_queries()
        print(f"Parsed {len(queries)} queries.")
        if 'R103' in queries:
             print(f"Example R103 processed terms: {queries['R103']}")
        if 'R101' in queries:
            print(f"Example R101 processed terms: {queries['R101']}")

    else:
        print("Queries-1.txt not found, skipping query parsing test.")

    # Test dataset parsing (requires DataSets/Dataset101 to exist)
    if os.path.exists(os.path.join("DataSets", "Dataset101")):
        dataset_coll_101 = parse_dataset_xml(os.path.join("DataSets", "Dataset101"))
        print(f"Parsed {len(dataset_coll_101.docs)} documents from Dataset101.")
    else:
        print("DataSets/Dataset101 not found, skipping dataset parsing test.")