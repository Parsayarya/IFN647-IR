import os
import string
import xml.etree.ElementTree as ET
from stemming import stem

class Doc:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.terms = {}

    def add_term(self, term):
        term = stem(term.lower())
        if len(term) > 2:
            self.terms[term] = self.terms.get(term, 0) + 1

def load_stop_words(filepath):
    with open(filepath, 'r') as f:
        return set(f.read().strip().split(','))

def parse_docs(dataset_path, stop_words):
    documents = {}
    for file in os.listdir(dataset_path):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join(dataset_path, file))
            root = tree.getroot()
            doc_id = root.attrib["itemid"]
            doc = Doc(doc_id)
            text_elements = root.findall(".//text//p")
            for elem in text_elements:
                line = elem.text if elem.text else ""
                line = line.translate(str.maketrans('', '', string.digits))
                line = line.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                for word in line.split():
                    if word not in stop_words:
                        doc.add_term(word)
            documents[doc_id] = doc
    return documents

def parse_query(raw_query, stop_words):
    query_terms = {}
    raw_query = raw_query.translate(str.maketrans('', '', string.digits))
    raw_query = raw_query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    for word in raw_query.split():
        word = stem(word.lower())
        if word not in stop_words and len(word) > 2:
            query_terms[word] = query_terms.get(word, 0) + 1
    return query_terms