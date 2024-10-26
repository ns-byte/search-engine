import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import Stemmer
import math

ps = Stemmer.Stemmer('porter')

# Process stop words document
with open("/Users/nanditas/Desktop/TTDS-Lab0/ttds_2023_english_stop_words.txt", 'r') as stop_words_file:
    stop_words = stop_words_file.read().splitlines()
stop_words = {word.strip().lower() for word in stop_words}

def preprocess_text(text):
    '''Preprocess text by tokenizing, removing punctuation, stop words, and normalizing.'''
    clean_text = re.sub(r"[-'/\\]", ' ', text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    tokens = clean_text.split()
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [ps.stemWord(token) for token in tokens]
    return tokens

def xml_parser(filename):
    '''Parse XML documents and return a dictionary of documents'''
    try:
        tree = ET.parse(filename)
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return {}
    root = tree.getroot()  
    documents = {}
    for doc in root.findall('DOC'):
        doc_id = doc.find('DOCNO').text.strip()
        text_element = doc.find('TEXT') 
        if text_element is not None:
            text = ''.join(text_element.itertext()).strip()
        else:
            text = '' 
        headline_element = doc.find('HEADLINE')
        if headline_element is not None:
            headline = ''.join(headline_element.itertext()).strip()
            text = f"{headline} {text}"
        documents[doc_id] = text
    return documents

def build_inverted_index(documents):
    '''Build an inverted index from a dictionary of documents'''
    inverted_index = defaultdict(lambda: {'doc_freq': 0, 'postings': defaultdict(list)})
    for doc_id, text in documents.items():
        processed_tokens = preprocess_text(text)
        for position, term in enumerate(processed_tokens, start=1):
            term_data = inverted_index[term]
            if doc_id not in term_data['postings']:
                term_data['doc_freq'] += 1
            term_data['postings'][doc_id].append(position)
    return inverted_index

def save_inverted_index_to_file(inverted_index, output_file):
    '''Save the inverted index to a file'''
    with open(output_file, 'w') as file:
        for term in sorted(inverted_index.keys()): 
            data = inverted_index[term]  
            file.write(f"{term}: {data['doc_freq']}\n")
            for doc_id, positions in data['postings'].items():
                positions_str = ', '.join(map(str, positions))
                file.write(f"\t{doc_id}: {positions_str}\n")
            file.write("\n")
            
def load_inverted_index_from_file(input_file):
    '''Load an inverted index from a file'''
    inverted_index = defaultdict(lambda: {'doc_freq': 0, 'postings': defaultdict(list)})
    with open(input_file, 'r') as file:
        term = None
        for line in file:
            if line.strip() == "": 
                continue
            if not line.startswith("\t"): 
                term, freq = line.strip().split(": ")
                freq = int(freq) 
                inverted_index[term]['doc_freq'] = freq
            else:
                doc_id, positions = line.strip().split(": ")
                positions = list(map(int, positions.split(",")))
                inverted_index[term]['postings'][doc_id] = positions
    return inverted_index


def preprocess_query(query, flag):
    '''Preprocess queries by tokenizing, removing punctuation and stop words, normalization'''
    clean_query = re.sub(r'[^\w\s]', '', query)  
    tokens = clean_query.split()
    tokens = [token.lower() for token in tokens]
    if flag:
        tokens = [token for token in tokens if token not in stop_words]
    tokens = [ps.stemWord(token) for token in tokens]
    return tokens

def boolean_search_parser(inverted_index, query):
    '''Parse and execute boolean search queries'''
    tokens = query.split()
    terms = []
    operators = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ['AND', 'OR', 'NOT']:
            # Check for "AND NOT" and "OR NOT"
            if i + 1 < len(tokens) and tokens[i+1] == 'NOT':
                combined_operator = f"{token} NOT"
                operators.append(combined_operator)
                i += 1
            else:
                operators.append(token)
        else:
            preprocessed_term = preprocess_query(token, flag=0)
            if preprocessed_term:
                terms.append(preprocessed_term[0])
        i += 1
    if len(terms) == 1 and len(operators) == 0:
        return set(inverted_index[terms[0]]['postings'].keys())
    return boolean_search(inverted_index, terms, operators)

def boolean_not(inverted_index, term):
    '''Return documents that do not contain the given term'''
    # Get all docs and subtract those containing the term
    all_docs = {doc for term_data in inverted_index.values() for doc in term_data['postings']}
    docs_with_term = set(inverted_index.get(term, {}).get('postings', {}))
    return all_docs - docs_with_term

def boolean_search(inverted_index, terms, operators):
    '''Execute a boolean search with AND, OR, NOT, AND NOT, OR NOT operators'''
    if operators and operators[0] == 'NOT':
        return boolean_not(inverted_index, terms[0])
    result_docs = set(inverted_index[terms[0]]['postings'].keys())
    for i, operator in enumerate(operators):
        if i + 1 >= len(terms):
            break
        next_term_docs = set(inverted_index[terms[i+1]]['postings'].keys())
        if operator == 'AND':
            result_docs = result_docs & next_term_docs  # Intersection
        elif operator == 'OR':
            result_docs = result_docs | next_term_docs  # Union
        elif operator == 'NOT':
            result_docs = result_docs - next_term_docs  # Exclusion
        elif operator == 'AND NOT':
            next_term_docs = boolean_not(inverted_index, terms[i+1])
            result_docs = result_docs & next_term_docs  # AND NOT
        elif operator == 'OR NOT':
            next_term_docs = boolean_not(inverted_index, terms[i+1])
            result_docs = result_docs | next_term_docs # OR NOT
    return result_docs

def phrase_search(inverted_index, preprocessed_phrase):
    '''Search for exact phrases'''
    terms = preprocessed_phrase.strip('"').split()    
    if len(terms) < 2:
        return set()
    result_docs = set(inverted_index[terms[0]]['postings'].keys())
    for term in terms[1:]:
        if term in inverted_index:
            result_docs &= set(inverted_index[term]['postings'].keys())
        else:
            print(f"'{term}' not in index")
    valid_docs = set()
    for doc in result_docs:
        positions_list = []
        for term in terms:
            positions = inverted_index[term]['postings'][doc]
            positions_list.append(positions)
        if phrase_positions(positions_list):
            valid_docs.add(doc)
    return valid_docs

def phrase_positions(positions_list):
    '''Check if the terms appear in exact phrase order'''
    for i in range(len(positions_list[0])):
        start_pos = positions_list[0][i]
        match = True
        for j in range(1, len(positions_list)):
            if start_pos + j not in positions_list[j]:
                match = False
                break
        if match:
            return True
    return False

def handle_proximity_query(proximity_query, inverted_index):
    '''Extract terms and preprocess proximity query'''
    proximity, terms = proximity_query[1:].split('(')
    max_distance = int(proximity)
    term1, term2 = terms.replace(')', '').split(', ')
    term1_processed = preprocess_query(term1, flag=0)[0]
    return proximity_search(inverted_index, term1_processed, term2, max_distance)

def proximity_search(inverted_index, term1, term2, max_distance):
    '''Search for terms within a given proximity'''
    docs_with_both_terms = set(inverted_index[term1]['postings'].keys()) & set(inverted_index[term2]['postings'].keys())
    valid_docs = set()
    for doc in docs_with_both_terms:
        position1 = inverted_index[term1]['postings'][doc]
        position2 = inverted_index[term2]['postings'][doc]
        if check_proximity(position1, position2, max_distance):
            valid_docs.add(doc)
    return valid_docs

def check_proximity(position1, position2, max_distance):
    '''Check if two terms appear within the given distance of each other'''
    for pos1 in position1:
        for pos2 in position2:
            if abs(pos1 - pos2) <= max_distance:
                return True
    return False
    
def process_combined_queries(inverted_index, preprocessed_query):
    '''Process queries that combine Boolean operators with phrase or proximity searches'''
    if ' AND NOT ' in preprocessed_query:
        phrases, remaining_query = preprocessed_query.split(' AND NOT ', 1)
        phrase_docs = phrase_search(inverted_index, phrases)
        remaining_docs = boolean_not(inverted_index, remaining_query.strip())
        return phrase_docs & remaining_docs
    if ' AND ' in preprocessed_query:
        phrases, remaining_query = preprocessed_query.split(' AND ', 1)
        phrase_docs = phrase_search(inverted_index, phrases)
        remaining_docs = query_handler(inverted_index, remaining_query.strip())
        return phrase_docs & remaining_docs
    elif ' OR ' in preprocessed_query:
        phrases, remaining_query = preprocessed_query.split(' OR ', 1)
        phrase_docs = phrase_search(inverted_index, phrases)
        remaining_docs = query_handler(inverted_index, remaining_query.strip())
        return phrase_docs | remaining_docs
    elif ' NOT ' in preprocessed_query:
        phrases, remaining_query = preprocessed_query.split(' NOT ', 1)
        phrase_docs = phrase_search(inverted_index, phrases)
        remaining_docs = query_handler(inverted_index, remaining_query.strip())
        return phrase_docs - remaining_docs
    return query_handler(inverted_index, preprocessed_query)

def query_handler(inverted_index, query):
    '''Handle phrase, proximity, and boolean queries'''
    tokens = re.findall(r'"[^"]*"|\S+', query)
    preprocessed_tokens = []
    for token in tokens:
        if token.startswith('"') and token.endswith('"'):  # Phrase search
            phrase = token.strip('"')
            preprocessed_tokens.append(f'"{" ".join(preprocess_query(phrase, flag=0))}"')
        elif token.startswith('#'):  # Proximity search
            preprocessed_tokens.append(token)
        elif token in ['AND', 'OR', 'NOT']:  # Boolean operators
            preprocessed_tokens.append(token)
        else:
            preprocessed_tokens.append(preprocess_query(token, flag=0)[0])  # Preprocess regular terms
    preprocessed_query = ' '.join(preprocessed_tokens)
    if '"' in preprocessed_query and any(op in preprocessed_query for op in ['AND', 'OR', 'NOT']):
        return process_combined_queries(inverted_index, preprocessed_query)
    elif preprocessed_query.startswith('#'):  # Proximity query
        return handle_proximity_query(preprocessed_query, inverted_index)
    elif preprocessed_query.startswith('"'):  # Phrase query
        return phrase_search(inverted_index, preprocessed_query)
    return boolean_search_parser(inverted_index, preprocessed_query)


def save_query_results(query_results, output_file):
    '''Save query results to a file'''
    with open(output_file, 'w') as file:
        for query_num, doc_ids in query_results.items():
            for doc_id in doc_ids:
                file.write(f"{query_num}: {doc_id}\n")

def query_and_save_results(inverted_index, queries, output_file):
    '''Process and save query results'''
    query_results = {}
    for query_num, query in queries.items():
        result_docs = query_handler(inverted_index, query)
        if isinstance(result_docs, set):
            query_results[query_num] = sorted(map(int, filter(str.isdigit, result_docs)))
        else:
            print(f"Invalid result for query {query_num}, {result_docs}")
    save_query_results(query_results, output_file)

###################################
###Ranked Queries Implementation###
###################################

def calculate_tfidf_weight(term, doc_id, tf, df, N):
    """Calculate the TFIDF weight for a term in a document."""
    term_frequency = 1 + math.log10(tf)
    inverse_document_frequency = math.log10(N / df)
    return term_frequency * inverse_document_frequency

def calculate_query_score(query_terms, doc_id, doc_terms, inverted_index, N):
    """Calculate the score for a query-document pair."""
    score = 0
    for term in query_terms:
        if term in doc_terms:
            tf = len(inverted_index[term]['postings'][doc_id])
            df = inverted_index[term]['doc_freq']
            score += calculate_tfidf_weight(term, doc_id, tf, df, N)
    return score

def rank_documents(query_num, query, inverted_index, N, documents):
    """Rank documents based on their TF-IDF score for the given query."""
    preprocessed_query = preprocess_query(query, flag=1)
    document_scores = {}
    for doc_id, doc_text in documents.items():
        preprocessed_doc = preprocess_text(doc_text)
        score = calculate_query_score(preprocessed_query, doc_id, preprocessed_doc, inverted_index, N)
        if score > 0:
            document_scores[doc_id] = score
    ranked_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:150]

def save_tfidf_results(queries, inverted_index, documents, output_file):
    """Process queries, rank documents, and save the results to a file."""
    N = len(documents)
    with open(output_file, 'w') as file:
        for query_num, query in queries.items():
            ranked_docs = rank_documents(query_num, query, inverted_index, N, documents)
            for doc_id, score in ranked_docs:
                file.write(f"{query_num},{doc_id},{score:.4f}\n")

document = xml_parser('/Users/nanditas/Downloads/cw1collection/trec.5000.xml')
inverted_index = build_inverted_index(document)
save_inverted_index_to_file(inverted_index, 'index.txt')
inverted_index = load_inverted_index_from_file('index.txt')

# Boolean Queries
queries = {
    1: "Sadness",
    2: "Glasgow AND SCOTLAND",
    3: "corporate AND taxes",
    4: '"corporate taxes"',
    5: '#30(corporate, taxes)',
    6: '"middle east" AND israel',
    7: '#5(Palestinian, organisations)',
    8: '"Financial times" AND NOT BBC',
    9: '"wall street" AND "dow jones"',
    10: '#20(dow, stocks)'
}

# Output results to results.boolean.txt
output_file = 'results.boolean.txt'
query_and_save_results(inverted_index, queries, output_file)

# Ranked Queries
queries_ranked = {
    1: "corporation tax reduction",
    2: "stock market in China",
    3: "health industries",
    4: "the artificial intelligence market",
    5: "the Israeli Palestinian conflict",
    6: "information retrieval",
    7: "Dow Jones industrial average stocks",
    8: "will be there a reduction in taxes for corporates?",
    9: "the gold prices versus the dollar value",
    10: "FT article on the deal between BBC and BSkyB"
}

# Output results to results.ranked.txt
output_file = 'results.ranked.txt'
save_tfidf_results(queries_ranked, inverted_index, document, output_file)
