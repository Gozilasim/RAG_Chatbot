from fucntions.preprocess_query import preprocess, expand_query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz



def word_match_score(query, document):
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())
    match_count = len(query_words.intersection(doc_words))

    if not query_words:
        return 0  # if empty matched words

    match_count = len(query_words.intersection(doc_words))
    return match_count / len(query_words)

def tf_idf_Filtering(db, n_doc, query, k):

    preprocess_query = preprocess(query)
    # Extract document contents
    all_docs = db.similarity_search(preprocess_query, k = n_doc)
    doc_texts = [doc.page_content for doc in all_docs]

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Calculate TF-IDF matrix for all documents
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)

    expanded_query = expand_query(preprocess_query)

    # Calculate TF-IDF scores for the query
    query_tfidf_vector = tfidf_vectorizer.transform([expanded_query])
    
    # Calculate cosine similarity between query TF-IDF vector and candidate TF-IDF vectors
    cosine_similarities = cosine_similarity(tfidf_matrix, query_tfidf_vector).flatten()

    # Calculate fuzzy matching scores
    fuzzy_scores = [fuzz.partial_ratio(expanded_query, doc.page_content) / 100.0 for doc in all_docs]

    # Calculate word matching scores
    word_match_scores = [word_match_score(expanded_query, doc.page_content) for doc in all_docs]

    # Combine the scores with additional word matching
    combined_scores = [
        ((0.5 * cos_sim + 0.8 * fuzz_score + 0.8 * word_match) / (0.5 + 0.8 + 0.8)) 
        for cos_sim, fuzz_score, word_match in zip(cosine_similarities, fuzzy_scores, word_match_scores)
    ]

    # Sort candidates based on TF-IDF cosine similarity
    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    filtered_docs = [all_docs[i] for i in sorted_indices]
    filtered_scores = [combined_scores[i] for i in sorted_indices]

    return filtered_docs[:k], filtered_scores[:k]