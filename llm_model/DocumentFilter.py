from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from llm_model.TextPreprocessor import TextPreprocessor
import pandas as pd
from llm_model.config import config
import requests

QueryPrep = TextPreprocessor()
# columns_to_round = ["Similirity Score", "cosine", "fuzzy", "word", "Ranking Score"]
class DocumentFilter:
    def __init__(self, headers):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.headers = headers

    def word_match_score(self, query, document):
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        match_count = len(query_words.intersection(doc_words))
        return match_count / len(query_words)
    

    def documentRetreival(self, query, index, return_k = 20):
        preprocessQuery = QueryPrep.preprocess(query)
        retriever = index.as_retriever(similarity_top_k=100)
        res_nodes = retriever.retrieve(preprocessQuery)
        doc_texts = [node.text for node in res_nodes]
        similarity_score = [node.score for node in res_nodes]


        tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
        query_tfidf_vector = self.tfidf_vectorizer.transform([preprocessQuery])


        cosine_similarities = cosine_similarity(tfidf_matrix, query_tfidf_vector).flatten()
        fuzzy_scores = [fuzz.partial_ratio(preprocessQuery, doc) / 100.0 for doc in doc_texts]
        word_match_scores = [self.word_match_score(preprocessQuery, doc) for doc in doc_texts]

        combined_scores = [(cos_sim + fuzz_score + word_match + semantic * 0.5) / 3.5 for cos_sim, fuzz_score, word_match, semantic in zip(cosine_similarities, fuzzy_scores, word_match_scores, similarity_score)]
        sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
        top_indices = sorted_indices[:return_k]


        return [res_nodes[i] for i in top_indices]
    
    # Tempporary not using this
    # def rerankJsonIndex(self, res_nodes, query):
    #     payload = {  
    #     "query": query,
    #     "texts": [node.get_content() for node in res_nodes],
    #     "return_text": True,
    #     "truncate" : True, 
    #     "return_scores": True
    #     } 

    #     endpoint = config['SERVICE_HOST_EMB'] + "/rerank"

    #     rerank_res = requests.post(
    #     endpoint,
    #     json=payload,
    #     verify='ca-bundle.crt',
    #     headers=self.headers
    #     )

    #     return rerank_res
