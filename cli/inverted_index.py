import pickle
import os
import math
from search_utils import tokenize_text, load_movies, load_stopwords
from collections import Counter
from constants import BM25_K1, BM25_B

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.stopwords = load_stopwords()
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text, self.stopwords)
        self.term_frequencies[doc_id] = Counter()


        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        avgdl = self._get_avg_doc_length()
        if avgdl == 0:
            return 0.0

        doc_length = self.doc_lengths[doc_id]

        length_norm = 1 - b + b * (doc_length / avgdl)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def _get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query, self.stopwords)
        scores = {}

        for doc_id in self.docmap:
            total_score = 0

            for token in tokens:
                total_score += self.bm25(doc_id, token)

            scores[doc_id] = total_score

        ranked_docs = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked_docs[:limit]

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"

            self._add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)
