import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def tokenize_text(text: str, stopwords: list[str]) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)

    tokens = text.lower().translate(translator).split()
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def tokenize_term(term: str, stopwords: list[str]) -> str:
    tokens = tokenize_text(term, stopwords)

    if len(tokens) != 1:
        raise ValueError("Expected exactly one token")

    return tokens[0]

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords():
    translator = str.maketrans("", "", string.punctuation)

    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        words = file.read().splitlines()

    return [word.lower().translate(translator) for word in words]
