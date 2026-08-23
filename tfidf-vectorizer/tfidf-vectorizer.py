import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """Return tfidf_matrix and vocabulary in a dictionary."""
    tokenized = [document.lower().split() for document in documents]
    vocab = sorted({token for tokens in tokenized for token in tokens})
    index = {token: position for position, token in enumerate(vocab)}
    matrix = np.zeros((len(documents), len(vocab)), dtype = float)

    #IDF
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens)) # use set to prevent double update, only one update can happen for each docs

    # TF
    for row, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for token, count in counts.items():
            tf = count / len(tokens)
            idf = math.log(len(documents) / doc_freq[token])
            matrix[row, index[token]] = tf * idf

    return {"tfidf_matrix": matrix, "vocabulary": vocab}