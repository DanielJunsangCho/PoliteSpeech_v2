import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
    


def extract_text_features(transcriptions):

    #Currently just using TFID vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(transcriptions)

    X = X.toarray()
    return X


