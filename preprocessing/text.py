import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
    

def ascii_text(transcription):
    return [ord(char) for char in transcription]   

def extract_text_features(transcription):

    #Currently just using TFID vectorizer
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(transcriptions)

    # X = X.toarray()
    # return X

    ascii_feature = ascii_text(transcription)
    return len(ascii_feature), ascii_feature
  



