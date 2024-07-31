import sys, os
import pandas as pd
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing.audio as Audio
import preprocessing.text as Text
import models.two_input as CNN

def main():
    path = sys.argv[1]
    df = pd.read_csv(path)

    audio_files = df['audio file']
    audio_features = []
    print("Extracting audio features...")
    for audio_file in audio_files:
        audio_feature = Audio.extract_audio_features(audio_file)
        audio_features.append(audio_feature)
    
    transcriptions = df['transcription']
    text_features = []
    lengths = []
    print("Extracting text features...")
    for transcription in transcriptions:
        if isinstance(transcription, float) and math.isnan(transcription):
            text_features.append([])
        else:
            length, features = Text.extract_text_features(transcription)
            text_features.append(features)
            lengths.append(length)
    
    max_length = max(lengths)
    real_text_features = []
    for text_feature in text_features:
        if len(text_feature) < max_length:
            diff = max_length - len(text_feature)
            text_feature = text_feature + [0] * diff
        real_text_features.append(text_feature)


    audio_features = np.stack(audio_features, axis=0)
    real_text_features = np.stack(real_text_features, axis=0)
    labels = df['label'].values

    print("Training model...")
    CNN.train_model(audio_features, real_text_features, labels)




if __name__ == '__main__':
	main()