import preprocessing.audio as Audio
import preprocessing.text as Text
import models.model1 as CNN
import models.model2 as LSTM
import sys, os
import pandas as pd
import numpy as np


def main():
    path = sys.argv[1]
    df = pd.read_csv(path)

    audio_files = df['audio file']
    audio_features = []
    for audio_file in audio_files:
        audio_feature = Audio.extract_audio_features(audio_file)
        audio_features.append(audio_feature)
    
    transcriptions = df['transcription']
    text_features = Text.extract_text_features(transcriptions)
    
    combined_features = []
    for i in range(len(audio_features)):
        combined_row = np.hstack((audio_features[i], text_features[i]))
        combined_features.append(combined_row)
    
    combined_features = np.stack(combined_features, axis=0)

    labels = df['label'].values
    CNN.train_model(combined_features, labels)

    # labels = df['label']
    # LSTM.train_LSTM(combined_features, labels)





if __name__ == '__main__':
	main()