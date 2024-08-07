import sys, os
import pandas as pd
import numpy as np
import math
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing.audio as Audio
import preprocessing.text as Text
import models.model3 as CNN

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

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

    
    combined_features = []
    for i in range(len(audio_features)):
        combined_row = np.hstack((audio_features[i], real_text_features[i]))
        combined_features.append(combined_row)
    combined_features = np.stack(combined_features, axis=0)


    labels = df['label'].values

    print("Setting up experiment...")
    mlflow.set_tracking_uri("http://localhost:5001")

    experiment_name = sys.argv[2]
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)



    print("Training model...")
    CNN.run_experiment(combined_features, labels, experiment_name)




if __name__ == '__main__':
	main()