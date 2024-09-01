# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import sys, os, time
import pandas as pd
import numpy as np
import math
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing.audio as Audio
import preprocessing.text as Text

mlflow.set_tracking_uri("http://localhost:5000")
logged_model = 'runs:/9fc065d6f5ff4d57abc82984bfe1281e/model'
loaded_model = mlflow.pytorch.load_model(logged_model)


path = sys.argv[1]
df = pd.read_csv(path)

audio_files = df['audio file']
audio_features = []
print("Extracting audio features...")
for audio_file in audio_files:
    curr = time.time()
    audio_feature = Audio.extract_audio_features(audio_file)
    audio_features.append(audio_feature)
    print(time.time() - curr)

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


# loaded_model.predict(pd.DataFrame(data))


