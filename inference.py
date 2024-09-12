import torch
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
# import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing.audio as Audio
import preprocessing.text as Text

# mlflow.set_tracking_uri("http://localhost:5000")
# logged_model = 'runs:/9fc065d6f5ff4d57abc82984bfe1281e/model'
# loaded_model = mlflow.pytorch.load_model(logged_model)

# save_path = "model.pth"
# torch.save(loaded_model, save_path)

model = torch.load("model.pth")


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
        curr = time.time()
        length, features = Text.extract_text_features(transcription)
        text_features.append(features)
        lengths.append(length)
        print(time.time() - curr)


# pad / truncate each transcription according to longest transcription
max_length = max(lengths)
real_text_features = []
for text_feature in text_features:
    if len(text_feature) < max_length:
        diff = max_length - len(text_feature)
        text_feature = text_feature + [0] * diff
    real_text_features.append(text_feature)

labels = df['label'].values
for i in range(len(audio_features)):
    combined_row = np.hstack((audio_features[i], real_text_features[i]))
    feature_tensor = torch.tensor(combined_row, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    ground_truth = labels[i]
    prediction = model(feature_tensor)
    prediction = (prediction > 0.5).float()

    print(ground_truth == 1, prediction)
    




