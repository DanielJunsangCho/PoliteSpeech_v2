import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys, os, time
import pandas as pd
import numpy as np
import math
import mlflow
import optuna
from optuna import multi_objective
from optuna.trial import TrialState

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing.audio as Audio
import preprocessing.text as Text

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
optuna.logging.set_verbosity(optuna.logging.ERROR)

DEVICE = torch.device("cpu")
EPOCHS = 100
BATCHSIZE = 32
N_TRAIN_EXAMPLES = BATCHSIZE * 50
N_VALID_EXAMPLES = BATCHSIZE * 20
params = {}
features, labels = None, None

def define_model_standard(trial):
    layers = [nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm1d(64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    
    nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    
    nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),

    nn.MaxPool1d(kernel_size=8, stride=2, padding=1),
    nn.Flatten(),
    nn.Linear(181248, 1),
    nn.Sigmoid()]

    return nn.Sequential(*layers)



def define_model(trial):
    global params 

    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers = []

    in_features = 1
    # input_dim = 11366
    input_dim = 11333
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 128)
        kernel_size = trial.suggest_int("kernel_size_l{}".format(i), 2, 8)
        stride = trial.suggest_int("stride_l{}".format(i), 2, 3)
        padding = trial.suggest_int("padding_l{}".format(i), 2, 3)
        layers.append(nn.Conv1d(in_features, out_features, kernel_size, stride, padding))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU(inplace=True))

        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
        input_dim = ((input_dim + (2 * padding) - kernel_size) // stride) + 1

        params["out_features_l{}".format(i)] = out_features
        params["kernel_size_l{}".format(i)] = kernel_size
        params["stride_l{}".format(i)] = stride
        params["padding_l{}".format(i)] = padding
    
    layers.append(nn.Flatten())
    final_dim = in_features * input_dim
    layers.append(nn.Linear(final_dim, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def split_data():
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_valid = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    valid_dataset = TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(y_valid))
    valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True)
    return train_loader, valid_loader

def objective(trial):
    global params
    with mlflow.start_run(nested=True):
        params = {}
        model = define_model_standard(trial).to(DEVICE)

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        decay = trial.suggest_float("decay", 1e-6, 1e-4)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=decay)
        params["optimizer"] = optimizer_name
        params["learning rate"] = lr
        params["decay"] = decay

        loss_func = nn.BCELoss()
        train_loader, valid_loader = split_data()
        for epoch in range(EPOCHS):
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                # Limiting training data for faster epochs
                if i * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(valid_loader):
                    # Limiting validation data
                    if i * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    preds = (outputs > 0.5).float()
                    all_preds.append(preds)
                    all_labels.append(labels)

            all_preds = torch.cat(all_preds).int()
            all_labels = torch.cat(all_labels).int()

            accuracy = binary_accuracy(all_preds, all_labels)
            precision = binary_precision(all_preds, all_labels)
            recall = binary_recall(all_preds, all_labels)

            mlflow.log_metric("accuracy", accuracy.item(), step=epoch)
            mlflow.log_metric("precision", precision.item(), step=epoch)
            mlflow.log_metric("recall", recall.item(), step=epoch)

            trial.report((accuracy.item(), precision.item(), recall.item()), epoch)

        mlflow.log_params(params)


        artifact_path = "model"
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            input_example=torch.randn(1, 1, 11366).numpy(),
        )

    return accuracy.item(), precision.item(), recall.item()


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def run_experiment(features_data, labels_data, experiment_id):
    global features, labels
    features = features_data
    labels = labels_data

    mlflow.set_experiment(experiment_id=experiment_id)

    study = multi_objective.create_study(directions=['maximize', 'maximize', 'maximize'])
    study.optimize(objective, n_trials=100)

    # for trial in study.trials:
    #     if trial.state == TrialState.COMPLETE:
    #         accuracy, precision, recall = trial.values
    #         if accuracy < 0.9 or precision < 0.9 or recall < 0.9:
    #             trial.state = TrialState.PRUNED

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    for trial in study.best_trials:
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            


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
    experiment_name = sys.argv[2]
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    print("Training model...")
    start = time.time()
    run_experiment(combined_features, labels, experiment_id)
    print(f"Latency: {time.time() - start}")


if __name__ == '__main__':
	main()