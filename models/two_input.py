import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# [32, 11206], [32, 160]
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.audio_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.MaxPool1d(kernel_size=8, stride=2, padding=1)
        )

        self.text_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.MaxPool1d(kernel_size=8, stride=2, padding=1)
        )
        
        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(180992, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio, text):
        audio = audio.unsqueeze(1)
        text = text.unsqueeze(1)

        audio_feat = self.audio_features(audio)
        text_feat = self.text_features(text)

        audio_feat = self.Flatten(audio_feat)
        text_feat = self.Flatten(text_feat)

        combined = torch.cat((audio_feat, text_feat), dim=1)

        output = self.FC(combined)
        output = self.sigmoid(output)
        return output



class trainCNN:
    def __init__(self):
        self.model = CNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compile(self, learning_rate=0.00001, weight_decay=1e-5):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = nn.BCELoss()

    def fit(self, audio_train, text_train, labels_train, epochs=200, batch_size=32, val_threshold=6, validation_data=None):
        train_dataset = TensorDataset(audio_train, text_train, labels_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        if validation_data:
            audio_val, text_val, labels_val = validation_data
            val_dataset = TensorDataset(audio_val, text_val, labels_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_no_improvement = 0
        self.best_model = None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, (audio, text, labels) in enumerate(train_loader, 0):
                audio, text, labels = audio.to(self.device), text.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model.forward(audio, text)
                outputs = outputs.squeeze()

                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:  # Print every 10 iterations
                    print(f'[epoch: {epoch + 1}, iteration: {i}] loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

            if validation_data:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for audio, text, labels in val_loader:
                        audio, text, labels = audio.to(self.device), text.to(self.device), labels.to(self.device)
                        outputs = self.model(audio, text)
                        outputs = outputs.squeeze()
                        loss = self.loss(outputs, labels)
                        val_loss += loss.item()


                val_loss /= len(val_loader)
                print(f'[epoch: {epoch + 1}] validation loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improvement = 0
                    self.best_model = self.model.state_dict()
                else:
                    epochs_no_improvement += 1
                
                if epochs_no_improvement == val_threshold:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        if self.best_model:
            self.model.load_state_dict(self.best_model)
        print("Finished training")

    def eval(self, audio_test, text_test, labels_test):
        audio_test = audio_test.to(self.device)
        text_test = text_test.to(self.device)
        labels_test = labels_test.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(audio_test, text_test)
            outputs = outputs.squeeze()

        predictions = (outputs > 0.5).bool()
        labels_test = labels_test.bool()
        
        accuracy = binary_accuracy(predictions, labels_test)
        precision = binary_precision(predictions, labels_test)
        recall = binary_recall(predictions, labels_test)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        return accuracy.item(), precision.item(), recall.item()


def train_model(audio_features, text_features, labels):
    # Standardize values
    audio_scaler = StandardScaler()
    text_scaler = StandardScaler()
    audio_features_scaled = audio_scaler.fit_transform(audio_features)
    text_features_scaled = text_scaler.fit_transform(text_features)

    # Split audio, text, and labels to train, val, test
    audio_train, audio_test, text_train, text_test, labels_train, labels_test = train_test_split(
        audio_features, text_features, labels, test_size=0.2, random_state=42
        )
    audio_train, audio_val, text_train, text_val, labels_train, labels_val = train_test_split(
        audio_train, text_train, labels_train, test_size=0.2, random_state=42
        )

    # Convert to Float Tensors
    audio_train = torch.FloatTensor(audio_train)
    audio_val = torch.FloatTensor(audio_val)
    audio_test = torch.FloatTensor(audio_test)

    text_train = torch.FloatTensor(text_train)
    text_val = torch.FloatTensor(text_val)
    text_test = torch.FloatTensor(text_test)

    labels_train = torch.FloatTensor(labels_train)
    labels_val = torch.FloatTensor(labels_val)
    labels_test = torch.FloatTensor(labels_test)


    model = trainCNN()
    model.compile()
    model.fit(audio_train, text_train, labels_train, validation_data=(audio_val, text_val, labels_val))
    accuracy, precision, recall = model.eval(audio_test, text_test, labels_test)

