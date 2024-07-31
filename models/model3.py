import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
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

            nn.MaxPool1d(kernel_size=8, stride=2, padding=1)
        )
        
        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(181248, 1) #with padding=2
        # self.FC = nn.Linear(180992, 1) #with padding=1
        # self.FC = nn.Linear(180736, 1) #without padding
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.Flatten(x)
        x = self.FC(x)
        x = self.sigmoid(x)
        return x



class trainCNN:
    def __init__(self):
        self.model = CNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compile(self, learning_rate=0.000001, weight_decay=1e-4):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = nn.BCELoss()

    def fit(self, X_train, y_train, epochs=500, batch_size=32, val_threshold=6, validation_data=None):
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        if validation_data:
            X_val, y_val = validation_data
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_no_improvement = 0
        self.best_model = None

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model.forward(inputs)
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
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
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

        self.model.load_state_dict(self.best_model)
        print("Finished training")

    def eval(self, X_test, y_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test)
            outputs = outputs.squeeze()

        predictions = (outputs > 0.5).float()
        y_test = y_test.long()
        
        accuracy = binary_accuracy(predictions, y_test)
        precision = binary_precision(predictions, y_test)
        recall = binary_recall(predictions, y_test)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        return accuracy.item(), precision.item(), recall.item()


def train_model(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = trainCNN()
    model.compile()
    model.fit(X_train, y_train, validation_data=(X_val, y_val))
    accuracy, precision, recall = model.eval(X_test, y_test)

