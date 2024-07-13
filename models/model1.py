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
        self.CNN1 = nn.Conv1d(1, 64, kernel_size=4, stride=2)
        self.Dropout1 = nn.Dropout(0.2)
        self.CNN2 = nn.Conv1d(64, 32, kernel_size=4, stride=2)
        self.Dropout2 = nn.Dropout(0.2)
        self.AdaptivePool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.CNN1(x)
        x = self.Dropout1(x)
        x = self.CNN2(x)
        x = self.Dropout2(x)
        x = self.AdaptivePool(x)
        x = x.squeeze()
        x = self.FC(x)
        x = self.sigmoid(x)
        return x


class trainCNN:
    def __init__(self):
        self.model = CNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compile(self, learning_rate=0.0001):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.BCELoss()

    def fit(self, X_train, y_train, epochs=50, batch_size=16, validation_data=None):
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if validation_data:
            print("validation_data")
            # gotta make a new dataset for validation data as well for evaluation time

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

                running_loss = loss.item()
                print(f'[epoch: {epoch + 1}, iteration: {i + 1:5d}] loss: {running_loss}')
                running_loss = 0.0

        print("Finished training")

    def eval(self, X_test, y_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test)
            outputs = outputs.squeeze()

        print(outputs, y_test)
        predictions = (outputs > 0.5).float()
        y_test = y_test.long()
        
        accuracy = binary_accuracy(predictions, y_test)
        precision = binary_precision(predictions, y_test)
        recall = binary_recall(predictions, y_test)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        return accuracy.item(), precision.item(), recall.item()

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


    model = trainCNN()
    model.compile()
    model.fit(X_train, y_train)
    accuracy, precision, recall = model.eval(X_test, y_test)
