from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LSTMModel:
    def __init__(self, X_train):
        self.model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics = metrics)
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs, batch_size, validation_data=validation_data)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {accuracy}, Loss: {loss}')


def train_LSTM(feature_vectors, labels):
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

    # # Normalize features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Define the model
    model = LSTMModel(X_train=X_train)

    model.compile()
    model.fit(X_train, y_train, validation_data=(X_test, y_test))
    model.evaluate(X_test, y_test)