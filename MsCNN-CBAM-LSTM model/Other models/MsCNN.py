import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import time

def load_data(data_path):
    data = []
    labels = []
    label_mapping = {'Guanine': 0, 'Adenine': 1, 'Cytosine': 2, 'Thymine': 3, '5-Methylcytosine': 4, 'Uracil': 5}
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            label = label_mapping.get(folder_name, None)
            if label is not None:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(folder_path, file_name)
                        data_matrix = np.loadtxt(file_path)
                        data.append(data_matrix[:, 1])
                        labels.append(label)

    return np.array(data), np.array(labels)
data_path = './'
X, y = load_data(data_path)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_mscnn(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 7, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_mscnn((X_train.shape[1], 1))
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
training_time = time.time() - start_time
start_time = time.time()
y_pred = np.argmax(model.predict(X_test), axis=1)
test_time = time.time() - start_time
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Training Time: {training_time:.2f} seconds")
print(f"Test Time: {test_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
