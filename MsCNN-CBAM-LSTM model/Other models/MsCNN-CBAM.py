import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import time

class CBAM(tf.keras.layers.Layer):
    def __init__(self, channel_attention_ratio=0.5, spatial_attention_ratio=0.5):
        super(CBAM, self).__init__()
        self.channel_attention_ratio = channel_attention_ratio
        self.spatial_attention_ratio = spatial_attention_ratio

    def build(self, input_shape):
        self.channel_attention = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(int(input_shape[-1] * self.channel_attention_ratio), activation='relu'),
            tf.keras.layers.Dense(input_shape[-1], activation='sigmoid')
        ])

        self.spatial_attention = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        ])

    def call(self, inputs):
        channel_weights = self.channel_attention(inputs)
        channel_weights = tf.expand_dims(tf.expand_dims(channel_weights, 1), 1)
        weighted_input = inputs * channel_weights

        spatial_weights = self.spatial_attention(weighted_input)
        output = weighted_input * spatial_weights
        return output

class MsCNN_CBAM(tf.keras.Model):
    def __init__(self):
        super(MsCNN_CBAM, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same')
        self.cbam = CBAM()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cbam(x)
        x = self.flatten(x)
        return self.dense(x)

def load_data(dataset_dir):
    data = []
    labels = []
    label_dict = {'Guanine': 0, 'Adenine': 1, 'Cytosine': 2, 'Thymine': 3, '5-Methylcytosine': 4, 'Uracil': 5}

    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(label_dir, file)
                    data_content = np.loadtxt(file_path)
                    data.append(data_content)
                    labels.append(label_dict[label])

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def preprocess_data(data, labels):
    data = np.expand_dims(data, axis=-1)
    labels = tf.keras.utils.to_categorical(labels, num_classes=6)
    return data, labels

dataset_dir = './'
data, labels = load_data(dataset_dir)
data, labels = preprocess_data(data, labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = MsCNN_CBAM()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_train_time = time.time()
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
end_train_time = time.time()
training_time = end_train_time - start_train_time

start_test_time = time.time()
y_pred = model.predict(X_test)
end_test_time = time.time()
test_time = end_test_time - start_test_time

y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred_labels)
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Training Time: {training_time:.2f} seconds')
print(f'Test Time: {test_time:.2f} seconds')
print('Confusion Matrix:')
print(conf_matrix)
