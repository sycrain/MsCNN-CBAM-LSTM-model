import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Concatenate, Input, \
    GlobalAveragePooling1D, Reshape, Multiply
from keras.callbacks import CSVLogger, Callback
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import itertools


# CBAM Implementation (Channel Attention + Spatial Attention)
def channel_attention(input_feature, ratio=8):
    channel_avg = GlobalAveragePooling1D()(input_feature)
    channel_max = GlobalAveragePooling1D()(input_feature)

    channel_avg = Dense(input_feature.shape[-1] // ratio, activation='relu')(channel_avg)
    channel_avg = Dense(input_feature.shape[-1], activation='sigmoid')(channel_avg)

    channel_max = Dense(input_feature.shape[-1] // ratio, activation='relu')(channel_max)
    channel_max = Dense(input_feature.shape[-1], activation='sigmoid')(channel_max)

    channel_attention = Multiply()([input_feature, channel_avg])
    channel_attention = Multiply()([channel_attention, channel_max])

    return channel_attention


def spatial_attention(input_feature):
    avg_pool = GlobalAveragePooling1D()(input_feature)
    max_pool = GlobalAveragePooling1D()(input_feature)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = Dense(input_feature.shape[1], activation='sigmoid')(concat)

    spatial_attention = Reshape((input_feature.shape[1], 1))(spatial_attention)
    spatial_attention = Multiply()([input_feature, spatial_attention])

    return spatial_attention


# 读取文件夹中的数据
def load_data_from_folders(folder_paths):
    X_data = []
    Y_labels = []

    for label, folder in enumerate(folder_paths):
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder, filename)
                data = np.loadtxt(file_path)  # 读取txt文件
                X_data.append(data[:, 1])  # 只取第二列作为特征
                Y_labels.append(label)  # 标签是文件夹的编号

    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels)
    return X_data, Y_labels


# 数据集的文件夹路径
folder_paths = ['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil']

# 载入数据
X, Y = load_data_from_folders(folder_paths)
X = X.reshape(-1, 500, 1)  # 将数据调整为 (样本数, 500, 1)

# 分类编码为One-hot
Y_onehot = np_utils.to_categorical(Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=0)


# 构建CNN-LSTM模型并加入CBAM模块
def create_model():
    # 定义输入层
    input_layer = Input(shape=(500, 1))

    # 多尺度卷积部分
    conv1 = Conv1D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv1 = channel_attention(pool1)  # Channel Attention

    conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same", activation='relu')(input_layer)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv2 = channel_attention(pool2)  # Channel Attention

    conv3 = Conv1D(filters=32, kernel_size=7, strides=1, padding="same", activation='relu')(input_layer)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv3 = channel_attention(pool3)  # Channel Attention

    # Concatenate不同尺度的卷积结果
    concat = Concatenate()([pool1, pool2, pool3])
    concat = spatial_attention(concat)  # Spatial Attention

    # LSTM层
    lstm = LSTM(20, return_sequences=False)(concat)

    output = Dense(6, activation='softmax')(lstm)
    # 使用Functional API构建模型
    model = Model(inputs=input_layer, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='model_structure.png', show_shapes=True)  # 保存网络结构图
    return model


# 自定义回调函数来保存每轮的训练集和测试集的损失与精度
class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses_train = []
        self.accuracy_train = []
        self.losses_val = []
        self.accuracy_val = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses_train.append(logs.get('loss'))
        self.accuracy_train.append(logs.get('accuracy'))
        self.losses_val.append(logs.get('val_loss'))
        self.accuracy_val.append(logs.get('val_accuracy'))


# 保存训练日志
csv_logger = CSVLogger('training_log.csv', append=False)
loss_history = LossHistory()

# 创建模型
model = create_model()

# 训练模型
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test),
                    callbacks=[csv_logger, loss_history])

# 保存每一轮的训练集和测试集的损失和精度到txt文件
#with open('train_val_loss_accuracy1.txt', 'w') as f:
#    for i in range(len(loss_history.losses_train)):
 #       f.write(f"{i + 1}, Train Loss: {loss_history.losses_train[i]}, Train Accuracy: {loss_history.accuracy_train[i]}, "
 #               f"Val Loss: {loss_history.losses_val[i]}, Val Accuracy: {loss_history.accuracy_val[i]}\n")

with open('train_val_loss_accuracy1.txt', 'w') as f:
    for i in range(len(loss_history.losses_train)):
        f.write(f"{i + 1},{loss_history.losses_train[i]},{loss_history.accuracy_train[i]}, "
                f"{loss_history.losses_val[i]},{loss_history.accuracy_val[i]}\n")


# 预测测试集
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# 计算分类指标
accuracy = accuracy_score(Y_true, Y_pred_classes)
precision = precision_score(Y_true, Y_pred_classes, average='weighted')
recall = recall_score(Y_true, Y_pred_classes, average='weighted')
f1 = f1_score(Y_true, Y_pred_classes, average='weighted')

print(f"Accuracy (Overall): {accuracy}")
print(f"Precision (Overall): {precision}")
print(f"Recall (Overall): {recall}")
print(f"F1 Score (Overall): {f1}")

# 按类别计算并打印每种物质的分类指标
target_names = ['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil']

print("\nPer-Class Metrics:")
for i, label in enumerate(target_names):
    class_accuracy = accuracy_score(Y_true == i, Y_pred_classes == i)
    class_precision = precision_score(Y_true, Y_pred_classes, average=None)[i]
    class_recall = recall_score(Y_true, Y_pred_classes, average=None)[i]
    class_f1 = f1_score(Y_true, Y_pred_classes, average=None)[i]
    print(f"{label}:")
    print(f"  Accuracy: {class_accuracy:.4f}")
    print(f"  Precision: {class_precision:.4f}")
    print(f"  Recall: {class_recall:.4f}")
    print(f"  F1 Score: {class_f1:.4f}")

# 混淆矩阵
cm = confusion_matrix(Y_true, Y_pred_classes)

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('matrix1.png', dpi=1200, bbox_inches='tight')
    plt.show()

# 绘制并保存混淆矩阵
plot_confusion_matrix(cm, classes=target_names)
