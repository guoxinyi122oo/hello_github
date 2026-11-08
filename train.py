import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化，并增加通道维度
x_train = x_train.astype("float32") /255.0
x_test = x_test.astype("float32") /255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

#搭建卷积神经网络 (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

# 测试准确率
print("\n正在测试集上评估...")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n测试集准确率: {acc*100:.2f}%")

# 保存模型
model.save("mnist_cnn.h5")
print(" 模型已保存为 mnist_cnn.h5")