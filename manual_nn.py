
import numpy as np
import tensorflow as tf

# 1. 加载 MNIST 数据集

print("正在加载 MNIST 数据集...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 将 28×28 图片展开为 784 个像素
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 将像素从 0~255 归一化到 0~1
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print("训练图片：", x_train.shape)
print("训练标签：", y_train.shape)
print("测试图片：", x_test.shape)
print("测试标签：", y_test.shape)



# 2. 初始化神经网络参数


# 输入层 784 个神经元
# 隐藏层 128 个神经元
# 输出层 10 个神经元

np.random.seed(42)

W1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 10) * np.sqrt(2.0 / 128)
b2 = np.zeros((1, 10))

print("\n网络参数初始化完成")
print("W1形状：", W1.shape)
print("b1形状：", b1.shape)
print("W2形状：", W2.shape)
print("b2形状：", b2.shape)



# 3. Softmax 函数


def softmax(z):

    # 减去最大值，防止指数计算溢出
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z, axis=1, keepdims=True
    )



# 4. 前向传播


def forward(x, W1, b1, W2, b2):

    # 输入层 -> 隐藏层
    z1 = np.dot(x, W1) + b1

    # ReLU 激活函数
    a1 = np.maximum(0, z1)

    # 隐藏层 -> 输出层
    z2 = np.dot(a1, W2) + b2

    # Softmax 得到10个类别的概率
    a2 = softmax(z2)

    return z1, a1, z2, a2



# 5. 计算交叉熵损失


def calculate_loss(probs, y):

    m = len(y)

    # 找到每张图片真实类别对应的预测概率
    correct_probs = probs[np.arange(m), y]

    # 交叉熵损失
    loss = -np.mean(
        np.log(correct_probs + 1e-12)
    )

    return loss



# 6. 反向传播 + 梯度下降


def train_step(x, y, W1, b1, W2, b2,
               learning_rate=0.1):

    m = x.shape[0]

    #前向传播

    z1, a1, z2, a2 = forward(
        x, W1, b1, W2, b2
    )

    # 计算损失

    loss = calculate_loss(a2, y)

    #输出层梯度

    dz2 = a2.copy()

    # 真实类别的梯度减去1
    dz2[np.arange(m), y] -= 1

    # 计算第二层权重和偏置梯度
    dW2 = np.dot(a1.T, dz2) / m

    db2 = np.sum(
        dz2, axis=0, keepdims=True
    ) / m

    #隐藏层梯度

    da1 = np.dot(dz2, W2.T)

    # ReLU 的导数
    dz1 = da1 * (z1 > 0)

    # 计算第一层权重和偏置梯度
    dW1 = np.dot(x.T, dz1) / m

    db1 = np.sum(
        dz1, axis=0, keepdims=True
    ) / m

    # 更新参数

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return loss



# 7. 预测函数


def predict(x, W1, b1, W2, b2):

    _, _, _, probs = forward(
        x, W1, b1, W2, b2
    )

    predictions = np.argmax(
        probs, axis=1
    )

    return predictions, probs



# 8. 计算准确率


def evaluate(x, y, W1, b1, W2, b2):

    predictions, _ = predict(
        x, W1, b1, W2, b2
    )

    accuracy = np.mean(
        predictions == y
    )

    return accuracy



# 9. 开始训练


epochs = 25
batch_size = 64
learning_rate = 0.1

print("\n开始训练神经网络...")

for epoch in range(epochs):

    # 每一轮都打乱训练数据
    indices = np.random.permutation(
        len(x_train)
    )

    x_train = x_train[indices]
    y_train = y_train[indices]

    total_loss = 0.0
    batch_count = 0

    # 按批次训练
    for i in range(0, len(x_train), batch_size):

        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        loss = train_step(
            x_batch,
            y_batch,
            W1, b1, W2, b2,
            learning_rate
        )

        total_loss += loss
        batch_count += 1

    average_loss = total_loss / batch_count

    # 每轮训练结束后测试准确率
    accuracy = evaluate(
        x_test, y_test,
        W1, b1, W2, b2
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"测试准确率: {accuracy * 100:.2f}%"
    )



# 10. 最终测试


print("\n训练完成！")

final_accuracy = evaluate(
    x_test, y_test,
    W1, b1, W2, b2
)

print(
    f"最终测试准确率："
    f"{final_accuracy * 100:.2f}%"
)



# 11. 查看一张图片的预测结果


index = 0

sample = x_test[index:index + 1]

predictions, probs = predict(
    sample, W1, b1, W2, b2)
print("\n单张图片测试：")
print("模型预测：", predictions[0])
print("真实答案：", y_test[index])
print("预测概率：", probs)



# 12. 保存训练的神经网络


np.savez(
    "manual_mnist_model.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("\n模型参数已保存：manual_mnist_model.npz")