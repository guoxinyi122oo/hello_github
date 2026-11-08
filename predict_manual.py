
import numpy as np
import cv2


# 1. 加载自己训练的神经网络参数

data = np.load("manual_mnist_model.npz")

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

print("模型参数加载成功！")


# 2. Softmax 函数

def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z, axis=1, keepdims=True
    )



# 3. 前向传播


def forward(x):

    # 输入层 -> 隐藏层
    z1 = np.dot(x, W1) + b1

    # ReLU
    a1 = np.maximum(0, z1)

    # 隐藏层 -> 输出层
    z2 = np.dot(a1, W2) + b2

    # Softmax
    a2 = softmax(z2)

    return a2



# 4. 读取并处理自己的手写图片


def preprocess(img_path):

    # 读取灰度图片
    img = cv2.imread(
        img_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(
            f"找不到图片：{img_path}"
        )

    # 二值化：让数字变成白色，背景变成黑色
    _, th = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    border=np.concatenate([th[0,:],th[-1,:],th[:,0],th[:,-1]])
    if np.mean(border)>127:
        th=255-th

    # 寻找数字轮廓
    points = cv2.findContours(th)

    if points is None:
        raise ValueError("图片中没有找到数字")

    # 获取数字边界
    x, y, w, h = cv2.boundingRect(points)

    # 裁剪数字
    digit = th[y:y+h, x:x+w]

    # 按比例缩放，最长边变成20
    if w > h:
        new_w = 20
        new_h = max(
            1,
            round(h * 20 / w)
        )
    else:
        new_h = 20
        new_w = max(
            1,
            round(w * 20 / h)
        )

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 创建28×28黑色画布
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # 计算居中位置
    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2

    # 把数字放到画布中央
    canvas[
        oy:oy+new_h,
        ox:ox+new_w
    ] = digit

    return canvas



# 5. 预测自己的手写数字


def predict(img_path):

    # 图片预处理
    digit = preprocess(img_path)
    print("处理后图片形状：",digit.shape)
    print("像素最大值：",digit.max())
    print("像素最小值：",digit.min())
    cv2.imwrite("processed_digit.png",digit)

    # 显示模型实际看到的图片
    cv2.imshow(
        "What Neural Network Sees",
        cv2.resize(
            digit,
            (280, 280),
            interpolation=cv2.INTER_NEAREST
        )
    )

    # 转换为0~1
    x = digit.astype(np.float32) / 255.0

    # 展开为784个输入
    x = x.reshape(1, 784)

    # 前向传播
    probs = forward(x)

    # 找到概率最大的数字
    label = int(np.argmax(probs, axis=1)[0])

    confidence = float(np.max(probs)) * 100

    print("\n预测结果：", label)
    print(f"模型输出置信度：{confidence:.2f}%")

    print("\n各数字的预测概率：")

    for i in range(10):
        print(f"数字 {i}：{probs[0, i] * 100:.2f}%")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 6. 程序入口


if __name__ == "__main__":

    predict("my_digit.png")