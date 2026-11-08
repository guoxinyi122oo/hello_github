import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载模型
model = load_model("mnist_cnn.h5")

def preprocess(img_path):
    # 读取为灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {img_path}，请检查文件名是否叫 my_digit.png")

    # 二值化
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找轮廓
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 取最大的轮廓
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = th[y:y+h, x:x+w]

    # 保持比例缩放，最长边为 20 像素
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中放到 28x28 的画布上
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = digit
    return canvas

def predict(img_path):
    digit = preprocess(img_path)
    if digit is None:
        print(" 没找到数字！请确认图片里有黑色的字。")
        return

    # 弹窗显示预处理后的图片
    cv2.imshow("What CNN Sees (28x28)", cv2.resize(digit, (280, 280), interpolation=cv2.INTER_NEAREST))
    print(" 在弹出的窗口上按任意键继续")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 预测
    x = digit.astype("float32") / 255.0
    x = x[None, ..., None] # 增加维度
    probs = model.predict(x, verbose=0)[0]
    
    label = int(np.argmax(probs))
    conf = probs[label] * 100
    print(f"\n 预测结果: {label}  (置信度: {conf:.2f}%)")

if __name__ == "__main__":
    predict("my_digit.png")