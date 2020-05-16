import tensorflow as tf
import numpy as np
from PIL import Image

def getTestPicArray(filename):
    # 生成文件的28*28矩阵
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    # 调整图片的大小
    im_arr = np.array(out.convert('L'))
    # 转换为灰度图，值在0～255之间
    return im_arr.reshape((1, 784))

model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

print("重构之前的网络")
model.summary()
# 重建网络
model.load_weights('/Users/yangnuocheng/Desktop/weights/weights.ckpt')
print('loaded weights!')

k = getTestPicArray('/Users/yangnuocheng/Desktop/models/sample_8.png')
print(type(k))
out = model.predict(k.reshape(1,28,28))
print(out)
print("模型的预测结果是：",np.argmax(model.predict(k.reshape(1,28,28)), axis=1))

