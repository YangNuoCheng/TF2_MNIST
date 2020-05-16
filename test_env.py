import tensorflow as tf

# print("tf.__version__：",tf.__version__)
# print("tf.__path__:",tf.__path__)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# 调用mnist数据集
x_train, x_test = x_train / 255.0, x_test / 255.0
# 对数据集做归一化
print(x_train[0].shape)