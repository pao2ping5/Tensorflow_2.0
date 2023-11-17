import tensorflow as tf
import numpy as np

input_xs = np.random.rand(1000)
input_ys = 3 * input_xs + 0.217
# 模型的定义
weight = tf.Variable(1., dtype=tf.float32, name="weight")
bias = tf.Variable(1., dtype=tf.float32, name="bias")


def model(xs):
    logits = tf.multiply(xs, weight) + bias
    return logits


opt = tf.keras.optimizers.Adam(1e-1)
for xs, ys in zip(input_xs, input_ys):
    xs = np.reshape(xs, [1])
    ys = np.reshape(ys, [1])
    _loss = lambda: tf.losses.MeanSquaredError()(model(xs), ys)
    # 匿名回调函数
    opt.minimize(_loss, [weight, bias])
    # 直接对回调函数进行更新
    print(_loss().numpy())
    # 打印函数计算值
print(weight)
print(bias)
