import tensorflow as tf
import numpy as np

# 生成数据
arr_list = np.arange(0, 100).astype(np.float32)
shape = arr_list.shape
# 使用Dataset API读取数据
dataset = tf.data.Dataset.from_tensor_slices(arr_list)
dataset_iterator = dataset.shuffle(shape[0]).batch(10)


# 创建计算模型
def model(xs):
    # ...编写一些函数
    outputs = tf.multiply(xs, 0.1)
    return outputs


# 数据的迭代输出
for it in dataset_iterator:
    logits = model(it)
    print(logits)
