import tensorflow as tf


@tf.function
def get_num(input_num):
    print("input is:", input_num)
    return input_num + input_num


print("result is :", get_num(tf.constant(1)))
print("---------")
print("result is :", get_num(tf.constant(1.0)))
print("---------")
print("result is :", get_num(tf.constant([1, 2])))
