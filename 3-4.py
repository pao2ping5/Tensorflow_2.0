import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
iris_data = np.float32(data.data)
iris_target = (data.target)
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target, num_classes=3))
train_data = tf.data.Dataset.from_tensor_slices((iris_data, iris_target)).batch(128)
input_xs = tf.keras.Input(shape=(4), name='input_xs')
out = tf.keras.layers.Dense(32, activation='relu', name='dense_1')(input_xs)
out = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(out)
logits = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(out)
model = tf.keras.Model(inputs=input_xs, outputs=logits)
opt = tf.optimizers.Adam(1e-3)
model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(train_data, epochs=500)
score = model.evaluate(iris_data, iris_target)
print("last score:", score)
