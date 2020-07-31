from sklearn import datasets
import numpy as np
import tensorflow as tf

def _ohe(targets):
    y = np.zeros((150, 3))
    for i, label in enumerate(targets):
        y[i, label] = 1.0
    return y

def train(X, y, epochs=10, batch_size=16):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs)
    return model

def freeze(model, path='../frozen'):
    model.save(f'{path}/0001')
    return True

if __name__ == '__main__':
    iris = datasets.load_iris()
    X, targets = iris.data, iris.target
    y = _ohe(targets)
    model = train(X, y, epochs=50)
    freeze(model)