from NeuralNet import *
import numpy as np
from pathlib import Path
import time

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

np.set_printoptions(linewidth=400)
p = Path().cwd().parent / 'Data/Fashion_Mnist'


files = []
for file in p.iterdir():
    if file.suffix == '.npy':
        files.append(file)

y_val = np.load(files[0])
y = np.load(files[1])
X = np.load(files[2]).astype(np.float64) / 255
X_val = np.load(files[3]).astype(np.float64) / 255

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = X.reshape(X.shape[0], 1, 28, 28)
X_val = X_val.reshape(X_val.shape[0], 1, 28, 28)

'''
for i in range(10):
    print(np.around(X[i].reshape(28,28), decimals=2))
    print(fashion_mnist_labels[y[i]])
    time.sleep(0.8)
'''
model = Model()

'''
model.add(Layer_Conv2D(32, (1, 5, 5),
          weight_regularizer_l2=5e-7, bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Maxpooling())
model.add(Layer_Conv2D(64, (32, 5, 5),
          weight_regularizer_l2=5e-7, bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Maxpooling())
'''
model.add(Layer_Flattening())
model.add(Layer_Dense(784, 512, weight_regularizer_l2=5e-7,
          bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Dense(512, 256, weight_regularizer_l2=5e-7,
          bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Dense(256, 128, weight_regularizer_l2=5e-7,
          bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(),
          optimizer=Optimizer_Adam(decay=1e-5),
          accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_val, y_val),
            epochs=17, print_every=100, batch_size=64)

model.save('models/PositiveDense.model')
