from NeuralNet import *
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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

np.set_printoptions(linewidth=300, suppress=True)
directory = Path.cwd().parent / "Data/fashion_mnist_images/train/"
png = []
for D in directory.iterdir():
    for P in D.iterdir():
        png.append(P)

params = np.load('models/PositiveDense.parameters', allow_pickle=True)
weights = []
biases = []
for item in params:
    weights.append(item[0])
    biases.append(item[1])

F = Layer_Flattening()
D1 = Layer_Dense(784, 512)
D2 = Layer_Dense(512, 256)
D3 = Layer_Dense(256, 128)
D4 = Layer_Dense(128, 10)
A = Activation_Softmax()

D1.set_parameters(weights[0], biases[0])
D2.set_parameters(weights[1], biases[1])
D3.set_parameters(weights[2], biases[2])
D4.set_parameters(weights[3], biases[3])


D4_avg = []
D4_std = []
D4_max = []
D4_min = []

for image_path in tqdm(png):
    image = Image.open(image_path)
    data = np.asarray(image)/255
    data = data.reshape(1, 1, 28, 28)

    F.forward(data, training=False)
    D1.forward(F.output, training=False)
    D2.forward(D1.output, training=False)
    D3.forward(D2.output, training=False)
    D4.forward(D3.output, training=False)
    A.forward(D4.output, training=False)

    D4_avg.append(np.mean(D4.output))
    D4_std.append(np.std(D4.output))
    D4_max.append(np.max(D4.output))
    D4_min.append(np.min(D4.output))


print('')
print("D4 avg: ", np.mean(D4_avg), "Std: ", np.std(D4_avg))
print("D4 std: ", np.mean(D4_std), "Std: ", np.std(D4_std))
print("D4 max: ", np.mean(D4_max), "Std: ", np.std(D4_max))
print("D4 min: ", np.mean(D4_min), "Std: ", np.std(D4_min))
