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
D1_avg = []
D2_avg = []
D3_avg = []
D4_avg = []

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

    D1_avg.append(np.average(D1.output))
    D2_avg.append(np.average(D2.output))
    D3_avg.append(np.average(D3.output))
    D4_avg.append(np.average(D4.output))

print("D1 avg: ", np.mean(D1_avg), "Std: ", np.std(D1_avg))
print("D2 avg: ", np.mean(D2_avg), "Std: ", np.std(D2_avg))
print("D3 avg: ", np.mean(D3_avg), "Std: ", np.std(D3_avg))
print("D4 avg: ", np.mean(D4_avg), "Std: ", np.std(D4_avg))
