from NeuralNet import *
import numpy as np
import random
from pathlib import Path
from PIL import Image

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

model = Model.load('models/PositiveDense.model')
np.set_printoptions(linewidth=300, suppress=True)
#directory = Path.cwd().parent / "Data/fashion_mnist_images/train"

#png = []
# for D in directory.iterdir():
#    for P in D.iterdir():
#        png.append(P)

image_path = '/home/brendan/S/JS/training/Code/pants.png'
image = Image.open(image_path)
data = np.asarray(image)[:, :, 1]
# print(data.shape)
# print('')
corrected_data = data/255
print(np.around(corrected_data, decimals=4))
corrected_data = corrected_data.reshape(1, 1, 28, 28).astype('float64')
print('')
prediction = model.predict(corrected_data)
print(prediction*100)
number = np.argmax(prediction)
#real_number, real_image = str(image_path).split("/")[8:10]
print("Prediction: ", fashion_mnist_labels[number],
      f"[{number}]", f"[{np.around(prediction[0][number] * 100, decimals=2)}% Confidence]")
#print("Real Image: ", fashion_mnist_labels[int(real_number)])
#print("Image ID: ", real_image)

# model.save_parameters('models/PositiveDense.parameters')
