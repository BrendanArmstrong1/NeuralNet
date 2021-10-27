from NeuralNet import *
from SnakeGame import *


model = Model()

model.add(Layer_Conv2D(64, (1, 3, 3),
                       weight_regularizer_l2=5e-7, bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Maxpooling())
model.add(Layer_Conv2D(128, (64, 3, 3),
                       weight_regularizer_l2=5e-7, bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Maxpooling())
model.add(Layer_Flattening())
model.add(Layer_Dense(128, 64, weight_regularizer_l2=5e-7,
                      bias_regularizer_l2=5e-7))
model.add(Activation_RelU())
model.add(Layer_Dropout(0.35))
model.add(Layer_Dense(64, 4))
model.add(Activation_Linear())

model.set(loss=Loss_MeanSquaredError(),
          optimizer=Optimizer_Adam(decay=4e-5))

model.finalize()


Game = Snake_Game()
done = False
while not done:
    print(Game.get_board())
    pred = model.predict(Game.get_board().reshape((1, 10, 10)))
    exp_values = np.exp(pred - np.max(pred, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    move = np.random.rand()
    if move > np.sum(probabilities[0][0:3]):
        Game.dirChange(np.array([-1, 0]))
    elif move > np.sum(probabilities[0][0:2]):
        Game.dirChange(np.array([1, 0]))
    elif move > np.sum(probabilities[0][0:1]):
        Game.dirChange(np.array([0, 1]))
    else:
        Game.dirChange(np.array([0, -1]))
    _, _, done = Game.forward()
    done = True

print(Game.get_board())
