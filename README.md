# Neural Network From Scratch

## What is this project?

This is the Neural Network project in Python, The purpose is to learn how Deep Neural networks function. The current end goal of this
project is to get my home made AI to win a game of snake(also home made ascii snake). Right now I'm aiming for a 6x6 size game board.

## Whats been done so far

### Neural Network

The framework for the fully connected dense layer and the convolutional layer is functional, I've made several different optimizers and utility layers such as
maxpooling and flattening. I've also built different activation functions. The network functions well and has achieved 88%
accuracy on fashion Mnist. The real substance behind this project is with the reinforcement learning 

### Reinforcement Learning

This is the bread and butter of this project. Teaching my AI to play snake on its own. Unfortunately I'm not satisfied with my
current position in this portion of the project. So far I have tried Deep Q learning with an epsilon greedy strategy, but this
only gave the network the intelligence of a 2 year old after 4 days of continuous training. The network went for fruit if it was
close enough, and it avoided walls, but the success can only be considered marginal. My new strategy is to implement advantage
actor critic learning to solve this problem. The more I research about it, the more I believe it will be successful. I am
currently searching for work as a new graduate so I wont be able to dedicate as much time to this project as I would like.

## File structure

There are a lot of files in this project and not all of them are super relevant, but they help me from time to time.

- NeuralNet.py \- This is the implementation of AI framework. 
- SnakeGame.py \- This is my custom ascii snake game that works in the terminal. This is what the AI will play.
- DQN.py \- This is the deep Q agent that I'm using as a trainer for reinforcement learning. Its been repurposed since I initially
    started with the epsilon greedy strategy. Don't look unless you want to go insane, I've left it a mess while I cool my brain
    down.

- snakeAI.py \- This is the implementation of the AI playing the snake game.

- utils \- This is the folder with utilities that I had used to create everything, This folder is fundamentally irrelevant to the
    project but I keep it there just in case. It contains scripts I wrote for testing the speed of the layers (speedtest.py), for
    testing the network with the Mnist data set(NetTest.py) a script for extracting data for the data set (extract.py), and the
    convolutional layer that I was working on separately in order to implement parallelization with the numba library. 

## Credits

Credit for everything but the convolutional layer goes to Harrison Kinsley and Daniel Kukiela from the [sentdex](https://www.youtube.com/c/sentdex) youtube channel. 
I purchased your book "[Neural Networks from Scratch in Python](https://nnfs.io/)" and it has been an amazing read.


