# This is the snake game that the ConvNet will train on.

from os import system
import numpy as np
import time
from pynput import keyboard

np.set_printoptions(linewidth=250)


class Snake_Game:
    def __init__(self, size=6, verbose=True):
        # initialize board, snake, snake direction, and fruit
        self.size = size
        self.reward = 0
        self.verbose = verbose
        self.board = np.zeros((size, size))
        self.direction = np.array((0, 1))
        self.prev_direction = np.array((0, 1))
        self.length = 5
        self.snake = np.array([[self.size//2, self.size//2]])
        for i in range(self.length - 1):
            self.snake = np.insert(
                self.snake, i+1, [self.snake[0][0], self.snake[0][1]-i-1], axis=0)
        self.board[self.snake[:, 0], self.snake[:, 1]] = 2
        self.board[self.snake[0, 0], self.snake[0, 1]] = 3
        self.fruitGen()
        if self.verbose:
            self.drawBoard()

    def forward(self):
        # step the game forward based snake direction
        self.reward = 0
        self.board[self.snake[0, 0], self.snake[0, 1]] = 2
        self.snake = np.insert(self.snake, 1, self.snake[0], axis=0)
        self.snake[0] += self.direction
        if (self.snake[0] == self.fruit).all():
            self.fruitGen()
            self.reward = 0.7
        else:
            self.board[self.snake[-1, 0], self.snake[-1, 1]] = 0
            self.snake = np.delete(self.snake, -1, axis=0)
        ##collision checking########
        if np.any(np.all(self.snake[0] == self.snake[1:], axis=1)):
            self.reward = -0.3
            if self.verbose:
                self.Game_Over(condition="Snake ate itself!")
            return self.reward, True
        if np.any(self.snake[0] == self.size) or np.any(self.snake[0] < 0):
            self.reward = -0.3
            if self.verbose:
                self.Game_Over(condition="Snake hit the wall!")
            return self.reward, True
        ############################
        self.board[self.snake[0, 0], self.snake[0, 1]] = 3
        self.prev_direction = self.direction
        if self.verbose:
            self.drawBoard()
        return self.reward, False

    def fruitGen(self):
        # generate a new fruit on the board
        # new fruit should be generated only in an empty space
        options = np.array(np.where(self.board == 0)).T
        rSel = int(np.floor(len(options)*np.random.rand()))
        self.fruit = np.array((options[rSel, 0], options[rSel, 1]))
        self.board[self.fruit[0], self.fruit[1]] = 1

    def dirChange(self, direction):
        # Takes input to change direction of snake
        if (self.prev_direction == direction*-1).all():
            return
        self.direction = direction

    def Game_Over(self, condition=""):
        print(f'\033[{self.size+2}E')
        print("Game Over!", condition)

    def get_board(self):
        return self.board

    def drawBoard(self):
        print((self.size+2)*'% ')
        for row in self.board:
            string = '% '
            for i in row:
                if i > 1:
                    string += '0 '
                elif i == 1:
                    string += 'F '
                else:
                    string += '  '
            string += '%'
            print(string)
        print((self.size+2)*'% ', f'\033[{self.size+2}F')


# def main():
#    B = Snake_Game()
#
#    def on_press(key):
#        if str(key) == 'Key.up':
#            B.dirChange(np.array([-1, 0]))
#        if str(key) == 'Key.down':
#            B.dirChange(np.array([1, 0]))
#        if str(key) == 'Key.left':
#            B.dirChange(np.array([0, -1]))
#        if str(key) == 'Key.right':
#            B.dirChange(np.array([0, 1]))
#    listener = keyboard.Listener(
#        on_press=on_press)
#    listener.start()
#    done = False
#    while not done:
#        done = B.forward()
#        time.sleep(0.4)
#
#
# if __name__ == "__main__":
#    main()
