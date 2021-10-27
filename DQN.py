from SnakeGame import Snake_Game
from NeuralNet import *
from collections import deque
import random
from tqdm import tqdm
import time

DISCOUNT = 0.7
MEMORY_SIZE = 10_000  # How many last steps to keep for model training
BATCH_SIZE = 128  # How many steps (samples) to use for training
EPISODES = 20000
MODEL_NAME = "Testing"

#  Stats settings
AGGREGATE_STATS_EVERY = 1000  # episodes
SHOW_PREVIEW = True


ep_rewards = []


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.critic, self.actor = self.create_model()

        self.replay_memory = []

    def create_model(self):

        critic = Model()
        actor = Model()

        #actor.add(Layer_Conv2D(64, (1, 3, 3)))
        actor.add(Layer_Dense(36, 80))
        actor.add(Activation_RelU())
        actor.add(Layer_Dropout(0.6))
        # actor.add(Layer_Maxpooling())
        #actor.add(Layer_Conv2D(128, (64, 3, 3)))
        actor.add(Layer_Dense(80, 64))
        actor.add(Activation_RelU())
        actor.add(Layer_Dropout(0.6))
        # actor.add(Layer_Maxpooling())
        # actor.add(Layer_Flattening())
        actor.add(Layer_Dense(64, 32))
        actor.add(Activation_RelU())
        actor.add(Layer_Dropout(0.6))
        actor.add(Layer_Dense(32, 4))
        actor.add(Activation_Softmax())

        actor.set(loss=Loss_CategoricalCrossentropy(),
                  optimizer=Optimizer_Adam(learning_rate=1e-3, decay=4e-5))

        #critic.add(Layer_Conv2D(64, (1, 3, 3)))
        critic.add(Layer_Dense(36, 80))
        critic.add(Activation_RelU())
        critic.add(Layer_Dropout(0.6))
        # critic.add(Layer_Maxpooling())
        #critic.add(Layer_Conv2D(128, (64, 3, 3)))
        critic.add(Layer_Dense(80, 64))
        critic.add(Activation_RelU())
        critic.add(Layer_Dropout(0.6))
        # critic.add(Layer_Maxpooling())
        # critic.add(Layer_Flattening())
        critic.add(Layer_Dense(64, 32))
        critic.add(Activation_RelU())
        critic.add(Layer_Dropout(0.6))
        critic.add(Layer_Dense(32, 1))
        critic.add(Activation_Linear())

        critic.set(loss=Loss_MeanSquaredError(),
                   optimizer=Optimizer_Adam(learning_rate=1e-3, decay=4e-5))

        critic.finalize()
        actor.finalize()

        return critic, actor

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MEMORY_SIZE:
            return

        X = []
        y = []
        l = []

        # Now we need to enumerate our batches
        for _, (current_state, action, reward, new_state, done) in enumerate(self.replay_memory):

            advantage = reward + (1-done) * DISCOUNT * \
                self.get_qs(new_state) - self.get_qs(current_state)
            probs = self.get_ps(current_state)
            logs = np.log(probs)
            logs[0][action] *= np.abs(advantage)
            exp_values = np.exp(logs - np.max(logs, axis=1, keepdims=True))
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

            # And append to our training data
            X.append(current_state)
            y.append(probs)
            l.append(advantage**2)

        # Fit on all samples as one batch, log only on terminal state
        self.actor.train(np.array(X)/3, np.array(y),
                         batch_size=BATCH_SIZE, verbose=False)
        self.critic.train(np.array(X)/3, np.array(l),
                          batch_size=BATCH_SIZE, verbose=False)

        self.replay_memory = []

    # Queries main network for Q values given current observation space (environment state)

    def get_qs(self, state):
        return self.critic.predict(np.array(state)/3)

    def get_ps(self, state):
        return self.actor.predict(np.array(state)/3)


agent = DQNAgent()
record = 4
random.seed(1)
np.random.seed(1)
# Iterate over episodes
for episode in range(1, EPISODES + 1):
    # for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        Game = Snake_Game()
    else:
        Game = Snake_Game(verbose=False)
    current_state = np.array(Game.get_board().flatten())

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        #        if np.random.random() > epsilon:
        #            # Get action from Q table
        #            action = np.argmax(agent.get_qs(current_state))
        #        else:
        #            # Get random action
        #            action = np.random.randint(0, 4)
        #        if action == 0:
        #            Game.dirChange(np.array([0, -1]))
        #        if action == 1:
        #            Game.dirChange(np.array([0, 1]))
        #        if action == 2:
        #            Game.dirChange(np.array([1, 0]))
        #        if action == 3:
        #            Game.dirChange(np.array([-1, 0]))

        probabilities = agent.get_ps(current_state)
        move = np.random.rand()
        if move > np.sum(probabilities[0][0:3]):
            Game.dirChange(np.array([-1, 0]))
            action = 3
        elif move > np.sum(probabilities[0][0:2]):
            Game.dirChange(np.array([1, 0]))
            action = 2
        elif move > np.sum(probabilities[0][0:1]):
            Game.dirChange(np.array([0, 1]))
            action = 1
        else:
            Game.dirChange(np.array([0, -1]))
            action = 0

        reward, done = Game.forward()
        new_state = np.array(Game.get_board().flatten())
        if Game.verbose:
            print(probabilities)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory(
            (current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(
            ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
