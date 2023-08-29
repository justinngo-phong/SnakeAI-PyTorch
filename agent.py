import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3
GAMMA = 0.9


class Agent:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0  # control randomness
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        # model, trainer
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # list of tuple
            random_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            random_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*random_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            init_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(init_state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old state
        old_state = agent.get_state(game)

        # Get move
        final_move = agent.get_action(old_state)

        # Execute move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # Train short memory for 1 step
        agent.train_short_memory(old_state,
                                 final_move,
                                 reward,
                                 new_state,
                                 done)
        agent.remember(old_state,
                       final_move,
                       reward,
                       new_state,
                       done)

        if done:
            # Train long memory (experienced replay) and plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save()

            print('Game', agent.number_of_games,
                  'Score', score, 'Record:', best_score)

            plot_scores.append(score)
            scores_length = len(plot_scores)
            if scores_length < 20:
                total_score = sum(plot_scores)
                plot_mean_scores.append(total_score / scores_length)
            else:
                total_score = sum(plot_scores[-20:])
                plot_mean_scores.append(total_score / 20)

            # total_score += score
            # plot_mean_scores.append(total_score / agent.number_of_games)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
