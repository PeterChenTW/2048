import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import datetime
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from game_2048 import Game, Board
from RL import DDQNAgent

sns.set(style='whitegrid')


def train_DDQN(episodes, model_name):
    agent = DDQNAgent(16, 4)
    record_freq = 50
    if os.path.exists(f'model/{model_name}'):
        print('exist')
        agent.load(f'model/{model_name}')
    scores = []
    turns = []
    invalid_move_ratio = []

    env = Board(4, 0)
    for e in range(episodes):
        state = torch.from_numpy(env.state).float()

        while True:
            action = agent.act(state, env.get_invalid_moves())
            next_state, reward, done = env.move(env.all_moves[action])
            next_state = torch.from_numpy(next_state).float()
            reward = torch.tensor(reward).float()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {:11}/{}, score: {:6}, score(v2): {:7}, turns: {:6}, invalid: {:6}"
                      .format(e, episodes, env.score, env.score_v2, env.turns - env.invalid_move, env.invalid_move))

                scores.append(env.score_v2)
                turns.append(env.turns-env.invalid_move)
                invalid_move_ratio.append(env.invalid_move/env.turns)
                break

        if e > 128:
            agent.replay(128)

        if e % record_freq == 0:
            agent.update_target_model()

        env.__init__(4)

        if e % 250 == 0 and e != 0:
            agent.save(f'model/{model_name}')

            scores_ma = pd.Series(scores).rolling(window=20).mean()
            losses_ma = pd.Series(agent.losses).rolling(window=20).mean()
            turns_ma = pd.Series(turns).rolling(window=20).mean()
            invalid_move_ma = pd.Series(
                invalid_move_ratio).rolling(window=20).mean()

            plt.figure(figsize=(12, 20))

            plt.subplot(4, 1, 1)
            plt.plot(np.arange(len(scores_ma)), scores_ma)
            plt.ylabel('Score')
            plt.xlabel('Episode')
            plt.title('Score over Episodes')
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(np.arange(len(losses_ma)), losses_ma)
            plt.yscale('log') 
            plt.ylabel('Loss (moving average, log scale)')
            plt.xlabel('Episode')
            plt.title('Loss over Episodes')
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(np.arange(len(turns_ma)), turns_ma)
            plt.ylabel('Turns')
            plt.xlabel('Episode')
            plt.title('Turns over Episodes')
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.plot(np.arange(len(invalid_move_ma)), invalid_move_ma)
            plt.ylabel('Invalid Moves Ratio')
            plt.xlabel('Episode')
            plt.title('Invalid Moves Ratio over Episodes')
            plt.grid(True)

            plt.tight_layout(pad=3.0) 
            plt.savefig(
                f'result/{model_name}_{datetime.datetime.now():%Y%m%d%H%M%S}.png')


if __name__ == '__main__':
    model_name = 'fixed_random_seed'

    while True:
        train_DDQN(10000000, model_name)

    pygame.init()
    screen = pygame.display.set_mode((410, 410))
    pygame.display.set_caption("2048")
    game = Game(4, role='AI', model_name=model_name)
    game.draw()
    game.run()
