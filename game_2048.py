import numpy as np
import random
from copy import deepcopy
import torch
import os
import pygame
import time
import time
from RL import DDQNAgent


class Board:
    def __init__(self, size, random_seed=None):
        self.size = size
        self.grid = [[0]*size for _ in range(size)]
        self.new_tile_position = None
        self.score = 0
        self.score_v2 = 0
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        for _ in range(2):
            self.score += self.add_random_tile()
        self.done = False
        self.merge_this_turn = None
        self.all_moves = ['left', 'down', 'right', 'up']
        self.get_score = 0
        self.turns = 0
        self.invalid_move = 0
        # self.state = self.one_hot_encode()
        self.state = np.array(self.grid)

    def one_hot_encode(self):
        one_hot = np.zeros((self.size, self.size, 16))
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] > 0:
                    index = int(np.log2(self.grid[i][j])) - 1
                    one_hot[i, j, index] = 1

        return one_hot

    def add_random_tile(self):
        self.empty_positions = [(x, y) for x in range(self.size)
                                for y in range(self.size) if self.grid[x][y] == 0]
        x, y = random.choice(self.empty_positions)
        self.grid[x][y] = random.choices([2, 4], weights=(90, 10), k=1)[0]
        self.new_tile_position = (x, y)
        return self.grid[x][y]

    def move(self, direction):
        self.get_score = 0
        self.turns += 1
        original_score = self.score
        original = deepcopy(self.grid)
        zero_count = 0
        self.merge_this_turn = False
        for _ in range(self.all_moves.index(direction)):
            self.rotate()

        for i in range(self.size):
            self.grid[i], it_zero = self.merge(self.grid[i])
            zero_count += it_zero

        for _ in range((4-self.all_moves.index(direction)) % 4):
            self.rotate()

        if not self.game_over():
            if original != self.grid and zero_count:
                self.score += self.add_random_tile()
        if zero_count == 1 and self.game_over():
            self.done = True

        self.score_v2 += self.get_score

        if self.score == original_score:
            reward = -1
            self.invalid_move += 1
        else:
            reward = np.log2(1+self.get_score)/16

        self.state = np.array(self.grid)
        return self.state, reward, self.done

    def rotate(self):
        for i in range(self.size//2):
            self.grid[i], self.grid[~i] = self.grid[~i], self.grid[i]

        for i in range(self.size):
            for j in range(i+1, self.size):
                self.grid[i][j], self.grid[j][i] = self.grid[j][i], self.grid[i][j]

    def merge(self, row):
        new = [num for num in row if num != 0]

        for i in range(len(new)-1):
            if new[i] == new[i+1]:
                new[i], new[i+1] = new[i]*2, 0
                self.merge_this_turn = True
                self.get_score += new[i]

        final = [num for num in new if num != 0]

        return final + [0]*(self.size - len(final)), self.size - len(final)

    def game_over(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    return False
                if i < self.size - 1 and self.grid[i][j] == self.grid[i+1][j]:
                    return False
                if j < self.size - 1 and self.grid[i][j] == self.grid[i][j+1]:
                    return False
        return True

    def get_invalid_moves(self):
        invalid_moves = []

        for i, move in enumerate(self.all_moves):
            original = deepcopy(self)
            original.move(move)
            if self.grid == original.grid:
                invalid_moves.append(i)
        return invalid_moves


class Game:
    def __init__(self, size, role='human', model_name='AI_model', random_seed=None):
        self.random_seed = random_seed
        self.board = Board(size, random_seed)
        self.role = role
        self.model_name = model_name
        self.icon_surface = pygame.image.load("icon.png")
        pygame.display.set_icon(self.icon_surface)
        self.screen = pygame.display.set_mode((410, 460))
        self.font = pygame.font.Font(None, 36)
        self.color_map = {
            0: (205, 193, 180),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 95, 64),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        pygame.mixer.music.load(os.path.join(
            self.current_dir, 'background.wav'))
        pygame.mixer.music.play(-1)

        self.sound_effect = pygame.mixer.Sound(
            os.path.join(self.current_dir, 'merge.wav'))
        self.sound_effect.set_volume(0.25)
        self.recommended_move = None
        self.action_weights = None

    def draw(self):
        if self.board.merge_this_turn:
            self.sound_effect.play()
        self.screen.fill((255, 255, 255))
        for i in range(self.board.size):
            for j in range(self.board.size):
                if i == self.board.new_tile_position[0] and j == self.board.new_tile_position[1]:
                    color = self.get_tile_color(self.board.grid[i][j])
                    color = (color[0], color[1], max(0, color[2] - 50))
                    pygame.draw.rect(self.screen, color,
                                     (j*100+10, i*100+10, 80, 80))
                else:
                    pygame.draw.rect(self.screen, self.get_tile_color(
                        self.board.grid[i][j]), (j*100+10, i*100+10, 80, 80))
                if self.board.grid[i][j] != 0:
                    text = self.font.render(
                        str(self.board.grid[i][j]), True, (119, 110, 101))
                    text_rect = text.get_rect(center=(j*100+50, i*100+50))
                    self.screen.blit(text, text_rect)

        if self.action_weights is not None:
            max_weight = max(self.action_weights.detach().numpy()[0])
            min_weight = min(self.action_weights.detach().numpy()[0])
            for i, action in enumerate(self.board.all_moves):
                if max_weight == min_weight:
                    color_intensity = 128
                else:
                    weight = self.action_weights.detach().numpy()[0][i]
                    color_intensity = int(
                        255 * (weight - min_weight) / (max_weight - min_weight))
                color = (255 - color_intensity, color_intensity, 0)
                action_text = self.font.render('O', True, color)
                if action == 'up':
                    p = (200, 10)
                elif action == 'right':
                    p = (400, 200)
                elif action == 'down':
                    p = (200, 400)
                else:
                    p = (5, 200)
                action_rect = action_text.get_rect(center=p)
                self.screen.blit(action_text, action_rect)

        score_context = "Score: {}".format(self.board.score_v2)
        if self.board.done:
            score_context += ", Game over!"

        score_text = self.font.render(score_context, True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(self.screen.get_width(
        ) // 2, self.screen.get_height() - 30))
        self.screen.blit(score_text, score_rect)

        score_box_color = self.get_score_box_color(self.board.score_v2)
        pygame.draw.rect(self.screen, score_box_color, (0, 410, 410, 50))

        score_context = "Score: {}".format(self.board.score_v2)
        if self.board.done:
            score_context += ", Game over!"

        score_text = self.font.render(score_context, True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(self.screen.get_width(
        ) // 2, self.screen.get_height() - 25)) 
        self.screen.blit(score_text, score_rect)
        if self.recommended_move is not None:
            recommended_move_text = self.font.render(
                "AI recommends: " + self.recommended_move, True, (0, 0, 0))
            recommended_move_rect = recommended_move_text.get_rect(
                center=(self.screen.get_width() // 2, 400))  
            self.screen.blit(recommended_move_text, recommended_move_rect)
        pygame.display.update()

    def get_tile_color(self, value):
        return self.color_map[value]

    def get_score_box_color(self, score):
        red = 255
        green = max(0, 255 - score // 20)
        blue = 192
        return (red, green, blue)

    def run(self):
        if self.role == 'human':
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.board.move("left")
                        elif event.key == pygame.K_RIGHT:
                            self.board.move("right")
                        elif event.key == pygame.K_UP:
                            self.board.move("up")
                        elif event.key == pygame.K_DOWN:
                            self.board.move("down")
                        self.draw()
        elif self.role == 'random':
            while not self.board.done:
                self.board.move(random.choices(
                    self.board.all_moves, (0.9, 0.007, 0.003, 0.1))[0])
                self.draw()
                time.sleep(0.1)
            print('score: ', self.board.score)
        elif self.role == 'AI':
            agent = DDQNAgent(16, 4)
            agent.load(f'model/{self.model_name}')
            agent.epsilon = 0
            state = torch.from_numpy(self.board.state).float()

            while not self.board.done:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_DOWN:
                            while not self.board.done:
                                time.sleep(0.1)
                                action = agent.act(
                                    state, self.board.get_invalid_moves())
                                self.board.move(self.board.all_moves[action])
                                state = torch.from_numpy(
                                    self.board.state).float()
                                self.action_weights = agent.model(
                                    torch.from_numpy(self.board.state).float())
                                # print(self.action_weights)
                                self.recommended_move = self.board.all_moves[torch.argmax(
                                    self.action_weights[0]).item()]
                                self.draw()

            print(self.random_seed, self.board.score_v2, np.max(self.board.grid))
