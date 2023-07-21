
import pygame
import random
import os
from copy import deepcopy
import time


class Board:
    def __init__(self, size):
        self.size = size
        self.grid = [[0]*size for _ in range(size)]
        self.score = 0
        for _ in range(2):
            self.score += self.add_random_tile()
        self.done = False
        self.merge_this_turn = None

    def add_random_tile(self):
        # Find all empty positions first
        empty_positions = [(x, y) for x in range(self.size)
                           for y in range(self.size) if self.grid[x][y] == 0]

        # Choose a random position from the empty ones
        x, y = random.choice(empty_positions)
        self.grid[x][y] = random.choices([2, 4], weights=(90, 10), k=1)[0]
        return self.grid[x][y]

    def move(self, direction):
        original = deepcopy(self.grid)
        zero_count = 0
        direction_list = ['left', 'down', 'right', 'up']
        self.merge_this_turn = False
        for _ in range(direction_list.index(direction)):
            self.rotate()

        for i in range(self.size):
            self.grid[i], it_zero = self.merge(self.grid[i])
            zero_count += it_zero

        for _ in range((4-direction_list.index(direction)) % 4):
            self.rotate()

        if not self.game_over():
            if original != self.grid and zero_count:
                self.score += self.add_random_tile()
        if zero_count == 1 and self.game_over():
            self.done = True

    def rotate(self):
        # up down
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


class Game:
    def __init__(self, size, screen):
        self.board = Board(size)
        # Increase the height of the window
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

    def draw(self):
        if self.board.merge_this_turn:
            self.sound_effect.play()
        self.screen.fill((255, 255, 255))
        for i in range(self.board.size):
            for j in range(self.board.size):
                pygame.draw.rect(self.screen, self.get_tile_color(
                    self.board.grid[i][j]), (j*100+10, i*100+10, 80, 80))
                if self.board.grid[i][j] != 0:
                    text = self.font.render(
                        str(self.board.grid[i][j]), True, (119, 110, 101))
                    text_rect = text.get_rect(center=(j*100+50, i*100+50))
                    self.screen.blit(text, text_rect)

        score_context = "Score: {}".format(self.board.score)
        if self.board.done:
            score_context += ", Game over!"

        score_text = self.font.render(score_context, True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(self.screen.get_width(
        ) // 2, self.screen.get_height() - 30))  # Position the score at the bottom
        self.screen.blit(score_text, score_rect)

        # Draw the score box
        score_box_color = self.get_score_box_color(self.board.score)
        pygame.draw.rect(self.screen, score_box_color, (0, 410, 410, 50))

        # Draw the score text
        score_context = "Score: {}".format(self.board.score)
        if self.board.done:
            score_context += ", Game over!"

        score_text = self.font.render(score_context, True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(self.screen.get_width(
        ) // 2, self.screen.get_height() - 25))  # Position the score in the middle of the score box
        self.screen.blit(score_text, score_rect)

        pygame.display.update()

    def get_tile_color(self, value):
        return self.color_map[value]

    def get_score_box_color(self, score):
        red = 255
        green = max(0, 255 - score // 20)
        blue = 192
        # The color changes from white to red as the score increases
        return (red, green, blue)

    def run(self):
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

        # # random play
        # while not self.board.done:
        #     self.board.move(random.choices(['left', 'right', 'up', 'down'], (0.9, 0.007, 0.003, 0.1))[0])
        #     self.draw()
        #     time.sleep(0.1)
        # print('score: ', self.board.score)


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((410, 410))
    pygame.display.set_caption("2048")
    game = Game(4, screen)
    game.draw()
    game.run()
