"""Fast 4x4 legacy-compatible board; legality queries never consume RNG.

Legacy rotations make 'down' move physically down and 'up' physically up.
Rows are memoized on demand, avoiding a fixed exponent limit or deepcopy.
"""
from functools import lru_cache
import random
import numpy as np


@lru_cache(maxsize=131072)
def _merge(row):
    compact = [v for v in row if v]
    score = 0
    for i in range(len(compact) - 1):
        if compact[i] == compact[i + 1]:
            compact[i] *= 2
            compact[i + 1] = 0
            score += compact[i]
    result = tuple(v for v in compact if v)
    return result + (0,) * (4 - len(result)), score


# Indices in slide-left order after each legacy clockwise rotation.
_LINES = (
    tuple(tuple(4*r+c for c in range(4)) for r in range(4)),
    tuple(tuple(4*r+c for r in range(3, -1, -1)) for c in range(4)),
    tuple(tuple(4*r+c for c in range(3, -1, -1)) for r in range(3, -1, -1)),
    tuple(tuple(4*r+c for r in range(4)) for c in range(3, -1, -1)),
)


class FastBoard:
    all_moves = ['left', 'down', 'right', 'up']
    size = 4

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self._cells = np.zeros(16, dtype=np.int64)
        self.score = self.score_v2 = self.turns = self.invalid_move = 0
        self.done = False
        self.get_score = 0
        self.merge_this_turn = None
        self.new_tile_position = None
        self.score += self.add_random_tile()
        self.score += self.add_random_tile()
        self.state = self._cells.reshape(4, 4).copy()

    @classmethod
    def from_grid(cls, grid, rng=None):
        """Create a board without spawning or consuming RNG; counters start at zero."""
        self = cls.__new__(cls)
        self.rng = rng if rng is not None else random.Random()
        self._cells = np.array(grid, dtype=np.int64).reshape(16).copy()
        self.score = self.score_v2 = self.turns = self.invalid_move = 0
        self.done = False
        self.get_score = 0
        self.merge_this_turn = None
        self.new_tile_position = None
        self.state = self._cells.reshape(4, 4).copy()
        return self

    @property
    def grid(self):
        return self._cells.reshape(4, 4).tolist()

    def grid_values(self):
        return self.grid

    def add_random_tile(self):
        self.empty_positions = [(i // 4, i % 4) for i, v in enumerate(self._cells.tolist()) if not v]
        x, y = self.rng.choice(self.empty_positions)
        value = self.rng.choices([2, 4], weights=(90, 10), k=1)[0]
        self._cells[4*x+y] = value
        self.new_tile_position = (x, y)
        return value

    def pure_move(self, direction):
        """Return flat values, merge score, and changed, without changing the board."""
        action = self.all_moves.index(direction) if isinstance(direction, str) else int(direction)
        if not 0 <= action < 4:
            raise ValueError('action must be in 0..3')
        original = self._cells.tolist()
        result = original.copy()
        score = 0
        for a, b, c, d in _LINES[action]:
            row, gained = _merge((original[a], original[b], original[c], original[d]))
            result[a], result[b], result[c], result[d] = row
            score += gained
        return result, score, result != original

    @staticmethod
    def _game_over(cells):
        for i, value in enumerate(cells):
            if not value or (i < 12 and value == cells[i+4]) or (i % 4 < 3 and value == cells[i+1]):
                return False
        return True

    def game_over(self):
        return self._game_over(self._cells.tolist())

    def get_invalid_moves(self):
        """Legacy action indices, computed without mutation, spawning, or RNG calls."""
        cells = self._cells.tolist()
        invalid = []
        for action, lines in enumerate(_LINES):
            for a, b, c, d in lines:
                row = (cells[a], cells[b], cells[c], cells[d])
                if _merge(row)[0] != row:
                    break
            else:
                invalid.append(action)
        return invalid

    def move(self, direction):
        cells, self.get_score, changed = self.pure_move(direction)
        self.turns += 1
        self.merge_this_turn = bool(self.get_score)
        zeros = cells.count(0)
        self._cells[:] = cells
        spawned = changed and zeros and not self._game_over(cells)
        if spawned:
            self.score += self.add_random_tile()
        # Deliberately preserve the legacy terminal-state quirk and sticky flag.
        if zeros == 1 and self.game_over():
            self.done = True
        self.score_v2 += self.get_score
        if not spawned:
            self.invalid_move += 1
            reward = -1
        else:
            reward = np.log2(1 + self.get_score) / 16
        self.state = self._cells.reshape(4, 4).copy()
        return self.state, reward, self.done
