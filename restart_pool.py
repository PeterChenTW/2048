"""Training-only restart boards and milestone collection for the curriculum."""


class RestartPool:
    def __init__(self, capacity, rng):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        self.rng = rng
        self._grids = []
        self._next = 0

    def add(self, grid):
        snapshot = [list(row) for row in grid]
        if len(self._grids) < self.capacity:
            self._grids.append(snapshot)
        else:
            self._grids[self._next] = snapshot
        self._next = (self._next + 1) % self.capacity

    def sample(self):
        if not self._grids:
            return None
        return [row.copy() for row in self.rng.choice(self._grids)]

    def __len__(self):
        return len(self._grids)


def collect_episode(grids):
    """Return first 512/1024 crossings and one late-game midpoint snapshot.

    ``grids[0]`` is the episode start; later grids follow actual moves. A
    threshold already present at the start is not collected again. Episodes
    starting at >=512 still contribute a midpoint, using step zero as the
    first-512 step. Odd midpoint indices round down. Snapshots may coincide
    when a short episode ends at a milestone; each collection rule still adds
    its own entry.
    """
    if not grids:
        return []
    maxima = [max(max(row) for row in grid) for grid in grids]
    selected = []
    first_512 = None
    for threshold in (512, 1024):
        first = next((i for i, maximum in enumerate(maxima) if maximum >= threshold), None)
        if threshold == 512:
            first_512 = first
        if maxima[0] < threshold and first is not None:
            selected.append(first)
    if first_512 is not None:
        selected.append((first_512 + len(grids) - 1) // 2)
    return [[list(row) for row in grids[i]] for i in selected]
