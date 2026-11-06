import numpy as np

class GridWorld:
    '''Configurable Gridworld.

    There are 50 states in total, however only 25 cells.
    The first 25 states have has_key = False whereas the last 25 states have has_key = True.
    has_key is a boolean that is True if the agent has visited the state with the key and False if it has not.

    Actions: 0 = up, 1 = down, 2 = left, 3 = right

    slip_prob: probability that the chosen action is replaced by a random one.
    
    shift: None, or dicts like "{"type": "reward", "at_episode": 50} or {"type": "transition", "at_episode": 100}"
    '''

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right

    def __init__(self, slip_prob=0.0, shift=None, max_steps=200, seed=None):
        self.slip_prob = slip_prob
        self.shift = shift
        self.max_steps = max_steps

        self.nS, self.nA = 50, 4
        self.start_cell = 0                   # top-left
        self.key_cell = 3
        self.door_cell = 24                   # bottom-right
        self.ghost_cell = 16                  
        self.walls_cells = [2, 7, 17, 18, 19]

        self.episode = 0

        self.has_key = False
        self.cell = self.start_cell
        self.state = (self.start_cell, self.has_key)
        self.t = 0

        self.rng = np.random.default_rng(seed)

    # ---------- core dynamics ----------
    def move(self, cell, a):
        r, c = divmod(cell, 5)
        dr, dc = self.ACTIONS[a]
        nr, nc = r + dr, c + dc
        next_cell = nr * 5 + nc
        if 0 <= nr < 5 and 0<= nc < 5 and next_cell not in self.walls_cells:
            return next_cell
        return cell

    def reset(self):
        self.episode += 1
        self.apply_shift()
        self.state = (self.start_cell, False)
        self.t = 0
        
        return self.state

    def step(self, a):
        if self.rng.random() < self.slip_prob:
            a = int(self.rng.choice([x for x in range(self.nA) if x != a]))

        self.cell = self.move(self.cell, a)
        self.t += 1

        if self.cell == self.key_cell:
            self.has_key = True

        self.state = (self.cell, self.has_key)

        if self.state == (self.door_cell, True):
            reward, done = 1.0, True
        elif self.cell == self.ghost_cell:
            reward, done = -1.0, True
        else:
            reward, done = 0.0, False

        truncated = self.t >= self.max_steps and not done

        return self.state, reward, done, truncated

    # ---------- shifts ----------
    def apply_shift(self):
        if self.shift is None or self.episode != self.shift["at_episode"]:
            return
        kind = self.shift["type"]
        if kind == "reward":
            # switches the goal state with the hazard state
            self.door_cell, self.ghost_cell = self.ghost_cell, self.door_cell
            return
        elif kind == "transition":
            self.walls_cells.append(8)
            self.walls_cells.remove(17)
            return
        else:
            raise ValueError(f"unknown shift type: {kind!r}")

    # ---------- true model, for sanity checks ----------
    def true_model(self):
        P = np.zeros((self.nS, self.nA, self.nS))
        R = np.zeros((self.nS, self.nA))
        terminals = {16, 41, 49}

        for s in range(self.nS):
            if s in terminals:
                P[s, :, s] = 1.0                      # absorbing, reward 0
                continue

            cell, has_key = s % 25, s // 25
            r, c = divmod(cell, 5)

            # where each action would take the agent, with no slipping
            dest = []
            for a in range(self.nA):
                dr, dc = self.ACTIONS[a]
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in self.walls_cells:
                    ncell = nr * 5 + nc
                else:
                    ncell = cell                      # wall or edge: stay
                nkey = 1 if (has_key or ncell == 3) else 0   # 3 = key cell
                dest.append(ncell + 25 * nkey)

            for a in range(self.nA):
                P[s, a, dest[a]] += 1 - self.slip_prob
                for a2 in range(self.nA):
                    if a2 != a:
                        P[s, a, dest[a2]] += self.slip_prob / 3
                R[s, a] = P[s, a, 49] - P[s, a, 16] - P[s, a, 41]

        return P, R
