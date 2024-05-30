from coordinates import Coordinate
import random
import sys
from python.agent import DQNAgent
import numpy as np

if sys.version_info >= (3,):
    xrange = range


class Player(object):
    def __init__(self):
        self.agent = DQNAgent(25, 4)
        self.vision = None
        self.genome = None

        self.state = None
        self.new_state = None
        self.reward = 0
        self.act = None
        self.done = None
        self.state_size = 25

    def get_data(self,state,new_state,reward,done):
        self.state = state
        self.new_state = new_state
        self.reward = reward
        self.done = done
        return self.learn()




    def take_turn(self, genome, vision):
        self.vision = vision
        self.genome = genome
        #self.reward = reward
        return self.turn()

    def bit_at(self, position):
        return (self.genome >> position) & 1

    def bit_range(self, start, stop):
        return (self.genome >> start) & ((1 << (stop - start)) - 1)

    def bit_chunk(self, start, length):
        return (self.genome >> start) & ((1 << length) - 1)

    def turn(self):
        return Coordinate(1, 0)

    def learn(self):
        batch_size = 32

        self.agent.remember(self.state, self.act, self.reward, self.new_state, self.done)
        if len(self.agent.memory) > batch_size:
            self.agent.replay(batch_size)


    def vision_at(self, x, y):
        return self.vision[2 + y][2 + x]

class aboba(Player):
    def __init__(self):
        Player.__init__(self)
        self.coords = [Coordinate(-1,-1),
                       Coordinate( 0,-1),
                       Coordinate( 1, 0),
                       Coordinate( 1,-1),
                       Coordinate(-1, 0),
                       Coordinate( 0, 0),
                       Coordinate(-1, 1),
                       Coordinate( 0, 1),
                       Coordinate( 1, 1)]


        self.n_moves = len(self.coords)


    def parse_map(self):
        # Собираем информацию из генома
        trap1 = self.bit_chunk(0, 4)
        trap1_offsetx = self.bit_chunk(4, 2)
        trap1_offsety = self.bit_chunk(6, 2)
        trap2 = self.bit_chunk(8, 4)
        trap2_offsetx = self.bit_chunk(12, 2)
        trap2_offsety = self.bit_chunk(14, 2)
        wall1 = self.bit_chunk(16, 4)
        wall2 = self.bit_chunk(20, 4)
        tel1 = self.bit_chunk(24, 4)
        tel1_offsetx = self.bit_chunk(28, 4)
        tel1_offsety = self.bit_chunk(32, 4)
        tel2 = self.bit_chunk(36, 4)
        tel2_offsetx = self.bit_chunk(40, 4)
        tel2_offsety = self.bit_chunk(44, 4)
        tel3 = self.bit_chunk(48, 4)
        tel3_offsetx = self.bit_chunk(52, 4)
        tel3_offsety = self.bit_chunk(56, 4)
        tel4 = self.bit_chunk(60, 4)
        tel4_offsetx = self.bit_chunk(64, 4)
        tel4_offsety = self.bit_chunk(68, 4)

        vis = np.arange(25)
        vis = vis.reshape(5,5)
        # Расширяем карту до всех возможных клеток
        for j in range(len(self.vision)):
            for i in range(len(self.vision[j])):
                if self.vision[j][i] == trap1:
                    vis[j][i] = 201
                elif self.vision[j][i] == trap2:
                    vis[j][i] = 202
                elif self.vision[j][i] == wall1:
                    vis[j][i] = 101
                elif self.vision[j][i] == wall2:
                    vis[j][i] = 102
                elif self.vision[j][i] == tel1:
                    vis[j][i] = 301
                elif self.vision[j][i] == tel2:
                    vis[j][i] = 302
                elif self.vision[j][i] == tel3:
                    vis[j][i] = 303
                elif self.vision[j][i] == tel4:
                    vis[j][i] = 304
                elif self.vision[j][i] == -1:
                    vis[j][i] = 203
                else:
                    vis[j][i] = 0
        return vis

    def learn(self):
        batch_size = 32
        state = self.parse_map()
        state = np.reshape(state, [1, self.state_size])
        new_state = self.parse_map()
        new_state = np.reshape(new_state, [1, self.state_size])
        self.agent.remember(state, self.act, self.reward, new_state, self.done)
        if len(self.agent.memory) > batch_size:
            self.agent.replay(batch_size)

    def turn(self):
        state = self.parse_map()
        state = np.reshape(state, [1, self.state_size])
        action = self.agent.act(state)
        self.act = action
        x = 0
        y = 0
        if action == 0:
            x = 1
            y = 0
        elif action == 1:
            x = -1
            y = 0
        elif action == 2:
            x = 0
            y = 1
        elif action == 3:
            x = 0
            y = -1
        return Coordinate(x,y)










class ForwardPlayer(Player):
    def turn(self):
        return Coordinate(1, 0)

class RandomPlayer(Player):
    def turn(self):
        return Coordinate(1, random.randint(-1, 1))


class LinearCombinationPlayer(Player):
    def __init__(self):
        Player.__init__(self)
        self.coords = [  # Coordinate(-1,-1),
            # Coordinate( 0,-1),
            Coordinate(1, 0),
            Coordinate(1, -1),
            # Coordinate(-1, 0),
            # Coordinate( 0, 0),
            # Coordinate(-1, 1),
            # Coordinate( 0, 1),
            Coordinate(1, 1)]
        self.n_moves = len(self.coords)

    def turn(self):
        restricted_coords = [c for c in self.coords if self.vision_at(c.x, c.y) > -1]
        restricted_n_moves = len(restricted_coords)
        s = 0
        for i in range(25):
            s += self.bit_range(2 * i, 2 * i + 2) * self.vision_at(int(i / 5) - 2, i % 5 - 2)
        return restricted_coords[s % restricted_n_moves]


class ColorScorePlayer(Player):
    def __init__(self):
        Player.__init__(self)
        self.coords = [  # Coordinate(-1,-1),
            # Coordinate( 0,-1),
            Coordinate(1, 0),
            Coordinate(1, -1),
            # Coordinate(-1, 0),
            # Coordinate( 0, 0),
            # Coordinate(-1, 1),
            # Coordinate( 0, 1),
            Coordinate(1, 1)]
        self.n_moves = len(self.coords)

    def turn(self):
        max_score = max(
            [self.bit_chunk(6 * self.vision_at(c.x, c.y), 6) for c in self.coords if self.vision_at(c.x, c.y) >= 0])
        restricted_coords = [c for c in self.coords if
                             self.vision_at(c.x, c.y) >= 0 and self.bit_chunk(6 * self.vision_at(c.x, c.y),
                                                                              6) == max_score]

        return random.choice(restricted_coords)


class LemmingPlayer(Player):
    def __init__(self):
        Player.__init__(self)
        self.coords = [Coordinate(-1, -1),
                       # Coordinate( 0,-1),
                       Coordinate(1, 0),
                       # Coordinate( 1,-1),
                       Coordinate(-1, 0),
                       # Coordinate( 0, 0),
                       Coordinate(-1, 1),
                       # Coordinate( 0, 1),
                       # Coordinate( 1, 1)
                       ]

    def turn(self):
        return random.choice(self.coords)


class IllegalPlayer(Player):
    def __init__(self):
        Player.__init__(self)
        self.coords = [Coordinate(2, -1),
                       Coordinate(2, 0),
                       Coordinate(2, 1)
                       ]

    def turn(self):
        return random.choice(self.coords)


class NeighborsOfNeighbors(Player):
    def __init__(self):
        Player.__init__(self)
        self.coords = [Coordinate(1, 0),
                       Coordinate(1, -1),
                       Coordinate(1, 1)
                       ]

    def turn(self):
        scores = [self.score(c.x, c.y) + 0.5 * self.adjacentScore(c.x, c.y) if self.vision_at(c.x, c.y) > -1 else 0
                  for c in self.coords]
        max_score = max(scores)
        return random.choice([c for s, c in zip(scores, self.coords) if s == max_score])

    def adjacentScore(self, x, y):
        adjacent = [(x + 1, y)]
        if self.vision_at(x, y + 1) > -1:
            adjacent += [(x + 1, y + 1)]
        if self.vision_at(x, y - 1) > -1:
            adjacent += [(x + 1, y - 1)]
        adjscores = [self.score(a, b) for a, b in adjacent]
        return sum(adjscores) / float(len(adjscores))

    def score(self, x, y):
        return -1 if self.vision_at(x, y) == -1 else self.bit_chunk(6 * self.vision_at(x, y), 6)




PLAYER_TYPE = aboba
