import numpy as np
import seaborn as sns
import warnings
from matplotlib.colors import to_hex, ListedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class TicTacToe:
    def __init__(self, actors=[]):
        if not actors:
            actors = [Actor(1, "blue"), Actor(-1, "red")]
        self.actors = actors
        self.board = np.zeros(shape=(3,3))

        self.values = [a.value for a in actors]
        self.vmap = {a.value : a for a in actors}
        assert 0 not in self.values, "WARNING : 0 is not a valid actor value"
        assert len(self.values) == len(set(self.values)), "WARNING : The actors do not have strictly unique values"
        self.values = [0] + self.values
        
        self.colors = [a.color for a in actors]
        assert len(self.colors) == len(set(self.colors)), "WARNING : The actors do not have strictly unique colors"
        assert "white" not in self.colors, "WARNING : white is not a valid actor color"
        self.colors = ["white"] + self.colors
        cmapping = {v:c for v,c in zip(self.values, self.colors)}
        self.cmap = ListedColormap([cmapping[v] for v in sorted(self.values)], N=len(self.colors))
        
        self.played = []
        self.round = 1
        self.turn = 1
        self.ended = False

    def reset(self):
        self.played = []
        self.round = 1
        self.turn = 1
        self.ended = False
        self.board = np.zeros(shape=(3,3))

    def verifyPlay(self, position):
        return self.board[position[0],position[1]] == 0
        
    def play(self, actor, position):
        if self.ended:
            warnings.warn(f"Game is over!")
            return False
        if len(self.played) == 2:
            self.played = []
            self.round += 1
        if actor.value in self.played:
            warnings.warn(f"Actor {actor.value} already played this round")
            return False
        if not self.verifyPlay(position):
            warnings.warn(f"Position {position} is not playable")
            return False
        if actor.value != self.turn:
            warnings.warn(f"It is not {actor}'s turn!")
        self.board[position[0],position[1]] = actor.value
        self.played.append(actor.value)
        self.turn *= -1
        if self.winState():
            self.ended = True
        if len(np.argwhere(self.board==0)) == 0:
            self.ended = True
        return True
    
    def winner(self):
        for i in range(3):
            if not (self.board[i,:] == 0).any() and (self.board[i, :] == self.board[i, 0]).all():
                return self.board[i,0]
            if not (self.board[:,i] == 0).any() and (self.board[:, i] == self.board[0, i]).all():
                return self.board[0,i]
        if not (np.diag(self.board) == 0).any() and (np.diag(self.board) == self.board[0,0]).all():
            return self.board[0,0]
        if not np.diag(self.board[:,::-1] == 0).any() and (np.diag(self.board[:,::-1]) == self.board[0,2]).all():
            return self.board[0,2]
        
        return 0

    def winState(self):
        return self.winner() != 0

    def reward(self):
        result = self.winner()
        # No winner yet
        if result == 0:
            self.vmap[1].feed(0.1)
            self.vmap[-1].feed(0.1)
            return

        self.vmap[result].feed(1)
        self.vmap[-result].feed(0)

    def copy(self):
        _copy = TicTacToe(self.actors)
        _copy.board = self.board.copy()
        return _copy

def hashBoard(board):
    return str(board.reshape(9))

class Actor:
    def __init__(self, name, color, value):
        self.value = value
        self.color = color

        self.name = name
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return self.name

def displayGenerator(**kwargs):
    def displayBoard(board):
        hm = sns.heatmap(board, **kwargs)
        return hm

    return displayBoard

def generateColor():
    color = hex(random.randint(0,255))
    while color in COLORS:
        color = hex(random.randint(0,255))
    return color