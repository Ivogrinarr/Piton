# All constants in one place for clarity

from __future__ import print_function
from math import exp, log

BOARD_WIDTH = 50
BOARD_HEIGHT = 15
UNSAFE_BOARD_WIDTH = BOARD_WIDTH - 1
BOARD_EXTENDED_WIDTH = BOARD_WIDTH + 3

OUT_OF_BOUNDS_COLOR = -1
NUMBER_OF_SAFE_COLORS = 8
NUMBER_OF_KILLER_TYPES = 2
NUMBER_OF_TELEPORTER_TYPES = 4
NUMBER_OF_WALL_TYPES = 2
MAX_TELEPORT_RADIUS = 4

from board import Board
from random import Random
from specimen import Specimen
import player as Player
import sys
import trap
import coordinates
import time

if sys.version_info >= (3,):
    xrange = range

#Display related constants:
TITLE = "The Genetic Rat Race"
CELL_SCALAR = 8
EMPTY_COLOR = (255, 255, 255)
SPECIMEN_COLOR = (0, 0, 0)
DEATH_COLOR = (255, 128, 128)
TELEPORT_COLOR = (128, 128, 255)
WALL_COLOR = (128, 128, 128)

#Pick one of the following:
#from pygame_display import Display  #Requires pygame
from tkinter_display import Display #tkinter comes installed with most versions of python
#from text_display import Display
#from no_display import Display

NUMBER_OF_BOARDS = 50

NUMBER_OF_COLORS = sum([trap_type.max_traps for trap_type in trap.trap_types])\
                   + NUMBER_OF_SAFE_COLORS

NUMBER_OF_TURNS = 10000

INITIAL_SPECIMENS = 1
SPECIMEN_LIFESPAN = 5000
REPRODUCTION_RATE = 0
NUM_PARENTS = 1 #2

GENOME_LENGTH = 100
GENOME_MAX_VALUE = (1 << GENOME_LENGTH) - 1
GENOME_CROSSOVER_RATE = .05
GENOME_MUTATION_RATE = .01

VISION_WIDTH = 5
VISION_DISTANCE = int(VISION_WIDTH/2)
VISION = [[coordinates.Coordinate(x, y)
           for x in xrange(-VISION_DISTANCE, VISION_DISTANCE+1)
           ]
          for y in xrange(-VISION_DISTANCE, VISION_DISTANCE+1)
          ]

RANDOM_SEED = 13722829
