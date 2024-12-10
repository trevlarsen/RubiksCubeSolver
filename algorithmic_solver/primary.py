import numpy as np
from random import randint
from condense import condense


def define_rotations():
    '''Defines the cycle structure for each of the six primitive rotations'''
    R = [[27,14,10,29],[26,18,30,28],[25,48,31, 7],[ 9,45,32, 2],[17,43,33, 6]]
    L = [[37,16,12,39],[36,20,40,38],[35,41,21, 5],[11,44,22, 4],[19,46,23, 8]]
    U = [[47,44,42,45],[46,41,43,48],[17,20,19,18],[13,16,15,14],[21,36,31,26]]
    D = [[ 1, 2, 3, 4],[ 6, 7, 8, 5],[23,28,33,38],[24,29,34,39],[25,30,35,40]]
    F = [[22,13, 9,24],[21,17,25,23],[40,46,26, 6],[12,47,27, 1],[20,48,28, 5]]
    B = [[32,15,11,34],[31,19,35,33],[30,43,36, 8],[10,42,37, 3],[18,41,38, 7]]
    return R, L, U, D, F, B


R, L, U, D, F, B = define_rotations()
moves_dict = {'R': R, 'L': L, 'U': U, 'D': D, 'F': F, 'B': B}


def apply(move, cube):
    '''Apply a single move to the cube that modifies it in place'''
    # Adapt for a reverse move
    if move.islower():
        move = move.upper()
        r = -1
    else: r = 1

    # Roll cycle structure groups
    for perm in moves_dict[move]:
        selected = cube[perm]
        rolled = np.roll(selected, r)
        cube[perm] = rolled


def execute(moves, cube):
    '''Apply a sequence of moves that modifies it in place'''
    applied = ''

    # Apply each move
    for move in moves:
        if move in 'RLUDFBrludfb':
            apply(move, cube)
            applied += move
    return applied


def scramble(cube, n=50):
    '''Scramble a cube using n random moves'''
    moves = 'RDFLBUrdflbu'
    sequence = ''
    condensed = ''

    while len(condensed) < n:
        move = moves[randint(0, 11)]
        sequence += move
        condensed = condense(sequence)
        
    execute(condensed, cube)
    return condensed


def new_cube():
    '''Return a solved cube'''
    return np.arange(49)


def is_solved(cube):
    '''Returns True is the cube is solved'''
    return np.allclose(cube, np.arange(49))

