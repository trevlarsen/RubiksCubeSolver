import numpy as np
from primary import execute, new_cube, is_solved


def test_sequence(sequence, show=False):
    '''Test whether a sequence applied once is equivalent to no moves'''
    cube1 = new_cube()
    cube2 = new_cube()
    execute(sequence, cube1)
    execute('', cube2)
    if show:
        print(cube1)
        print(cube2)
    return np.allclose(cube1, cube2)


def sequence_order(sequence):
    '''Determine the order of any given sequence'''
    cube = new_cube()
    solved = new_cube()
    execute(sequence, cube)
    order = 1
    while not np.allclose(cube, solved):
        execute(sequence, cube)
        order += 1
    return order


def reverse_sequence(sequence):
    '''Return the reverse of a given sequence'''
    reverse = ''
    for move in sequence[::-1]:
        if move in 'RDFLBUrdflbu':
            if move.isupper():
                reverse += move.lower()
            else:
                reverse += move.upper()
    return reverse


def verify_rotations():
    '''Various sequences to ensure that all rotations function properly'''
    assert sequence_order('RF') == 105, 'Rotations are off'
    assert sequence_order('rdL') == 180, 'Rotations are off'
    assert sequence_order('FFURb') == 144, 'Rotations are off'
    assert sequence_order('RDFLBUrdflbu') == 360, 'Rotations are off'


def valid_solution(scramble_sequence, solution):
    test_cube = new_cube()
    execute(scramble_sequence, test_cube)
    execute(solution, test_cube)
    return is_solved(test_cube)
