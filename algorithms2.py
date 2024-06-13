


import numpy as np
import time as t
import json
from primary import apply, execute, new_cube, scramble


with open('algorithms.json', 'r') as json_file:
        algs = json.load(json_file)


def align(cube, target):
    '''Align the target piece'''
    i = np.where(cube == target)[0][0]
    moves = algs['align'+str(target)].get(str(i))
    return execute(moves, cube), i


# Step 1: Make the Cross ====================================================================================================

def make_cross(cube):
    '''Make the yellow cross on the bottom by aligning pieces 1-4'''
    solution = ''
    for i in range(1,5):
        solution += align(cube, i)[0]
    return solution, 0


# Step 2: Solve Bottom Corners ====================================================================================================

def bottom_corners(cube):
    '''Align the bottom corners (pieces 5-8)'''
    solution = ''
    for i in range(5,9):
        solution += align(cube, i)[0]
    return solution, 0


# Step 3: Solve Middle Edges ====================================================================================================

def middle_edges(cube):
    '''Align the middle edges (pieces 9-12)'''
    solution = ''
    for i in range(9,13):
        solution += align(cube, i)[0]
    return solution, 0


# Step 4: Orient Top Edges ====================================================================================================

def orient_top_edges(cube):
    '''Orient the top edges of the cube'''
    solution = ''
    
    # Determine which pieces are already oriented
    current = cube[np.array([47,45,42,44])]
    oriented = []
    for index, piece in zip([47,45,42,44], current):
        if piece in [47,45,42,44]:
            oriented.append(index)

    # Case: None are oriented
    if not oriented:
        solution += execute('FRUrufBULulb', cube)
    # Case: Two are oriented
    elif len(oriented) == 2:
        so = set(oriented)
        diff = abs(oriented[0] - oriented[1])
        # Case: L shape
        if diff in [2,3]:
            if so == set([42,44]):
                solution += execute('UUBULulb', cube)
            elif so == set([42,45]):
                solution += execute('UBULulb', cube)
            elif so == set([45,47]):
                solution += execute('BULulb', cube)
            else:
                solution += execute('uBULulb', cube)
        # Case: Bar shape
        elif 45 in so:
            solution += execute('FRUruf', cube)
        else:
            solution += execute('UFRUruf', cube)
    return solution, oriented


# Step 5: Permute Top Edges ====================================================================================================

def permute_top_edges(cube):
    '''Permute already oriented top edges'''
    solution = ''

    # Determine which edges are already permuted
    edges = np.array(sorted([13,14,15,16]))
    permuted = edges[edges==cube[edges]]

    # Manipulate until 2 or 4 edges are permuted
    while len(permuted) not in [2,4]:
        apply('U', cube)
        solution += 'U'
        permuted = edges[edges==cube[edges]]

    # Case: All permuted
    if len(permuted) == 4:
        return solution, 0
    # Case: Two permuted
    else:
        diff = abs(permuted[0] - permuted[1])
        # Case: Bar shape
        if diff == 2:
            if 14 in permuted:
                solution += execute('URUUruRurURUUruRurU', cube)
            else:
                solution += execute('RUUruRurURUUruRurUU', cube)
        # Case: L shape
        else:
            if np.allclose(permuted, np.array([15,16])):
                solution += execute('RUUruRuru', cube)
            elif np.allclose(permuted, np.array([14,15])):
                solution += execute('uRUUruRur', cube)
            elif np.allclose(permuted, np.array([13,14])):
                solution += execute('UURUUruRurU', cube)
            else:
                solution += execute('URUUruRurUU', cube)
        return solution, 0


# Step 6: Orient Top Corners ====================================================================================================

def corner_state(cube):
    '''Determine the current orientation of the top corners'''
    corners = [[21,46,20],[19,41,36],[43,18,31],[48,17,26]]
    oriented = []
    for corner in corners:
        if set(corner) == set(cube[corner]):
            oriented.append(corner)
    return oriented


def orient_top_corners(cube):
    '''Orient the top corners of the cube'''
    solution = ''
    
    # Determine which corners are already oriented
    oriented = corner_state(cube)

    # Case: All oriented
    if len(oriented) == 4:
        return solution, 0
    # Case: None oriented
    if not len(oriented):
        solution += execute('lURuLUru', cube)
        oriented = corner_state(cube)
    # Rotate for one oriented case
    if 17 in oriented[0]: c = ''
    elif 18 in oriented[0]: c = 'U'
    elif 19 in oriented[0]: c = 'UU'
    else: c = 'u'
    # Case: One oriented
    while len(oriented) != 4:
        solution += execute(c+'lURuLUru'+c*3, cube)
        oriented = corner_state(cube)
    return solution, 0


# Step 7: Permute Top Corners ====================================================================================================

def permute_top_corners(cube):
    '''Permute the top corners of the cube'''
    solution = ''
    finished = set([41,43,48,46])
    permuted = set()

    # Cycle corners until all are permuted
    while permuted != finished:
        if cube[48] not in finished:
            solution += execute('rdRDrdRD', cube)
            continue
        permuted.add(cube[48])
        solution += execute('U', cube)
    return solution, 0



# Functions for testing individual algorithms =================================================================================

def test_align(function, target_pieces, prev_functions=[], preserve_pieces=[]):
    '''Test whether an alignment function aligns the desired pieces while keeping other specified in alignment'''
    cube = new_cube()
    scramble(cube)

    # Apply previous functions
    for func in prev_functions:
        func(cube)
    a, i = function(cube)

    # Check for successful alignment of target pieces
    success = True
    for piece in target_pieces:
        if cube[piece] != piece:
            success = False
            break

    # Check for successful preservation of specified pieces
    ignored = True
    for piece in preserve_pieces:
        if cube[piece] != piece:
            ignored = False
            break

    return success, ignored, i


def trial_align(function, target_pieces, prev_functions=[], preserve_pieces=[], n=1000):
    '''Run a sequence of trials to test an alignment function'''
    if type(target_pieces) != list or type(preserve_pieces) != list or type(prev_functions) != list:
        raise ValueError('Enter pieces as a list')
    
    # Initializing trial
    success = 0
    aligned = 0
    failed = {}
    print()
    print('\nRunning trials.......', end='')

    # Apply alignment function
    start = t.time()
    for i in range(n):
        s, a, i= test_align(function, target_pieces, prev_functions=prev_functions, preserve_pieces=preserve_pieces)
        # Record successes/failures
        if s: success += 1
        else:
            if i not in failed: failed[i] = 1
            else: failed[i] += 1
        if not a:
            if i not in failed: failed[i] = 1
            else: failed[i] += 1
        aligned += a
    time = t.time() - start

    # Print results
    print('Finished.')
    print(f'\nResult: {function.__name__} worked {100*success/n}% of the time and kept solved pieces aligned {100*aligned/n}% of the time in {time:.4f} seconds.')
    print(f'\nFailures occurred in routines for positions {failed}\n')


# Corners Template
# algs = {5:'',    6:'',   7:'',   8:'',  # Yellow corners
#         21:'',  17:'',  25:'',  23:'',  # Green corners
#         41:'',  43:'',  48:'',  46:'',  # White corners
#         35:'',  33:'',  31:'',  19:'',  # Blue corners
#         36:'',  20:'',  40:'',  38:'',  # Orange corners
#         26:'',  18:'',  30:'',  28:'',  # Red corners
#     }

# Edges Template
# algs = { 1:'',  2:'',  3:'',  4:'', # Yellow edges
#         24:'',  9:'', 13:'', 22:'', # Green edges
#         47:'', 45:'', 42:'', 44:'', # White edges
#         34:'', 32:'', 15:'', 11:'', # Blue edges
#         12:'', 16:'', 37:'', 39:'', # Orange edges
#         27:'', 14:'', 10:'', 29:''  # Red edges 
# }