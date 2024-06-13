


import numpy as np
import time as t
from primary import apply, execute, new_cube, scramble


# Step 1: Make the Cross ====================================================================================================

def align1(cube):
    '''Aligns edge piece 1'''
    i = np.where(cube == 1)[0][0]
    algs = { 1:'s',    2:'d',    3:'DD',    4:'D',   # Yellow edges
            24:'frd',  9:'rd',  13:'Frd',  22:'LD',  # Green edges
            47:'FF',  45:'UFF', 42:'UUFF', 44:'uFF', # White edges
            34:'dRF', 32:'Rd',  15:'UrF',  11:'lD',  # Blue edges
            12:'f',   16:'Lf',  37:'LLf',  39:'lf',  # Orange edges
            27:'F',   14:'rF',  10:'rrF',  29:'RF'   # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align2(cube):
    '''Aligns edge piece 2'''
    i = np.where(cube == 2)[0][0]
    algs = { 1:'n',    2:'s',    3:'Fdf',   4:'FDDf',  # Yellow edges
            24:'n',    9:'r',   13:'Frf',  22:'FFrFF', # Green edges
            47:'uRR', 45:'RR',  42:'URR',  44:'UURR',  # White edges
            34:'BR',  32:'R',   15:'bR',   11:'BBR',   # Blue edges
            12:'fDF', 16:'UbR', 37:'bURR', 39:'lfDF',  # Orange edges
            27:'FDf', 14:'ubR', 10:'Dbd',  29:'rDbd'   # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align3(cube):
    '''Aligns edge piece 3'''
    i = np.where(cube == 3)[0][0]
    algs = { 1:'n',     2:'n',     3:'s',    4:'LDld', # Yellow edges
            24:'n',     9:'RuBBr', 13:'UlB', 22:'DLd', # Green edges
            47:'UUBB',  45:'uBB',  42:'BB',  44:'UBB', # White edges
            34:'BdRD',  32:'dRD',  15:'ulB', 11:'Dld', # Blue edges
            12:'LLB',   16:'lB',   37:'B',   39:'LB',  # Orange edges
            27:'RRbRR', 14:'Rbr',  10:'b',   29:'n'    # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align4(cube):
    '''Aligns edge piece 4'''
    i = np.where(cube == 4)[0][0]
    algs = { 1:'n',   2:'n',       3:'n',    4:'s',    # Yellow edges
            24:'n',   9:'DDrDD',  13:'fLF', 22:'L',    # Green edges
            47:'ULL', 45:'UULL',  42:'uLL', 44:'LL',   # White edges
            34:'n',   32:'DDRDD', 15:'Blb', 11:'l',    # Blue edges
            12:'Dfd', 16:'ufLF',  37:'dBD', 39:'DFdL', # Orange edges
            27:'DFd', 14:'UfLF',  10:'dbD', 29:'n'     # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def make_cross(cube):
    solution = ''
    solution += align1(cube)[0]
    solution += align2(cube)[0]
    solution += align3(cube)[0]
    solution += align4(cube)[0]
    return solution, 0



# Step 2: Solve Bottom Corners ====================================================================================================

def align5(cube):
    '''Aligns corner piece 5'''
    i = np.where(cube == 5)[0][0]
    algs = {5:'s',        6:'RUrfLFl',  7:'BFUUbf',    8:'buBLflF',  # Yellow corners
            21:'LflF',   17:'UfLFl',   25:'fUUFFuf',  23:'FUUflUUL', # Green corners
            41:'LLfLLF', 43:'uLLfLLF', 48:'FFLFFl',   46:'lULFUUf',  # White corners
            35:'FbufB',  33:'BlUULb',  31:'lUUL',     19:'Fuf',      # Blue corners
            36:'uLflF',  20:'fLFl',    40:'lUULFUUf', 38:'LUULLUL',  # Orange corners
            26:'lUL',    18:'FUUf',    30:'rFUUfR',   28:'lRULr',    # Red corners
    }
    
    moves = algs.get(i)
    return execute(moves, cube), i


def align6(cube):
    '''Aligns corner piece 6'''
    i = np.where(cube == 6)[0][0]
    algs = {5:'n',         6:'s',       7:'BUbrFRf',   8:'LUUlrFRf', # Yellow corners
            21:'uFrfR',   17:'rFRf',   25:'fUUFRUUr', 23:'n',        # Green corners
            41:'URRFRRf', 43:'RRFRRf', 48:'uRRFRRf',  46:'UURRFRRf', # White corners
            35:'bRUUrB',  33:'BfUbF',  31:'fUF',      19:'RUUr',     # Blue corners
            36:'fUUF',    20:'Rur',    40:'n',        38:'LUlfUF',   # Orange corners
            26:'FrfR',    18:'UrFRf',  30:'rUURRur',  28:'RUruFrfR', # Red corners
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align7(cube):
    '''Aligns corner piece 7'''
    i = np.where(cube == 7)[0][0]
    algs = {5:'n',         6:'n',        7:'s',         8:'buBrUUR', # Yellow corners
            21:'rUUR',    17:'Bub',     25:'n',        23:'n',       # Green corners
            41:'BBRBBr',  43:'uBBRBBr', 48:'UUBBRBBr', 46:'UBBRBBr', # White corners
            35:'bUUBBub', 33:'rURRbrB', 31:'RbrB',     19:'UbRBr',   # Blue corners
            36:'rUR',     20:'BUUb',    40:'n',        38:'LrUlR',   # Orange corners
            26:'uRbrB',   18:'bRBr',    30:'BuBBRBr',  28:'n',       # Red corners
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align8(cube):
    '''Aligns corner piece 8'''
    i = np.where(cube == 8)[0][0]
    algs = {5:'n',         6:'n',         7:'n',          8:'s',        # Yellow corners
            21:'bUB',     17:'LUUl',     25:'n',         23:'n',        # Green corners
            41:'LulbUUB', 43:'uLulbUUB', 48:'UULulbUUB', 46:'ULulbUUB', # White corners
            35:'LuLLBLb', 33:'n',        31:'uBlbL',     19:'lBLb',     # Blue corners
            36:'BlbL',    20:'UlBLb',    40:'n',         38:'bUBBlbL',  # Orange corners
            26:'bUUB',    18:'Lul',      30:'n',         28:'n',        # Red corners
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def bottom_corners(cube):
    solution = ''
    solution += align5(cube)[0]
    solution += align6(cube)[0]
    solution += align7(cube)[0]
    solution += align8(cube)[0]
    return solution, 0



# Step 3: Solve Middle Edges ====================================================================================================

def align9(cube):
    '''Aligns edge piece 9'''
    i = np.where(cube == 9)[0][0]
    algs = { 1:'n',                2:'n',                 3:'n',                4:'n',                # Yellow edges
            24:'n',                9:'s',                13:'URurFrfR',        22:'lULfLFlfUFrFRf',    # Green edges
            47:'UUfUFrFRf',       45:'ufUFrFRf',         42:'fUFrFRf',         44:'UfUFrFRf',         # White edges
            34:'n',               32:'rURbRBrUUfUFrFRf', 15:'uRurFrfR',        11:'LulBlbLUUfUFrFRf', # Blue edges
            12:'lULfLFluRurFrfR', 16:'RurFrfR',          37:'LulBlbLURurFrfR', 39:'n',                # Orange edges
            27:'fUFuRUUrUrFRf',   14:'UURurFrfR',        10:'BubRbrBUfUFrFRf', 29:'n'                 # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align10(cube):
    '''Aligns edge piece 10'''
    i = np.where(cube == 10)[0][0]
    algs = { 1:'n',                 2:'n',              3:'n',               4:'n',               # Yellow edges
            24:'n',                 9:'n',             13:'BubRbrB',        22:'lULfLFlurURbRBr', # Green edges
            47:'UrURbRBr',         45:'UUrURbRBr',     42:'urURbRBr',       44:'rURbRBr',         # White edges
            34:'n',                32:'rURuBUUbUbRBr', 15:'UUBubRbrB',      11:'LulBlbLUrURbRBr', # Blue edges
            12:'lULfLFlUUBubRbrB', 16:'uBubRbrB',      37:'LulBlbLBubRbrB', 39:'n',               # Orange edges
            27:'n',                14:'UBubRbrB',      10:'s',              29:'n'                # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align11(cube):
    '''Aligns edge piece 11'''
    i = np.where(cube == 11)[0][0]
    algs = { 1:'n',                2:'n',          3:'n',              4:'n',                # Yellow edges
            24:'n',                9:'n',         13:'uLulBlbL',      22:'lULfLFlUUbUBlBLb', # Green edges
            47:'bUBlBLb',         45:'UbUBlBLb',  42:'UUbUBlBLb',     44:'ubUBlBLb',         # White edges
            34:'n',               32:'n',         15:'ULulBlbL',      11:'s',                # Blue edges
            12:'lULfLFlULulBlbL', 16:'UULulBlbL', 37:'LulUbUUBuBlbL', 39:'n',                # Orange edges
            27:'n',               14:'LulBlbL',   10:'n',             29:'n'                 # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def align12(cube):
    '''Aligns edge piece 12'''
    i = np.where(cube == 12)[0][0]
    algs = { 1:'n',         2:'n',         3:'n',          4:'n',             # Yellow edges
            24:'n',         9:'n',        13:'UUFufLflF', 22:'lULuFUUfUfLFl', # Green edges
            47:'ulULfLFl', 45:'lULfLFl',  42:'UlULfLFl',  44:'UUlULfLFl',     # White edges
            34:'n',        32:'n',        15:'FufLflF',   11:'n',             # Blue edges
            12:'s',         16:'UFufLflF', 37:'n',         39:'n',             # Orange edges
            27:'n',        14:'uFufLflF', 10:'n',         29:'n'              # Red edges 
    }
    moves = algs.get(i)
    return execute(moves, cube), i


def middle_edges(cube):
    solution = ''
    solution += align9(cube)[0]
    solution += align10(cube)[0]
    solution += align11(cube)[0]
    solution += align12(cube)[0]
    return solution, 0



# Step 4: Orient Top Edges ====================================================================================================

def orient_top_edges(cube):
    '''Orient the top edges of the cube'''
    solution = ''
    current = cube[np.array([47,45,42,44])]
    oriented = []
    for index, piece in zip([47,45,42,44], current):
        if piece in [47,45,42,44]:
            oriented.append(index)

    if not oriented:
        solution += execute('FRUrufBULulb', cube)
    elif len(oriented) == 2:
        so = set(oriented)
        diff = abs(oriented[0] - oriented[1])
        if diff in [2,3]:
            if so == set([42,44]):
                solution += execute('UUBULulb', cube)
            elif so == set([42,45]):
                solution += execute('UBULulb', cube)
            elif so == set([45,47]):
                solution += execute('BULulb', cube)
            else:
                solution += execute('uBULulb', cube)
        elif 45 in so:
            solution += execute('FRUruf', cube)
        else:
            solution += execute('UFRUruf', cube)
    return solution, oriented



# Step 5: Permute Top Edges ====================================================================================================

def permute_top_edges(cube):
    '''Permute already oriented top edges'''
    solution = ''
    edges = np.array(sorted([13,14,15,16]))
    permuted = edges[edges==cube[edges]]
    while len(permuted) not in [2,4]:
        apply('U', cube)
        solution += 'U'
        permuted = edges[edges==cube[edges]]
    if len(permuted) == 4:
        return solution
    else:
        diff = abs(permuted[0] - permuted[1])
        if diff == 2:
            if 14 in permuted:
                solution += execute('URUUruRurURUUruRurU', cube)
            else:
                solution += execute('RUUruRurURUUruRurUU', cube)
        else:
            if np.allclose(permuted, np.array([15,16])):
                solution += execute('RUUruRuru', cube)
            elif np.allclose(permuted, np.array([14,15])):
                solution += execute('uRUUruRur', cube)
            elif np.allclose(permuted, np.array([13,14])):
                solution += execute('UURUUruRurU', cube)
            else:
                solution += execute('URUUruRurUU', cube)
        return solution



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
    oriented = corner_state(cube)
    if len(oriented) == 4:
        return solution
    if not len(oriented):
        solution += execute('lURuLUru', cube)
        oriented = corner_state(cube)
    if 17 in oriented[0]: c = ''
    elif 18 in oriented[0]: c = 'U'
    elif 19 in oriented[0]: c = 'UU'
    else: c = 'u'
    while len(oriented) != 4:
        solution += execute(c+'lURuLUru'+c*3, cube)
        oriented = corner_state(cube)
    return solution



# Step 7: Permute Top Corners ====================================================================================================

def permute_top_corners(cube):
    '''Permute the top corners of the cube'''
    solution = ''
    finished = set([41,43,48,46])
    permuted = set()
    while permuted != finished:
        if cube[48] not in finished:
            solution += execute('rdRDrdRD', cube)
            continue
        permuted.add(cube[48])
        solution += execute('U', cube)
    return solution



# Functions for testing individual algorithms =================================================================================

def test_align(function, target_pieces, prev_functions=[], preserve_pieces=[]):
    '''Test whether an alignment function aligns the desired pieces while keeping other specified in alignment'''
    cube = new_cube()
    scramble(cube)
    for func in prev_functions:
        func(cube)
    a, i = function(cube)

    success = True
    for piece in target_pieces:
        if cube[piece] != piece:
            success = False
            break
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
    success = 0
    aligned = 0
    failed = {}
    print()
    print('\nRunning trials.......', end='')
    start = t.time()
    for i in range(n):
        s, a, i= test_align(function, target_pieces, prev_functions=prev_functions, preserve_pieces=preserve_pieces)
        if s:
            success += 1
        else:
            if i not in failed:
                failed[i] = 1
            else:
                failed[i] += 1
        if not a:
            if i not in failed:
                failed[i] = 1
            else:
                failed[i] += 1
        aligned += a
    time = t.time() - start
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