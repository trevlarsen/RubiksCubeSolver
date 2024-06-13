


import numpy as np
import re
import string
from primary import execute, new_cube


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


def replace_three_same_with_opposite_case(text):
    return re.sub(r'([a-zA-Z])\1{2}', lambda m: m.group(1).lower() if m.group(1).isupper() else m.group(1).upper(), text)


def remove_uppercase_lowercase_pairs(text):
    pattern = '|'.join([f'{c}{c.lower()}|{c.lower()}{c}' for c in string.ascii_uppercase])
    return re.sub(pattern, '', text)


def replace_two_lowercase_with_two_uppercase(text):
    return re.sub(r'([a-z])\1', lambda m: m.group(1).upper() * 2, text)


def condense(solution):
    solution = replace_three_same_with_opposite_case(solution)
    solution = remove_uppercase_lowercase_pairs(solution)
    solution = replace_three_same_with_opposite_case(solution)
    solution = remove_uppercase_lowercase_pairs(solution)
    solution = replace_two_lowercase_with_two_uppercase(solution)
    return solution


# replacements_list = [
#     {'Rr':'', 'Ll':'', 'Dd':'', 'Uu':'', 'Ff':'', 'Bb':''},
#     {'rR':'', 'lL':'', 'dD':'', 'uU':'', 'fF':'', 'bB':''},
#     {'rrr':'R', 'lll':'L', 'ddd':'D', 'uuu':'U', 'fff':'F', 'bbb':'B'},
#     {'RRR':'r', 'LLL':'l', 'DDD':'d', 'UUU':'u', 'FFF':'f', 'BBB':'b'},
#     {'Rr':'', 'Ll':'', 'Dd':'', 'Uu':'', 'Ff':'', 'Bb':''},
#     {'rR':'', 'lL':'', 'dD':'', 'uU':'', 'fF':'', 'bB':''},
#     {'rrr':'R', 'lll':'L', 'ddd':'D', 'uuu':'U', 'fff':'F', 'bbb':'B'},
#     {'RRR':'r', 'LLL':'l', 'DDD':'d', 'UUU':'u', 'FFF':'f', 'BBB':'b'},
#     {'rr':'RR', 'll':'LL', 'dd':'DD', 'uu':'UU', 'ff':'FF', 'bb':'BB', }
# ]

# def replace_match(match, replacements):
#     return replacements[match.group(0)]    


# def condense(solution):
#     '''Remove redundancies from a solution'''
#     for replacements in replacements_list:
#         pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
#         solution = re.sub(pattern, lambda match: replace_match(match, replacements), solution)
#     pattern = re.compile(r'(.)\1\1')
#     if bool(pattern.search(solution)):
#         return condense(solution)
#     return solution