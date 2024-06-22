


import numpy as np
import time as t
import textwrap
import curses
from matplotlib import pyplot as plt

from primary import apply, execute, scramble, new_cube, is_solved
from sequences import test_sequence, sequence_order, reverse_sequence, verify_rotations, condense
from algorithms2 import make_cross, bottom_corners, middle_edges, orient_top_edges, permute_top_edges, orient_top_corners, permute_top_corners
from show_cube import show_cube
# from interface import side_data


def solve(cube):
    '''Implements the beginner's method to solve a scrambled cube'''
    solution = ''
    solution += make_cross         (cube)[0]
    solution += bottom_corners     (cube)[0]
    solution += middle_edges       (cube)[0]
    solution += orient_top_edges   (cube)[0]
    solution += permute_top_edges  (cube)[0]
    solution += orient_top_corners (cube)[0]
    solution += permute_top_corners(cube)[0]
    return solution


def one_solve(n=30, details=True, follow=False, plot=False, skip=1, both_sides=False):
    '''Performs a single solve and outputs data about the solve'''
    cube = new_cube()
    scramble_sequence = scramble(cube,n)

    if details:
        print()
        print('Solving.......', end='')

    # Perform and time solution
    start = t.time()
    solution = condense(solve(cube))
    duration = t.time() - start
    move_count = len(solution)

    # Record result
    if is_solved(cube):
        solved = True
    else:
        solved = False
    
    # Print details
    if details:
        if solved:
            print('Solve Successful!')
        else:
            print('Solve failed')
        print()
        print('Details:')
        print(f'Time: {1000*duration:.2f} milliseconds')
        print('Scramble: ' + scramble_sequence)

        # Format solution for user to follow easily
        if follow:
            solution = textwrap.wrap(solution, 5)
            print(f'Solution: {solution[0]}')
            for s in solution[1:]:
                print(f'\t  {s}')
        else:
            print('Solution:  ' + solution)
        print(f'Move Count: {move_count}')
    
    if plot:
        plt.ion()
        plt_cube = new_cube()
        for move in scramble_sequence:
            show_cube(plt_cube, both_sides=both_sides, speed=.05, title='Scrambling...')
            apply(move, plt_cube)
        show_cube(plt_cube, both_sides=both_sides, speed=.05, title='Ready')
        plt.pause(2)
        for i, move in enumerate(solution):
            if i % skip == 0:
                show_cube(plt_cube, both_sides=both_sides, speed=.005, title='Solving...')
            apply(move, plt_cube)
        show_cube(plt_cube, both_sides=both_sides, title='Solved!')
        plt.ioff()
        plt.show()
    return duration, move_count, solved


def performance_test(n, m, details=True, full_data=False):
    '''Analyzes the performance of a solution method over a specified number of solves'''
    # Initialize variables
    time = []
    moves = []
    solved = 0
    print()

    # Record the performance of n solves
    start = t.time()
    for i in range(n):
        print(f'\rCubes Solved: {solved}', end='', flush=True)
        a, b, s= one_solve(m, details=False)
        time.append(a)
        moves.append(b)
        solved += s
    end = t.time() - start
    print(f'\rCubes Solved: {n}')

    # Process trial data
    fast_time = 1000*min(time)
    avg_time = 1000*sum(time)/n
    slow_time = 1000*max(time)
    least_moves = min(moves)
    avg_moves = sum(moves)/n
    most_moves = max(moves)
    
    # Print performance details
    if details:
        print(f"Performance of Beginner's Method over {n} solves:\n")
        print(f'Solved {100*solved/n}% of scrambles in {end:.2f} seconds')
        print(f'Average solve time: {avg_time:.2f} milliseconds')
        print(f'Average move count: {avg_moves:.1f}\n')

    # Return data
    if full_data:
        return fast_time, avg_time, slow_time, least_moves, avg_moves, most_moves
    else:
        return avg_time, avg_moves


def scramble_test(n, lim=30):
    '''Computes statistics on time and move count over many trials when scramble length is varied'''
    lengths = np.concatenate((np.arange(20), np.arange(20,lim+1,5)))
    fast_times = []
    avg_times = []
    slow_times = []
    least_moves = []
    avg_moves = []
    most_moves = []

    # Performance tests for different scramble lengths
    for m in lengths:
        ft, at, st, lm, am, mm = performance_test(n, m, details=False, full_data=True)
        fast_times.append(ft)
        avg_times.append(at)
        slow_times.append(st)
        least_moves.append(lm)
        avg_moves.append(am)
        most_moves.append(mm)
    
    # Plot times
    plt.figure(figsize=(14,7))
    plt.subplot(121)
    plt.plot(lengths, fast_times, label='Fastest', linestyle='--', color='green')
    plt.plot(lengths, slow_times, label='Slowest', linestyle='--', color='red')
    plt.plot(lengths, avg_times, label='Average', color='black')
    plt.fill_between(lengths, avg_times, fast_times, alpha=.1, color='green')
    plt.fill_between(lengths, slow_times, avg_times, alpha=.1, color='red')
    plt.title('Scramble Length vs. Solve Time')
    plt.xlabel('Scramble Length')
    plt.ylabel('Time (milliseconds)')
    plt.xlim(0,lim)
    plt.ylim(0,None)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    plt.legend()

    # Plot move counts
    plt.subplot(122)
    plt.plot(lengths, least_moves, label='Least', linestyle='--', color='green')
    plt.plot(lengths, most_moves, label='Most', linestyle='--', color='red')
    plt.plot(lengths, avg_moves, label='Average', color='black')
    plt.fill_between(lengths, avg_moves, least_moves, alpha=.1, color='green')
    plt.fill_between(lengths, most_moves, avg_moves, alpha=.1, color='red')
    plt.title('Scramble Length vs. Move Count')
    plt.xlabel('Scramble Length')
    plt.ylabel('Moves')
    plt.xlim(0,lim)
    plt.ylim(0,None)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


# def user_solve():
#     '''Allow the user to input values of their cube'''
#     data = []
#     print('\nInstructions:')
#     print('Enter the value for each subface of the cube one face at a time')
#     print("\n\t'g' for green\t'r' for red\n\t'b' for blue\t'o' for orange\n\t'w' for white\t'y' for yellow")
#     print('Make sure to orient the cube as instructed')
#     print('Backspace to correct errors')
#     print('Confirm entries after filling each face')
#     for face in ['GREEN', 'RED', 'BLUE', 'ORANGE']:
#         print(f'\nEnter values for the {face} side')
#         print(f'Make sure the {face} center is facing you and the white center is on the top')
#         print(f"Your middle entry should be '{face[0].lower()}'")
#         data.append(curses.wrapper(side_data))
#     print()
#     print(data)



one_solve(plot=True, skip=2, both_sides=False)

# performance_test(5000,30, details=True, full_data=False)

# scramble_test(100)
    
# user_solve()

