import numpy as np
from matplotlib import pyplot as plt
import time as t

from main import solve, one_solve



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



# performance_test(5000,30, details=True, full_data=False)

# scramble_test(100)