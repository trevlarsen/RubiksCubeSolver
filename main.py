import numpy as np
import time as t
import textwrap
from matplotlib import pyplot as plt

from primary import apply, scramble, new_cube, is_solved
from sequences import condense
from algorithms2 import make_cross, bottom_corners, middle_edges, orient_top_edges, permute_top_edges, orient_top_corners, permute_top_corners
from show_cube import show_cube


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
    
    # 3d plotting
    if plot:
        # Display initial cube
        plt.ion()
        plt_cube = new_cube()
        show_cube(plt_cube, both_sides=both_sides, title='Initializing...')
        plt.pause(1.5)

        # Display scramble
        for move in scramble_sequence:
            show_cube(plt_cube, both_sides=both_sides, speed=.03, title='Scrambling...')
            apply(move, plt_cube)
        show_cube(plt_cube, both_sides=both_sides, speed=.05, title='Ready')
        plt.pause(3)
        
        # Display solution
        for i, move in enumerate(solution):
            if i % skip == 0:
                show_cube(plt_cube, both_sides=both_sides, speed=.005, title='Solving...')
            apply(move, plt_cube)
        show_cube(plt_cube, both_sides=both_sides, title='Solved!')
        plt.ioff()
        plt.show()
    
    return duration, move_count, solved


# Function to display the main menu using Matplotlib
# def display_main_menu():
#     fig, ax = plt.subplots()
#     ax.text(0.5, 0.9, "Rubik's Cube Solver", transform=ax.transAxes,
#             ha='center', fontsize=20, bbox={'facecolor': 'lightblue', 'alpha': 0.5})
#     ax.text(0.5, 0.6, "Main Menu", transform=ax.transAxes,
#             ha='center', fontsize=16, bbox={'facecolor': 'lightgray', 'alpha': 0.5})
#     ax.text(0.5, 0.4, "1. View Instructions", transform=ax.transAxes,
#             ha='center', fontsize=14)
#     ax.text(0.5, 0.3, "2. Begin Cube Solving", transform=ax.transAxes,
#             ha='center', fontsize=14)
#     ax.text(0.5, 0.2, "3. Exit", transform=ax.transAxes,
#             ha='center', fontsize=14)
#     ax.axis('off')
#     plt.draw()


# Function to display instructions using Matplotlib
# def display_instructions():
#     fig, ax = plt.subplots()
#     ax.text(0.5, 0.5, "Rubik's Cube Instructions", transform=ax.transAxes,
#             ha='center', va='center', fontsize=20, bbox={'facecolor': 'lightblue', 'alpha': 0.5})
#     ax.axis('off')
#     plt.draw()


# def interactive():
#     display_main_menu()
#     while True:
#         plt.ion()
#         choice = input("Enter your choice (1-3): ")
#         if choice == '1':
#             display_instructions()
#         elif choice == '2':
#             one_solve(plot=True, skip=2, both_sides=False)
#         elif choice == '3':
#             print("Exiting program.")
#             plt.ioff()
#             break
#         else:
#             print("Invalid choice. Please enter a number from 1 to 3.")



# interactive()

# test
one_solve(plot=True, skip=2, both_sides=False)


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
