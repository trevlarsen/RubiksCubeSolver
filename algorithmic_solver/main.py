import numpy as np
import time as t
import textwrap
from matplotlib import pyplot as plt
from time import perf_counter

from primary import apply, scramble, new_cube, is_solved
from sequences import valid_solution
from condense import condense
from algorithms2 import make_cross, bottom_corners, middle_edges, orient_top_edges, permute_top_edges, orient_top_corners, permute_top_corners
from show_cube import show_cube

import sys
import os

# Add the parent directory of 'classification_solver' to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Now you can import from classification_solver
from classification_solver.larger_model import solve_cube_with_model
from classification_solver.classifier import Cube


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
    if is_solved(cube) and valid_solution(scramble_sequence, solution):
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


# one_solve(plot=True, skip=2, both_sides=False)

def convert_scramble(move_string):
    moves = []
    for char in move_string:
        if char.islower():
            moves.append(char.upper() + "'")
        else:
            moves.append(char)
    return moves


def revert_solution(move_list):
    move_string = ""
    for move in move_list:
        if move.endswith("2"):  # Handle moves like D2
            move_string += move[0] * 2
        elif "'" in move:  # Handle primed moves
            move_string += move[0].lower()
        else:  # Handle regular moves
            move_string += move
    return move_string

def display_solution(solver_name, scramble, solution, both_sides, skip, speed):
    # Display initial cube
        plt_cube = new_cube()
        show_cube(plt_cube, both_sides=both_sides, title=solver_name + ' Solver')
        plt.pause(3)

        # Display scramble
        for move in scramble:
            show_cube(plt_cube, both_sides=both_sides, speed=.03, title='Scrambling...')
            apply(move, plt_cube)
        show_cube(plt_cube, both_sides=both_sides, speed=.05, title='Ready')
        plt.pause(3)
        
        # Display solution
        for i, move in enumerate(solution):
            if i % skip == 0:
                show_cube(plt_cube, both_sides=both_sides, speed=speed, title='Solving...')
            apply(move, plt_cube)
        show_cube(plt_cube, both_sides=both_sides, title=f'Solved in {len(solution)} moves!')
        plt.pause(3)


def compare_solvers(plot=True, both_sides=False, skip=1):
    class_time = 0.0
    while True:
        alg_cube = new_cube()
        alg_scramble = scramble(alg_cube, n=13)
        start = perf_counter()
        alg_solution = condense(solve(alg_cube))
        alg_time = perf_counter() - start
        alg_solution_length = len(alg_solution)

        class_scramble = convert_scramble(alg_scramble)
        os.chdir("..")
        os.chdir("classification_solver")
        start = perf_counter()
        success, class_solution = solve_cube_with_model(class_scramble, model_path='larger_imitation_model_best.pth')
        class_time += perf_counter() - start
        os.chdir("..")
        os.chdir("algorithmic_solver")
        class_solution = condense(revert_solution(class_solution))
        class_solution_length = len(class_solution)
        if not success:
            continue
        break
    if plot:
        plt.ion()
        display_solution('Algorithmic', alg_scramble, alg_solution, both_sides, skip, speed=.005)
        display_solution('Classification', alg_scramble, class_solution, both_sides, skip, speed=.35)
        plt.ioff()
        plt.show()
    
    print()
    print("="*40)
    print(f"Scramble: {alg_scramble}")
    print("="*40)
    print()

    print("Algorithmic Solver Results")
    print("-"*40)
    print(f"Solution:    {alg_solution}")
    print(f"Move Count:  {alg_solution_length}")
    print(f"Solve Time:  {alg_time:.6f} seconds")
    print()

    print("Classification Solver Results")
    print("-"*40)
    print(f"Solution:    {class_solution}")
    print(f"Move Count:  {class_solution_length}")
    print(f"Solve Time:  {class_time:.6f} seconds")
    print("="*40)
    print()

if __name__ == "__main__":
    compare_solvers()

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
