import sys

def get_side_inputs(side_name, top_name):
    inputs = []
    print(f'\nEnter values for the {side_name} side')
    print(f'Make sure the {side_name} center is facing you and the {top_name} center is on the top')
    print(f"Your middle entry should be '{side_name[0].lower()}'")
    print()
    while len(inputs) < 9:
        display = "".join(inputs)
        rows = len(display)//9
        for i in range(0, len(display), 9):
            line = display[i:i+9]
            if line:  # Ensure line is not empty
                print(line, flush=True)
        inputs.append(input()+'  ')
        # ANSI escape sequence to move cursor up
        sys.stdout.write(f"\033[F"*(rows+1))
        # # ANSI escape sequence to clear from cursor to end of screen
        # sys.stdout.write("\033[J")
        # sys.stdout.flush()

# print(get_side_inputs('green', 'white'))

    