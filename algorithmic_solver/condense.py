import re
import string


def replace_three_same_with_opposite_case(text):
    return re.sub(r'([a-zA-Z])\1{2}', lambda m: m.group(1).lower() if m.group(1).isupper() else m.group(1).upper(), text)


def remove_uppercase_lowercase_pairs(text):
    pattern = '|'.join([f'{c}{c.lower()}|{c.lower()}{c}' for c in string.ascii_uppercase])
    return re.sub(pattern, '', text)


def replace_two_lowercase_with_two_uppercase(text):
    return re.sub(r'([a-z])\1', lambda m: m.group(1).upper() * 2, text)


def simplify_pair_sequences(text, primary, secondary):
    """
    Simplify sequences like (primary...primary opposite case) or (primary opposite case...primary)
    with (any number of secondary or its opposite case) in between
    to just (any number of secondary or its opposite case), ensuring at least one character in between.

    Parameters:
        text (str): The input string.
        primary (str): The primary letter (e.g., F, R, U).
        secondary (str): The independent letter (e.g., B, L, D).

    Returns:
        str: The simplified string.
    """
    pattern = fr'({primary}([{secondary}{secondary.lower()}]+){primary.lower()}|{primary.lower()}([{secondary}{secondary.lower()}]+){primary})'
    simplified = re.sub(pattern, lambda m: m.group(2) or m.group(3), text)
    return simplified


def simplify_asymmetric_sequences(text, primary, secondary):
    """
    Simplify sequences like (primary primary or primary lower)(any number of secondary or its opposite case)
    (primary primary or primary lower) to (primary opposite case)(any number of secondary or its opposite case),
    ensuring:
    - Same case on either side.
    - One side has two letters, the other has one.
    - At least one intermediate character.

    Parameters:
        text (str): The input string.
        primary (str): The primary letter (e.g., F, R, U).
        secondary (str): The independent letter (e.g., B, L, D).

    Returns:
        str: The simplified string.
    """
    # Regex pattern to enforce one side has 2 letters, the other has 1, with at least one valid intermediate
    pattern = (
        fr'({primary}{primary})([{secondary}{secondary.lower()}]+?)({primary})'  # Uppercase: FF on one side, F on the other
        fr'|'
        fr'({primary})([{secondary}{secondary.lower()}]+?)({primary}{primary})'  # Uppercase: F on one side, FF on the other
        fr'|'
        fr'({primary.lower()}{primary.lower()})([{secondary}{secondary.lower()}]+?)({primary.lower()})'  # Lowercase: ff on one side, f on the other
        fr'|'
        fr'({primary.lower()})([{secondary}{secondary.lower()}]+?)({primary.lower()}{primary.lower()})'  # Lowercase: f on one side, ff on the other
    )

    def replacement(match):
        # Extract groups for uppercase and lowercase cases
        left, middle, right = (
            match.group(1) or match.group(4) or match.group(7) or match.group(10),  # Left side
            match.group(2) or match.group(5) or match.group(8) or match.group(11),  # Middle sequence
            match.group(3) or match.group(6) or match.group(9) or match.group(12)  # Right side
        )
        # Determine replacement based on case of the left group
        if left.isupper():
            return primary.lower() + middle  # Replace with lowercase
        else:
            return primary.upper() + middle  # Replace with uppercase
    
    # Apply the regex with the replacement function
    simplified = re.sub(pattern, replacement, text)
    return simplified



def condense(solution):
    move_pairs = [
        ('F', 'B'),  # F and B are independent
        ('R', 'L'),  # R and L are independent
        ('U', 'D')   # U and D are independent
    ]
    
    while True:
        old_solution = solution
        solution = replace_three_same_with_opposite_case(solution) #-6 moves on average
        solution = remove_uppercase_lowercase_pairs(solution) #-4 moves on average
        solution = replace_two_lowercase_with_two_uppercase(solution)
        for primary, secondary in move_pairs: #-.6 moves on average
            solution = simplify_pair_sequences(solution, primary, secondary)
            solution = simplify_asymmetric_sequences(solution, primary, secondary)
        if solution == old_solution:
            break
    return solution