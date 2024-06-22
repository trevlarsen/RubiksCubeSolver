import numpy as np
from matplotlib import pyplot as plt


UP = (0,0,1)
FRONT = (1,0,0)
RIGHT = (0,1,0)
LEFT = (0,-1,0)
BACK = (-1,0,0)
DOWN = (0,0,-1)

WHITE = (0.95,0.95,0.95)
YELLOW = (1.0,1.0,0.1)
RED = (0.9,0.1,0.1)
GREEN = (0.3,0.9,0.1)
BLUE = (0.2,0.1,0.8)
ORANGE = (242/255,177/255,0.1)

CELL_LOCS = [
    None,
    ((1,0,-1),DOWN),
    ((0,1,-1),DOWN),
    ((-1,0,-1),DOWN),
    ((0,-1,-1),DOWN),
    ((1,-1,-1),DOWN), #
    ((1,1,-1),DOWN),
    ((-1,1,-1),DOWN),
    ((-1,-1,-1),DOWN),
    ((1,1,0),FRONT),
    ((-1,1,0),RIGHT), #
    ((-1,-1,0),BACK),
    ((1,-1,0),LEFT),
    ((1,0,1),FRONT),
    ((0,1,1),RIGHT),
    ((-1,0,1),BACK), #
    ((0,-1,1),LEFT),
    ((1,1,1),FRONT),
    ((-1,1,1),RIGHT),
    ((-1,-1,1),BACK),
    ((1,-1,1),LEFT), #
    ((1,-1,1),FRONT),
    ((1,-1,0),FRONT),
    ((1,-1,-1),FRONT),
    ((1,0,-1),FRONT),
    ((1,1,-1),FRONT), #
    ((1,1,1),RIGHT),
    ((1,1,0),RIGHT),
    ((1,1,-1),RIGHT),
    ((0,1,-1),RIGHT),
    ((-1,1,-1),RIGHT), #
    ((-1,1,1),BACK),
    ((-1,1,0),BACK),
    ((-1,1,-1),BACK),
    ((-1,0,-1),BACK),
    ((-1,-1,-1),BACK), #
    ((-1,-1,1),LEFT),
    ((-1,-1,0),LEFT),
    ((-1,-1,-1),LEFT),
    ((0,-1,-1),LEFT),
    ((1,-1,-1),LEFT), #
    ((-1,-1,1),UP),
    ((-1,0,1),UP),
    ((-1,1,1),UP),
    ((0,-1,1),UP),
    ((0,1,1),UP), #
    ((1,-1,1),UP),
    ((1,0,1),UP),
    ((1,1,1),UP),
]

def get_perp_dirs(face):
    if face == UP:
        return RIGHT, FRONT
    elif face == FRONT:
        return RIGHT, DOWN
    elif face == RIGHT:
        return BACK, DOWN
    elif face == LEFT:
        return UP, FRONT
    elif face == BACK:
        return RIGHT, UP
    elif face == DOWN:
        return RIGHT, BACK

def get_face_color(face):
    if face == UP:
        return WHITE
    elif face == FRONT:
        return GREEN
    elif face == RIGHT:
        return RED
    elif face == LEFT:
        return ORANGE
    elif face == BACK:
        return BLUE
    elif face == DOWN:
        return YELLOW

def _draw_square(ax, x0, face, s, color):
        dx, dy = get_perp_dirs(face)
        x0, dx, dy = np.array(x0), np.array(dx), np.array(dy)
        hs = 0.5 * s
        
        mesh = np.array([
            [
                x0 - hs * dx - hs * dy,
                x0 - hs * dx + hs * dy,
            ],
            [
                x0 + hs * dx - hs * dy,
                x0 + hs * dx + hs * dy,
            ],
        ])[::-1].transpose(2, 0, 1)
        
        ax.plot_surface(*mesh, color=color, shade=False)

def _render_cube(ax, cube):
    ax.clear()  # Clear the previous plot instead of clearing the entire figure
    # Render cube base and center faces
    for d in [UP, FRONT, RIGHT, LEFT, BACK, DOWN]:
        darr = np.array(d)
        dx, dy = get_perp_dirs(d)
        dx, dy = np.array(dx), np.array(dy)
        for i in range(-1, 2):
            for j in range(-1, 2):
                _draw_square(ax, darr * 1.5 + i * dx + j * dy, d, 1, 'k')
        
        center_color = get_face_color(d)
        _draw_square(ax, darr * 1.65, d, 0.9, center_color)
        
    # Render side faces
    for i, cell in enumerate(cube[1:], start=1):
        pos, facing = CELL_LOCS[i]
        color = get_face_color(CELL_LOCS[cell][1])
        act_pos = np.array(pos) + 0.65 * np.array(facing)
        _draw_square(ax, act_pos, facing, 0.9, color)
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])
    plt.draw()


def show_cube(cube, both_sides=True, speed=.1):
    plt.clf()
    if both_sides:
        ax = plt.subplot(1,2,1, projection='3d')
    else:
        ax = plt.subplot(1,1,1, projection='3d')
    _render_cube(ax, cube)
    ax.axis('off')
    ax.axis('equal')
    if both_sides:
        ax = plt.subplot(1,2,2, projection='3d')
        _render_cube(ax, cube)
        ax.view_init(azim=135, elev=-30)
        ax.axis('off')
        ax.axis('equal')
    plt.draw()
    plt.pause(speed)
