import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as clt
from matplotlib.animation import FuncAnimation

import numpy as np
import random
import time

from block_automaton_py import apply_rule

import ctypes

# prepare c-side `apply_rule` function for use
c_automaton = ctypes.CDLL("./simd_block/block_automaton_c.so")
c_automaton.apply_rule.restype = ctypes.c_int
c_automaton.apply_rule.argtypes = (
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
)


def update(state, idx):
    """Calls c-side function to update state"""

    # to prepare for c, convert
    # numpy array to string
    n_columns = state.shape[1]
    flattened = "".join(state.flatten())

    # allocate memory for out variable in python side so that it gets
    # automatically gc'd by python
    out = ctypes.create_string_buffer(len(flattened) + 1)

    # call the C function
    ret = c_automaton.apply_rule(flattened.encode("utf-8"), n_columns, idx, out)
    if ret != 0:
        raise Exception("Something went wrong in the c side")

    # return a nice little numpy array
    return np.reshape(list(out.value.decode("utf-8")), state.shape)


def update_py(state, idx):
    """Calls python-side function to update state"""

    return np.reshape(
        apply_rule(list(state.flatten()), state.shape[1], idx), state.shape
    )


if __name__ == "__main__":
    """Run as a script."""

    # Get a random initial pattern
    width = 100
    height = 80
    initial_pattern = np.empty((height, width), dtype=str)
    for i in range(height):
        for j in range(width):
            if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                initial_pattern[i, j] = "w"
                continue
            if random.random() > 0.1:
                initial_pattern[i, j] = "o"
            else:
                initial_pattern[i, j] = "x"
    state = initial_pattern

    n_rows = state.shape[0]
    n_columns = state.shape[1]

    # Create a tickless figure
    fig, ax = plt.subplots()
    ax.axis("off")

    # Prepare the permanent plot properties
    ax.set_xlim([0 - 0.5, n_columns + 0.5])
    ax.set_ylim([0 - 0.5, n_rows + 0.5])

    # create a initial configuration of cells in a PatchCollection for superfast plotting
    paths = []
    for column_idx in range(n_columns):
        for row_idx in range(n_rows):
            rect = patches.Rectangle(
                (column_idx, n_rows - row_idx - 1),
                1,
                1,
            )
            paths.append(rect)
    collection = clt.PatchCollection(
        paths, edgecolors="green", facecolors=["white"], linewidths=0.01
    )
    ax.add_collection(collection)

    def animation_func(idx):
        """Helper to update a single frame."""

        # time it
        beginning = time.time()

        # Get the global state
        global state

        # And update it via the automaton rule.

        # c version
        # state = update(state, idx)

        # # python version a bit slower
        state = update_py(state, idx)

        middlepoint = time.time()

        # Figure out the color of each cell
        colors = []
        for column_idx in range(n_columns):
            for row_idx in range(n_rows):
                cell = state[row_idx, column_idx]
                if cell == "x":
                    facecolor = "black"
                elif cell == "o":
                    facecolor = "white"
                else:
                    facecolor = "grey"
                colors.append(facecolor)
        # And only update them
        collection.set_facecolors(colors)

        # timing information
        times = [middlepoint - beginning, time.time() - middlepoint]
        print(f"Computation: {times[0]}, plotting: {times[1]}")

    # Let matplotlib do the real work
    ani = FuncAnimation(fig, animation_func, frames=10000, interval=80, blit=False)
    plt.show()
