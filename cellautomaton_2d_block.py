import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

import numpy as np

from enum import Enum


class Cell(Enum):
    """Cells can be of two types."""

    ALIVE = 1
    DEAD = 0
    WALL = -1

class Rule2DBlock:
    """Base class for 2D block rules."""

    NAME = ""

    def __init__(self):
        self.counter = 0

    def apply_block(self, block):
        """Apply rule to a single quad."""
        raise Exception("Not implemented.")

    def apply(self, state):
        """Apply rule to data."""

        data = state.data
        n_rows = data.shape[0]
        n_columns = data.shape[1]

        if self.counter % 2 == 0:
            blocks = np.empty((int(n_rows/2), int(n_columns/2)), dtype=np.ndarray)
        else:
            blocks = np.empty((int(n_rows/2) - 1, int(n_columns/2) - 1), dtype=np.ndarray)

        for row_idx in range(n_rows):
            for column_idx in range(n_columns):
                if self.counter % 2 == 0:
                    if row_idx % 2 == 0 and column_idx % 2 == 0:
                        blocks[int(row_idx/2), int(column_idx/2)] = data[row_idx:row_idx+2, column_idx:column_idx+2]
                else:
                    if row_idx % 2 == 1 and row_idx < n_rows - 1 and column_idx % 2 == 1 and column_idx < n_columns - 1:
                        blocks[int((row_idx-1)/2), int((column_idx-1)/2)] = data[row_idx:row_idx+2, column_idx:column_idx+2]

        new_data = np.copy(data)
        for block_row_idx in range(blocks.shape[0]):
            for block_column_idx in range(blocks.shape[1]):
                if self.counter % 2 == 0:
                    new_data[2*block_row_idx:(2*block_row_idx+2), 2*block_column_idx:(2*block_column_idx+2)] = (
                        self.apply_block(blocks[block_row_idx, block_column_idx])
                    )
                else:
                    new_data[(2*block_row_idx+1):(2*block_row_idx+3), (2*block_column_idx+1):(2*block_column_idx+3)] = (
                        self.apply_block(blocks[block_row_idx, block_column_idx])
                    )

        self.counter += 1

        return State2D(new_data)


class State2D:
    """Stores a state."""

    def __init__(self, data):
        if data.shape[0] % 2 != 0:
            raise Exception('Shape[0] should be divisible by 2')

        if data.shape[1] % 2 != 0:
            raise Exception('Shape[1] should be divisible by 2')

        self.data = data


class RuleSimulation(Rule2DBlock):
    """Implementation of Gas simulation."""

    NAME = "Rule Gas"

    RULES = [
        ### collision rules
        (np.array([
            [Cell.ALIVE, Cell.DEAD], 
            [Cell.DEAD,  Cell.DEAD]
        ]),
        np.array([
            [Cell.DEAD, Cell.DEAD], 
            [Cell.DEAD, Cell.ALIVE]
        ])),
        ###
        (np.array([
            [Cell.DEAD,  Cell.DEAD], 
            [Cell.ALIVE, Cell.DEAD]
        ]),
        np.array([
            [Cell.DEAD,  Cell.ALIVE], 
            [Cell.DEAD,  Cell.DEAD]
        ])),
        ###
        (np.array([
            [Cell.DEAD,  Cell.DEAD], 
            [Cell.ALIVE, Cell.ALIVE]
        ]),
        np.array([
            [Cell.ALIVE, Cell.ALIVE], 
            [Cell.DEAD,  Cell.DEAD]
        ])),
        ###
        (np.array([
            [Cell.DEAD, Cell.ALIVE], 
            [Cell.DEAD, Cell.ALIVE]
        ]),
        np.array([
            [Cell.ALIVE, Cell.DEAD], 
            [Cell.ALIVE, Cell.DEAD]
        ])),
        ###
        (np.array([
            [Cell.DEAD,  Cell.ALIVE], 
            [Cell.ALIVE, Cell.DEAD]
        ]),
        np.array([
            [Cell.ALIVE, Cell.DEAD], 
            [Cell.DEAD,  Cell.ALIVE]
        ])),
        ###
        (np.array([
            [Cell.DEAD,  Cell.ALIVE], 
            [Cell.ALIVE, Cell.ALIVE]
        ]),
        np.array([
            [Cell.ALIVE, Cell.ALIVE], 
            [Cell.ALIVE, Cell.DEAD]
        ])),
        ###
        (np.array([
            [Cell.ALIVE, Cell.ALIVE], 
            [Cell.DEAD,  Cell.ALIVE]
        ]),
        np.array([
            [Cell.ALIVE, Cell.DEAD], 
            [Cell.ALIVE, Cell.ALIVE]
        ])),
        ### wall rules
        (np.array([
            [Cell.DEAD, Cell.ALIVE], 
            [Cell.WALL, Cell.WALL]
        ]),
        np.array([
            [Cell.ALIVE, Cell.DEAD], 
            [Cell.WALL,  Cell.WALL]
        ])),
        ###
        (np.array([
            [Cell.WALL, Cell.ALIVE], 
            [Cell.WALL, Cell.DEAD]
        ]),
        np.array([
            [Cell.WALL, Cell.DEAD], 
            [Cell.WALL, Cell.ALIVE]
        ])),
        ###
        (np.array([
            [Cell.WALL,  Cell.WALL], 
            [Cell.ALIVE, Cell.DEAD]
        ]),
        np.array([
            [Cell.WALL, Cell.WALL], 
            [Cell.DEAD, Cell.ALIVE]
        ])),
        ###
        (np.array([
            [Cell.DEAD,  Cell.WALL], 
            [Cell.ALIVE, Cell.WALL]
        ]),
        np.array([
            [Cell.ALIVE, Cell.WALL], 
            [Cell.DEAD,  Cell.WALL]
        ])),
    ]

    def apply_block(self, block):
        """Apply rule to a single block."""
        for rule in self.RULES:
            if np.array_equal(block, rule[0]):
                return rule[1]
            if np.array_equal(block, rule[1]):
                return rule[0]
        return block


def subplots(n_plots, page_size):
    """Return figs and axes."""
    figs, axes = [], []

    n_rows = int(np.floor(np.sqrt(page_size)))
    n_columns = int(np.ceil((page_size / n_rows)))

    n_plots_remaining = n_plots
    while True:
        if n_plots_remaining <= 0:
            break

        if n_plots_remaining >= page_size:
            n_current_plots = page_size
        else:
            n_current_plots = ((n_plots_remaining - 1) % page_size) + 1

        fig, fig_axes = plt.subplots(nrows=n_rows, ncols=n_columns)

        for idx in range(n_rows * n_columns):
            x = idx % n_columns
            y = idx // n_columns
            ax = fig_axes[y, x]
            ax.axis("off")
            if idx < n_current_plots:
                axes.append(ax)
            else:
                pass

        figs.append(fig)

        n_plots_remaining -= page_size

    return figs, axes


def plot_history_2d(generations, rule, page_size):
    """Makes matplotlib plot of all generations."""

    figs, axes = subplots(len(generations), page_size)

    for gen_idx, generation in enumerate(generations):
        ax = axes[gen_idx]

        ax.set_title(f"Generation {gen_idx+1}")

        data = generation.data
        n_rows = data.shape[0]
        n_columns = data.shape[1]

        ax.set_xlim([0 - 0.5, n_columns + 0.5])
        ax.set_ylim([0 - 0.5, n_rows + 0.5])

        for column_idx in range(n_columns):
            for row_idx in range(n_rows):
                cell = data[row_idx, column_idx]
                if cell == Cell.ALIVE:
                    facecolor = "black"
                elif cell == Cell.DEAD:
                    facecolor = "white"
                else:
                    facecolor = "grey"

                rect = patches.Rectangle(
                    (column_idx, n_rows - row_idx - 1),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="green",
                    facecolor=facecolor,
                )
                ax.add_patch(rect)

    for fig in figs:
        fig.suptitle(rule.NAME)
        fig.tight_layout()

    plt.show()


def animate_history_2d(generations, rule):
    """Makes matplotlib animation from generations."""

    fig, ax = plt.subplots()
    ax.axis("off")

    def _animation_frame(generation_idx):
        """Helper to generate a single frame."""

        generation = generations[generation_idx]

        data = generation.data
        n_rows = data.shape[0]
        n_columns = data.shape[1]

        ax.clear()
        ax.axis("off")
        ax.set_xlim([0 - 0.5, n_columns + 0.5])
        ax.set_ylim([0 - 0.5, n_rows + 0.5])
        ax.set_title(rule.NAME)

        for column_idx in range(n_columns):
            for row_idx in range(n_rows):
                cell = data[row_idx, column_idx]
                if cell == Cell.ALIVE:
                    facecolor = "black"
                elif cell == Cell.DEAD:
                    facecolor = "white"
                else:
                    facecolor = "grey"

                rect = patches.Rectangle(
                    (column_idx, n_rows - row_idx - 1),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="green",
                    facecolor=facecolor,
                )
                ax.add_patch(rect)

    ani = FuncAnimation(fig, _animation_frame, frames=(len(generations)))
    plt.show()


if __name__ == "__main__":
    """Run as a script."""

    page_size = 16

    rule = RuleSimulation()

    initial_pattern = np.array(
        [
            [Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL, Cell.WALL,  Cell.WALL,  Cell.WALL, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.ALIVE, Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.ALIVE, Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.DEAD,  Cell.DEAD,  Cell.DEAD, Cell.WALL],
            [Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL,  Cell.WALL, Cell.WALL,  Cell.WALL,  Cell.WALL, Cell.WALL],
        ]
    )

    n_generations = 100
    state = State2D(initial_pattern)

    # Compute n_generations generations
    generations = [state]
    for idx in range(n_generations - 1):
        state = rule.apply(state)
        generations.append(state)

    print("Hello automaton.")

    # print("Plotting history of " + str(rule.NAME))
    # plot_history_2d(generations, rule, page_size)

    print("Animating history of " + str(rule.NAME))
    animate_history_2d(generations, rule)
