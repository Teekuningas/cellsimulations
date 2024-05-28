import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

import numpy as np

from enum import Enum


class Cell(Enum):
    """Cells can be of two types."""

    ALIVE = 1
    DEAD = 0

    def __add__(self, other):
        if isinstance(other, Cell):
            return self.value + other.value
        else:
            return self.value + other

    def __radd__(self, other):
        if isinstance(other, Cell):
            return self.value + other.value
        else:
            return self.value + other


class Rule2D:
    """Base class for 2D rules."""

    NAME = ""

    def apply_cell(self, neighbourhood):
        """Apply rule to a single neighbourhood."""
        raise Exception("Not implemented.")

    def apply(self, state):
        """Apply rule to data."""
        data = state.data
        n_rows = data.shape[0]
        n_columns = data.shape[1]

        new_data = np.empty((n_rows, n_columns), dtype=Cell)
        for row_idx in range(n_rows):
            for column_idx in range(n_columns):
                # find the neighbourhood
                if column_idx < 1:
                    lc = -1
                    mc = 0
                    rc = 1
                elif column_idx == n_columns - 1:
                    lc = -2
                    mc = -1
                    rc = 0
                else:
                    lc = column_idx - 1
                    mc = column_idx
                    rc = column_idx + 1
                if row_idx < 1:
                    tr = -1
                    mr = 0
                    br = 1
                elif row_idx == n_rows - 1:
                    tr = -2
                    mr = -1
                    br = 0
                else:
                    tr = row_idx - 1
                    mr = row_idx
                    br = row_idx + 1
                neighbourhood = np.array(
                    [
                        [data[tr, lc], data[tr, mc], data[tr, rc]],
                        [data[mr, lc], data[mr, mc], data[mr, rc]],
                        [data[br, lc], data[br, mc], data[br, rc]],
                    ]
                )
                new_data[row_idx, column_idx] = self.apply_cell(neighbourhood)

        return State2D(new_data)


class State2D:
    """Stores a state."""

    def __init__(self, data):
        self.data = data


class RuleGameOfLife(Rule2D):
    """Implementation of Rule 1."""

    NAME = "Rule Game Of Life"

    def apply_cell(self, neighbourhood):
        """Apply rule to a single neighbourhood."""

        # Any live cell with fewer than two live neighbours dies, as if by underpopulation
        if neighbourhood[1, 1] == Cell.ALIVE:
            count = np.sum(neighbourhood) - 1
            if count < 2:
                return Cell.DEAD

        # Any live cell with two or three live neighbours lives on to the next generation
        if neighbourhood[1, 1] == Cell.ALIVE:
            count = np.sum(neighbourhood) - 1
            if count >= 2 and count < 4:
                return Cell.ALIVE

        # Any live cell with two or three live neighbours lives on to the next generation
        if neighbourhood[1, 1] == Cell.ALIVE:
            count = np.sum(neighbourhood) - 1
            if count > 3:
                return Cell.DEAD

        # Any dead cell with exactly three live neighbours becomes a live cell,
        # as if by reproduction
        if neighbourhood[1, 1] == Cell.DEAD:
            count = np.sum(neighbourhood)
            if count == 3:
                return Cell.ALIVE

        # Otherwise just keep dead:
        return Cell.DEAD


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
                else:
                    facecolor = "white"

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
                else:
                    facecolor = "white"

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

    rule = RuleGameOfLife()

    initial_pattern_flower = np.array(
        [
            [Cell.DEAD, Cell.ALIVE, Cell.DEAD],
            [Cell.ALIVE, Cell.DEAD, Cell.ALIVE],
            [Cell.ALIVE, Cell.DEAD, Cell.ALIVE],
            [Cell.ALIVE, Cell.DEAD, Cell.ALIVE],
            [Cell.DEAD, Cell.ALIVE, Cell.DEAD],
        ]
    )

    initial_pattern_glider = np.array(
        [
            [Cell.DEAD, Cell.ALIVE, Cell.DEAD],
            [Cell.DEAD, Cell.DEAD, Cell.ALIVE],
            [Cell.ALIVE, Cell.ALIVE, Cell.ALIVE],
        ]
    )

    def _pad_to_shape(arr, out_shape):
        """Helper to pad a pattern to a specific size with pattern
        in the center."""
        m, n = out_shape
        x, y = arr.shape
        out = np.zeros(out_shape, dtype=Cell)
        out.fill(Cell.DEAD)
        mx, my = (m - x) // 2, (n - y) // 2
        out[mx : mx + x, my : my + y] = arr
        return out

    # n_generations = 20
    # n_rows = 19
    # n_columns = 17
    # state = State2D(_pad_to_shape(initial_pattern_flower, (n_columns, n_rows)))

    n_generations = 500
    n_rows = 11
    n_columns = 11
    state = State2D(_pad_to_shape(initial_pattern_glider, (n_columns, n_rows)))

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
