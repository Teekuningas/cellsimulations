import matplotlib.pyplot as plt
import matplotlib.patches as patches

from enum import Enum


class Cell(Enum):
    """Cells can be of two types."""

    ALIVE = 1
    DEAD = 0


class Rule1D:
    """Base class for 1D rules."""

    NAME = ""
    PATTERNS = []

    def apply_cell(self, neighbourhood):
        """Apply rule to single neighbourhood."""
        for (pattern, result) in self.PATTERNS:
            if neighbourhood == pattern:
                return result

        raise Exception("Found unknown pattern.")

    def apply(self, state):
        """Apply rule to data."""
        data = state.data

        new_data = []
        for idx, elem in enumerate(data):
            # Wrap around if at a edge
            if idx == 0:
                neighbourhood = [data[-1], data[0], data[1]]
            elif idx == len(data) - 1:
                neighbourhood = [data[-2], data[-1], data[1]]
            # Otherwise pick the neighbourhood straightforwardly
            else:
                neighbourhood = [data[idx - 1], data[idx], data[idx + 1]]
            new_data.append(self.apply_cell(neighbourhood))

        return State1D(new_data)


class State1D:
    """Stores a state."""

    def __init__(self, data):
        self.data = data


class Rule1(Rule1D):
    """Implementation of Rule 1."""

    NAME = "Rule 1"

    PATTERNS = [
        ([Cell.ALIVE, Cell.ALIVE, Cell.ALIVE], Cell.DEAD),
        ([Cell.ALIVE, Cell.ALIVE, Cell.DEAD], Cell.DEAD),
        ([Cell.ALIVE, Cell.DEAD, Cell.ALIVE], Cell.DEAD),
        ([Cell.ALIVE, Cell.DEAD, Cell.DEAD], Cell.DEAD),
        ([Cell.DEAD, Cell.ALIVE, Cell.ALIVE], Cell.DEAD),
        ([Cell.DEAD, Cell.ALIVE, Cell.DEAD], Cell.DEAD),
        ([Cell.DEAD, Cell.DEAD, Cell.ALIVE], Cell.DEAD),
        ([Cell.DEAD, Cell.DEAD, Cell.DEAD], Cell.ALIVE),
    ]


class Rule18(Rule1D):
    """Implementation of Rule 18."""

    NAME = "Rule 18"

    PATTERNS = [
        ([Cell.ALIVE, Cell.ALIVE, Cell.ALIVE], Cell.DEAD),
        ([Cell.ALIVE, Cell.ALIVE, Cell.DEAD], Cell.DEAD),
        ([Cell.ALIVE, Cell.DEAD, Cell.ALIVE], Cell.DEAD),
        ([Cell.ALIVE, Cell.DEAD, Cell.DEAD], Cell.ALIVE),
        ([Cell.DEAD, Cell.ALIVE, Cell.ALIVE], Cell.DEAD),
        ([Cell.DEAD, Cell.ALIVE, Cell.DEAD], Cell.DEAD),
        ([Cell.DEAD, Cell.DEAD, Cell.ALIVE], Cell.ALIVE),
        ([Cell.DEAD, Cell.DEAD, Cell.DEAD], Cell.DEAD),
    ]


class Rule105(Rule1D):
    """Implementation of Rule 105."""

    NAME = "Rule 105"

    PATTERNS = [
        ([Cell.ALIVE, Cell.ALIVE, Cell.ALIVE], Cell.DEAD),
        ([Cell.ALIVE, Cell.ALIVE, Cell.DEAD], Cell.ALIVE),
        ([Cell.ALIVE, Cell.DEAD, Cell.ALIVE], Cell.ALIVE),
        ([Cell.ALIVE, Cell.DEAD, Cell.DEAD], Cell.DEAD),
        ([Cell.DEAD, Cell.ALIVE, Cell.ALIVE], Cell.ALIVE),
        ([Cell.DEAD, Cell.ALIVE, Cell.DEAD], Cell.DEAD),
        ([Cell.DEAD, Cell.DEAD, Cell.ALIVE], Cell.DEAD),
        ([Cell.DEAD, Cell.DEAD, Cell.DEAD], Cell.ALIVE),
    ]


def stringify_state(state):
    """Makes a string representation of a state."""
    res = ""
    for cell in state.data:
        if cell == Cell.ALIVE:
            res += "[x]"
        else:
            res += "[ ]"
    return res


def stringify_history(generations):
    """Makes a string representation of all generations."""
    res = ""
    for state in generations:
        res += stringify_state(state) + "\n"

    return res


def plot_history_1d(generations, rule):
    """Makes matplotlib plot of all generations."""
    fig, ax = plt.subplots()
    plt.axis("off")
    ax.set_xlim([0 - 0.5, len(generations[0].data) + 0.5])
    ax.set_ylim([0 - 0.5, len(generations) + 0.5 + 4])
    ax.set_title(rule.NAME)

    # plot the rule patterns
    for pattern_idx, pattern in enumerate(rule.PATTERNS):
        # grouping box
        scaling_factor = 32 / (len(generations[0].data))
        group_x = (4 * pattern_idx) / scaling_factor
        group_y = len(generations) + 4 - 3
        group_width = 4 / scaling_factor
        group_height = 3
        rect = patches.Rectangle(
            (group_x, group_y), group_width, group_height, linewidth=1, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        # and the small ones

        facecolor = "black" if pattern[0][0] == Cell.ALIVE else "white"
        rect = patches.Rectangle(
            (group_x + (0.5 / scaling_factor), group_y + 1.5),
            1 / scaling_factor,
            1,
            linewidth=1,
            edgecolor="green",
            facecolor=facecolor,
        )
        ax.add_patch(rect)

        facecolor = "black" if pattern[0][1] == Cell.ALIVE else "white"
        rect = patches.Rectangle(
            (group_x + (1.5 / scaling_factor), group_y + 1.5),
            1 / scaling_factor,
            1,
            linewidth=1,
            edgecolor="green",
            facecolor=facecolor,
        )
        ax.add_patch(rect)

        facecolor = "black" if pattern[0][2] == Cell.ALIVE else "white"
        rect = patches.Rectangle(
            (group_x + (2.5 / scaling_factor), group_y + 1.5),
            1 / scaling_factor,
            1,
            linewidth=1,
            edgecolor="green",
            facecolor=facecolor,
        )
        ax.add_patch(rect)

        facecolor = "black" if pattern[1] == Cell.ALIVE else "white"
        rect = patches.Rectangle(
            (group_x + (1.5 / scaling_factor), group_y + 0.5),
            1 / scaling_factor,
            1,
            linewidth=1,
            edgecolor="green",
            facecolor=facecolor,
        )
        ax.add_patch(rect)

    # plot the generations
    for gen_idx in range(len(generations)):
        for cell_idx, cell in enumerate(generations[gen_idx].data):
            if cell == Cell.ALIVE:
                facecolor = "black"
            else:
                facecolor = "white"

            rect = patches.Rectangle(
                (cell_idx, len(generations) - gen_idx - 1),
                1,
                1,
                linewidth=1,
                edgecolor="green",
                facecolor=facecolor,
            )
            ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    """Run as a script."""

    # # Real automaton
    # n_generations = 50
    # n_cells = 101

    # Exercise
    n_generations = 20
    n_cells = 21

    rule = Rule105()

    # select initial state
    state = State1D(
        int((n_cells - 1) / 2) * [Cell.DEAD]
        + [Cell.ALIVE]
        + int((n_cells - 1) / 2) * [Cell.DEAD]
    )

    # Compute n_generations generations
    generations = [state]

    # # Real automaton
    # for idx in range(n_generations - 1):
    #     state = rule.apply(state)
    #     generations.append(state)

    # Exercise sheet
    for idx in range(n_generations - 1):
        state = State1D(
            n_cells * [Cell.DEAD]
        )
        generations.append(state)


    print("Hello automaton.")

    print("History of " + str(rule.NAME) + ": ")
    plot_history_1d(generations, rule)

