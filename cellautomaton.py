import matplotlib.pyplot as plt

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
                neighbourhood = [data[idx-1], data[idx], data[idx+1]]
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

if __name__ == '__main__':
    """Run as a script."""

    n_generations = 10

    rule = Rule1()

    # select initial state
    state = State1D(5*[Cell.DEAD] + [Cell.ALIVE] + 5*[Cell.DEAD])

    # Compute n_generations generations
    generations = [state]
    for idx in range(n_generations-1):
        state = rule.apply(state)
        generations.append(state)

    print("Hello automaton.")

    print("History of " + str(rule.NAME) + ": ")
    print(stringify_history(generations))
