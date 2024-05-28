def apply_rule(state, n_columns):
    """Given state, compute next one."""

    n_rows = len(state) / n_columns

    # copy previous state as the base
    new_state = state.copy()

    i = 0
    while i < n_rows:
        j = 0
        while j < n_columns:

            # current cell idx and value
            idx = n_columns * i + j
            curr = state[idx]

            # top neighbour
            if i > 0:
                tn_idx = n_columns * (i - 1) + j
            else:
                tn_idx = idx
            tn = state[tn_idx]

            # bottom neighbour
            if i < n_rows - 1:
                bn_idx = n_columns * (i + 1) + j
            else:
                bn_idx = idx
            bn = state[bn_idx]

            # left neighbour
            if j > 0:
                ln_idx = n_columns * i + (j - 1)
            else:
                ln_idx = idx
            ln = state[ln_idx]

            # right neighbour
            if j < n_columns - 1:
                rn_idx = n_columns * i + (j + 1)
            else:
                rn_idx = idx
            rn = state[rn_idx]

            # neighbourhood size
            n = 5

            # update state by a neighbourhood average (convolution)
            new_state[idx] = (tn + bn + ln + rn + curr) / n

            j += 1
        i += 1

    # renormalize to not lose or increase energy
    previous_sum = sum(state)
    new_sum = sum(new_state)
    for idx in range(len(new_state)):
        new_state[idx] = new_state[idx] * (previous_sum / new_sum)

    return new_state
