def match_pattern(tl, tr, bl, br):
    """Helper to match patterns."""
    ###
    if [tl, tr, bl, br] == ["x", "o", "o", "o"]:
        return ["o", "o", "o", "x"]
    if [tl, tr, bl, br] == ["o", "o", "o", "x"]:
        return ["x", "o", "o", "o"]
    ###
    if [tl, tr, bl, br] == ["o", "o", "x", "o"]:
        return ["o", "x", "o", "o"]
    if [tl, tr, bl, br] == ["o", "x", "o", "o"]:
        return ["o", "o", "x", "o"]
    ###
    if [tl, tr, bl, br] == ["x", "x", "o", "o"]:
        return ["o", "o", "x", "x"]
    if [tl, tr, bl, br] == ["o", "o", "x", "x"]:
        return ["x", "x", "o", "o"]
    ###
    if [tl, tr, bl, br] == ["o", "x", "o", "x"]:
        return ["x", "o", "x", "o"]
    if [tl, tr, bl, br] == ["x", "o", "x", "o"]:
        return ["o", "x", "o", "x"]
    ###
    if [tl, tr, bl, br] == ["o", "x", "x", "o"]:
        return ["x", "o", "o", "x"]
    if [tl, tr, bl, br] == ["x", "o", "o", "x"]:
        return ["o", "x", "x", "o"]
    ###
    if [tl, tr, bl, br] == ["x", "x", "x", "o"]:
        return ["o", "x", "x", "x"]
    if [tl, tr, bl, br] == ["o", "x", "x", "x"]:
        return ["x", "x", "x", "o"]
    ###
    if [tl, tr, bl, br] == ["x", "o", "x", "x"]:
        return ["x", "x", "o", "x"]
    if [tl, tr, bl, br] == ["x", "x", "o", "x"]:
        return ["x", "o", "x", "x"]
    ### wall rules
    if [tl, tr, bl, br] == ["o", "x", "w", "w"]:
        return ["x", "o", "w", "w"]
    if [tl, tr, bl, br] == ["x", "o", "w", "w"]:
        return ["o", "x", "w", "w"]
    ###
    if [tl, tr, bl, br] == ["w", "x", "w", "o"]:
        return ["w", "o", "w", "x"]
    if [tl, tr, bl, br] == ["w", "o", "w", "x"]:
        return ["w", "x", "w", "o"]
    ###
    if [tl, tr, bl, br] == ["w", "w", "x", "o"]:
        return ["w", "w", "o", "x"]
    if [tl, tr, bl, br] == ["w", "w", "o", "x"]:
        return ["w", "w", "x", "o"]
    ###
    if [tl, tr, bl, br] == ["x", "w", "o", "w"]:
        return ["o", "w", "x", "w"]
    if [tl, tr, bl, br] == ["o", "w", "x", "w"]:
        return ["x", "w", "o", "w"]

    return [tl, tr, bl, br]


def apply_rule(state, n_columns, idx):
    """Given state, compute next one."""

    n_rows = len(state) / n_columns

    if n_rows % 2 != 0:
        raise Exception("Height should be divisible by 2")

    if n_columns % 2 != 0:
        raise Exception("Width should be divisible by 2")

    # precompute block division variables
    if idx % 2 == 0:
        start_idx = 0
        block_count_columns = int(n_columns / 2)
        block_count_rows = int(n_rows / 2)
    else:
        start_idx = 1
        block_count_columns = int((n_columns / 2 - 1))
        block_count_rows = int((n_rows / 2 - 1))

    # copy previous state as the base
    new_state = state.copy()

    # then replace the contents that can change
    i = 0
    while i < block_count_rows:
        j = 0
        while j < block_count_columns:
            # implement c-like replace system
            idx_11_i = i * 2 + start_idx
            idx_11_j = j * 2 + start_idx
            idx_11 = n_columns * idx_11_i + idx_11_j
            idx_21_i = idx_11_i + 1
            idx_21_j = idx_11_j
            idx_21 = n_columns * idx_21_i + idx_21_j
            idx_12_i = idx_11_i
            idx_12_j = idx_11_j + 1
            idx_12 = n_columns * idx_12_i + idx_12_j
            idx_22_i = idx_11_i + 1
            idx_22_j = idx_11_j + 1
            idx_22 = n_columns * idx_22_i + idx_22_j
            tl = state[idx_11]
            bl = state[idx_21]
            tr = state[idx_12]
            br = state[idx_22]
            new_tl, new_tr, new_bl, new_br = match_pattern(tl, tr, bl, br)
            new_state[idx_11] = new_tl
            new_state[idx_21] = new_bl
            new_state[idx_12] = new_tr
            new_state[idx_22] = new_br
            j += 1
        i += 1

    return new_state
