#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int match_pattern(char tl, char tr, char bl, char br, char* res) {
        // normal rules
	if (tl == 'x' && tr == 'o' && bl == 'o' && br == 'o') {
		res[0] = 'o';
		res[1] = 'o';
		res[2] = 'o';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'o' && tr == 'o' && bl == 'o' && br == 'x') {
		res[0] = 'x';
		res[1] = 'o';
		res[2] = 'o';
		res[3] = 'o';
		return 0;
	}
        // //
	if (tl == 'o' && tr == 'o' && bl == 'x' && br == 'o') {
		res[0] = 'o';
		res[1] = 'x';
		res[2] = 'o';
		res[3] = 'o';
		return 0;
	}
	if (tl == 'o' && tr == 'x' && bl == 'o' && br == 'o') {
		res[0] = 'o';
		res[1] = 'o';
		res[2] = 'x';
		res[3] = 'o';
		return 0;
	}
	// //
	if (tl == 'x' && tr == 'x' && bl == 'o' && br == 'o') {
		res[0] = 'o';
		res[1] = 'o';
		res[2] = 'x';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'o' && tr == 'o' && bl == 'x' && br == 'x') {
		res[0] = 'x';
		res[1] = 'x';
		res[2] = 'o';
		res[3] = 'o';
		return 0;
	}
	// //
	if (tl == 'o' && tr == 'x' && bl == 'o' && br == 'x') {
		res[0] = 'x';
		res[1] = 'o';
		res[2] = 'x';
		res[3] = 'o';
		return 0;
	}
	if (tl == 'x' && tr == 'o' && bl == 'x' && br == 'o') {
		res[0] = 'o';
		res[1] = 'x';
		res[2] = 'o';
		res[3] = 'x';
		return 0;
	}
	// //
	if (tl == 'o' && tr == 'x' && bl == 'x' && br == 'o') {
		res[0] = 'x';
		res[1] = 'o';
		res[2] = 'o';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'x' && tr == 'o' && bl == 'o' && br == 'x') {
		res[0] = 'o';
		res[1] = 'x';
		res[2] = 'x';
		res[3] = 'o';
		return 0;
	}
	// //
	if (tl == 'x' && tr == 'x' && bl == 'x' && br == 'o') {
		res[0] = 'o';
		res[1] = 'x';
		res[2] = 'x';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'o' && tr == 'x' && bl == 'x' && br == 'x') {
		res[0] = 'x';
		res[1] = 'x';
		res[2] = 'x';
		res[3] = 'o';
		return 0;
	}
	// //
	if (tl == 'x' && tr == 'o' && bl == 'x' && br == 'x') {
		res[0] = 'x';
		res[1] = 'x';
		res[2] = 'o';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'x' && tr == 'x' && bl == 'o' && br == 'x') {
		res[0] = 'x';
		res[1] = 'o';
		res[2] = 'x';
		res[3] = 'x';
		return 0;
	}
	// wall rules
	if (tl == 'o' && tr == 'x' && bl == 'w' && br == 'w') {
		res[0] = 'x';
		res[1] = 'o';
		res[2] = 'w';
		res[3] = 'w';
		return 0;
	}
	if (tl == 'x' && tr == 'o' && bl == 'w' && br == 'w') {
		res[0] = 'o';
		res[1] = 'x';
		res[2] = 'w';
		res[3] = 'w';
		return 0;
	}
	// //
	if (tl == 'w' && tr == 'x' && bl == 'w' && br == 'o') {
		res[0] = 'w';
		res[1] = 'o';
		res[2] = 'w';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'w' && tr == 'o' && bl == 'w' && br == 'x') {
		res[0] = 'w';
		res[1] = 'x';
		res[2] = 'w';
		res[3] = 'o';
		return 0;
	}
	// //
	if (tl == 'w' && tr == 'w' && bl == 'x' && br == 'o') {
		res[0] = 'w';
		res[1] = 'w';
		res[2] = 'o';
		res[3] = 'x';
		return 0;
	}
	if (tl == 'w' && tr == 'w' && bl == 'o' && br == 'x') {
		res[0] = 'w';
		res[1] = 'w';
		res[2] = 'x';
		res[3] = 'o';
		return 0;
	}
	// //
	if (tl == 'x' && tr == 'w' && bl == 'o' && br == 'w') {
		res[0] = 'o';
		res[1] = 'w';
		res[2] = 'x';
		res[3] = 'w';
		return 0;
	}
	if (tl == 'o' && tr == 'w' && bl == 'x' && br == 'w') {
		res[0] = 'x';
		res[1] = 'w';
		res[2] = 'o';
		res[3] = 'w';
		return 0;
	}

        // otherwise just user the original values
	res[0] = tl;
	res[1] = tr;
	res[2] = bl;
	res[3] = br;
	return 0;
}

int apply_rule(const char *state, const int n_columns, const int idx, char *out) {

	int n_rows = strlen(state) / n_columns;

	if (n_rows % 2 != 0 ) {
		return -1;
	}

	if (n_columns % 2 != 0 ) {
		return -1;
	}

	int start_idx;
	int block_count_columns;
	int block_count_rows;

	if (idx % 2 == 0) {
		start_idx = 0;
		block_count_columns = n_columns / 2;
		block_count_rows = n_rows / 2;
	} else {
		start_idx = 1;
		block_count_columns = n_columns / 2 - 1;
		block_count_rows = n_rows / 2 - 1;
	}

	// Initialize output with the current state
	strcpy(out, state);

	// And then fill in the contents that can change
	for (int i = 0; i < block_count_rows; i++) {
		for (int j = 0; j < block_count_columns; j++) {
			int idx_11_i = i * 2 + start_idx;
			int idx_11_j = j * 2 + start_idx;
			int idx_11 = n_columns * idx_11_i + idx_11_j;
			int idx_21_i = idx_11_i + 1;
			int idx_21_j = idx_11_j;
			int idx_21 = n_columns * idx_21_i + idx_21_j;
			int idx_12_i = idx_11_i;
			int idx_12_j = idx_11_j + 1;
			int idx_12 = n_columns * idx_12_i + idx_12_j;
			int idx_22_i = idx_11_i + 1;
			int idx_22_j = idx_11_j + 1;
			int idx_22 = n_columns * idx_22_i + idx_22_j;

			char tl = state[idx_11];
			char bl = state[idx_21];
			char tr = state[idx_12];
			char br = state[idx_22];

			char match[4];
			match_pattern(tl, tr, bl, br, match);

			out[idx_11] = match[0];
			out[idx_21] = match[2];
			out[idx_12] = match[1];
			out[idx_22] = match[3];
		}
	}

	return 0;
}
