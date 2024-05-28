.PHONY: shell
shell:
	nix develop
	
.PHONY: build
start: build_block

.PHONY: start
start: start_block

.PHONY: build_block
build_block: block_automaton_c.c
	gcc -O3 -fPIC -shared -o block_automaton_c.so block_automaton_c.c

.PHONY: start_block
start_block: build_block
	python3 block_main.py

.PHONY: start_diff
start_diff:
	python3 diff_main.py

.PHONY: start_wind
start_wind:
	python3 wind_main.py


.PHONY: format
format:
	black *.py
	nix fmt flake.nix
