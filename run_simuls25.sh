#!/bin/zsh

J_values=(50 100)
M_values=(1 2 3)

# Loop over all combinations
for j in ${J_values[@]}; do
  for m in ${M_values[@]}; do
    uv run simuls_misspecif/simuls_driver.py -J$j -M$m --bounds
  done
done
