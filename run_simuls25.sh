#!/bin/zsh

J_values=(5)
M_values=(3 4)

# Loop over all combinations
for j in ${J_values[@]}; do
  for m in ${M_values[@]}; do
    uv run simuls_misspecif/simuls_driver.py -J$j -M$m
  done
done
