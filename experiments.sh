#!/bin/bash

# Define lists of parameter values
paths_list=("8_layer_8_head_corruption_quadratic_with_imbalance/checkpoint_40000.pth")
#paths_list=("checkpoint_10000.pth")
#timesteps_list=(20 30 50 60)
timesteps_list=(20 100 200 300 400 600 800 1000 1200 1400 1600 1999)

# Create output directory
mkdir -p outputs

# Clear output file once at start
RESULTS_FILE="outputs/all_results_8_layers_2000_timesteps_imbalance.txt"
echo "" > "$RESULTS_FILE"

num_heads=8
num_layers=8

# Loop over all combinations
for path in "${paths_list[@]}"; do
  for timestep in "${timesteps_list[@]}"; do
      
    echo "Running with path=$path, timestep=$timestep"
    
    # Run generate.py
    python generate.py --path_checkpoint "checkpoints/$path" --timesteps "$timestep" --num_heads "$num_heads" --num_layers "$num_layers"
    
    # Run validate.py and append output
    {
    echo "=== PARAMS: checkpoint=$path timesteps=$timestep ==="
    python validate.py 
    echo ""
    } >> "$RESULTS_FILE"
    
    echo "Appended results to $RESULTS_FILE"
      
  done
done
