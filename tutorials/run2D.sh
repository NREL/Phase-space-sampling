# Run the downsampling script
mpiexec -np 4 python tests/main_from_input.py -i input2D

# Plot the loss
python postProcess/plotLoss.py -i input2D

# Plot the sampling results
python postProcess/visualizeDownSampled_subplots.py -i input2D

