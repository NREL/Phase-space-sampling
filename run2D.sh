# Run the downsampling script
mpiexec -np 4 python main_iterative.py input

# Plot the loss
python plotLoss.py input

# Plot the sampling results
python visualizeDownSampled_subplots.py input

