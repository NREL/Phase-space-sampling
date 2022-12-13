# Run the downsampling script
mpiexec -np 1 python main_iterative.py input2D_bins

# Plot the loss
python plotLoss.py input2D_bins

# Plot the sampling results
python visualizeDownSampled_subplots.py input2D_bins

