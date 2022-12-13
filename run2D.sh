# Run the downsampling script
mpiexec -np 4 python main_iterative.py input2D

# Plot the loss
python plotLoss.py input2D

# Plot the sampling results
python visualizeDownSampled_subplots.py input2D

