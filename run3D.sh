# Run the downsampling script
mpiexec -np 4 python main_iterative.py input3D

# Plot the loss
python plotLoss.py input3D

# Plot the sampling results
python visualizeDownSampled_subplots.py input3D

