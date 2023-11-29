# Run the downsampling script
mpiexec -np 4 python main_iterative.py -i input3D

# Plot the loss
python plotLoss.py -i input3D

# Plot the sampling results
python visualizeDownSampled_subplots.py -i input3D

