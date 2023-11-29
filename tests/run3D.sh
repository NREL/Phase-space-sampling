# Run the downsampling script
mpiexec -np 4 python main.py -i input3D

# Plot the loss
python postProcess/plotLoss.py -i input3D

# Plot the sampling results
python postProcess/visualizeDownSampled_subplots.py -i input3D

