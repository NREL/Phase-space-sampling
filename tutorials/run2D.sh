# Run the downsampling script
mpiexec -np 4 python tests/test_parallel.py -i input2D

# Plot the loss
python postProcess/plotLoss.py -i input2D

# Plot the sampling results
python postProcess/visualizeDownSampled_subplots.py -i input2D

