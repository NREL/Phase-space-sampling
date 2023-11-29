# Run the downsampling script
mpiexec -np 4 python main_iterative.py -i input2D

# Plot the loss
python phaseSpaceSampling/postProcess/plotLoss.py -i input2D

# Plot the sampling results
python phaseSpaceSampling/postProcess/visualizeDownSampled_subplots.py -i input2D

