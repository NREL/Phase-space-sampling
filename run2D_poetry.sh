poetry update

# Run the downsampling script
poetry run mpiexec -np 4 python main_iterative.py input2D

# Plot the loss
poetry run python plotLoss.py input2D

# Plot the sampling results
poetry run python visualizeDownSampled_subplots.py input2D

