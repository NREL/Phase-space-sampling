# Run the downsampling script
mpiexec -np 4 python main_iterative.py -i input2D_bins

# Plot the sampling results
python visualizeDownSampled_subplots.py -i input2D_bins

