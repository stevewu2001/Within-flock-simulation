import subprocess
from generate_param_grid import generate_param_grid
import numpy as np

seed = 42
np.random.seed(seed)
param_grid  = generate_param_grid(bins = 50) # shape (50,8)
np.save("param_grid.npy", param_grid)


for i in range(param_grid.shape[0]):
    subprocess.run([
        "python", "../Mass_simulation.py",
        "--tot_chicken_popul", "0",
        "--tot_no_bird", "40",        # total flock size
        "--duck_sym_prob", "0.2",    # example value
        "--random_parameter_toggle", "1",
        "--row_index", str(i),  # Pass the current row index
        "--frequency_dependent_toggle", "0"  # Use frequency-dependent transmission
    ])






