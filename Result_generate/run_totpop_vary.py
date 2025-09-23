import subprocess
from generate_param_grid import generate_param_grid
import numpy as np

seed = 42
np.random.seed(seed)

totpop_range = range(20, 201, 20)  # Example: total population from 20 to 200 in steps of 20
chicken_popul_proportion = [0, 0.25, 0.5, 0.75, 1.0]  # Proportion of chickens in the total population

for t in totpop_range:
    for p in chicken_popul_proportion:
        tot_chicken_popul = int(t * p)
        subprocess.run([
        "python", "../Mass_simulation.py",
        "--tot_chicken_popul", str(tot_chicken_popul),
        "--tot_no_bird", str(t),        # total flock size
        "--duck_sym_prob", "0.2",    # example value
        "--random_parameter_toggle", "0",
        "--row_index", "0",  # Pass the current row index
        "--frequency_dependent_toggle", "0"  # Use frequency-dependent transmission
    ])








