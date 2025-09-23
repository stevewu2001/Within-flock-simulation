import subprocess
import numpy as np



# Define the parameter grids

tot_chicken_popul = [0, 10, 20, 30, 40]
duck_symptomatic_probability = [0.0, 0.2, 0.4, 0.6, 0.8]
seed = 36750


# Loop through all combinations
for c in tot_chicken_popul:
    for d in duck_symptomatic_probability:
        # if c == 40 and d != 0.0:
        #     break

        print(f"Running with tot_chicken_popul={c}, duck_symptomatic_probability={d}")
        subprocess.run([
            "python", "../Mass_simulation.py",

            "--tot_chicken_popul", str(c),
            "--tot_no_bird", "40",        # total flock size
            "--duck_sym_prob", str(d),    # example value
            "--random_parameter_toggle", "0",
            "--frequency_dependent_toggle", "1",  # Use frequency-dependent transmission
            "--seed", str(seed)
        ])
