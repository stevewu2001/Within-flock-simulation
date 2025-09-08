# batch_run.py
import subprocess

# Uncomment the following code to see sensitivity analysis for beta

# # # Define the parameter grids
# chicken_factors = [0.5, 1.0, 1.5]
# duck_factors = [0.5, 1.0, 1.5]
# total_chicken_population = [0, 20, 40]

# # Loop through all combinations
# for c in chicken_factors:
#     for d in duck_factors:
#         for t in total_chicken_population:
#             print(f"Running with chicken_factor={c}, duck_factor={d}, total_chicken_population={t}")
#             subprocess.run([
#                 "python", "Mass_simulation.py",
#                 "--chicken_transmission_sensitivity_factor", str(c),
#                 "--duck_transmission_sensitivity_factor", str(d),
#                 "--tot_chicken_popul", str(t),
#                 "--tot_no_bird", "40",        # total flock size
#                 "--duck_sym_prob", "0.2",    # example value
#                 "--asym_duck_param_rescale", "3.0"
#             ])

# Uncomment the following code to see sensitivity analysis for total bird population

# Define the parameter grids

total_population = [40, 400, 1000]
total_chicken_proportion = [0.0, 0.25, 0.5, 0.75, 1.0]

# Loop through all combinations
for t in total_population:
    for p in total_chicken_proportion:
        c = int(p * t)

        subprocess.run([
            "python", "Mass_simulation.py",
            "--chicken_transmission_sensitivity_factor", "1.0",
            "--duck_transmission_sensitivity_factor", "1.0",
            "--tot_chicken_popul", str(c),
            "--tot_no_bird", str(t),        # total flock size
            "--duck_sym_prob", "0.2",    # example value
            "--asym_duck_param_rescale", "3.0"
        ])
