import numpy as np
import h5py
import multiprocessing as mp
import argparse
import os
from tqdm import tqdm

num_simu = 5000  # number of total simulations, adjust as needed

def parse_args():
    parser = argparse.ArgumentParser(description="Set parameters for the simulation.")
    parser.add_argument('--tot_chicken_popul', type=int, default=40, help='Total chicken population do not exceed total num of birds')
    parser.add_argument('--duck_sym_prob', type=float, default=0.0, help='Probability of a duck being symptomatic, from 0 to 1')
    parser.add_argument('--tot_no_bird', type=int, default=40, help='Total number of birds in a flock')
    parser.add_argument('--random_parameter_toggle', type=int, default=0, help='Whether to randomly sample parameters from distributions, choose 0 or 1')
    parser.add_argument('--row_index', type=int, default=0, help='Row index for parameter grid if random_parameter_toggle is 1')
    parser.add_argument('--frequency_dependent_toggle', type=int, default=0, help='Whether to use frequency-dependent transmission (1) or density-dependent (0)')
    parser.add_argument('--seed', type=int, default=36750, help='Random seed for reproducibility')
    return parser.parse_args()

# ---- constants that do NOT depend on args ----
num_flocks = 1
num_species = 4  # chicken, duck, vaccinated chicken, vaccinated duck
max_events = 50000

# ------------------- Definitions of Transitions -------------------
def S_to_E(current_val):
    S = current_val[:,:,0].copy()
    I = current_val[:,:,3].copy()
    I_asym = current_val[:,:,4].copy()

    if frequency_dependent_toggle == 1:
        tot_alive_birds = np.sum(current_val[:,:,:-1])
        denom = max(tot_alive_birds - 1, 1)  # avoid division by zero
        beta_eff = beta / denom
        beta_asym_eff = beta_asym / denom
    else:
        beta_eff = beta
        beta_asym_eff = beta_asym

    FoI = np.einsum('ijab,ij->ab', beta_eff, I) + np.einsum('ijab,ij->ab', beta_asym_eff, I_asym)
    weight = np.einsum('ab,ab->ab', FoI, S)
    weight_sym = np.einsum('ab,b->ab', weight, p)
    weight_asym = np.einsum('ab,b->ab', weight, q)
    return weight_sym, weight_asym

def E_to_I(current_val):
    E = current_val[:,:,1].copy()
    E_asym = current_val[:,:,2].copy()
    weight_sym = np.einsum('ab,b->ab', E, sigma)
    weight_asym = np.einsum('ab,b->ab', E_asym, sigma)
    return weight_sym, weight_asym

def I_to_RD(current_val):
    I = current_val[:,:,3].copy()
    I_asym = current_val[:,:,4].copy()
    weight_sym_recover = np.einsum('ab,b->ab', I, gamma * delta_comp)
    weight_asym_recover = np.einsum('ab,b->ab', I_asym, gamma_asym * delta_asym_comp)
    weight_sym_fatal = np.einsum('ab,b->ab', I, gamma * delta)
    weight_asym_fatal = np.einsum('ab,b->ab', I_asym, gamma_asym * delta_asym)
    return weight_sym_recover, weight_asym_recover, weight_sym_fatal, weight_asym_fatal

def Gillespie_simu(init_val, max_events=max_events):
    num_event = 0
    current_val = init_val.copy()
    t = [0] + [None] * max_events
    y = [init_val] + [None] * max_events

    while (num_event < max_events) and (np.sum(current_val[:,:,1:5]) > 0):
        num_event += 1
        all_events = np.zeros((num_flocks, num_species, 8))
        all_events[:,:,0], all_events[:,:,1] = S_to_E(current_val)
        all_events[:,:,2], all_events[:,:,3] = E_to_I(current_val)
        all_events[:,:,4], all_events[:,:,5], all_events[:,:,6], all_events[:,:,7] = I_to_RD(current_val)

        tot_rate = np.sum(all_events)
        if tot_rate <= 0:
            break

        r1 = np.random.uniform()
        t[num_event] = t[num_event-1] - np.log(r1) / tot_rate

        r2 = np.random.uniform()
        event_rate = np.einsum('abc->c', all_events)
        event_rate /= np.sum(event_rate)
        event_rate_cum = np.cumsum(event_rate)
        type_event = np.searchsorted(event_rate_cum, r2)

        r3 = np.random.uniform()
        flock_rate = np.einsum('ab->a', all_events[:,:,type_event])
        flock_rate /= np.sum(flock_rate)
        flock_rate_cum = np.cumsum(flock_rate)
        flock_to_update = np.searchsorted(flock_rate_cum, r3)

        r4 = np.random.uniform()
        species_rate = all_events[flock_to_update,:,type_event]
        species_rate /= np.sum(species_rate)
        species_rate_cum = np.cumsum(species_rate)
        species_to_update = np.searchsorted(species_rate_cum, r4)

        if type_event == 0:
            current_val[flock_to_update, species_to_update, 0] -= 1
            current_val[flock_to_update, species_to_update, 1] += 1
        elif type_event == 1:
            current_val[flock_to_update, species_to_update, 0] -= 1
            current_val[flock_to_update, species_to_update, 2] += 1
        elif type_event == 2:
            current_val[flock_to_update, species_to_update, 1] -= 1
            current_val[flock_to_update, species_to_update, 3] += 1
        elif type_event == 3:
            current_val[flock_to_update, species_to_update, 2] -= 1
            current_val[flock_to_update, species_to_update, 4] += 1
        elif type_event == 4:
            current_val[flock_to_update, species_to_update, 3] -= 1
            current_val[flock_to_update, species_to_update, 5] += 1
        elif type_event == 5:
            current_val[flock_to_update, species_to_update, 4] -= 1
            current_val[flock_to_update, species_to_update, 5] += 1
        elif type_event == 6:
            current_val[flock_to_update, species_to_update, 3] -= 1
            current_val[flock_to_update, species_to_update, 6] += 1
        elif type_event == 7:
            current_val[flock_to_update, species_to_update, 4] -= 1
            current_val[flock_to_update, species_to_update, 6] += 1

        y[num_event] = current_val.copy()

    t = np.array(t[0:num_event+1])
    y = np.array(y[0:num_event+1])
    return t, y

# set_init_val uses globals 'tot_no_bird', 'tot_chicken_popul', and 'q' which will be provided to workers
def set_init_val(set_popul):
    init_val = set_popul.copy()
    init_infect = np.random.randint(1, tot_no_bird+1)
    init_infect_is_duck = int(init_infect > tot_chicken_popul)
    init_infect_is_asym = int(np.random.uniform() < q[init_infect_is_duck])
    init_val[0,init_infect_is_duck,0] -= 1
    init_val[0,init_infect_is_duck,1+init_infect_is_asym] += 1
    return init_val

# Worker initializer: receives a dict of precomputed parameter arrays and sets them as module globals
def init_worker(params):
    globals().update(params)

def run_simulation(sim_id, base_seed):
    np.random.seed(base_seed + sim_id)  # deterministic per simulation
    t, y = Gillespie_simu(set_init_val(set_popul))
    return sim_id, t, y

# ------------------- MAIN -------------------
if __name__ == "__main__":
    args = parse_args()

    # Build all parameters (exact same logic as in original code) and collect them into a dict
    tot_chicken_popul = args.tot_chicken_popul
    duck_symptomatic_probability = args.duck_sym_prob
    tot_no_bird = args.tot_no_bird
    random_parameter_toggle = args.random_parameter_toggle
    row_index = args.row_index
    frequency_dependent_toggle = args.frequency_dependent_toggle
    seed = args.seed


    print(f"Total Number of Birds in a Flock: {tot_no_bird}")
    print(f"Total Chicken Population: {tot_chicken_popul}")
    print(f"Duck Symptomatic Probability: {duck_symptomatic_probability}")

    np.random.seed(seed)

    tot_duck_popul = tot_no_bird - tot_chicken_popul

    if random_parameter_toggle == 0:

        chicken_to_chicken_transmission_rate = 1.15
        chicken_latency_period = 0.24
        chicken_infectious_period = 2.1

        symptomatic_duck_to_duck_transmission_rate = 4.3
        duck_latency_period = 0.17
        symptomatic_duck_infectious_period = 4.7
        symptomatic_duck_case_fatality_probability = 0.7

        rescale_factor = 3.0

    else:

        param_grid = np.load("param_grid.npy")
        chicken_to_chicken_transmission_rate = param_grid[row_index, 0]
        chicken_latency_period = param_grid[row_index, 1]
        chicken_infectious_period = param_grid[row_index, 2]

        symptomatic_duck_to_duck_transmission_rate = param_grid[row_index, 3]
        duck_latency_period = param_grid[row_index, 4]
        symptomatic_duck_infectious_period = param_grid[row_index, 5]
        symptomatic_duck_case_fatality_probability = param_grid[row_index, 6]

        rescale_factor = param_grid[row_index, 7]

    chicken_symptomatic_probability = 1
    chicken_case_fatality_probability = 1


    asymptomatic_duck_to_duck_transmission_rate = symptomatic_duck_to_duck_transmission_rate / rescale_factor
    asymptomatic_duck_infectious_period = rescale_factor * symptomatic_duck_infectious_period
    asymptomatic_duck_case_fatality_probability = 0

    chicken_to_duck_transmission_rate = symptomatic_duck_to_duck_transmission_rate
    symptomatic_duck_to_chicken_transmission_rate = chicken_to_chicken_transmission_rate
    asymptomatic_duck_to_chicken_transmission_rate = (
        asymptomatic_duck_to_duck_transmission_rate * chicken_to_chicken_transmission_rate
        / symptomatic_duck_to_duck_transmission_rate
    )

    # Build beta tensors (same shapes)
    beta = np.zeros((num_flocks, num_species, num_flocks, num_species))
    beta[:, 0, :, 0] = chicken_to_chicken_transmission_rate
    beta[:, 1, :, 1] = symptomatic_duck_to_duck_transmission_rate
    beta[:, 0, :, 1] = chicken_to_duck_transmission_rate
    beta[:, 1, :, 0] = symptomatic_duck_to_chicken_transmission_rate

    beta_asym = np.zeros((num_flocks, num_species, num_flocks, num_species))
    beta_asym[:, 1, :, 1] = asymptomatic_duck_to_duck_transmission_rate
    beta_asym[:, 1, :, 0] = asymptomatic_duck_to_chicken_transmission_rate

    idx = np.arange(num_flocks)
    if args.frequency_dependent_toggle == 0:
        beta[idx, :, idx, :] /= tot_no_bird - 1
        beta_asym[idx, :, idx, :] /= tot_no_bird - 1
 
    latency_period = np.array([chicken_latency_period, duck_latency_period, chicken_latency_period, duck_latency_period])
    sigma = 1 / latency_period

    infectious_period = np.array([chicken_infectious_period, symptomatic_duck_infectious_period, chicken_infectious_period, symptomatic_duck_infectious_period])
    gamma = 1 / infectious_period
    infectious_period_asym = np.array([chicken_infectious_period, asymptomatic_duck_infectious_period, chicken_infectious_period, asymptomatic_duck_infectious_period])
    gamma_asym = 1 / infectious_period_asym

    p = np.array([chicken_symptomatic_probability, duck_symptomatic_probability, chicken_symptomatic_probability, duck_symptomatic_probability])
    q = 1 - p

    delta = np.array([chicken_case_fatality_probability, symptomatic_duck_case_fatality_probability, chicken_case_fatality_probability, symptomatic_duck_case_fatality_probability])
    delta_comp = 1 - delta
    delta_asym = np.array([chicken_case_fatality_probability, asymptomatic_duck_case_fatality_probability, chicken_case_fatality_probability, asymptomatic_duck_case_fatality_probability])
    delta_asym_comp = 1 - delta_asym

    set_popul = np.zeros((num_flocks, num_species, 7))
    set_popul[:,0,0] += tot_chicken_popul
    set_popul[:,1,0] += tot_duck_popul
    tot_popul = set_popul[:,:,0].copy()

    # Prepare params dict to pass to workers
    params = {
        'set_popul': set_popul,
        'beta': beta,
        'beta_asym': beta_asym,
        'sigma': sigma,
        'gamma': gamma,
        'gamma_asym': gamma_asym,
        'p': p,
        'q': q,
        'delta': delta,
        'delta_asym': delta_asym,
        'delta_comp': delta_comp,
        'delta_asym_comp': delta_asym_comp,
        'tot_no_bird': tot_no_bird,
        'tot_chicken_popul': tot_chicken_popul,
        'random_parameter_toggle': random_parameter_toggle,
        'frequency_dependent_toggle': frequency_dependent_toggle,
    }

    output_file = f'Raw_simu_files/chicken_{tot_chicken_popul}_symprob_{duck_symptomatic_probability}_totpop_{tot_no_bird}_rand_{random_parameter_toggle}_index_{row_index}_freq_{frequency_dependent_toggle}.h5'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Use Pool with initializer so worker processes receive parameter globals
    num_workers = min(mp.cpu_count(), num_simu)

    with mp.Pool(num_workers, initializer=init_worker, initargs=(params,)) as pool:
        results = list(tqdm(
            pool.starmap(run_simulation, [(i, seed) for i in range(num_simu)]),
            total=num_simu,
            desc="Running Simulations"
        ))
    # Save results
    with h5py.File(output_file, 'w') as f:
        f.attrs['total_chicken_population'] = tot_chicken_popul
        f.attrs['duck_symptomatic_probability'] = duck_symptomatic_probability
        f.attrs['num_simulations'] = num_simu
        for sim_id, t, y in results:
            sim_group = f.create_group(f"simulation_{sim_id+1}")
            sim_group.create_dataset("time", data=np.array(t))
            sim_group.create_dataset("state", data=np.array(y), compression="gzip")

    print("Simulation completed and saved!")
