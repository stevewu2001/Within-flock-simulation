import numpy as np
import h5py
import multiprocessing as mp
import argparse

num_simu = 10000 # number of total simulations, smaller number for testing, larger number for data generation.

def parse_args():
    parser = argparse.ArgumentParser(description="Set parameters for the simulation.")
    
    # Argument for total duck population
    parser.add_argument('--tot_duck_popul', type=int, default=3000, help='Total duck population') # <---- set total duck population (vary between [300, 1500, 3000, 5000])
    
    # Argument for the number of vaccinated chickens
    parser.add_argument('--vaccinated', type=int, default=0, help='Number of vaccinated chickens')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()

    # Use the parsed values for tot_duck_popul and vaccinated
    tot_duck_popul = args.tot_duck_popul
    vaccinated = args.vaccinated
    
    # Print values to check
    print(f"Total Duck Population: {tot_duck_popul}")
    print(f"Vaccinated Chickens: {vaccinated}")

######## Changable Values ########
# set number of species and flocks
num_flocks = 1 # One single flock, need to reflect in beta, sigma, and gamma 
num_types = 4 # chicken, sentinel chicken, vaccinated chicken, duck

tot_chicken_popul = 3000 # total chicken population

######## Surveillance Strategy ########
surveillance = 30 # how many chickens are sentinel birds / how many chickens to randomly sample
testing_period = 7 # how many days to test the sentinel birds / do random testing.

####### Here we may adjust the key parameters for the simulation ########
same_species_symptomatic_infection_rate = 1.13 
same_species_asymptomatic_infection_rate = 1.07
different_species_symptomatic_infection_rate = 0.3 # vary between [0.3, 0.6, 1.13]
different_species_asymptomatic_infection_rate = 0.25 # vary between [0.25, 0.5, 1.07]

chicken_symptomatic_latency_period = 1
duck_symptomatic_latency_period = 1
chicken_asymptomatic_latency_period = 1
duck_asymptomatic_latency_period = 1

chicken_symptomatic_infectious_period = 1.4
duck_symptomatic_infectious_period = 1.4
chicken_asymptomatic_infectious_period = 15
duck_asymptomatic_infectious_period = 15

chicken_symptomatic_prob = 0.95
duck_symptomatic_prob = 0.05

# the habitable area of one flock, for density dependence
farm_area = 5000


#####################################################################
#####################################################################

######## set initial conditions ######## 

# set initial conditions 

# the following convention will be used: first dimension will represent which flock, second is which species, third is the compartment.
init_val = np.zeros((num_flocks, num_types, 6)) # Six possible compartment: S, E_sym, E_asym, I_sym, I_asym, R

# first let all birds to start susceptible, also choose population size here.
init_val[:,0,0] += tot_chicken_popul # <---- set total chicken population
init_val[:,3,0] += tot_duck_popul # <---- set total duck population

init_val[0,0,0] -= surveillance
init_val[0,1,0] += surveillance # chicken under surveillance 

init_val[0,0,0] -= vaccinated
init_val[0,2,0] += vaccinated # vaccinated chicken

# store the total population for each flock and each species
tot_popul = init_val[:,:,0].copy()

# then choose a bird to be exposed (symptomatic), here we assume it to be a chicken
init_val[0,0,0] -= 1
init_val[0,0,1] += 1

# this is the maximum number of events that would occur, typically the number will not be reached, but 
# for diseases that does not die out this is necessary to not fall into an infinite while loop.
max_events = 500000


# initialise the infection rate tensor
beta_sym = np.zeros((num_flocks, num_types, num_flocks, num_types))

beta_sym[:, :2, :, :2] = same_species_symptomatic_infection_rate # within-chicken infection
beta_sym[:, :2, :, 3] = different_species_symptomatic_infection_rate # chicken-to-duck infection
beta_sym[:, 3, :, :2] = different_species_asymptomatic_infection_rate # duck-to-chicken infection
beta_sym[:, 3, :, 3] = same_species_symptomatic_infection_rate # within-duck infection

beta_asym = np.zeros((num_flocks, num_types, num_flocks, num_types))

beta_asym[:, :2, :, :2] = same_species_asymptomatic_infection_rate # within-chicken infection
beta_asym[:, :2, :, 3] = different_species_asymptomatic_infection_rate # chicken-to-duck infection
beta_asym[:, 3, :, :2] = different_species_asymptomatic_infection_rate # duck-to-chicken infection
beta_asym[:, 3, :, 3] = same_species_asymptomatic_infection_rate # within-duck infection

# the habitable area of one flock, for density dependence
farm_areas = np.ones(num_flocks) * farm_area

# for within-flock infection rate, divide by the habitable area of the flock. 
# for between-flock infection rate, divide by the total population of the flock (where the infection comes from).
for i in range(num_flocks):
    for j in range(num_flocks):
        if i == j:
            beta_sym[i, :, j, :] /= farm_areas[i]
            beta_asym[i, :, j, :] /= farm_areas[i]
        else:
            beta_sym[i, :, j, :] /= tot_popul[i]
            beta_asym[i, :, j, :] /= tot_popul[j]


# latency and infectious period
latency_period_sym = np.array([chicken_symptomatic_latency_period, chicken_symptomatic_latency_period, 1, duck_symptomatic_latency_period])
sigma_sym = 1 / latency_period_sym
latency_period_asym = np.array([chicken_asymptomatic_latency_period, chicken_asymptomatic_latency_period, 1, duck_asymptomatic_latency_period])
sigma_asym = 1 / latency_period_asym

infectious_period_sym = np.array([chicken_symptomatic_infectious_period, chicken_symptomatic_infectious_period, 1, duck_symptomatic_infectious_period])
gamma_sym = 1 / infectious_period_sym
infectious_period_sym = np.array([chicken_asymptomatic_infectious_period, chicken_asymptomatic_infectious_period, 1, duck_asymptomatic_infectious_period])
gamma_asym = 1 / infectious_period_sym

# probability of displaying symptoms
p_sym = np.array([chicken_symptomatic_prob, chicken_symptomatic_prob, 0, duck_symptomatic_prob])
p_asym = np.ones(num_types) - p_sym


######## Define update rules to be used in the Gillespie Algorithm ########

def S_to_E(current_val, symptomatic = True):
    S = current_val[:,:,0].copy()
    I_sym = current_val[:,:,3].copy()
    I_asym = current_val[:,:,4].copy()

    output = np.zeros((num_flocks, num_types))
    for a in range(num_flocks):
        for b in range(num_types):
            for i in range(num_flocks):
                output[a, b] += np.sum(beta_sym[i, :, a, b] * I_sym[i, :] + beta_asym[i, :, a, b] * I_asym[i, :]) * S[a, b]
                

            if symptomatic:
                output[a, b] *= p_sym[b]
            else:
                output[a, b] *= p_asym[b]
    return output

def E_to_I(current_val, symptomatic = True):
    E_sym = current_val[:,:,1].copy()
    E_asym = current_val[:,:,2].copy()
    if symptomatic:
        return E_sym * sigma_sym
    else:
        return E_asym * sigma_asym

def I_to_R(current_val, symptomatic = True):
    I_sym = current_val[:,:,3].copy()
    I_asym = current_val[:,:,4].copy()
    if symptomatic:
        return I_sym * gamma_sym
    else:
        return I_asym * gamma_asym


######## Gillespie Algorithm ########

def Gillespie_simu(max_events=max_events, init_val=init_val):

    # initialise the event count and current values

    num_event = 0
    current_val = init_val.copy()

    # set the time and state sequence
    t = [0] + [None] * max_events
    y = [init_val] + [None] * max_events


    while (num_event < max_events) and (np.sum(current_val[:,:,1:5]) > 0): # stop the loop if: 1. maximum event number is reached, or 2. no more infections can possibly occur.
        
        num_event += 1

        ##### create an event tensor ####

        all_events = np.zeros((num_flocks, num_types, 6)) # six types of update rules in total
        all_events[:,:,0] = S_to_E(current_val, True)
        all_events[:,:,1] = S_to_E(current_val, False)
        all_events[:,:,2] = E_to_I(current_val, True)
        all_events[:,:,3] = E_to_I(current_val, False)
        all_events[:,:,4] = I_to_R(current_val, True)   
        all_events[:,:,5] = I_to_R(current_val, False)

        # store total rate to rescale later
        tot_rate = np.sum(all_events)
        
        # do a time leap
        
        r1 = np.random.uniform()
        t[num_event] = t[num_event-1] - np.log(r1) / tot_rate
        
        # then choose events, first choose the type of events (S to E_S, S to E_A, E_S to I_S, E_A to I_A, I_S to R, or I_A to R)
        
        r2 = np.random.uniform()

        for event in range(6):
            if r2 < np.sum(all_events[:,:,0:event+1]) / tot_rate:
                type_event = event
                break

        # then choose which flock gets updated
        
        r3 = np.random.uniform()
        spec_event_rate = np.sum(all_events[:,:,type_event]) # total rate of a specific event occurring

        for i in range(num_flocks):
            if r3 < np.sum(all_events[0:i+1,:,type_event]) / spec_event_rate:
                flock_to_update = i
                break

        # finally choose which species get updated

        r4 = np.random.uniform()
        spec_event_flock_rate = np.sum(all_events[flock_to_update,:,type_event])

        for j in range(num_types):
            if r4 < np.sum(all_events[flock_to_update,0:j+1,type_event]) / spec_event_flock_rate:
                species_to_update = j
                break

        # do the updating
        if type_event == 0:
            current_val[flock_to_update, species_to_update, 0] -= 1
            current_val[flock_to_update, species_to_update, 1] += 1
        if type_event == 1:
            current_val[flock_to_update, species_to_update, 0] -= 1
            current_val[flock_to_update, species_to_update, 2] += 1
        if type_event == 2:
            current_val[flock_to_update, species_to_update, 1] -= 1
            current_val[flock_to_update, species_to_update, 3] += 1
        if type_event == 3:
            current_val[flock_to_update, species_to_update, 2] -= 1
            current_val[flock_to_update, species_to_update, 4] += 1
        if type_event == 4:
            current_val[flock_to_update, species_to_update, 3] -= 1
            current_val[flock_to_update, species_to_update, 5] += 1
        if type_event == 5:
            current_val[flock_to_update, species_to_update, 4] -= 1
            current_val[flock_to_update, species_to_update, 5] += 1

        # store the updated value

        y[num_event] = current_val.copy()

    # get rid of none value if there is any:
    t = np.array(t[0:num_event+1])
    y = np.array(y[0:num_event+1])

    return t, y # y format: [time, flock, species, compartment]


simu_params = [{
'num_simu': num_simu,
'num_flocks': num_flocks,
'num_species': num_types,
'tot_chicken_popul': tot_chicken_popul,
'tot_duck_popul': tot_duck_popul,
'vaccinated': vaccinated,
'surveillance': surveillance,
'testing_period': testing_period,
'same_species_symptomatic_infection_rate': same_species_symptomatic_infection_rate,
'same_species_asymptomatic_infection_rate': same_species_asymptomatic_infection_rate,
'different_species_symptomatic_infection_rate': different_species_symptomatic_infection_rate,
'different_species_asymptomatic_infection_rate': different_species_asymptomatic_infection_rate,
'chicken_symptomatic_latency_period': chicken_symptomatic_latency_period,
'duck_symptomatic_latency_period': duck_symptomatic_latency_period,
'chicken_asymptomatic_latency_period': chicken_asymptomatic_latency_period,
'duck_asymptomatic_latency_period': duck_asymptomatic_latency_period,
'chicken_symptomatic_infectious_period': chicken_symptomatic_infectious_period,
'duck_symptomatic_infectious_period': duck_symptomatic_infectious_period,
'chicken_asymptomatic_infectious_period': chicken_asymptomatic_infectious_period,
'duck_asymptomatic_infectious_period': duck_asymptomatic_infectious_period,
'chicken_symptomatic_prob': chicken_symptomatic_prob,
'duck_symptomatic_prob': duck_symptomatic_prob,
'farm_areas': farm_areas} for _ in range(num_simu)]

# File to save the simulations
output_file = 'simulation_results.h5'


