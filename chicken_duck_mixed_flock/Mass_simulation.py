import numpy as np
import pandas as pd

num_simu = 10 # number of total simulations, smaller number for testing, larger number for data generation.

######## Changable Values ########
# set number of species and flocks
num_flocks = 1 # One single flock, need to reflect in beta, sigma, and gamma 
num_species = 4 # chicken, sentinel chicken, vaccinated chicken, duck

tot_chicken_popul = 3000 # total chicken population
tot_duck_popul = 3000 # <---- set total duck population
vaccinated = 0 # <---- choose how many chickens to be vaccinated

######## Surveillance Strategy ########
surveillance = 30 # how many chickens are sentinel birds / how many chickens to randomly sample
testing_period = 7 # how many days to test the sentinel birds / do random testing.

######## Changable Values ########
# Here we may adjust the key parameters for the simulation
same_species_symptomatic_infection_rate = 1.13
same_species_asymptomatic_infection_rate = 1.07
different_species_symptomatic_infection_rate = 0.3
different_species_asymptomatic_infection_rate = 0.25

chicken_symptomatic_latency_period = 1
duck_symptomatic_latency_period = 1
chicken_asymptomatic_latency_period = 1
duck_asymptomatic_latency_period = 1

chicken_symptomatic_infectious_period = 3.2
duck_symptomatic_infectious_period = 3.2
chicken_asymptomatic_infectious_period = 4.9
duck_asymptomatic_infectious_period = 4.9

chicken_symptomatic_prob = 0.95
duck_symptomatic_prob = 0.05

# initialise the infection rate tensor
beta_S = np.zeros((num_flocks, num_species, num_flocks, num_species))

beta_S[:, :2, :, :2] = same_species_symptomatic_infection_rate # within-chicken infection
beta_S[:, :2, :, 3] = different_species_symptomatic_infection_rate # chicken-to-duck infection
beta_S[:, 3, :, :2] = different_species_asymptomatic_infection_rate # duck-to-chicken infection
beta_S[:, 3, :, 3] = same_species_symptomatic_infection_rate # within-duck infection

beta_A = np.zeros((num_flocks, num_species, num_flocks, num_species))

beta_A[:, :2, :, :2] = same_species_asymptomatic_infection_rate # within-chicken infection
beta_A[:, :2, :, 3] = different_species_asymptomatic_infection_rate # chicken-to-duck infection
beta_A[:, 3, :, :2] = different_species_asymptomatic_infection_rate # duck-to-chicken infection
beta_A[:, 3, :, 3] = same_species_asymptomatic_infection_rate # within-duck infection

# latency and infectious period
latency_period_S = np.array([chicken_symptomatic_latency_period, chicken_symptomatic_latency_period, 1, duck_symptomatic_latency_period])
sigma_S = 1 / latency_period_S
latency_period_A = np.array([chicken_asymptomatic_latency_period, chicken_asymptomatic_latency_period, 1, duck_asymptomatic_latency_period])
sigma_A = 1 / latency_period_A

infectious_period_S = np.array([chicken_symptomatic_infectious_period, chicken_symptomatic_infectious_period, 1, duck_symptomatic_infectious_period])
gamma_S = 1 / infectious_period_S
infectious_period_A = np.array([chicken_asymptomatic_infectious_period, chicken_asymptomatic_infectious_period, 1, duck_asymptomatic_infectious_period])
gamma_A = 1 / infectious_period_A

# probability of displaying symptoms
p_S = np.array([chicken_symptomatic_prob, chicken_symptomatic_prob, 0, duck_symptomatic_prob])
p_A = np.ones(num_species) - p_S

######## set initial conditions ######## 

# set initial conditions 

# the following convention will be used: first dimension will represent which flock, second is which species, third is the compartment.
init_val = np.zeros((num_flocks, num_species, 6)) # Six possible compartment: S, E_S, E_A, I_S, I_A, R

# first let all birds to start susceptible, also choose population size here.
init_val[0,0,0] += tot_chicken_popul # <---- set total chicken population
init_val[0,3,0] += tot_duck_popul # <---- set total duck population

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


######## Define update rules to be used in the Gillespie Algorithm ########

def S_to_E(a, b, current_val, symptomatic = True, tot_popul=tot_popul, beta_S=beta_S, beta_A=beta_A, p_S=p_S, p_A=p_A, num_flocks=num_flocks, num_species=num_species):
    val = 0
    for i in range(num_flocks):
        for j in range(num_species):
            val += (beta_S[i,j,a,b] * current_val[i,j,3] + beta_A[i,j,a,b] * current_val[i,j,4]) / np.sum(tot_popul[i,:])

    if symptomatic:
        val = val * current_val[a,b,0] * p_S[b]
    else:
        val = val * current_val[a,b,0] * p_A[b]
    return val

def E_to_I(a, b, current_val, symptomatic = True, tot_popul=tot_popul, sigma_S=sigma_S, sigma_A=sigma_A, num_flocks=num_flocks, num_species=num_species):
    if symptomatic:
        return current_val[a,b,1] * sigma_S[b]
    else:
        return current_val[a,b,2] * sigma_A[b]

def I_to_R(a, b, current_val, symptomatic = True, tot_popul=tot_popul, gamma_S=gamma_S, gamma_A=gamma_A, num_flocks=num_flocks, num_species=num_species):
    if symptomatic:
        return current_val[a,b,3] * gamma_S[b]
    else:
        return current_val[a,b,4] * gamma_A[b]
    

######## Gillespie Algorithm ########

def Gillespie_simu(max_events=max_events, init_val=init_val, tot_popul=tot_popul, 
                   beta_S=beta_S, beta_A=beta_A, sigma_S=sigma_S, sigma_A=sigma_A,
                   gamma_S=gamma_S, gamma_A=gamma_A, p_S=p_S, p_A=p_A, num_flocks=num_flocks, 
                   num_species=num_species):

    # initialise the event count and current values

    num_event = 0
    current_val = init_val.copy()

    # set the time and state sequence
    t = [0] + [None] * max_events
    y = [init_val] + [None] * max_events


    while (num_event < max_events) and (np.sum(current_val[:,:,1:5]) > 0): # stop the loop if: 1. maximum event number is reached, or 2. no more infections can possibly occur.
        
        num_event += 1

        ##### create an event tensor ####

        all_events = np.zeros((num_flocks, num_species, 6)) # six types of update rules in total
        for i in range(num_flocks):
            for j in range(num_species):
                all_events[i, j, 0] = S_to_E(i, j, current_val, True)
                all_events[i, j, 1] = S_to_E(i, j, current_val, False)
                all_events[i, j, 2] = E_to_I(i, j, current_val, True)
                all_events[i, j, 3] = E_to_I(i, j, current_val, False)
                all_events[i, j, 4] = I_to_R(i, j, current_val, True)
                all_events[i, j, 5] = I_to_R(i, j, current_val, False)


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

        for j in range(num_species):
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

    return t, y


######## Outbreak statistics ########
def outbreak_statistics(t, y, outbreak_threshold = 5):
    # Outbreak or not
    outbreak = len([state for state in y if np.sum(state[:,:,3:5]) >= outbreak_threshold]) > 0

    # if there is an outbreak, when does it happen
    if outbreak:
        outbreak_time = t[[i for i, state in enumerate(y) if np.sum(state[:,:,3:5]) >= outbreak_threshold][0]]
    else:
        outbreak_time = None

    # Species-specific outbreak
    outbreak_indiv = [len([state for state in y if np.sum(state[:,i,3:5]) >= outbreak_threshold]) > 0 for i in range(num_species)]
    outbreak_time_indiv = [t[[j for j, state in enumerate(y) if np.sum(state[:,i,3:5]) >= outbreak_threshold][0]] if outbreak_indiv[i] else None for i in range(num_species)]
    return outbreak, outbreak_time, outbreak_indiv, outbreak_time_indiv

######## Peak size statistics ########
def peak_size(t, y):
    ######## Peak size for all flocks as a whole ########
    peak_size_whole = np.max([np.sum(state[:,:,3:5]) for state in y])
    peak_time_whole = t[np.argmax([np.sum(state[:,:,3:5]) for state in y])]

    ######## Individual species peak size ########
    peak_size_indiv = [np.amax([np.sum(state[:,i,3:5]) for state in y], axis=0) for i in range(num_species)]
    peak_time_indiv = [t[j] for j in [np.argmax([np.sum(state[:,i,3:5]) for state in y], axis=0) for i in range(num_species)]]
    return peak_size_whole, peak_time_whole, peak_size_indiv, peak_time_indiv

######## Final size and end time ########
def final_size_end_time(t, y):
    ######## Final size ########
    final_size = np.sum(y[-1,:,:,-1], axis=0)
    end_time = t[-1]
    return final_size, end_time

######## Obtain time of surveillance outcomes ########

def sentinel_outcomes(t, y, testing_period=testing_period):

    testing_time = np.array(range(0, int(max(t)), testing_period))
    testing_index = np.zeros(len(testing_time))

    for test in range(len(testing_time)):
        i = np.searchsorted(t, testing_time[test], side='right') - 1
        testing_index[test] = i

    # Obtain the result of all testing:
    testing_result = [y[int(i), 0, 1, 3] for i in testing_index] # test int(i), flock 0, sentinel chicken, I_S
    detection_time = next((testing_time[i] for i, x in enumerate(testing_result) if x >= 1), None)

    return detection_time   

def random_sample_outcomes(t, y, surveillance=surveillance, testing_period=testing_period):

    testing_time = np.array(range(0, int(max(t)), testing_period))
    testing_index = np.zeros(len(testing_time))

    for test in range(len(testing_time)):
        i = np.searchsorted(t, testing_time[test], side='right') - 1
        testing_index[test] = i

    # Obtain the result of all testing:
    testing_result = [np.random.uniform() < np.sum(y[int(i), 0, 0:2, 3]) / np.sum(y[int(i), 0, 0:3, :]) for i in testing_index] # test int(i), flock 1, chicken, I_S
    detection_time = next((testing_time[i] for i, x in enumerate(testing_result) if x >= 1), None)

    return detection_time

######## Collect all statistics for a number of simulations ########
def mass_simu(num_simu, max_events=max_events, init_val=init_val, tot_popul=tot_popul, 
              beta_S=beta_S, beta_A=beta_A, sigma_S=sigma_S, sigma_A=sigma_A,
              gamma_S=gamma_S, gamma_A=gamma_A, p_S=p_S, p_A=p_A, num_flocks=num_flocks, 
              num_species=num_species):
    t_mass_simu = []
    y_mass_simu = []

    param = max_events, init_val, tot_popul, beta_S, beta_A, sigma_S, sigma_A, gamma_S, gamma_A, p_S, p_A, num_flocks, num_species
    result = list(map(lambda p: Gillespie_simu(*p), [param]*num_simu))

    t_mass_simu, y_mass_simu = zip(*result)
    return list(t_mass_simu), list(y_mass_simu)

def mass_outbreak_statistics(t_mass_simu, y_mass_simu, outbreak_threshold=5):
    simu = [(t_mass_simu[i], y_mass_simu[i], outbreak_threshold) for i in range(len(t_mass_simu))]
    result = list(map(lambda s: outbreak_statistics(*s), simu))

    mass_outbreak, mass_outbreak_time, mass_outbreak_indiv, mass_outbreak_time_indiv = zip(*result)

    return np.array(mass_outbreak), np.array(mass_outbreak_time), np.array(mass_outbreak_indiv), np.array(mass_outbreak_time_indiv)

def mass_peak_size(t_mass_simu, y_mass_simu):
    simu = [(t_mass_simu[i], y_mass_simu[i]) for i in range(len(t_mass_simu))]
    result = list(map(lambda s: peak_size(*s), simu))

    mass_peak_size_whole, mass_peak_time_whole, mass_peak_size_indiv, mass_peak_time_indiv = zip(*result)

    return np.array(mass_peak_size_whole), np.array(mass_peak_time_whole), np.array(mass_peak_size_indiv), np.array(mass_peak_time_indiv)

def mass_final_size_end_time(t_mass_simu, y_mass_simu):
    simu = [(t_mass_simu[i], y_mass_simu[i]) for i in range(len(t_mass_simu))]
    result = list(map(lambda s: final_size_end_time(*s), simu))

    mass_final_size, mass_end_time = zip(*result)

    return np.array(mass_final_size), np.array(mass_end_time)

def mass_sentinel_outcomes(t_mass_simu, y_mass_simu):
    simu = [(t_mass_simu[i], y_mass_simu[i]) for i in range(len(t_mass_simu))]
    result = list(map(lambda s: sentinel_outcomes(*s), simu))
    
    mass_detection_time = np.array(result)

    return mass_detection_time
    
def mass_random_sample_outcomes(t_mass_simu, y_mass_simu):
    simu = [(t_mass_simu[i], y_mass_simu[i]) for i in range(len(t_mass_simu))]
    result = list(map(lambda s: random_sample_outcomes(*s), simu))
    
    mass_detection_time = np.array(result)

    return mass_detection_time


######## Run the simulation ########
t_mass_simu, y_mass_simu = mass_simu(num_simu)

mass_outbreak, mass_outbreak_time, mass_outbreak_indiv, mass_outbreak_time_indiv = mass_outbreak_statistics(t_mass_simu, y_mass_simu)
mass_peak_size_whole, mass_peak_time_whole, mass_peak_size_indiv, mass_peak_time_indiv = mass_peak_size(t_mass_simu, y_mass_simu)
mass_final_size, mass_end_time = mass_final_size_end_time(t_mass_simu, y_mass_simu)
mass_detection_time_sentinel = mass_sentinel_outcomes(t_mass_simu, y_mass_simu)
mass_detection_time_random = mass_random_sample_outcomes(t_mass_simu, y_mass_simu)

######## Create a csv file to store the results ########
df = pd.DataFrame({'Outbreak': mass_outbreak,
                   'Outbreak Time': mass_outbreak_time,
                   'Outbreak Chicken': mass_outbreak_indiv[:, 0],
                   'Outbreak Duck': mass_outbreak_indiv[:, 3], 
                   'Outbreak Time Chicken': mass_outbreak_time_indiv[:, 0], 
                   'Outbreak Time Duck': mass_outbreak_time_indiv[:, 3], 
                   'Peak Size Whole': mass_peak_size_whole, 
                   'Peak Time Whole': mass_peak_time_whole, 
                   'Peak Size Chicken': np.sum(mass_peak_size_indiv[:, :3], axis=1), 
                   'Peak Time Chicken': np.sum(mass_peak_time_indiv[:, :3], axis=1), 
                   'Peak Size Duck': mass_peak_size_indiv[:, 3], 
                   'Peak Time Duck': mass_peak_time_indiv[:, 3], 
                   'Final Size Chicken': np.sum(mass_final_size[:, :3], axis=1), 
                   'Final Size Duck': mass_final_size[:, 3], 
                   'End Time': mass_end_time,
                   'Detection Time Sentinel': mass_detection_time_sentinel,
                   'Detection Time Random': mass_detection_time_random})

df.to_csv('Results.csv', index=False)