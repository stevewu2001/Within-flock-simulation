import numpy as np
from scipy.stats import qmc, uniform


def generate_param_grid(bins = 100):

    num_params = 8 # 8 parameters to do LHS on, if edited, also edit the param_grid creation below

    # LHS in [0,1]^n
    sampler = qmc.LatinHypercube(d=num_params)
    lhs_unit = sampler.random(n=bins)  # shape (100,8)

    
    chicken_to_chicken_transmission_rate = uniform(1.02, 1.30 - 1.02).ppf(lhs_unit[:,0])
    chicken_latency_period = uniform(0.0099, 0.48 - 0.0099).ppf(lhs_unit[:,1])
    chicken_infectious_period = uniform(1.8, 2.3 - 1.8).ppf(lhs_unit[:,2])

    symptomatic_duck_to_duck_transmission_rate = uniform(2.8, 5.8 - 2.8).ppf(lhs_unit[:,3])
    duck_latency_period = uniform(0.03, 0.38 - 0.03).ppf(lhs_unit[:,4])
    symptomatic_duck_infectious_period = uniform(2.8, 5.7 - 2.8).ppf(lhs_unit[:,5])
    symptomatic_duck_case_fatality_probability = uniform(0.61, 0.78 - 0.61).ppf(lhs_unit[:,6])
    asym_duck_param_rescale = uniform(1.0, 5.0 - 1.0).ppf(lhs_unit[:,7])

    param_grid = np.column_stack((
        chicken_to_chicken_transmission_rate,
        chicken_latency_period,
        chicken_infectious_period,
        symptomatic_duck_to_duck_transmission_rate,
        duck_latency_period,
        symptomatic_duck_infectious_period,
        symptomatic_duck_case_fatality_probability,
        asym_duck_param_rescale
    ))

    print(np.shape(param_grid))  # Should be (bins, 8)

    return param_grid

