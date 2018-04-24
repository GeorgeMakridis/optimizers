'''
Initiates the parameters (r, alpha, a, b) using np.random.normal() function by picking 4 samples of normal
distribution with mean=1 and std =0.05. The normalization factor ==  10. Finally, implements the two optimizers
(nelder-mead and genetic algorithms) and saves the the optimal values of the parameters, among with the optimal
output of the objective function.
Nelder_mead or denetic algorithm can be chosen by setting the param :is_gen_preffered accordingly.
'''

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from MyCostFunction import CostFunction
    from Optimizer import Optimizer

    #:TODO here import the data

    sum_data = pd.read_csv('................')
    params_final = []
    scores = []

    params_final_gen = []
    scores_gen = []

    norm_factor = 10.0

    is_gen_preffered = True

    while 1:

        ini_r, ini_alpha, ini_a, ini_b = np.random.normal(1, 0.05, 4)

        cost_function = CostFunction(sum_data)
        optimizer = Optimizer(sum_data, np.array([ini_r, ini_alpha, ini_a, ini_b]), norm_factor)

        params_gen, score_gen =  optimizer.genetics(cost_function.neg_log_likelihood, sum_data, np.array([ini_r, ini_alpha, ini_a, ini_b]),
                                 norm_factor)

        params, score = optimizer.nelder_mead(cost_function.neg_log_likelihood, sum_data, np.array([ini_r, ini_alpha, ini_a, ini_b]),
                                    norm_factor)

        if not is_gen_preffered and score_gen<4.515:
            params_final.append(params)
            scores.append(score)
            break;
        if is_gen_preffered and score<4.515:
            params_final_gen.append(params_gen)
            scores_gen.append(score_gen)
            break;

    if not is_gen_preffered:
        df = pd.DataFrame(params_final,columns=['r','alpha','a','b'])
        df['scores'] = scores
        df.to_csv('input/estimated_parameters.csv')

    if is_gen_preffered:
        df = pd.DataFrame(params_final_gen,columns=['r','alpha','a','b'])
        df['scores'] = scores_gen
        df.to_csv('input/estimated_parameters_gen.csv')