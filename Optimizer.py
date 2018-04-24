import copy
import random
import numpy as np


class Optimizer:
    """
        Optimizer class contains optimization functions. This version supports Nelder-Mead and Genetic algorithm.
    """

    def __init__(self, features, x_start, norm_factor):
        self.fearures = features
        self.x_start = x_start
        self.norm_factor = norm_factor

    def __cal_centroid(self, dim, solutions_dict):
        """
        Calculates the centroid.
        :param dim: (int) the dimension of objective function parameters which we should find in order to minimize the
        cost function
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :return: (array) the centroid of the given batch of solutions (except for the worst of them)
        """
        x0 = [0.] * dim
        for solution in solutions_dict[:-1]:
            for i, c in enumerate(solution):
                x0[i] += c / (len(solutions_dict) - 1)
        return x0

    def __reflection(self, x0, alpha, solutions_dict, f, features, norm_factor, scores):
        """
        Computes the reflected point. If the reflected point is better than the second worst, but not better than the
        best, then obtain a new simplex by replacing the worst point.
        :param x0: (array) the centroid
        :param alpha: (float) reflection coefficient
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param f: (function) the objective function
        :param features: (pandas DataFrame)
        :param norm_factor: (float) normalization factor used in objective function
        :param scores: (list) a list of scores of the given solutions_dict
        :return: a flag (bool), reflected point , reflected score, and the updated solutions_dict and scores
        """
        xr = x0 + alpha * (x0 - solutions_dict[-1])
        rscore = f(features, xr, norm_factor)
        if scores[0] <= rscore < scores[-2]:
            del scores[-1]
            del solutions_dict[-1]
            scores.append(rscore)
            solutions_dict.append(xr)
            return True, xr, rscore, solutions_dict, scores
        return False, xr, rscore, solutions_dict, scores

    def __expansion(self, x0, xr, rscore, gamma, solutions_dict, f, features, norm_factor, scores):
        """
        If the reflected point is the best point so far, then compute the expanded point. If the expanded point is
        better than the reflected point, then obtain a new simplex by replacing the worst point,
        else obtain a new simplex by replacing the worst point.
        :param x0: (array) the centroid
        :param xr: (array) reflected point
        :param rscore: (float) the output of objective function of reflected point
        :param gamma: (float) expansion coefficient
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param f: the objective function
        :param features: (pandas DataFrame)
        :param norm_factor: (float) normalization factor used in objective function
        :param scores: (list) a list of scores of the given solutions_dict
        :return: a flag (bool) and the updated solutions_dict and scores
        """
        if rscore < scores[0]:
            xe = x0 + gamma * (x0 - solutions_dict[-1])
            escore = f(features, xe, norm_factor)
            if escore < rscore:
                del scores[-1]
                del solutions_dict[-1]
                scores.append(escore)
                solutions_dict.append(xe)
                return True, solutions_dict, scores
            else:
                del scores[-1]
                del solutions_dict[-1]
                scores.append(rscore)
                solutions_dict.append(xr)
                return True, solutions_dict, scores
        return False, solutions_dict, scores

    def __contraction(self, x0, rho, solutions_dict, f, features, norm_factor, scores):
        """
        Here it is certain that reflected point is bigger than second worst point,Compute contracted point. If the
        contracted point is better than the worst point,then obtain a new simplex by replacing the worst point.
        :param x0:(array) the centroid
        :param rho: (float) contraction coefficient
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param f: the objective function
        :param features: (pandas DataFrame)
        :param norm_factor: (float) normalization factor used in objective function
        :param scores: (list) a list of scores of the given solutions_dict
        :return: a flag (bool) and the updated solutions_dict and scores
        """
        xc = x0 + rho * (solutions_dict[-1] - x0)
        cscore = f(features, xc, norm_factor)
        if cscore < scores[-1]:
            del scores[-1]
            del solutions_dict[-1]
            scores.append(cscore)
            solutions_dict.append(xc)
            return True, solutions_dict, scores
        return False, solutions_dict, scores

    def __shrink(self, sigma, solutions_dict, f, features, norm_factor):
        """
        Replace all points except the best
        :param sigma: (float) shrink coefficient
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param f: the objective function
        :param features: (pandas DataFrame)
        :param norm_factor: (float) normalization factor used in objective function
        :return: the updated solutions_dict and scores
        """
        x1 = solutions_dict[0]
        scores_new = []
        solutions_dict_new = []
        for solution in solutions_dict:
            redx = x1 + sigma * (solution - x1)
            score_2 = f(features, redx, norm_factor)
            scores_new.append(score_2)
            solutions_dict_new.append(redx)
        return solutions_dict_new, scores_new

    def nelder_mead(self, f, features, x_start, norm_factor,
                    step=0.1, no_improve_thr=10e-6,
                    no_improv_break=10, max_iter=1000,
                    alpha=1., gamma=2., rho=0.5, sigma=0.5):
        """
        Heavily Inspired by https://github.com/fchollet/nelder-mead implementation
        :param f: (function): function to optimize, must return a scalar score
                and operate over a numpy array of the same dimensions as x_start
        :param features: (pandas DataFrame)git commit -m "first commit"
        :param x_start: (numpy array) initial solution (position)
        :param norm_factor: normalization factor used in objective function
        :param step: (float): look-around radius in initial step
        :param no_improv_break, no_improve_thr: break after no_improv_break iterations with
                an improvement lower than no_improv_thr
        :param max_iter: (int): always break after this number of iterations
        :param alpha, gamma, rho, sigma: (float) parameters of the algorithm
        :return: (numpy array) optimal position (solution)
        """

        dim = len(x_start)
        prev_best = f(features, x_start, norm_factor)
        no_improv = 0
        solutions_dict = list()
        scores = list()
        solutions_dict.append(x_start)
        scores.append(prev_best)

        for i in range(dim):
            x = copy.copy(x_start)
            x[i] += step
            score = f(features, x, norm_factor)
            scores.append(score)
            solutions_dict.append(x)

        # simplex iter
        for i in range(max_iter):
            ind = np.argpartition(scores, range(dim))[:]
            solutions_dict_new = solutions_dict
            scores_new = scores
            scores = []
            solutions_dict = []
            for k in range(dim):
                solutions_dict.append(solutions_dict_new[ind[k]])
                scores.append(scores_new[ind[k]])

            best = scores[0]

            prev_best, no_improv = self.__update_previous_best(best, prev_best, no_improve_thr, no_improv)

            if no_improv >= no_improv_break:
                return solutions_dict[0], scores[0]

            x0 = self.__cal_centroid(dim, solutions_dict)

            reflection_flag, xr, rscore, solutions_dict, scores = self.__reflection(x0, alpha, solutions_dict, f,
                                                                                    features, norm_factor, scores)
            if reflection_flag:
                continue

            expansion_flag, solutions_dict, scores = self.__expansion(x0, xr, rscore, gamma, solutions_dict, f,
                                                                      features, norm_factor, scores)
            if expansion_flag:
                continue

            contraction_flag, solutions_dict, scores = self.__contraction(x0, rho, solutions_dict, f, features,
                                                                          norm_factor, scores)
            if contraction_flag:
                continue

            solutions_dict, scores = self.__shrink(sigma, solutions_dict, f, features, norm_factor)

        return solutions_dict[0], scores[0]
    def __create_random_initial_solutions(self, num_solutions, dim, ):
        """
        Creates the initial batch of random solutions.
        :param num_solutions: (int) number of solutions
        :param dim: (int) the dimension of objective function parameters which we should find in order to minimize the
        cost function
        :return: (numpy array) batch of random initial solutions
        """
        solutions_dict = dict()
        for i in range(num_solutions):
            solutions_dict[i] = np.random.normal(1, 0.05, dim)
        return solutions_dict

    def __scoring_the_solutions(self, num_solutions, solutions_dict, f, features, norm_factor):
        """
        Calculates the scores of each solution
        :param num_solutions: (int) size of batch of solutions
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param f: the objective function
        :param features: (pandas DataFrame)
        :param norm_factor: (float) normalization factor used in objective function
        :return: (array) scores of the given solution_dict
        """
        solutions_scores = np.zeros(num_solutions)
        for k in range(num_solutions):
            solution = solutions_dict[k]

            results = f(features, solution, norm_factor)

            solutions_scores[k] = results

        return solutions_scores

    def __update_previous_best(self, best, prev_best, no_improve_thr, no_improv):
        """
        Updates (or not) the previously saved best score if the new one is lower
        :param best: (float) the new score
        :param prev_best: (float) the previous best score
        :param no_improve_thr: (float) the
        :param no_improv: (int) number of consecutive no improvements
        :return: updated prev_best and no_improv
        """
        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        return prev_best, no_improv

    def __selection(self, solutions_scores, NUMBER_OF_SELECTED_SOLUTIONS, solutions_dict):
        """
        Selects half of the solutions with the smallest score, using np.argpartition which returns the sorted indexes
        ##TODO: implement a temperature mode for selecting solutions for the next phase
        :param solutions_scores: (array) the scores of the given solutions
        :param NUMBER_OF_SELECTED_SOLUTIONS: (int) number of the solution for the next phase
        :param solutions_dict: (array) the solutions which are gonna be parents
        :return: sorted solutions_dict
        """

        ind = np.argpartition(solutions_scores, -NUMBER_OF_SELECTED_SOLUTIONS)[:NUMBER_OF_SELECTED_SOLUTIONS]
        solutions_dict_new = solutions_dict
        for a in range(NUMBER_OF_SELECTED_SOLUTIONS):
            solutions_dict[a] = solutions_dict_new[ind[a]]
        return solutions_dict

    def __crossover(self, NUMBER_OF_SELECTED_SOLUTIONS, num_solutions, solutions_dict, alpha):
        """
        Implements arithmetic crossover.
        :param NUMBER_OF_SELECTED_SOLUTIONS: (int) number of the solution  that are gooing to be parents
        :param num_solutions: (int) size of batch of solutions
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param alpha: (float) coeficient default 0.5
        :return: (array) of possible solutions consisted of parents and childs
        """
        for b in range(NUMBER_OF_SELECTED_SOLUTIONS, num_solutions, 2):
            solutions_dict[b] = alpha * solutions_dict[b - NUMBER_OF_SELECTED_SOLUTIONS] + (1 - alpha) * solutions_dict[
                b - NUMBER_OF_SELECTED_SOLUTIONS + 1]
            solutions_dict[b + 1] = alpha * solutions_dict[b - NUMBER_OF_SELECTED_SOLUTIONS + 1] + (1 - alpha) * \
                                                                                                   solutions_dict[
                                                                                                       b - NUMBER_OF_SELECTED_SOLUTIONS]
        return solutions_dict

    def __mutation(self, num_solutions, solutions_dict, dim, mutation_thr):
        """
        :param num_solutions: (int) size of batch of solutions
        :param solutions_dict: (list) a list of possible solutions of our cost function
        :param dim: (int) size of a solution
        :param mutation_thr: (float) mutation threshold expressing the possibility of a mutation
        :return: (array) of after mutation - possible solutions
        """
        for c in range(num_solutions):
            mutation_random = random.uniform(0, 1)
            mutation_index = random.randint(0, dim - 1)
            if mutation_random < mutation_thr:
                solutions_dict[c][mutation_index] += np.random.normal(0, 0.15, 1)
        return solutions_dict

    def genetics(self, f, features, params, norm_factor,
                 no_improve_thr=10e-6, num_solutions=20,
                 no_improv_break=10, max_iter=1000,
                 mutation_thr=0.15, alpha=0.5):
        """
        In a genetic algorithm, a population of candidate solutions to an optimization problem is evolved toward better
        solutions. Each candidate solution has a set of properties which can be mutated and altered;
        It is implemented a real code encodings. Initialization , selection, crossover and mutation are implemented and
        both max iterrations and early stopping (successive iterations no longer produce better results) as termination
        criteria are implemented.

        :param f: (function): function to optimize, must return a scalar score
                and operate over a numpy array of the same dimensions as x_start
        :param features: (pandas DataFrame)
        :param params: (numpy array) initial solution (position) only used for getting the size of a solution
        :param norm_factor: normalization factor used in objective function
        :param step: (float): look-around radius in initial step
        :param no_improv_break, no_improve_thr: break after no_improv_break iterations with
                an improvement lower than no_improv_thr
        :param max_iter: (int): always break after this number of iterations
        :param mutation_thr: (float) mutation threshold expressing the possibility of a mutation
        :param alpha: (float) crossover coefficient
        :return:the best solution , and the its score
        """

        dim = len(params)
        no_improv = 0

        NUMBER_OF_SELECTED_SOLUTIONS = num_solutions / 2

        solutions_dict = self.__create_random_initial_solutions(num_solutions, dim)

        for i in range(max_iter):

            solutions_scores = self.__scoring_the_solutions(num_solutions, solutions_dict, f, features, norm_factor)

            best = solutions_scores.mean()

            if i == 0:
                prev_best = best

            prev_best, no_improv = self.__update_previous_best(best, prev_best, no_improve_thr, no_improv)

            if no_improv >= no_improv_break:
                ind = np.argpartition(solutions_scores, 0)
                solutions_dict_new = solutions_dict

                solutions_dict[0] = solutions_dict_new[ind[0]]
                return solutions_dict[0], solutions_scores[ind[0]]

            solutions_dict = self.__selection(solutions_scores, NUMBER_OF_SELECTED_SOLUTIONS, solutions_dict)

            solutions_dict = self.__crossover(NUMBER_OF_SELECTED_SOLUTIONS, num_solutions, solutions_dict, alpha)

            solutions_dict = self.__mutation(num_solutions, solutions_dict, dim, mutation_thr)

        ind = np.argpartition(solutions_scores, 0)
        solutions_dict_new = solutions_dict

        solutions_dict[0] = solutions_dict_new[ind[0]]
        return solutions_dict[0], solutions_scores[ind[0]]